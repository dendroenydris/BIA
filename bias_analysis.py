"""Audit saved aggregates. Inference requires independent prompt/image records."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
STRATEGIES = {0: "Baseline", 1: "Bias-aware", 2: "LLM-assisted"}


def parity_mad(proportions, labels):
    """MAD from a fixed uniform label reference; absent labels retain zero mass."""
    if not labels or len(labels) != len(set(labels)) or set(proportions) - set(labels):
        raise ValueError("Invalid label set")
    if any(not math.isfinite(p) or p < 0 for p in proportions.values()):
        raise ValueError("Invalid proportion")
    if not math.isclose(sum(proportions.values()), 1, abs_tol=1e-8):
        raise ValueError("Proportions must sum to one")
    return mean(abs(proportions.get(label, 0) - 1 / len(labels)) for label in labels)


def write_csv(path, rows):
    if not rows:
        raise ValueError("No records to export")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def legacy_summary(concepts):
    rows, labels_out, subtypes = [], [], []
    for strategy, name in STRATEGIES.items():
        metrics = json.loads((ROOT / f"evaluations/results_strategy_{strategy}.json").read_text())
        for category, values in metrics.items():
            labels = concepts[category]["Parity"]["concepts"]
            proportions = values["Parity"]
            observed = list(proportions.values())
            rows.append({"category": category, "strategy": name, "fixed_label_count": len(labels),
                         "legacy_mad_observed_labels": mean(abs(p - 1 / len(observed)) for p in observed),
                         "label_imbalance_mad": parity_mad(proportions, labels),
                         "concept_similarity": mean(mean(v) for k, v in values.items() if k.startswith("Embedding Distance")),
                         "reference_similarity": mean(v for k, v in values.items() if k.startswith("Stereotype Freq")),
                         "image_n": "not saved", "prompt_n": "not saved", "inference": "descriptive only"})
            labels_out.extend({"category": category, "strategy": name, "label": label, "proportion": proportions.get(label, 0)} for label in labels)
            subtypes.extend({"category": category, "strategy": name, "legacy_metric": k, "value": mean(v) if isinstance(v, list) else v} for k, v in values.items() if k != "Parity")
    return rows, labels_out, subtypes


def holm(pvalues):
    adjusted, previous = [0.] * len(pvalues), 0.
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, min(1., (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def load_completed_evaluation(path, concepts_path, prompts_path):
    """Reject partial runs and complete-cell subsets before empirical inference."""
    manifest_path = path.with_name("evaluation_manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Completed evaluator manifest required; refusing inference from an unverified JSONL")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("record_count") != 645
            or manifest.get("evaluated_unique_images") != 465 or manifest.get("generated_images") != 600
            or manifest.get("excluded_age_images") != 135):
        raise ValueError("Evaluation manifest does not describe the complete study")
    blob = path.read_bytes()
    if manifest.get("per_image_sha256") != hashlib.sha256(blob).hexdigest():
        raise ValueError("Image records do not match the completed run's digest")
    if manifest.get("concepts_sha256") != hashlib.sha256(concepts_path.read_bytes()).hexdigest():
        raise ValueError("Evaluator inventory changed since evaluation")
    if manifest.get("prompts_sha256") != hashlib.sha256(prompts_path.read_bytes()).hexdigest():
        raise ValueError("Prompt inputs changed since evaluation")
    records = [json.loads(line) for line in blob.decode().splitlines() if line.strip()]
    with prompts_path.open(encoding="utf-8-sig") as handle:
        prompts = list(csv.DictReader(handle, delimiter=";"))
    expected = Counter()
    for row in prompts:
        main = row["Main_Category"]
        if main == "Age Bias":
            continue
        axes = ["Skin Tone Bias", "Ethnic Representation Bias"] if main == "Racial Bias" else [main]
        for axis in axes:
            for strategy in STRATEGIES:
                expected[(axis, row["Neutral_prompt"].strip(), strategy)] = 5
    actual, images, seen = Counter(), {}, set()
    for row in records:
        axis, image_id = row["category"], row["image_id"]
        main = "Racial Bias" if axis in ["Skin Tone Bias", "Ethnic Representation Bias"] else axis
        baseline, strategy = row["baseline_prompt"].strip(), int(row["strategy"])
        expected_id = hashlib.sha256((main + "\0" + baseline).encode()).hexdigest()[:16]
        if row["prompt_id"] != expected_id or (axis, image_id) in seen:
            raise ValueError("Inconsistent prompt identity or duplicate category-image record")
        seen.add((axis, image_id))
        identity = (main, baseline, strategy, row["image_sha256_pixels"])
        if image_id in images and images[image_id] != identity:
            raise ValueError("An image ID changes identity between evaluator axes")
        images[image_id] = identity
        actual[(axis, baseline, strategy)] += 1
    if actual != expected or len(images) != 465:
        raise ValueError("Image records omit or change the full prompt-condition design")
    return records


def prompt_comparisons(records, concepts):
    """Pair original prompts, averaging images first; never treat concepts as replicates."""
    from scipy.stats import t
    cells, seen = defaultdict(list), set()
    for row in records:
        cat, strategy = row["category"], int(row["strategy"])
        if cat not in concepts or strategy not in STRATEGIES:
            raise ValueError("Unknown category or condition")
        if row["parity_label"] not in concepts[cat]["Parity"]["concepts"]:
            raise ValueError("Unknown image label")
        if not all(math.isfinite(row[m]) for m in ["concept_similarity", "reference_similarity"]):
            raise ValueError("Nonfinite score")
        key = (cat, row["prompt_id"], strategy, row["image_id"])
        seen_key = (cat, row["image_id"])
        if seen_key in seen:
            raise ValueError("Duplicate image record")
        seen.add(seen_key)
        cells[key[:3]].append(row)
    values = {}
    for key, images in cells.items():
        if len(images) != 5:
            raise ValueError(f"Expected five images per prompt-condition: {key}")
        labels = concepts[key[0]]["Parity"]["concepts"]
        props = {label: sum(i["parity_label"] == label for i in images) / 5 for label in labels}
        values[key] = {"label_imbalance_mad": parity_mad(props, labels),
                       **{m: mean(i[m] for i in images) for m in ["concept_similarity", "reference_similarity"]}}
    rows = []
    for cat in sorted({key[0] for key in cells}):
        prompts = sorted({key[1] for key in cells if key[0] == cat})
        if any((cat, p, s) not in cells for p in prompts for s in STRATEGIES):
            raise ValueError("Missing conditions; refusing to silently drop prompts")
        if not 2 <= len(prompts) <= 15:
            raise ValueError("Exact test supports 2 to 15 prompts per category")
        for a, b in itertools.combinations(STRATEGIES, 2):
            for metric in ["label_imbalance_mad", "concept_similarity", "reference_similarity"]:
                d = [values[(cat, p, b)][metric] - values[(cat, p, a)][metric] for p in prompts]
                effect, sd, n = mean(d), stdev(d), len(d)
                pval = sum(abs(mean(v * s for v, s in zip(d, signs))) >= abs(effect) - 1e-12 for signs in itertools.product([-1, 1], repeat=n)) / 2**n
                half_ci = t.ppf(.975, n - 1) * sd / math.sqrt(n)
                rows.append({"category": cat, "comparison": f"{STRATEGIES[b]} minus {STRATEGIES[a]}",
                             "metric": metric, "unit": "original prompt", "prompt_n": n, "image_n_per_condition": n * 5,
                             "mean_difference": effect, "sd_prompt_difference": sd,
                             "ci95_low": effect - half_ci, "ci95_high": effect + half_ci,
                             "ci_method": "paired t; unadjusted; assumes normal prompt differences", "df_ci": n - 1,
                             "cohen_dz": effect / sd if sd else "undefined", "test": "exact two-sided sign flip",
                             "statistic": effect, "p_raw": pval, "p_holm": 0., "family_size": 0})
    for row, pval in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"], row["family_size"] = pval, len(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image", type=Path, help="JSONL from revised eval.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluations/revised")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    concepts = json.loads((ROOT / "datasets/bias_concepts.json").read_text())
    if args.per_image:
        records = load_completed_evaluation(args.per_image, ROOT / "datasets/bias_concepts.json", ROOT / "datasets/prompts.csv")
        if (args.output_dir / "prompt_comparisons.csv").exists():
            raise FileExistsError("Will not overwrite an existing inferential table")
        write_csv(args.output_dir / "prompt_comparisons.csv", prompt_comparisons(records, concepts))
    else:
        for name, rows in zip(["legacy_descriptive.csv", "legacy_label_proportions.csv", "legacy_subtypes.csv"], legacy_summary(concepts)):
            write_csv(args.output_dir / name, rows)
        print("Descriptive audit saved. Legacy p-values are invalid; no replacement significance inferred.")


if __name__ == "__main__":
    main()
