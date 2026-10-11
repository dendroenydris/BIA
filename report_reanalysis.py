"""Derive complete descriptive and diagnostic tables from a verified image run.

Primary contrasts are unchanged. Bootstrap and directional sign tests are
exploratory sensitivity checks, not a search for replacement significance.
"""
import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats

from bias_analysis import ROOT, STRATEGIES, holm, load_completed_evaluation, parity_mad, prompt_comparisons, write_csv

METRICS = ["label_imbalance_mad", "concept_similarity", "reference_similarity"]


def report(input_dir, output_dir):
    records = load_completed_evaluation(input_dir / "per_image.jsonl", ROOT / "datasets/bias_concepts.json", ROOT / "datasets/prompts.csv")
    concepts = json.loads((ROOT / "datasets/bias_concepts.json").read_text())
    if output_dir.exists():
        raise FileExistsError("Use a new report directory; source evidence is never overwritten")
    output_dir.mkdir(parents=True)
    groups, cells = defaultdict(list), defaultdict(list)
    for row in records:
        groups[(row["category"], row["strategy"])].append(row)
        cells[(row["category"], row["prompt_id"], row["strategy"])].append(row)
    cell_rows, values = [], {}
    for (axis, prompt_id, strategy), images in sorted(cells.items()):
        labels = concepts[axis]["Parity"]["concepts"]
        proportions = {label: sum(r["parity_label"] == label for r in images) / 5 for label in labels}
        row = {"category": axis, "prompt_id": prompt_id, "baseline_prompt": images[0]["baseline_prompt"],
               "strategy": STRATEGIES[strategy], "image_n": 5,
               "label_imbalance_mad": parity_mad(proportions, labels),
               **{m: mean(r[m] for r in images) for m in METRICS[1:]}}
        cell_rows.append(row)
        values[(axis, prompt_id, strategy)] = row
    descriptions, proportions_out, subtypes = [], [], []
    for (axis, strategy), images in sorted(groups.items()):
        labels = concepts[axis]["Parity"]["concepts"]
        counts = Counter(r["parity_label"] for r in images)
        prompt_rows = [r for (a, p, s), r in values.items() if a == axis and s == strategy]
        row = {"category": axis, "strategy": STRATEGIES[strategy], "prompt_n": len(prompt_rows),
               "image_n": len(images), "pooled_label_mad": parity_mad({l: counts[l] / len(images) for l in labels}, labels)}
        for metric in METRICS:
            observed = [r[metric] for r in prompt_rows]
            row[metric + "_mean"] = mean(observed)
            row[metric + "_sd_prompts"] = stdev(observed)
        descriptions.append(row)
        for label in labels:
            proportions_out.append({"category": axis, "strategy": STRATEGIES[strategy], "label": label,
                                    "count": counts[label], "image_n": len(images), "proportion": counts[label] / len(images)})
        for key in sorted(images[0]["scores"]):
            operation = mean if key.startswith("Embedding Distance") else max
            subtypes.append({"category": axis, "strategy": STRATEGIES[strategy], "subtype": key,
                             "value": mean(operation(r["scores"][key]["cosine_similarities"]) for r in images)})
    primary = prompt_comparisons(records, concepts)
    diagnostics, differences = [], []
    rng = np.random.default_rng(20261003)
    all_d = {}
    for index, contrast in enumerate(primary, 1):
        axis, metric = contrast["category"], contrast["metric"]
        b_name, a_name = contrast["comparison"].split(" minus ")
        a = next(s for s, name in STRATEGIES.items() if name == a_name)
        b = next(s for s, name in STRATEGIES.items() if name == b_name)
        prompt_ids = sorted(p for ax, p, s in values if ax == axis and s == a)
        d = np.array([values[(axis, p, b)][metric] - values[(axis, p, a)][metric] for p in prompt_ids])
        all_d[index] = d
        positive, negative = int(np.sum(d > 1e-12)), int(np.sum(d < -1e-12))
        zeros = len(d) - positive - negative
        sign_p = stats.binomtest(positive, positive + negative, .5).pvalue if positive + negative else 1.
        normality = stats.shapiro(d) if np.ptp(d) > 1e-12 else None
        # Resample paired prompt differences, never individual images or reference phrases.
        bootstrap = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
        low, high = np.quantile(bootstrap, [.025, .975])
        row = {"comparison_id": index, "category": axis, "comparison": contrast["comparison"], "metric": metric,
               "prompt_n": len(d), "positive": positive, "negative": negative, "zero": zeros,
               "shapiro_W": float(normality.statistic) if normality else "undefined_constant",
               "shapiro_p": float(normality.pvalue) if normality else "undefined_constant",
               "skewness": float(stats.skew(d, bias=False)) if normality else "undefined_constant",
               "bootstrap_mean_ci95_low": float(low), "bootstrap_mean_ci95_high": float(high),
               "bootstrap_method": "paired prompt percentile; 20000 resamples; exploratory; unadjusted",
               "directional_sign_p": float(sign_p), "directional_sign_p_holm": 0., "sign_family_size": 36}
        diagnostics.append(row)
        for p, delta in zip(prompt_ids, d):
            differences.append({"comparison_id": index, "category": axis, "comparison": contrast["comparison"],
                                "metric": metric, "prompt_id": p, "difference": float(delta)})
    assert len(primary) == len(diagnostics) == 36
    for row, adjusted in zip(diagnostics, holm([r["directional_sign_p"] for r in diagnostics])):
        row["directional_sign_p_holm"] = adjusted
    for name, rows in [("prompt_cells.csv", cell_rows), ("condition_descriptives.csv", descriptions),
                       ("label_proportions.csv", proportions_out), ("subtype_means.csv", subtypes),
                       ("primary_comparisons.csv", primary), ("contrast_diagnostics.csv", diagnostics),
                       ("paired_differences.csv", differences)]:
        write_csv(output_dir / name, rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_dir / "paired_difference_QQ.pdf", metadata={"Author": "", "Title": "Paired-difference diagnostics"}) as pdf:
        for axis in sorted(groups.keys()):
            if axis[1] != 0:
                continue
            selected = [r for r in diagnostics if r["category"] == axis[0]]
            fig, axs = plt.subplots(3, 3, figsize=(8.3, 10.5))
            for ax, row in zip(axs.flat, selected):
                d = all_d[row["comparison_id"]]
                (theoretical, observed), (slope, intercept, _) = stats.probplot(d)
                ax.scatter(theoretical, observed, s=13, color="#264b6a")
                ax.plot(theoretical, intercept + slope * theoretical, color="#8e635b", linewidth=.8)
                ax.set_title(f"C{row['comparison_id']:02d}: {row['metric']}\n{row['comparison']}", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.set_xlabel("Normal quantile", fontsize=7)
                ax.set_ylabel("Observed paired difference", fontsize=7)
            fig.suptitle(axis[0] + " — exploratory assumption diagnostics", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .96))
            pdf.savefig(fig)
            plt.close(fig)
    summary = {"status": "complete", "source_per_image_sha256": hashlib.sha256((input_dir / "per_image.jsonl").read_bytes()).hexdigest(),
               "source_manifest_sha256": hashlib.sha256((input_dir / "evaluation_manifest.json").read_bytes()).hexdigest(),
               "prompt_cells": len(cell_rows), "comparisons": 36, "bootstrap_seed": 20261003,
               "bootstrap_resamples": 20000, "bootstrap_unit": "paired original prompt differences",
               "primary_holm_below_05": [i + 1 for i, r in enumerate(primary) if r["p_holm"] < .05],
               "sign_sensitivity_holm_below_05": [r["comparison_id"] for r in diagnostics if r["directional_sign_p_holm"] < .05],
               "shapiro_below_05": [r["comparison_id"] for r in diagnostics if isinstance(r["shapiro_p"], float) and r["shapiro_p"] < .05],
               "interpretation": "exploratory selected-prompt inference; not population sampling, randomized causal inference, or an ethical fairness test"}
    (output_dir / "report_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.input_dir, args.output_dir)
