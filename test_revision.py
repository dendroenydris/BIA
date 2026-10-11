"""Small regression check: scientific units, zero labels, and prompt extraction."""
import math
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from bias_analysis import parity_mad, holm, prompt_comparisons, load_completed_evaluation
from eval import reference_texts, validate_design, file_sha256, load_study_dataset
from sample import extract_revised_prompt


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Invalid input was accepted")


def completion_gate_fixture():
    """Even a partial stream ending at a whole cell must not support inference."""
    root = Path(__file__).resolve().parent
    concepts_path, prompts_path = root / "datasets/bias_concepts.json", root / "datasets/prompts.csv"
    with prompts_path.open(encoding="utf-8-sig") as handle:
        prompts = list(csv.DictReader(handle, delimiter=";"))
    records, image_id = [], 0
    for row in prompts:
        main, baseline = row["Main_Category"], row["Neutral_prompt"].strip()
        prompt_id = hashlib.sha256((main + "\0" + baseline).encode()).hexdigest()[:16]
        for strategy in range(3):
            for i in range(5):
                if main != "Age Bias":
                    axes = ["Skin Tone Bias", "Ethnic Representation Bias"] if main == "Racial Bias" else [main]
                    for axis in axes:
                        records.append({"category": axis, "image_id": image_id, "baseline_prompt": baseline,
                                        "strategy": strategy, "prompt_id": prompt_id,
                                        "image_sha256_pixels": f"SYNTHETIC_FIXTURE_{image_id}"})
                image_id += 1
    assert len(records) == 645
    with tempfile.TemporaryDirectory(prefix="bias-fixture-") as temp:
        path = Path(temp) / "per_image.jsonl"
        manifest_path = path.with_name("evaluation_manifest.json")
        def write(rows, status="complete"):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = {"status": status, "record_count": 645, "evaluated_unique_images": 465,
                        "generated_images": 600, "excluded_age_images": 135,
                        "per_image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "concepts_sha256": hashlib.sha256(concepts_path.read_bytes()).hexdigest(),
                        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest()}
            manifest_path.write_text(json.dumps(manifest))
        load = lambda: load_completed_evaluation(path, concepts_path, prompts_path)
        path.write_text("")
        rejected(load)
        write(records)
        assert len(load()) == 645
        path.write_text(path.read_text() + "\n")
        rejected(load)
        write(records, status="running")
        rejected(load)
        write(records[15:])
        rejected(load)
        altered = [dict(row) for row in records]
        next(row for row in altered if row["category"] == "Ethnic Representation Bias")["image_sha256_pixels"] = "CHANGED_FIXTURE"
        write(altered)
        rejected(load)
        write(records)
        other_inventory = Path(temp) / "changed_concepts.json"
        other_inventory.write_bytes(concepts_path.read_bytes() + b"\n")
        rejected(lambda: load_completed_evaluation(path, other_inventory, prompts_path))
        other_prompts = Path(temp) / "changed_prompts.csv"
        other_prompts.write_bytes(prompts_path.read_bytes() + b"\n")
        rejected(lambda: load_completed_evaluation(path, concepts_path, other_prompts))


def local_dataset_fixture():
    """Exercise both actual file readers without weights, network, or research data."""
    from datasets import Dataset
    with tempfile.TemporaryDirectory(prefix="bias-local-fixture-") as temp:
        root = Path(temp)
        original = Dataset.from_dict({"Prompt": ["SYNTHETIC FIXTURE"], "Prompt Strategies": [0]})
        parquet = root / "fixture.parquet"
        original.to_parquet(parquet)
        original.save_to_disk(str(root / "fixture"))
        arrow = next((root / "fixture").glob("*.arrow"))
        for path in [parquet, arrow]:
            loaded = load_study_dataset(str(path), root / "cache", local_only=True)
            assert loaded.to_dict() == original.to_dict()
        rejected(lambda: load_study_dataset("UNAVAILABLE_FIXTURE", root / "cache", local_only=True))
        unsupported = root / "fixture.csv"
        unsupported.write_text("not an accepted input")
        rejected(lambda: load_study_dataset(str(unsupported), root / "cache", local_only=True))
        assert file_sha256(unsupported) == hashlib.sha256(unsupported.read_bytes()).hexdigest()


def main():
    assert parity_mad({"a": .5, "b": .5}, ["a", "b"]) == 0
    assert math.isclose(parity_mad({"a": 1}, ["a", "b"]), .5)
    assert math.isclose(parity_mad({"medium": 33/56, "dark": 23/56}, ["light", "medium", "dark"]), 2/9)
    rejected(lambda: parity_mad({"a": .4}, ["a", "b"]))
    rejected(lambda: parity_mad({"a": 1}, ["b"]))
    assert holm([.01, .03, .04]) == [.03, .06, .06]
    assert reference_texts(["[PROMPT] shows a man."], "A chef. Avoid bias.") == ["A chef shows a man."]
    inputs = [{"Main_Category": "Gender Bias", "Neutral_prompt": "A chef."}]
    metadata = [{"Main Category": "Gender Bias", "Prompt": "A chef.", "Prompt Strategies": s}
                for s in [0, 1, 2] for i in range(5)]
    validate_design(metadata, inputs)
    rejected(lambda: validate_design(metadata[:-1], inputs))
    rejected(lambda: validate_design(metadata + [metadata[0]], inputs))
    rejected(lambda: validate_design([{**r, "Prompt": "A different task."} for r in metadata], inputs))
    assert extract_revised_prompt("thought chain: reason\nrevised prompt: A portrait.\nWith detail.") == "A portrait.\nWith detail."
    rejected(lambda: extract_revised_prompt("revised prompt: [bias-reduced prompt]"))
    rejected(lambda: extract_revised_prompt("no response marker"))
    rejected(lambda: extract_revised_prompt("revised prompt: one\nrevised prompt: two"))
    completion_gate_fixture()
    try:
        from datasets import Dataset
    except ImportError:
        print("Datasets unavailable; local file reader fixture not run")
    else:
        local_dataset_fixture()
    try:
        import scipy
    except ImportError:
        print("Core checks passed; scipy unavailable, inference fixture not run")
        return
    concepts = {"Gender Bias": {"Parity": {"concepts": ["a man", "a woman"]}}}
    records = [{"category": "Gender Bias", "prompt_id": f"p{p}", "strategy": s,
                "image_id": f"{p}-{s}-{i}", "parity_label": "a man" if i < 3 else "a woman",
                "concept_similarity": .2 + p * .01 + s * (.005 + p * .001),
                "reference_similarity": .25 + p * .005} for p in range(3) for s in range(3) for i in range(5)]
    result = prompt_comparisons(records, concepts)
    assert len(result) == 9 and all(r["prompt_n"] == 3 and r["image_n_per_condition"] == 15 for r in result)
    assert all(r["family_size"] == 9 and r["p_holm"] >= r["p_raw"] for r in result)
    assert all(r["p_raw"] == 1 for r in result if r["metric"] == "reference_similarity")
    rejected(lambda: prompt_comparisons(records[:-1], concepts))
    rejected(lambda: prompt_comparisons(records + [records[0]], concepts))
    rejected(lambda: prompt_comparisons(records[:-1] + [{**records[-1], "image_id": records[0]["image_id"]}], concepts))
    rejected(lambda: prompt_comparisons([r for r in records if not (r["prompt_id"] == "p0" and r["strategy"] == 2)], concepts))
    print("Regression checks passed")


if __name__ == "__main__":
    main()
