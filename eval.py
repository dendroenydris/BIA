"""Preserve per-image units and evaluate every non-age image using fixed references."""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def file_sha256(path):
    """Hash large local inputs without loading an entire weight file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_model_hashes(directory):
    """Return SHA-256 hashes for local model and processor files."""
    return {str(p.relative_to(directory)): file_sha256(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".txt", ".bin", ".safetensors"}}


def load_study_dataset(source, cache_dir, revision=None, local_only=False):
    from datasets import Dataset, load_dataset
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() == ".arrow":
            return Dataset.from_file(str(path))
        if path.suffix.lower() == ".parquet":
            return Dataset.from_parquet(str(path), cache_dir=str(cache_dir))
        raise ValueError("Local data must be a Parquet or Hugging Face streaming Arrow file")
    if local_only:
        raise ValueError("Offline evaluation requires an existing local dataset file")
    return load_dataset(source, split="train", revision=revision, cache_dir=str(cache_dir))


def reference_texts(phrases, baseline_prompt):
    short = re.split(r"[.,]", baseline_prompt)[0].strip()
    return [p.replace("[PROMPT]", short) for p in phrases]


def validate_design(metadata, prompt_rows):
    """Validate all prompt-condition cells before loading model weights."""
    expected = {(r["Main_Category"], r["Neutral_prompt"].strip(), s)
                for r in prompt_rows for s in [0, 1, 2]}
    actual = Counter((r["Main Category"], r.get("Baseline Prompt", r["Prompt"]).strip(),
                      int(r["Prompt Strategies"])) for r in metadata)
    if set(actual) != expected or any(n != 5 for n in actual.values()):
        raise ValueError("Dataset differs from the 40-prompt, three-condition, five-image design")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Local parquet/streaming arrow or Hugging Face dataset ID")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="ID or local directory")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Require local dataset and model/processor; no Hub loading")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluations/revised")
    args = parser.parse_args()
    import datasets
    import torch
    import transformers
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.local_files_only and not Path(args.model).is_dir():
        raise ValueError("Offline evaluation requires an existing local CLIP model/processor directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "per_image.jsonl"
    if output.exists():
        raise FileExistsError(f"Will not overwrite evidence: {output}")
    data = load_study_dataset(args.dataset, args.output_dir / "dataset_cache",
                              args.dataset_revision, args.local_files_only)
    required_columns = {"Main Category", "Prompt", "Prompt Strategies", "Image"}
    if not required_columns.issubset(data.column_names):
        raise ValueError(f"Dataset is missing required columns: {sorted(required_columns - set(data.column_names))}")
    if dict(Counter(data["Main Category"])) != {"Gender Bias": 165, "Racial Bias": 180, "Age Bias": 135, "Geo-Cultural Bias": 120}:
        raise ValueError("Dataset counts differ from the 600-image design; audit before evaluating")
    with (ROOT / "datasets/prompts.csv").open(encoding="utf-8-sig") as handle:
        prompts = list(csv.DictReader(handle, delimiter=";"))
    baseline_column = "Baseline Prompt" if "Baseline Prompt" in data.column_names else "Prompt"
    metadata = ({"Main Category": c, "Prompt": p, "Prompt Strategies": s} for c, p, s in
                zip(data["Main Category"], data[baseline_column], data["Prompt Strategies"]))
    validate_design(metadata, prompts)
    concepts = json.loads((ROOT / "datasets/bias_concepts.json").read_text())
    input_hashes = {"dataset_file_sha256": file_sha256(Path(args.dataset)) if Path(args.dataset).is_file() else None,
                    "model_files_sha256": local_model_hashes(Path(args.model)) if Path(args.model).is_dir() else None,
                    "prompts_sha256": file_sha256(ROOT / "datasets/prompts.csv")}
    model_options = {"revision": args.model_revision, "local_files_only": args.local_files_only,
                     "cache_dir": str(args.output_dir / "model_cache")}
    model = CLIPModel.from_pretrained(args.model, **model_options).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model, use_fast=False, **model_options)
    text_cache = {}

    def text_features(texts):
        key = tuple(texts)
        if key not in text_cache:
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=False).to(device)
            if inputs["input_ids"].shape[-1] > model.config.text_config.max_position_embeddings:
                raise ValueError("Evaluation reference exceeds CLIP text limit")
            features = model.get_text_features(**inputs)
            text_cache[key] = features / features.norm(dim=-1, keepdim=True)
        return text_cache[key]

    with output.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(data):
            if row["Main Category"] == "Age Bias":
                continue
            baseline = row.get("Baseline Prompt", row["Prompt"])
            strategy = int(row["Prompt Strategies"])
            if strategy not in [0, 1, 2]:
                raise ValueError("Unknown condition")
            prompt_id = hashlib.sha256((row["Main Category"] + "\0" + baseline.strip()).encode()).hexdigest()[:16]
            image = row["Image"]
            if isinstance(image, dict):
                image = Image.open(io.BytesIO(image["bytes"])) if image.get("bytes") else Image.open(image["path"])
            elif not isinstance(image, Image.Image):
                image = Image.open(image)
            image = image.convert("RGB")
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            features = model.get_image_features(**processor(images=image, return_tensors="pt").to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            categories = ["Skin Tone Bias", "Ethnic Representation Bias"] if row["Main Category"] == "Racial Bias" else [row["Main Category"]]
            for category in categories:
                definition = concepts[category]
                labels = definition["Parity"]["concepts"]
                label_sims = (features @ text_features(labels).T).squeeze(0)
                scores = {}
                for metric in ["Embedding Distance", "Stereotype Frequency Score"]:
                    for subtype, phrases in definition[metric].items():
                        texts = reference_texts(phrases, baseline)
                        sims = (features @ text_features(texts).T).squeeze(0).cpu().tolist()
                        scores[f"{metric} - {subtype}"] = {"references": texts, "cosine_similarities": sims}
                result = {"image_id": index, "image_sha256_pixels": image_hash, "prompt_id": prompt_id,
                          "baseline_prompt": baseline, "category": category, "strategy": strategy,
                          "parity_label": labels[int(label_sims.argmax())], "label_cosines": label_sims.cpu().tolist(),
                          "concept_similarity": mean(mean(v["cosine_similarities"]) for k, v in scores.items() if k.startswith("Embedding Distance")),
                          "reference_similarity": mean(max(v["cosine_similarities"]) for k, v in scores.items() if k.startswith("Stereotype Frequency")), "scores": scores}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Evaluated image {index + 1}/{len(data)}", flush=True)
    manifest = {"status": "complete", "dataset": args.dataset, "dataset_revision_requested": args.dataset_revision,
                "dataset_fingerprint": data._fingerprint, "model": args.model,
                "model_commit": getattr(model.config, "_commit_hash", None), "device": device,
                "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__,
                "tokenizer_class": type(processor.tokenizer).__name__,
                "image_processor_class": type(processor.image_processor).__name__, "processor_use_fast": False,
                "generated_images": len(data), "evaluated_unique_images": 465, "excluded_age_images": 135,
                "record_count": 645, "reference_rule": "same baseline first clause for every condition", "fairface_used": False,
                "per_image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "concepts_sha256": file_sha256(ROOT / "datasets/bias_concepts.json"),
                "local_files_only": args.local_files_only, **input_hashes}
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
