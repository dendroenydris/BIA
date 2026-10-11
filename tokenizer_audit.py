"""Audit generation inputs with an explicitly supplied local tokenizer; no download."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.tokenizer.is_dir() or args.max_length < 3:
        raise ValueError("Supply an existing local tokenizer directory and valid limit")
    import transformers
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if tokenizer.truncation_side != "right":
        raise ValueError("This audit expects the standard right-truncation policy")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "tokenizer_audit.jsonl"
    if out.exists():
        raise FileExistsError(out)
    with (ROOT / "datasets/prompts.csv").open(encoding="utf-8-sig") as f, out.open("x") as handle:
        for index, row in enumerate(csv.DictReader(f, delimiter=";"), 1):
            for condition, column in [(0, "Neutral_prompt"), (1, "Bias_aware_prompt_engineering"),
                                      (2, "Prompt_engineering_with_the_help_of_LLM")]:
                text = row[column]
                full = tokenizer(text, truncation=False)["input_ids"]
                encoded = tokenizer(text, truncation=True, max_length=args.max_length)["input_ids"]
                content = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                budget = args.max_length - tokenizer.num_special_tokens_to_add(pair=False)
                record = {"prompt_entry": index, "category": row["Main_Category"], "condition": condition,
                          "submitted_text": text, "full_token_count_with_special": len(full),
                          "encoded_token_ids": encoded, "encoded_tokens": tokenizer.convert_ids_to_tokens(encoded),
                          "encoded_text": tokenizer.decode(encoded, skip_special_tokens=True),
                          "removed_content_token_ids": content[budget:],
                          "removed_text": tokenizer.decode(content[budget:], skip_special_tokens=True),
                          "truncated": len(full) > args.max_length}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"status": "complete", "records": 120, "max_length": args.max_length,
                "truncation_side": tokenizer.truncation_side, "transformers": transformers.__version__,
                "tokenizer_class": type(tokenizer).__name__,
                "preprocessing_path": "ftfy.fix_text" if tokenizer.fix_text is not None else "BasicTokenizer fallback; ftfy unavailable",
                "historical_revision_verified": False,
                "tokenizer_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in args.tokenizer.iterdir() if p.is_file()}}
    (args.output_dir / "tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
