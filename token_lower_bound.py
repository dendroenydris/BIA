"""Conditional CLIP pretokenizer lower bound, not an actual tokenization audit."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/clip/tokenization_clip.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import regex
    pattern = regex.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", regex.IGNORECASE)
    rows = []
    with (ROOT / "datasets/prompts.csv").open(encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter=";"), 1):
            for strategy, column in enumerate(["Neutral_prompt", "Bias_aware_prompt_engineering", "Prompt_engineering_with_the_help_of_LLM"]):
                text = row[column]
                # The saved inputs contain no markup/entities or special-token strings.
                # Non-ASCII content is quotation marks, an em dash, and café's accented e.
                if any(c in text for c in ["<", "&"]) or any(ord(c) > 127 and c not in "’‘“”—é" for c in text):
                    raise ValueError("Unexpected normalization case; do not assume this lower bound")
                normalized = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})).lower()
                chunks = list(pattern.finditer(normalized))
                rows.append({"entry": index, "category": row["Main_Category"], "strategy": strategy,
                             "whitespace_word_count": len(text.split()), "regex_chunk_count": len(chunks),
                             "standard_clip_token_lower_bound_with_special": len(chunks) + 2,
                             "exceeds_standard_77_limit": len(chunks) + 2 > 77,
                             "suffix_cannot_fit_under_standard_77": normalized[chunks[75].start():] if len(chunks) > 75 else "",
                             "full_text": text})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "token_lower_bound.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {"kind": "conditional pretokenizer lower-bound audit; not actual tokenization",
                "source": SOURCE, "model_context_assumption": 77, "special_tokens": 2,
                "vocab_and_merges_used": False, "historical_revision_verified": False,
                "normalization": "lowercase and uncurl quotation marks; whitespace ignored by regex",
                "why_lower_bound": "Each regex chunk is independently BPE encoded into at least one token; merges cannot cross chunks. Add two special tokens. Exact IDs and cut point require the actual tokenizer. Rows below this bound may still truncate.",
                "definitely_exceed_under_assumption": sum(r["exceeds_standard_77_limit"] for r in rows)}
    (args.output_dir / "token_lower_bound_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Conditional lower-bound audit: {manifest['definitely_exceed_under_assumption']} inputs exceed 77. Exact audit remains pending.")


if __name__ == "__main__":
    main()
