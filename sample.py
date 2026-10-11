"""Generate and record category-specific prompt rewrites with a supplied LLM."""
import argparse
import csv
import json
import re
from pathlib import Path


def generate_prompt(original_prompt, bias_name):
    return f"""Apply step-by-step Chain of Thought reasoning to the following prompt to reduce {bias_name}.
Prompt: {original_prompt}
Output the result in the format:
thought chain: [reasoning steps]
revised prompt: [bias-reduced prompt]"""


def extract_revised_prompt(response_text):
    matches = re.findall(r"(?:^|\n)\s*revised prompt:\s*(.+?)(?=\n\s*(?:thought chain|revised prompt):|$)", response_text, re.IGNORECASE | re.DOTALL)
    if len(matches) != 1 or matches[0].strip() == "[bias-reduced prompt]":
        raise ValueError("Missing, ambiguous, or unfilled revised prompt")
    return matches[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "datasets/prompts.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import pipeline
    generator = pipeline("text-generation", model=args.model, revision=args.model_revision,
                         device_map="auto", model_kwargs={"torch_dtype": "auto"})
    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig"), delimiter=";"))
    with args.output.open("x", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            instruction = generate_prompt(row["Neutral_prompt"], row["Main_Category"])
            # Decode continuation only, so the template is never mistaken for the answer.
            response = generator(instruction, max_new_tokens=512, do_sample=False, return_full_text=False)[0]["generated_text"]
            revised = extract_revised_prompt(response)
            f.write(json.dumps({"row_id": index, "original": row["Neutral_prompt"], "category": row["Main_Category"],
                                "instruction": instruction, "response": response, "revised_prompt": revised,
                                "model": args.model, "model_commit": getattr(generator.model.config, "_commit_hash", None),
                                "max_new_tokens": 512, "do_sample": False, "chat_template_applied": False}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
