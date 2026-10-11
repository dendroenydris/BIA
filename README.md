# Prompt engineering and bias in generated images

Code for auditing saved aggregate results, evaluating the original images with CLIP, and comparing prompt conditions. The corrected image-level analysis replaces the original significance classifications and mitigation rankings. Results and interpretation are documented in `evaluations.md`.

The inputs are 40 prompt-category entries, three conditions, and five images per cell (600 images). The CSV's historical `Neutral_prompt` column is called baseline in the revision. The historical Step 3 is a saved prompt-rewrite condition; the separate Zephyr script does not establish its provenance. Exact original LLM and image-generation versions/settings were not recorded completely. Corrected scripts are prospective, not proof of what ran historically.

## Installation and checks

Use Python 3.10 or later. Install the analysis and evaluation dependencies:

```sh
python -m pip install -r requirements.txt
```

From this directory:

```sh
python bias_analysis.py --output-dir evaluations/revised
python test_revision.py
```

The aggregate audit writes corrected MAD, complete label proportions, and subtype summaries using the standard library. The regression checks use synthetic fixtures and exercise input validation, prompt extraction, and paired inference.

## Re-evaluate the original images

Supply a local copy of the original 600-row dataset and local CLIP model/processor files. These commands do not regenerate images:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python eval.py --local-files-only --dataset /path/to/original/train.parquet --model /path/to/clip-vit-base-patch32 --output-dir evaluations/image_reanalysis
python bias_analysis.py --per-image evaluations/image_reanalysis/per_image.jsonl --output-dir evaluations/image_reanalysis
```

`eval.py` checks category totals, excludes age, holds baseline-derived references fixed across conditions, evaluates all 465 non-age images, and saves 645 records with IDs, pixel hashes, label cosines, and reference scores. `evaluation_manifest.json` is written only after completion and binds the JSONL to its SHA-256 digest. The statistical CLI requires that manifest, verifies the inventory digest and every expected prompt-condition cell, and rejects partial, altered, or subset streams, even when a partial run stops at a complete cell. The script refuses to overwrite an existing JSONL. A new output directory is needed for a retry. CLIP references exceeding the text limit cause an error rather than silent truncation. It uses CLIP only; FairFace is not used.

Local inputs may be Parquet or a Hugging Face streaming `.arrow` file; Feather/Arrow IPC files are not supported by this reader. `--local-files-only` requires an existing dataset file and model directory and tells both model and processor loaders to use local files. Dataset/model caches are placed inside the chosen output directory. The completion manifest fingerprints the local data file, prompt CSV, and model/processor snapshot files. The statistical reader also rejects a changed prompt CSV. These hashes establish the resources used for the new analysis; they do not establish the unrecorded historical model revision.

The processor explicitly uses `use_fast=False`, preserving the slow processor path verified for the local snapshot rather than following a changing library default. Its actual tokenizer/image-processor classes are recorded in the run manifest. This specifies the new evaluation environment, not the historical generation environment.

## Statistical units and interpretation

Images are averaged within each original prompt-condition cell. The three pairwise condition comparisons use original prompts as paired units within each evaluator axis. The exact two-sided sign-flip test assumes exchangeability of signs of prompt differences; it is not justified as a randomized treatment test by the historical seed policy. All 36 axis × metric × pair comparisons share one Holm correction when all axes are present. Report the full family, including non-significant comparisons. Mean differences, prompt-difference SD, standardized paired effects, and unadjusted paired-t intervals are exported. Normal-difference assumptions for the intervals must be examined before use with these small prompt counts. These exploratory intervals are conditional on the selected prompt set, not population guarantees.

Per-prompt MAD is a different estimand from pooled legacy MAD. With five images and ten geo-cultural labels its minimum is 0.1, even if all five assignments differ. It must not be compared numerically with pooled legacy MAD as though it were the same quantity. Repeated baseline sentences across categories and repeated racial images across axes are not independent evidence; axes are not pooled into one efficacy estimate. No similarity change is automatically an ethical improvement.

## Prompt rewriting and truncation

`sample.py` is an optional prospective rewriting tool, requiring locally available weights. It fixes continuation parsing, retains the full response, uses the row's actual category, and records deterministic decoding settings. It must not be used to overwrite historical generation inputs or described as a re-run of the historical ChatGPT procedure.

`tokenizer_audit.py` audits full and truncated prompt strings using an explicitly supplied local tokenizer. Its resource identity matters: replay with the evaluation CLIP tokenizer is a sensitivity check, not an audit of verified historical Stable Diffusion encoding. Image data are distributed separately from this code directory in the anonymous empirical archive; model weights are not bundled.

## Completed revision analysis

The anonymous empirical archive includes 600 pixel-preserved images, 645 current axis-image scores from all 465 non-age images, and all 36 planned paired comparisons. Its archive README distinguishes original and metadata-stripped dataset hashes. Historical aggregates in this code directory remain provenance material, not the new primary results.

```sh
python report_reanalysis.py --input-dir /path/to/completed/evaluation --output-dir /path/to/new/report
```

This command verifies the complete image run before deriving prompt-cell means/SDs, pooled counts, every primary contrast, diagnostic Shapiro/Q–Q summaries, and exploratory 20,000 paired-prompt bootstrap and directional sign-test sensitivity. Primary and sensitivity Holm families are separate, with all 36 comparisons retained. These checks do not prove sign exchangeability or independent population sampling of the selected prompts.

The complete public SD 2.1 tokenizer replay is also distributed in the anonymous archive, using mirror revision `bb2154823665391b4fb29b0b9cf82a198964ee05`. `report_tokenizer_audit.py` independently replays every saved input and checks the source/file hashes before producing a clearly post-hoc no-overflow racial subset. It verifies the full image stream before selecting the four retained racial prompt entries in all conditions. All 18 subset contrasts are reported with their own Holm family; primary comparisons remain unchanged. Current mirror agreement with the evaluation CLIP tokenizer is not proof of historical version identity. The BasicTokenizer fallback preprocessing is explicit. See the archive README for reproduction commands.

The original notebook and old outputs are historical material. The anonymous code package includes the relevant generation-code excerpt without authentication cells or notebook outputs. Use the separate anonymous empirical package to reproduce the corrected analysis from the complete saved image scores; model weights are not bundled.

A model-free conditional check is also available:

```sh
python token_lower_bound.py --output-dir evaluations/length_audit
```

Under the documented CLIP pretokenizer/BPE and 77-position context, eight saved LLM-assisted inputs exceed that limit even at the minimum one-token-per-regex-chunk bound. This does not reconstruct exact token IDs, historical encoder input, or the precise cut point. Inputs below the bound can still truncate. See the exported manifest and full local-tokenizer audit for the distinction.
