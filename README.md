# Is prompt engineering able to mitigate bias in generated images?

Code, data and results for the paper *Is Prompt Engineering able to Mitigate Bias in Generated Images?*

The study asks whether rewriting a text-to-image prompt changes who is depicted. Each of 40 everyday prompts (for
example "A doctor treating a patient.") exists in three versions:

| Condition | Short name | How it was made |
|---|---|---|
| Baseline | B | the plain prompt |
| Manual instruction | M | the baseline followed by a fixed sentence for its category, e.g. "Avoid stereotypical depictions of gender." |
| LLM-assisted rewrite | L | a rewrite produced by GPT-4o (ChatGPT) with a chain-of-thought instruction; only the text after "revised prompt:" was used |

The prompts cover four categories: gender (11 prompts), race (12), age (9) and geo-culture (8). Images are labelled
for perceived gender presentation, skin tone, ethnic appearance, apparent age and cultural setting, and each condition
is compared with the others on **label MAD**, the mean absolute deviation of the label shares of a prompt from an equal
share for every label. Lower label MAD means the labels are spread more evenly; it is a measure of imbalance, not of
fairness.

## Repository map

| Folder | Contents |
|---|---|
| `datasets/` | The 40 prompts in three versions, the CLIP label sets, and the generation schedule of every image set (prompt, condition, repetition, seed) |
| `generation/` | The notebook that generated the original archive, and the scripts that generated every other image set |
| `evaluation/` | The label guide, the Qwen3-VL prompt and labelling script, and the CLIP labelling scripts |
| `human_ratings/` | The raw rating sheets of the human raters and the keys that link each rated image to its prompt and condition |
| `records/` | Per-image records: generation receipts (settings, seed, initial-noise hash, file hash), raw evaluator output and labels |
| `analysis/` | All statistics: label MAD and paired comparisons, evaluator agreement, sensitivity analyses, table compilation |
| `results/` | Every reported number, as written by the analysis code |
| `plan/` | The evaluation plan fixed before the new evaluators were run, and the protocol of the matched-clause pilot |
| `tests/` | Checks on synthetic data and on the tokenizer |
| `paths.py` | Every repository path used by the scripts |

All scripts take their repository paths from `paths.py` and run from any working directory, e.g.
`python analysis/label_imbalance.py`. Inputs that are not stored here (images, model weights, external statistics) are
command-line arguments.

## Image sets

| Image set | Generator | Design | Images | Schedule | Records |
|---|---|---|---|---|---|
| Original archive | SD 2.1 | 40 prompts × 3 conditions × 5 images, independent noise per image | 600 | `datasets/original_archive_schedule.csv` | `records/qwen3vl/original_archive.jsonl`, `records/clip/original_archive_b32/` |
| Main experiment | SD 2.1 and FLUX.1-schnell | 40 prompts × 3 conditions × 50 images per model; the three conditions of a prompt share 50 initial noises | 12,000 | `datasets/main_experiment/` | `records/generation/main_{sd21,flux}/` |
| Matched-clause experiment | SD 2.1 (and FLUX.1-schnell, partial) | 31 non-age prompts × 2 versions × 50 images | 3,100 + 1,550 | `datasets/matched_clause/` | `records/generation/matched_{sd21,flux}/` |
| Matched-clause pilot | SD 2.1 | 31 prompts × 2 versions × 5 images | 310 | `datasets/matched_clause/pilot/` | `records/matched_clause_pilot/` |
| Truncation regeneration | SD 2.1 | 8 long rewrites × 2 encodings × 5 images | 80 | `datasets/truncation_regeneration/` | `records/truncation_regeneration/` |

**Original archive.** `generation/original_archive_sd21.ipynb` generated five images per prompt and condition with
stabilityai/stable-diffusion-2-1 (fp16, DPMSolverMultistepScheduler, pipeline defaults of 50 steps and guidance 7.5,
768 × 768), drawing the initial noise independently for each image. Its manual prompts are the strings of
`datasets/prompts.csv`.

**Main experiment.** `datasets/main_experiment/build_schedule.py` writes the schedule: 50 seeds per prompt derived
from master seed 20261117, shared by the three conditions. The manual prompts are standardised to the baseline string
followed by the category sentence; the nine prompts whose original manual version differed are listed in
`datasets/main_experiment/bias_aware_corrections.csv`. SD 2.1 used sd2-community/stable-diffusion-2-1 at revision
bb2154823665391b4fb29b0b9cf82a198964ee05 (fp16, 25 steps, guidance 7.5, empty negative prompt); FLUX.1-schnell used
black-forest-labs/FLUX.1-schnell at revision 741f7c3ce8b383c54771c7003378a50191e9efe9 (bfloat16, 4 steps, guidance 0,
256-token T5 sequence). Both at 768 × 768 on NVIDIA A100 GPUs; versions and settings are in each `execution_config.json`.

**Matched-clause experiment.** Each prompt pair (`datasets/matched_clause/prompt_pairs.csv`) keeps the baseline and
adds the same detail sentence; version I then adds "Avoid demographic stereotypes in this depiction.", version C adds
"Avoid visual artifacts in this depiction." Both versions have the same token length and fit the 77-token context, so
the comparison isolates the anti-stereotype clause. Both versions of a block share one initial noise. The first five
repetitions reuse the seeds of the pilot (50 steps, Apple MPS), which makes the step-count check possible. The
FLUX.1-schnell run covers the 11 gender prompts and the first four racial prompts in both versions (1,500 images) and one
version of a fifth racial prompt (50 images, unpaired and not analysed).

**Truncation regeneration.** SD 2.1 reads at most 77 tokens, and eight LLM-assisted racial rewrites (A14–A19, A22, A23)
are longer. `generation/truncation_regeneration.py` renders each with the standard truncated input (T) and with the full
text encoded in consecutive 77-token chunks (F, `generation/long_prompt.py`), under five shared initial noises.

## Evaluators

**Human raters.** Two raters independently labelled 60 archive images (`human_ratings/original_archive/`, H01–H60: the
chef, portrait, birthday and wedding prompts, A10, A12, A32 and A33, three conditions, five images each) with the label
guide `evaluation/label_guide.md`, without seeing prompt or condition. One rater did not label ethnic appearance. Two
further independent raters labelled 60 FLUX.1-schnell images of the same four prompts from the main experiment
(`human_ratings/flux/`, F01–F60). The keys link each image to its prompt, condition and pixel hash.

**Qwen3-VL (primary evaluator).** Qwen/Qwen3-VL-8B-Instruct at revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b labels
every image with the label guide's categories (`evaluation/qwen3vl_prompt.txt`, greedy decoding, one re-ask for
invalid values). `records/qwen3vl/<set>.jsonl` holds, per image, the raw answer, any re-ask, the labels of the live
run (`codes_live`) and the final labels (`codes`), which `evaluation/reparse_qwen3vl_labels.py` re-derives from the
raw answers without the model.

**CLIP (secondary evaluator).** The original evaluator, openai/clip-vit-base-patch32, assigns the most similar label of
each attribute's label set (`datasets/bias_concepts.json`); `evaluation/clip_original_archive.py` applies it to the
original archive together with the exploratory similarity measures. CLIP ViT-L/14 (revision
32bd64288804d66eefd0ccbe215aa642df71cc41) with five prompt templates per label is the second CLIP variant.
`evaluation/label_images_clip.py` labels the generated sets with both (`records/clip/<set>_{b32,l14ens}.jsonl`).

**Agreement threshold.** `plan/PLAN.md` fixed, before the new evaluators were run, the agreement rule: an evaluator's
labels are read as perceived appearance on an attribute when its Cohen's kappa with the raters reaches the lower 95%
bootstrap limit of the raters' agreement with each other. `plan/PLAN.sha256` records the hash and time of each version
of the plan. File names in the plan refer to the working copies of the scripts listed under "Script names" below.

## Reproducing the reported numbers

Analysis needs Python 3.10 or later with numpy and scipy (`pip install -r requirements.txt` installs everything,
including the model libraries). The commands below rewrite the files in `results/` from the stored records; the
results are byte-identical with scipy 1.14 (other scipy versions change the last digits of t-quantiles; under Python
3.12 and later `step_count_check.py` writes one zero difference as `0.0` instead of `-0.0`).

| Result | Command | Output |
|---|---|---|
| Rater–rater agreement and label counts next to Table 1, archive images | `python analysis/human_agreement.py` | `results/agreement/archive_raters/` |
| Qwen3-VL vs raters, archive images (Table 2, Table E1) | `python analysis/qwen3vl_human_agreement.py` (add `--first-parser` for the labels of the live run) | `results/agreement/qwen3vl_vs_raters*/` |
| CLIP vs raters, archive images (Table 2) | `python analysis/clip_human_agreement.py {b32,l14ens} --images ARCHIVE.parquet` | `results/agreement/clip_{b32,l14}_vs_raters/` |
| Raters and Qwen3-VL, FLUX.1-schnell images (Table E2) | `python analysis/flux_human_agreement.py` | `results/agreement/flux_raters/` |
| Label MAD, all paired comparisons and MDEs, every image set and evaluator | `python analysis/label_imbalance.py` | `results/qwen3vl/<set>/`, `results/clip/<set>_<variant>/` |
| Results tables quoted in the paper | `python analysis/compile_results.py` | `results/results_tables.md` |
| FLUX.1-schnell minus SD 2.1 at baseline | `python analysis/model_baseline_comparison.py` | `results/model_baseline/` |
| Population-reference sensitivity (world and U.S.) | `python analysis/population_reference.py [--wpp-dir DIR]` | `results/population_reference/` |
| Step-count check (25 vs 50 steps) | `python analysis/step_count_check.py` | `results/step_count_check/` |
| Matched-clause pilot tables (Appendix D) | `python generation/matched_clause_pilot.py --analyze-only --out DIR`, then `python analysis/matched_clause_pilot_audit.py` | `results/matched_clause_pilot/` |
| Truncation regeneration tables (Appendix C) | `python generation/truncation_regeneration.py --analyze-only --out DIR` | `results/truncation_regeneration/` |
| CLIP analysis of the original archive (Appendix G) | `python analysis/original_archive_clip_report.py --output-dir DIR` | `results/clip/original_archive_b32/` |
| Tokenizer replay of all 120 prompt strings (Appendix C) | `python analysis/tokenizer_audit.py --tokenizer SD21_TOKENIZER --output-dir DIR`, then `python analysis/tokenizer_audit_report.py --tokenizer SD21_TOKENIZER --comparison-tokenizer CLIP_B32 --output-dir DIR` | `records/tokenizer_audit/`, `results/tokenizer_audit/` |
| Token lower bound without a tokenizer | `python analysis/token_lower_bound.py --output-dir DIR` | `results/tokenizer_audit/token_lower_bound/` |
| Audit of the original study's saved aggregates | `python analysis/clip_statistics.py --output-dir DIR` | `results/original_study/aggregate_audit/` |

Scripts whose stored output must not be overwritten take a new `--output-dir`; compare it with the stored folder. The
tokenizer replays and the truncation schedule were made with the tokenizer's BasicTokenizer fallback (the `ftfy`
package not installed); with `ftfy` installed some LLM-assisted rewrites count two more tokens. The population
references are stored in `results/population_reference/reference_*.json` and used when `--wpp-dir` is not given; to
recompute them, download `WPP2024_Demographic_Indicators_Medium.csv.gz` and
`WPP2024_PopulationByAge5GroupSex_Medium.csv.gz` from the UN World Population Prospects 2024; the U.S. figures are in
`datasets/population_reference/us_sources.json`.

**Statistics.** Images are summarised per prompt and condition; prompts are the paired units. p-values are exact
two-sided sign-flip values over all sign patterns of the prompt differences, adjusted with Holm within each design's
primary set of tests (main experiment: 5 attributes × 3 comparisons = 15; matched-clause experiment: 4 attributes, I − C).
Images labelled mixed, indeterminate or no person are left out of label MAD and reported as a share. The minimum
detectable effect uses the observed SD of the prompt differences (paired t, two-sided α = 0.05, power 0.80).

**Tests.** `python tests/test_clip_analysis.py` and `python tests/test_matched_clause_pilot.py` run on synthetic data;
`python tests/test_long_prompt.py [SD21_TOKENIZER]` checks the chunked encoding against the SD 2.1 tokenizer.

## Labelling and generating again

These steps need the images or a GPU.

```sh
# original archive: write the 600 images (and the 60 rated ones) from the archive parquet, then label them
python evaluation/export_archive_images.py --images ARCHIVE.parquet --out ARCHIVE_DIR
python evaluation/label_images_qwen3vl.py --manifest ARCHIVE_DIR/manifest_original_archive.csv --root ARCHIVE_DIR \
    --codebook evaluation/label_guide.md --out RUN_DIR
python evaluation/clip_original_archive.py --dataset ARCHIVE.parquet --output-dir RUN_DIR

# main experiment and matched-clause experiment
python generation/generate_sd21.py --schedule datasets/main_experiment/generation_schedule.csv --out RUN_DIR
python generation/generate_sd21.py --schedule datasets/matched_clause/generation_schedule.csv \
    --pairs datasets/matched_clause/prompt_pairs.csv --out RUN_DIR
python generation/generate_flux.py --schedule datasets/main_experiment/generation_schedule.csv --out RUN_DIR
python evaluation/label_images_clip.py {b32,l14ens} RUN_DIR OUT.jsonl
python evaluation/label_images_qwen3vl.py --manifest MANIFEST.csv --root RUN_DIR --codebook evaluation/label_guide.md --out OUT_DIR
python evaluation/reparse_qwen3vl_labels.py OUT_DIR/codes.jsonl
```

A labelling manifest is a CSV with columns `key,path`; for a generation run the key of image `images/NNNN.png` is
`genNNNN`. `generation/generate_sd21.py`, `generation/generate_flux.py` and `evaluation/label_images_qwen3vl.py` are
byte-identical to the versions that produced the stored records: their SHA-256 equals `script_sha256` in
`records/generation/main_*/execution_config.json` and `records/qwen3vl/*_config.json`. Generation receipts record each
image's file hash and initial-noise hash, so regenerated images can be checked against them. `generation/rewrite_prompts_llm.py`
applies the chain-of-thought rewriting template with an open model (Zephyr); it is a tool for new rewrites, the stored
LLM-assisted prompts are the GPT-4o outputs.

## What is not stored here

- **Images.** The 600 archive images are distributed with the paper's supplementary material as one parquet file. The
  16,650 images of the main and matched-clause experiments, the 310 pilot images and the 80 truncation images are
  identified by file hash in the generation receipts and are available on request; they will be deposited with a
  persistent identifier on publication.
- **Model weights.** SD 2.1, FLUX.1-schnell, Qwen3-VL and CLIP are public; the revisions and file hashes are in the
  configuration files under `records/`.
- **Population statistics.** The UN WPP 2024 files (see above).

## Script names

The plan in `plan/PLAN.md` was written against working copies of the scripts. The correspondence is:

| Name in the plan | Script |
|---|---|
| `run_vlm_codebook.py` | `evaluation/label_images_qwen3vl.py` |
| `reparse_vlm_codes.py` | `evaluation/reparse_qwen3vl_labels.py` |
| `analyze.py` | `analysis/human_agreement.py` |
| `vlm_human_agreement.py` | `analysis/qwen3vl_human_agreement.py` |
| `clip_human_agreement.py` | `analysis/clip_human_agreement.py` |
| `analyze_rescored.py` | `analysis/label_imbalance.py` |
| `issued_codebook.md` | `evaluation/label_guide.md` |
| `BIA/datasets/bias_concepts.json` | `datasets/bias_concepts.json` |

## Data formats

- `datasets/prompts.csv` (`;`-separated): `Main_Category`, `Sub_Category`, `Neutral_prompt` (baseline),
  `Bias_aware_prompt_engineering` (manual instruction), `Prompt_engineering_with_the_help_of_LLM` (LLM-assisted rewrite).
  Prompt entries A1–A40 are the rows in order.
- Generation schedules: one row per image with `run_order`, `entry`, `category`, `repetition`, `latent_seed`, `arm`
  (the condition: `baseline`, `bias_aware`, `llm_assisted`; or the version `I`, `C`) and `submitted_prompt`.
- Result tables use the same `arm` column and the attribute names `gender`, `skin_tone`, `ethnicity`, `culture`, `age`.
