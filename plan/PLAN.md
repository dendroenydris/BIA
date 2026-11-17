# Evaluator revision: pre-specified plan (editor comment 1)

Written 2026-11-16, before any revised evaluator was run on any image. Its SHA-256 and UTC time are recorded in
`PLAN.sha256` when frozen. Changes after freezing go in a dated amendment section below, never as silent edits.

## Why

On the 60 human-coded archive images, the original CLIP ViT-B/32 evaluator agrees with the human raters on gender
(kappa ~0.94) but not on skin tone (~0.15 / ~0.00), ethnic appearance (~0.25, rater A only) or geo-culture
(~0.21 / ~0.03); the raters agree with each other at 0.48 and 0.47 on the last two
(`../clip_human_agreement/agreement.csv`). Its label sets also cannot express `mixed`, `indeterminate` or
no-person, which the raters use often. We therefore add two revised evaluators, fixed here in advance.

## Evaluators

| Role | Evaluator | Fixed settings |
|---|---|---|
| **Primary** | Qwen3-VL-8B-Instruct, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (Apache-2.0), bf16 | Codes each image with the raters' codebook (`human_reannotation_20261024/old_archive/source/issued_codebook.md`): the "Columns and allowed values" and "Special cases" sections verbatim, fields `person_count`, `gender_presentation`, `skin_tone`, `ethnic_appearance`, `age_appearance`, `cultural_scene`. Answer as one JSON object. Greedy decoding, max 200 new tokens. A value outside the allowed list triggers one re-ask stating the allowed values; still invalid -> missing (counted, reported). No prompt text, label wording or example images are tuned on the 60 human-coded images. |
| Sensitivity | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`, revision `32bd64288804d66eefd0ccbe215aa642df71cc41`) | The original label lists of `BIA/datasets/bias_concepts.json` unchanged, each embedded through five templates and averaged: "a photo of {}.", "a cropped photo of {}.", "a close-up photo of {}.", "a picture of {}.", "an image of {}.". Argmax label per axis, as before. |
| Reported as is | Original CLIP ViT-B/32 evaluator | Unchanged; its agreement is reported because the editor asked for it. |

Only one primary evaluator; the choice is made here and is not revisited after seeing results.

## Validity check on the 60 human-coded images (one pass per evaluator)

- Primary (VLM): Cohen's kappa against each rater on all 60 images per field, in the shared codebook vocabulary
  (`mixed`, `indeterminate`, `no_person` are ordinary categories). Raters' codes normalised by `analyze.py`.
- CLIP evaluators: as in `../clip_human_agreement/clip_human_agreement.py` (human codes mapped to CLIP labels;
  codes CLIP cannot express are counted and left out).
- Evaluator-human kappa for an axis = mean of kappa vs rater A and vs rater B (ethnic appearance: rater A only;
  rater B left it empty). Bootstrap CIs as in `analyze.py`.
- **Decision rule.** An evaluator is *validated* on an axis if its evaluator-human kappa is at least the lower 95%
  bound of the human-human kappa on that field (all 60 images): gender 0.650, skin tone 0.347, age 0.219,
  cultural scene 0.382; ethnic appearance (no human-human kappa) uses the skin-tone bound 0.347.
  On a non-validated axis that evaluator's results are reported descriptively only, with no inferential claim.

## Prompt-engineering analyses rescored

Rescored with every evaluator: the original 600 images, the FLUX.1-schnell and SD 2.1 matched-clause runs
(3,100 images each). Axis assignment as before (racial prompts -> skin tone and ethnicity); the primary evaluator also
scores the age prompts on the age axis (editor comment 10).

Primary outcome: label MAD against a uniform reference over the substantive labels of each axis:
gender {woman, man}; skin {light, medium, dark}; ethnicity {white, black, east_asian, south_southeast_asian,
middle_eastern, latino}; culture {western_european, south_asian, east_asian, middle_eastern, african, latin_american};
age {child, teenager, young_adult, middle_aged, older_adult}. Images coded `mixed`, `indeterminate`, `no_person`
(culture: `mixed`, `none_or_indeterminate`) or missing are left out of the MAD denominator, and their share is
reported per cell. A cell with no substantive image leaves that prompt out of that axis's paired test (counted).

Tests as before: paired by prompt, exact sign-flip p-values, Holm within the primary label-MAD family
(main study: 5 axes x 3 contrasts = 15; matched-clause: 4 axes, I minus C), MDE from the observed paired SD
(two-sided paired t, alpha 0.05, power 0.80). Similarity measures stay exploratory and CLIP-only.

## Amendments

**A1 (2026-11-16, before any revised evaluator was run on any image).** The consequence attached to the decision
rule is replaced. Inferential results (paired tests, Holm, MDE) are reported for every axis and every evaluator,
whatever its agreement. The rule now governs interpretation only: on an axis where an evaluator meets the
human-human reference, its results may be read as changes in perceived appearance; where it does not, they are
read as changes in that evaluator's labels, reported next to its agreement figures. Reason: the paired tests are
valid tests of the evaluator's label distributions regardless of agreement; agreement bears on what those labels
measure. Human codes are not treated as ground truth: skin tone and ethnicity are socially constructed categories
and both raters and CLIP apply particular conventions; human-human agreement is used only as the reference for how
consistently perceived appearance can be coded.

**A2 (2026-11-17, after 48 images of the first generation set had been coded, before any human-coded image was
coded).** Two implementation fixes to `run_vlm_codebook.py`. (1) The model answers `person_count` as a number
("3"); the codebook bins three or more people as `3plus`. Numeric counts >= 3 and "3+" are now mapped to `3plus`
before validation. Before the fix such answers failed validation and, after the re-ask, became missing (16 of the
first 48 images). (2) Throughput: the processor's fast image path (`use_fast=True`) and batches of 32. The fast
path resizes with a different implementation, which can change pixel values marginally; it applies to every image
alike. The 48 records coded before the fix are kept in `vlm_buggy_run1/` and not used. No prompt text, label wording
or decoding setting changed.

**A3 (2026-11-17, found in the validity run on the 60 human-coded images).** A second output-format fix. In 2 of
the 60 answers (H13, H48) the model left codebook values unquoted (`"person_count": 3plus`), which is invalid JSON,
so every field failed and the re-ask failed the same way. The parser now quotes bare words before parsing. All
codes, including those already produced, are re-derived from the stored raw output with this parser
(`reparse_vlm_codes.py`; no model call). This fix was made after the validity run had been seen, so both versions
are reported: before (58 images; mean evaluator-human kappa gender 0.835, skin 0.510, ethnicity 0.646, age 0.279,
culture 0.579) and after (60 images; 0.831, 0.491, 0.658, 0.286, 0.591). Every axis meets its reference bound in
both. No prompt, decoding setting or label changed.

**A4 (written 2026-11-17, documenting a change of image sets made on 2026-11-16, before the vision-language model
coded any image).** The section "Prompt-engineering analyses rescored" listed the original 600 images and FLUX.1-schnell
and SD 2.1 matched-clause runs of 3,100 images each. Before any of these were coded, the authors redirected the
generation budget to the three-condition comparison, which the editor had asked to be regenerated: it was rendered with
50 images per prompt and condition on both SD 2.1 and FLUX.1-schnell (6,000 images each), with shared initial
latents. The SD 2.1 matched-clause run was completed (3,100 images). The FLUX.1-schnell matched-clause run was stopped at
1,550 of 3,100 images, covering only part of the prompt entries; it was not coded or analysed. The outcome, the test
families (15 contrasts per model for the three-condition comparison, four for the matched clause), the evaluators and
the decision rule are as specified above. This amendment records the change of image sets; it is written after results
were seen and is reported as such.

**A5 (written 2026-11-17, before any image of the FLUX.1-schnell matched-clause run was coded).** The 1,550 images of
the FLUX.1-schnell matched-clause run (A4) are coded with Qwen3-VL using the same script, codebook, model revision and
settings as every other set, and analysed with the matched-clause design: I − C contrasts of prompt-level label MAD on
each axis with at least one complete prompt entry. The run contains both arms, 50 images each, for the 11 gender entries
and the first four racial entries, and one arm of a fifth racial entry, which is unpaired and not used; it contains no
geo-cultural entry. The testable contrasts (gender, skin tone, ethnic appearance) form their own Holm family. With four
racial entries, the smallest attainable two-sided sign-flip p-value is 0.125, so the racial contrasts are descriptive.
This amendment is written after the results of every other image set were seen and is reported as such.
