# Matched instruction probe — draft, no images generated

This prospective supplementary experiment addresses the confounding of demographic instructions with prompt length and descriptive detail. It does not recreate the original LLM rewriting process or identify a reasoning mechanism. The existing 600-image results remain historical-image reanalysis, separate from this experiment.

## Question and intervention

On the 31 original non-age prompt-category entries, does replacing a token-matched visual-quality instruction with an anti-stereotyping instruction change the specified CLIP representation diagnostics, with the requested role/action/scene and common detail held fixed?

Both arms preserve the baseline string verbatim, including historical spelling. Both append the same sentence: `Use natural lighting and clear visual detail.` The only substituted sentence is:

- I: `Avoid demographic stereotypes in this depiction.`
- C: `Avoid visual artifacts in this depiction.`

The current SD 2.1 mirror tokenizer encodes each sentence to nine positions in isolation, including special tokens. Before generation, verify full-prompt token-count equality for every pair and complete retention below 77 positions. Do not pad with meaningless tokens or silently truncate. The exact encoded input tables accompany the schedule.

The control is an active aesthetic instruction, not an untreated baseline. Interpretation is limited to this explicit clause substitution; a difference could depend on the particular control wording. Original task content and common detail are held fixed at the text level, but image-level semantic fidelity and ethical improvement are not thereby validated. Uniform label mass remains an evaluator diagnostic rather than a universal fairness target. This probe does not attribute effects to LLM assistance.

## Blocking and execution

Use five independent latent seeds per prompt entry, reused in both arms by resetting the generator before each render. Both arms appear once in every prompt × seed block, with a reproducible randomized order. Shuffle block order separately. This gives 31 × 5 × 2 = 310 images, of which racial images are evaluated on two axes. Preserve generated-image pixels, full/encoded prompts, seeds, run order, public model/tokenizer hashes, package versions, device/dtype, scheduler configuration and actual settings.

The proposed generator is the public SD 2.1 768 variant, with a pinned snapshot and DPMSolverMultistepScheduler, 768 × 768 pixels, 50 steps, guidance 7.5 and an empty negative prompt. These are newly specified settings, not inferred historical values. The full model files and a generation runtime are not currently available in the verified local assets. Generation requires a permitted normal download and a local resource preflight; no paid service or remote compute is authorized. Run a single technical pilot solely to measure memory/time and verify saved resources; do not tune prompts or settings using outcome scores.

## Analysis and scope

Average five images within each prompt × arm cell, then pair I−C by original prompt entry. Primary diagnostic: prompt-cell label MAD on the four axes, one four-test Holm family, with all differences/SDs/dz/unadjusted CIs/conditional sign-flip p-values reported. Secondary concept/reference similarities form a separate eight-test exploratory family; neither has a universal ethical direction. Racial images are not pooled across axes. Common latent seeds are nuisance controls, not independent prompt replicates.

The selected prompt counts remain gender 11, each racial axis 12 and geo-cultural eight. This is a resource-constrained selected-prompt probe, not a powered population-level fairness study. Before treating it as confirmatory, specify a scientifically meaningful effect and perform a prospective power analysis; five image repetitions cannot compensate for a small number of independent tasks. Do not stop early for significance, select only favorable prompts, or treat non-rejection as equivalence. Human reliability work remains deferred by the author's instruction.

## Current status

Prompt pairs and the 310-row schedule are prepared and token-validated in this folder. This is design preparation only. No new research images, pilot, outcome scores or inferential results exist for the probe, and the manuscript/response do not report it as conducted.
