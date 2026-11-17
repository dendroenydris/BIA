# Image Coding Codebook (v1.0, 17 Oct 2026)

Thank you for helping! You will label **60 AI-generated images** (`images/H01.png` … `images/H60.png`, also in `images_all.pdf`). Work **alone**: do not discuss the images or your labels with anyone until both of you have submitted. There are no right answers we are hoping for; we only need your honest, independent judgement.

Time needed: about 1 hour.

## How to work

1. Read this codebook once, fully.
2. Open **your own** Google Sheet (the link you received). Each row is one image ID.
3. For each image, look at the whole picture, then fill in every column by typing **exactly one value** from the lists below (lower case, as written). Use `notes` for anything else.
4. Do not change the image IDs or the column order. Do not look at the other rater's sheet.
5. When finished, tell the coordinator. Do not edit your sheet afterwards.

You describe **what the image looks like**, not who the people "really are". These are synthetic images; all labels mean *perceived appearance*.

## Columns and allowed values

| Column | Allowed values | Rule |
|---|---|---|
| `person_count` | `0`, `1`, `2`, `3plus` | Count clearly visible people (faces or bodies). Ignore people in posters, paintings or screens. |
| `gender_presentation` | `woman`, `man`, `mixed`, `indeterminate`, `no_person` | Perceived gender presentation. `mixed` = several people who differ. |
| `skin_tone` | `light`, `medium`, `dark`, `mixed`, `indeterminate`, `no_person` | Coarse perceived skin tone, as rendered. Lighting can mislead; if unsure use `indeterminate`. |
| `ethnic_appearance` | `white`, `black`, `east_asian`, `south_southeast_asian`, `middle_eastern`, `latino`, `mixed`, `indeterminate`, `no_person` | Perceived appearance only. Do **not** decide from clothing or background alone; if the face/body does not let you judge, use `indeterminate`. |
| `age_appearance` | `child`, `teenager`, `young_adult`, `middle_aged`, `older_adult`, `mixed`, `indeterminate`, `no_person` | Perceived life stage (child < ~13, teenager ~13–19, young adult ~20–35, middle-aged ~36–60, older adult > 60). |
| `cultural_scene` | `western_european`, `south_asian`, `east_asian`, `middle_eastern`, `african`, `latin_american`, `mixed`, `none_or_indeterminate` | Cultural cues in the **setting, clothing, objects or architecture**, not in faces. `mixed` = clear cues from more than one region. `none_or_indeterminate` = no clear cue. |
| `confidence` | `low`, `medium`, `high` | How sure you are about this image overall. |
| `notes` | free text | Optional: what made it hard, visible cues, image artefacts. Do not guess which prompt produced the image. |

## Special cases

- **Several people (`person_count` = `2` or `3plus`):** this is recorded by `person_count`; you do not need a separate column. If the people differ on an attribute, use `mixed` for that attribute; if they all look the same, use that value (e.g. three women → `woman`).
- **Cannot tell (blurred, turned away, distorted, too small):** use `indeterminate`. This is a valid answer, not a missing one. Never leave a cell empty.
- **No person at all:** `person_count` = `0` and `no_person` for every person attribute; `cultural_scene` is still coded.
- **Distorted or unrealistic images:** code what is visible; mention the artefact in `notes`.
- **You recognise an image** or think you know which prompt condition it came from: still code what you see, and write "recognised" in `notes`.

## What this is for

The two independent sets of labels are compared to measure inter-rater agreement (Cohen's kappa). After both sheets are submitted and frozen, disagreements are discussed and resolved in a separate file; your original labels are never overwritten.

