# Aggregate CLIP outputs of the original study

These files are the outputs of the original CLIP evaluation of the 600-image archive: pooled label proportions and
reference-subtype averages per category and condition (`results.json`, `results_strategy_{0,1,2}.json` for baseline,
manual instruction and LLM-assisted rewrite), the plots, and `significance_tests_summary.csv`. The significance tests
and mitigation rankings in that file are not used in the paper: the saved aggregates contain no independent image or
prompt scores, so no valid test, interval or sample size can be derived from them. The paper's CLIP analysis of the
archive is the image-level evaluation in `records/clip/original_archive_b32` and `results/clip/original_archive_b32`.

`aggregate_audit/` (written by `analysis/clip_statistics.py`) recomputes label MAD from these aggregates over the full
label set, counting absent labels as zero:

| Pooled label MAD | Baseline | Manual instruction | LLM-assisted |
|---|---:|---:|---:|
| Gender | 0.034884 | 0.175000 | 0.166667 |
| Skin tone | 0.274510 | 0.156863 | 0.222222 |
| Ethnic references | 0.130196 | 0.210196 | 0.110000 |
| Geo-culture | 0.081176 | 0.101818 | 0.110000 |

Computed over the observed labels only, the LLM-assisted skin-tone MAD is 0.089286, because no LLM-assisted image was
labelled light-skinned; this is the absent-label effect discussed in Section 4.5 of the paper. No condition has the
lowest MAD on every attribute.

`Embedding Distance` is a cosine similarity and `Stereotype Frequency` an average maximum reference cosine, not a count.
The reference phrases include counter-stereotypical ones, so neither measure has a fixed better or worse direction.

The original evaluator scored 500 of the 600 images before excluding age. The image-level evaluation covers all 465
non-age images (645 image-attribute records, because racial images are scored for both skin tone and ethnic
references); its mean prompt-level label MAD is:

| Mean prompt-level label MAD | Baseline | Manual instruction | LLM-assisted |
|---|---:|---:|---:|
| Gender | 0.300000 | 0.409091 | 0.354545 |
| Skin tone | 0.325926 | 0.300000 | 0.288889 |
| Ethnic references | 0.180000 | 0.273333 | 0.213333 |
| Geo-culture | 0.147500 | 0.150000 | 0.162500 |

Pooled and mean prompt-level MAD are different quantities. None of the 12 label-MAD comparisons passes the Holm
correction; all 36 comparisons, including the exploratory similarity measures, are in
`results/clip/original_archive_b32/primary_comparisons.csv`.
