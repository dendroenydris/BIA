# Historical aggregate audit and current revision results

The original plots and `significance_tests_summary.csv` are retained as historical outputs. Their significance classifications and mitigation rankings are withdrawn. Do not use them as current results.

The saved JSON files contain pooled label proportions and reference-subtype averages, not independent image or prompt scores. This audit fixes the label set in mean absolute deviation (MAD), including zero-frequency labels. Lower MAD means closer to the configured uniform label reference; it does not establish ethical fairness.

| Axis | Baseline | Bias-aware | LLM-assisted |
|---|---:|---:|---:|
| Gender MAD | 0.034884 | 0.175000 | 0.166667 |
| Skin-tone MAD | 0.274510 | 0.156863 | 0.222222 |
| Ethnic-reference MAD | 0.130196 | 0.210196 | 0.110000 |
| Geo-cultural MAD | 0.081176 | 0.101818 | 0.110000 |

The omitted light-skin label previously made LLM-assisted skin MAD appear to be 0.089286. No condition has the lowest MAD on every axis. These taxonomies mix person and scene labels and require construct validation.

`Embedding Distance` is a cosine similarity, and `Stereotype Frequency` is an average maximum reference cosine, not a count. The inventory includes counter-stereotypical references; neither measure has a universal better/worse direction. Reference subtypes are not experimental replicates. No valid significance, confidence interval, or sampling denominator can be reconstructed from these aggregates alone.

The historical evaluator selected 500 of 600 images before excluding age. The complete non-age design contains 465 unique images and 645 category-image records, because 180 racial images receive two evaluations. Those design counts do not establish the historical sampled denominator.

The full-design image-level re-evaluation has now completed: all 465 non-age images, 645 axis-image records, and 129 prompt-condition-axis cells. These current results supersede the historical sampled aggregate values above as primary manuscript results.

| Mean per-prompt MAD | Baseline | Bias-aware | LLM-assisted |
|---|---:|---:|---:|
| Gender | 0.300000 | 0.409091 | 0.354545 |
| Skin tone | 0.325926 | 0.300000 | 0.288889 |
| Ethnic references | 0.180000 | 0.273333 | 0.213333 |
| Geo-cultural | 0.147500 | 0.150000 | 0.162500 |

Pooled and mean-per-prompt MAD are different estimands. All 36 planned paired comparisons are reported. None of the MAD comparisons passes the common Holm correction. The sole corrected association is lower gender concept similarity, LLM-assisted minus baseline (mean difference −0.022825; Holm p 0.035156), which is not established ethical improvement. Unadjusted intervals and conditional tests retain small-selected-prompt, sign-exchangeability, and evaluator-validity limitations. See the anonymous empirical archive for complete primary and exploratory diagnostic/sensitivity outputs and the README for reproduction commands. No matched generation or repeat human annotation is claimed.
