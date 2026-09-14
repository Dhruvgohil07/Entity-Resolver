# Feature diagnostics

What each column of the pair vector is worth on its own, before any model sees
it. `features/` has no classifier in it, so there is no F1 here — the number
that answers `reports/synth/baseline_tfidf.md`'s **test F1 0.2867** arrives with
`model/`. This table exists to catch the failures that are invisible from
inside a model: a constant column, a column whose sign is backwards, and a
column imputed on almost every pair all read as "the model did not improve".

Regenerate with:

```bash
python -m dedup.features.evaluate --dataset synth-20k --out reports/synth/features.md
```

## Setup

- Dataset: `synth-20k` — 18829 records, 8401 ground-truth entities, 21284 true pairs.
- Split: entity-grouped, `test_fraction=0.3`, `seed=0` → 13166 train / 5663 test records.
- Featurizer fit on **train only**, measured on **test**. 33 columns; semantic block off (`--semantic` enables it).
- Candidate pairs come from the full blocker union, run **inside each split**:

  | split | records | true pairs | candidates | PC | RR |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | train | 13166 | 14962 | 903,939 | 0.9296 | 0.9896 |
  | test | 5663 | 6322 | 289,516 | 0.9453 | 0.9819 |

- The test PC above is the **ceiling every number in this report inherits**. 5976 of 6322 true pairs reached the candidate set; any that did not are counted as missed recall, never dropped from the denominator.
- Class balance on test: 1 positive per 46 candidate pairs.

## Per-feature diagnostics

Ranked by PR-AUC. **Coverage** is the fraction of candidate pairs where the column holds a
real value rather than the fill — the mean of the column's own `companion_indicator`, and
1.0 for columns that are always defined. **Sep** is positive mean minus negative mean, and
**n+** is how many covered pairs are true matches. ⚠ marks a column that separates the
classes in the opposite direction to how it was declared.

| feature | kind | coverage | PR-AUC | pos mean | neg mean | sep | n+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `code_token_jaccard` |  | 0.7759 | 0.5532 | 0.7293 | 0.0222 | +0.7071 | 5134 |
| `model_number_prefix_ratio` |  | 0.7283 | 0.3644 | 0.8122 | 0.1332 | +0.6790 | 4823 |
| `model_number_exact` |  | 0.7283 | 0.3394 | 0.7383 | 0.0118 | +0.7265 | 4823 |
| `title_tfidf_cosine` |  | 1.0000 | 0.2788 | 0.6564 | 0.2074 | +0.4490 | 5976 |
| `code_token_shared_count` |  | 1.0000 | 0.1945 | 0.8148 | 0.0474 | +0.7674 | 5976 |
| `title_idf_overlap` |  | 1.0000 | 0.1884 | 0.4200 | 0.1438 | +0.2762 | 5976 |
| `code_best_ratio` |  | 0.7759 | 0.1874 | 0.9217 | 0.3493 | +0.5724 | 5134 |
| `desc_tfidf_cosine` |  | 0.6326 | 0.1480 | 0.7086 | 0.2493 | +0.4593 | 3735 |
| `desc_token_jaccard` |  | 0.6326 | 0.1397 | 0.5501 | 0.1928 | +0.3573 | 3735 |
| `title_token_set_ratio` |  | 1.0000 | 0.1214 | 0.8159 | 0.5346 | +0.2814 | 5976 |
| `title_token_jaccard` |  | 1.0000 | 0.1205 | 0.4516 | 0.2076 | +0.2439 | 5976 |
| `title_token_sort_ratio` |  | 1.0000 | 0.1167 | 0.7107 | 0.4757 | +0.2350 | 5976 |
| `cross_title_desc_cosine_max` |  | 0.9585 | 0.1121 | 0.5902 | 0.2136 | +0.3766 | 5738 |
| `cross_title_desc_cosine_min` |  | 0.9585 | 0.1089 | 0.4828 | 0.1638 | +0.3190 | 5738 |
| `title_ratio` |  | 1.0000 | 0.1002 | 0.6775 | 0.4672 | +0.2103 | 5976 |
| `title_token_containment` |  | 1.0000 | 0.0990 | 0.6899 | 0.3710 | +0.3189 | 5976 |
| `title_partial_ratio` |  | 1.0000 | 0.0845 | 0.7704 | 0.5436 | +0.2267 | 5976 |
| `title_common_prefix_ratio` |  | 1.0000 | 0.0360 | 0.1636 | 0.1040 | +0.0597 | 5976 |
| `digit_token_shared_count` |  | 1.0000 | 0.0281 | 0.1446 | 0.0336 | +0.1109 | 5976 |
| `desc_len_ratio` |  | 0.6326 | 0.0234 | 0.5275 | 0.4738 | +0.0537 | 3735 |
| `title_len_ratio` |  | 1.0000 | 0.0215 | 0.7663 | 0.7417 | +0.0246 | 5976 |
| `code_tokens_both_present` | indicator | 1.0000 | 0.0213 | 0.8591 | 0.7741 | +0.0850 | 5976 |
| `model_number_both_present` | indicator | 1.0000 | 0.0212 | 0.8071 | 0.7267 | +0.0804 | 5976 |
| `price_neither_present` | indicator | 1.0000 | 0.0209 | 0.4287 | 0.3651 | +0.0636 | 5976 |
| `price_both_present` | indicator | 1.0000 | 0.0205 | 0.2271 | 0.1860 | +0.0411 | 5976 |
| `digit_tokens_both_present` | indicator | 1.0000 | 0.0202 | 0.1401 | 0.1118 | +0.0283 | 5976 |
| `desc_any_present` | indicator | 1.0000 | 0.0195 | 0.9602 | 0.9585 | +0.0017 | 5976 |
| `desc_both_present` | indicator | 1.0000 | 0.0194 | 0.6250 | 0.6327 | -0.0077 | 5976 |
| `brand_both_present` | indicator | 1.0000 | 0.0193 | 0.2542 | 0.2658 | -0.0116 | 5976 |
| `price_abs_log_ratio` | distance | 0.1868 | 0.0174 | 0.2075 | 1.1688 | -0.9613 | 1357 |
| `price_rel_diff` | distance | 0.1868 | 0.0174 | 0.1800 | 0.5450 | -0.3650 | 1357 |
| `digit_token_jaccard` |  | 0.1123 | 0.0108 | 0.9259 | 0.2736 | +0.6523 | 837 |
| `brand_equal` |  | 0.2656 | 0.0060 | 0.6880 | 0.4907 | +0.1972 | 1519 |

Two things about how these are computed:

- **Every figure is over the covered pairs only.** Averaging the fill into a class mean
  reads a null as a mismatch — the error CLAUDE.md's missingness invariant exists to
  prevent, and it is just as wrong in a report as in a vector. The same restriction applies
  to the ranking, where a fill of 0.0 ranks last for a similarity column but — once negated
  — ranks *first* for a distance column, sorting an absent price as a perfect price match.
- **`distance` columns are negated before ranking**, so their PR-AUC is comparable with the
  rest. Without it `price_abs_log_ratio` would report as useless for being strong.

The recall denominator stays every true pair in the split, so a narrow column cannot look
strong by being asked less: its PR-AUC is bounded by the share of the split's true pairs it
covers, its `n+`. Read PR-AUC as *how much of the problem this column can reach*, and the
means as *whether it points the right way where it applies*. They answer different
questions and a column can score well on one and badly on the other.

## Columns pointing the wrong way

- None.

## Near-duplicate columns

- `digit_token_jaccard` / `digit_token_shared_count` — r = +0.971
- `title_token_jaccard` / `title_idf_overlap` — r = +0.955
- `title_token_jaccard` / `title_token_containment` — r = +0.953

## Reading this honestly

- **These are univariate numbers and the model is not.** A low PR-AUC here means the
  column cannot separate the classes *alone*. `code_best_ratio` near 1.0 means one
  thing when `model_number_exact` is 1 and the opposite when it is 0, and no single-column
  metric can show that. Use this table to find broken columns, not to prune the vector.
- **No PR-AUC here is comparable to the baseline's 0.2302.** `title_tfidf_cosine` is
  very nearly the baseline's own similarity function, and it scores *higher* in this
  table — which measures the candidate set, not the column. The baseline ranks the full
  N² upper triangle; this ranks the 289,516 pairs blocking left,
  having already discarded 98.19% of them, almost all
  negatives. Precision rises because the negatives are gone, not because the column got
  better. The like-for-like comparison against 0.2867 is a trained, calibrated model
  scored end to end, and it does not exist until `model/`.
- **The baseline row was scored above a similarity floor of 0.2.** Pairs at or below that
  cosine were never scored, so the baseline reaches recall 0.9813 at most rather than the
  full triangle's. Its F1, precision and recall are exact — the train-chosen threshold
  0.7060 sits above the floor, so no discarded pair could have passed it — but its PR-AUC
  ends at that ceiling and is a lower bound.
- **PR-AUC, not ROC-AUC.** At 1 positive per 46 pairs, ROC-AUC reads ~0.99
  for a column with no practical value (CLAUDE.md, Invariants).
- **Recall divides by every true pair in the split**, including the ones blocking never
  emitted. So no column's PR-AUC can exceed the test PC above, and pruning candidates
  cannot flatter these numbers.
- **Coverage below 1.0 is not a defect.** Every imputed column ships a companion indicator,
  so the model can tell a computed 0.0 from a filled one, and that pairing is enforced when
  the featurizer is constructed rather than by review. A low-coverage column still
  contributes more inside the vector than its row here suggests, because the model gets the
  indicator alongside it and this table scores the column alone.
- **The blocker parameters were tuned on Abt-Buy, not on this catalog**, so the PC figures
  inherited here measure how those settings transfer rather than a best-of-sweep ceiling.

