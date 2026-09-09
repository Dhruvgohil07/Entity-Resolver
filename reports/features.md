# Feature diagnostics

What each column of the pair vector is worth on its own, before any model sees
it. `features/` has no classifier in it, so there is no F1 here — the number
that answers `reports/baseline_tfidf.md`'s **test F1 0.5204** arrives with
`model/`. This table exists to catch the failures that are invisible from
inside a model: a constant column, a column whose sign is backwards, and a
column imputed on almost every pair all read as "the model did not improve".

Regenerate with:

```bash
python -m dedup.features.evaluate --dataset abt-buy --out reports/features.md
```

## Setup

- Dataset: `abt-buy` — 2173 records, 1076 ground-truth entities, 1118 true pairs.
- Split: entity-grouped, `test_fraction=0.3`, `seed=0` → 1521 train / 652 test records.
- Featurizer fit on **train only**, measured on **test**. 33 columns; semantic block off (`--semantic` enables it).
- Candidate pairs come from the full blocker union, run **inside each split**:

  | split | records | true pairs | candidates | PC | RR |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | train | 1521 | 777 | 55,021 | 0.9949 | 0.9524 |
  | test | 652 | 341 | 18,819 | 1.0000 | 0.9113 |

- The test PC above is the **ceiling every number in this report inherits**. 341 of 341 true pairs reached the candidate set; any that did not are counted as missed recall, never dropped from the denominator.
- Class balance on test: 1 positive per 55 candidate pairs.

## Per-feature diagnostics

Ranked by PR-AUC. **Coverage** is the fraction of candidate pairs where the column holds a
real value rather than the fill — the mean of the column's own `companion_indicator`, and
1.0 for columns that are always defined. **Sep** is positive mean minus negative mean, and
**n+** is how many covered pairs are true matches. ⚠ marks a column that separates the
classes in the opposite direction to how it was declared.

| feature | kind | coverage | PR-AUC | pos mean | neg mean | sep | n+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `code_token_jaccard` |  | 0.7456 | 0.6274 | 0.6584 | 0.0032 | +0.6552 | 268 |
| `model_number_prefix_ratio` |  | 0.6917 | 0.5784 | 0.8310 | 0.0431 | +0.7880 | 246 |
| `title_tfidf_cosine` |  | 1.0000 | 0.5222 | 0.6648 | 0.1153 | +0.5495 | 341 |
| `model_number_exact` |  | 0.6917 | 0.4714 | 0.6707 | 0.0005 | +0.6703 | 246 |
| `code_best_ratio` |  | 0.7456 | 0.4646 | 0.9610 | 0.2859 | +0.6751 | 268 |
| `cross_title_desc_cosine_max` |  | 0.9494 | 0.4473 | 0.4246 | 0.0915 | +0.3331 | 338 |
| `title_idf_overlap` |  | 1.0000 | 0.4318 | 0.4099 | 0.0716 | +0.3383 | 341 |
| `code_token_shared_count` |  | 1.0000 | 0.4261 | 0.6891 | 0.0075 | +0.6817 | 341 |
| `title_token_containment` |  | 1.0000 | 0.4099 | 0.7082 | 0.2517 | +0.4565 | 341 |
| `title_token_set_ratio` |  | 1.0000 | 0.4041 | 0.8138 | 0.4346 | +0.3792 | 341 |
| `cross_title_desc_cosine_min` |  | 0.9494 | 0.3301 | 0.2737 | 0.0590 | +0.2147 | 338 |
| `title_token_jaccard` |  | 1.0000 | 0.2876 | 0.4486 | 0.1271 | +0.3215 | 341 |
| `title_token_sort_ratio` |  | 1.0000 | 0.2451 | 0.7016 | 0.4030 | +0.2986 | 341 |
| `title_partial_ratio` |  | 1.0000 | 0.2162 | 0.7220 | 0.4628 | +0.2592 | 341 |
| `title_ratio` |  | 1.0000 | 0.1897 | 0.6591 | 0.4085 | +0.2506 | 341 |
| `title_common_prefix_ratio` |  | 1.0000 | 0.1350 | 0.2465 | 0.0649 | +0.1816 | 341 |
| `digit_token_shared_count` |  | 1.0000 | 0.0456 | 0.1789 | 0.0160 | +0.1629 | 341 |
| `digit_token_jaccard` |  | 0.1068 | 0.0333 | 0.8677 | 0.1285 | +0.7392 | 63 |
| `desc_tfidf_cosine` |  | 0.6447 | 0.0310 | 0.2673 | 0.1170 | +0.1503 | 200 |
| `desc_token_jaccard` |  | 0.6447 | 0.0209 | 0.1180 | 0.0710 | +0.0470 | 200 |
| `digit_tokens_both_present` | indicator | 1.0000 | 0.0206 | 0.1848 | 0.1054 | +0.0794 | 341 |
| `price_abs_log_ratio` | distance | 0.2145 | 0.0202 | 0.2469 | 1.3558 | -1.1090 | 62 |
| `price_rel_diff` | distance | 0.2145 | 0.0201 | 0.2018 | 0.6177 | -0.4159 | 62 |
| `title_len_ratio` |  | 1.0000 | 0.0195 | 0.7478 | 0.7416 | +0.0062 | 341 |
| `desc_any_present` | indicator | 1.0000 | 0.0189 | 0.9912 | 0.9486 | +0.0426 | 341 |
| `code_tokens_both_present` | indicator | 1.0000 | 0.0189 | 0.7859 | 0.7448 | +0.0411 | 341 |
| `model_number_both_present` | indicator | 1.0000 | 0.0187 | 0.7214 | 0.6912 | +0.0302 | 341 |
| `brand_both_present` | indicator | 1.0000 | 0.0177 | 0.0235 | 0.2434 | -0.2200 | 341 |
| `price_both_present` | indicator | 1.0000 | 0.0176 | 0.1818 | 0.2151 | -0.0333 | 341 |
| `price_neither_present` | indicator | 1.0000 | 0.0175 | 0.2727 | 0.3099 | -0.0372 | 341 |
| `desc_both_present` | indicator | 1.0000 | 0.0172 | 0.5865 | 0.6458 | -0.0593 | 341 |
| `desc_len_ratio` ⚠ |  | 0.6447 | 0.0060 | 0.2213 | 0.4763 | -0.2550 | 200 |
| `brand_equal` |  | 0.2394 | 0.0001 | 0.7500 | 0.3655 | +0.3845 | 8 |

Two things about how these are computed, both of which changed a published number here:

- **Every figure is over the covered pairs only.** Averaging the fill into a class mean
  reads a null as a mismatch — the error CLAUDE.md's missingness invariant exists to
  prevent, and it is just as wrong in a report as in a vector. `brand_equal` is covered on
  2.3% of positives against 24.3% of negatives, so averaging 0.0 in at those rates reported
  it as pointing *backwards*, when on pairs that actually have a brand it points forwards
  and strongly. The same restriction applies to the ranking, where a fill of 0.0 ranks last
  for a similarity column but — once negated — ranks *first* for a distance column, sorting
  an absent price as a perfect price match.
- **`distance` columns are negated before ranking**, so their PR-AUC is comparable with the
  rest. Without it `price_abs_log_ratio` would report as useless for being strong.

The recall denominator stays every true pair in the split, so a narrow column cannot look
strong by being asked less: `brand_equal` can address 8 of 341 true pairs, and its PR-AUC
is bounded near 0.02 accordingly. Read PR-AUC as *how much of the problem this column can
reach*, and the means as *whether it points the right way where it applies*. They answer
different questions and a column can score well on one and badly on the other.

## Columns pointing the wrong way

- `desc_len_ratio` — positives 0.2213 against negatives 0.4763, over 12,133 covered pairs (200 of them positive).

This is **not** a bug, and flipping the sign would be the wrong fix. It is the
deduplication framing showing through: every pair of the combined catalog is a candidate,
so same-side pairs (Buy against Buy) sit in the table alongside the cross-source pairs that
carry almost all the true matches. Abt descriptions average 249 characters against Buy's
34, so two descriptions of *similar* length are evidence of being same-side — which on this
dataset means evidence of being a non-match. A tree model uses that correctly. A human
reading the column name does not, which is why it is called out here.

Check `n+` before drawing a conclusion from any row in this section. A column covered on a
handful of true pairs can land here on noise, and the fix for that is more data, not a sign
flip. `brand_equal` was listed here in an earlier version of this report for a worse reason
than noise: the class means included the fill, so a missing brand was being counted as a
brand mismatch.

## Near-duplicate columns

- `digit_token_jaccard` / `digit_token_shared_count` — r = +0.959
- `title_token_jaccard` / `title_idf_overlap` — r = +0.953

## Reading this honestly

- **These are univariate numbers and the model is not.** A low PR-AUC here means the
  column cannot separate the classes *alone*. `code_best_ratio` near 1.0 means one
  thing when `model_number_exact` is 1 and the opposite when it is 0, and no single-column
  metric can show that. Use this table to find broken columns, not to prune the vector.
- **No PR-AUC here is comparable to the baseline's 0.4720.** `title_tfidf_cosine` is
  very nearly the baseline's own similarity function, and it scores *higher* in this
  table — which measures the candidate set, not the column. The baseline ranks the full
  N² upper triangle; this ranks the 18,819 pairs blocking left,
  having already discarded 91.13% of them, almost all
  negatives. Precision rises because the negatives are gone, not because the column got
  better. The like-for-like comparison against 0.5204 is a trained, calibrated model
  scored end to end, and it does not exist until `model/`.
- **PR-AUC, not ROC-AUC.** At 1 positive per 55 pairs, ROC-AUC reads ~0.99
  for a column with no practical value (CLAUDE.md, Invariants).
- **Recall divides by every true pair in the split**, including the ones blocking never
  emitted. So no column's PR-AUC can exceed the test PC above, and pruning candidates
  cannot flatter these numbers.
- **Coverage below 1.0 is not a defect.** Every imputed column ships a companion indicator,
  so the model can tell a computed 0.0 from a filled one, and that pairing is enforced when
  the featurizer is constructed rather than by review. A low-coverage column still
  contributes more inside the vector than its row here suggests, because the model gets the
  indicator alongside it and this table scores the column alone.
- **The blocker parameters were swept on the full catalog**, so the PC figures inherited
  here are best-of-sweep rather than a clean estimate. Small bias, still unquantified —
  it is recorded as an open question in CLAUDE.md and not resolved by this report.

