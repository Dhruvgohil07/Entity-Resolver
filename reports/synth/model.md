# Model: LightGBM over the pair vector, calibrated, cost-banded

The first number in this project comparable to the baseline's **test F1 0.2867** (`reports/synth/baseline_tfidf.md`). Per-column PR-AUCs in
`reports/synth/features.md` are not that number and never were.

Regenerate with:

```bash
python -m dedup.model.evaluate --dataset synth-20k --out reports/synth/model.md
```

## Setup

- Dataset: `synth-20k` — 18,829 records, 8,401 entities, 21,284 true pairs.
- Split: entity-grouped, `test_fraction=0.3`, `seed=0` → 13,166 train / 5,663 test records.
- Model: LightGBM over the 33-column pair vector, featurizer fit on train only.
- Calibration: Platt, fit on out-of-fold predictions over 5 entity-grouped folds of train — 471,135 pairs, 14,441 positives.
- Cost model: `C_fm=20 C_fs=2 C_review=1 -> p_hi=0.9500 p_lo=0.5000`.

| split | records | candidates | true pairs found | in split | PC | RR | neg/pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 13,166 | 903,939 | 13,908 | 14,962 | 0.9296 | 0.9896 | 64.0 |
| test | 5,663 | 289,516 | 5,976 | 6,322 | 0.9453 | 0.9819 | 47.4 |

## Results — threshold chosen on train, spent on test

The baseline's protocol, so this row is the like-for-like comparison. The
threshold is chosen on the **out-of-fold** train predictions, because in-sample
train scores from a fitted booster are near-perfect and a threshold picked on
them would mean nothing.

| | Test F1 | Test P | Test R | PR-AUC | P@10 | P@100 | R-prec | Threshold | Oracle F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model | **0.7422** | 0.8655 | 0.6496 | 0.7544 | 0.000 | 0.560 | 0.723 | 0.3023 | 0.7436 |
| baseline (TF-IDF) | 0.2867 | 0.2262 | 0.3915 | 0.2302 | 0.789 | 0.810 | 0.286 | 0.7060 | 0.2897 |

F1 +0.4555 against the baseline. **Precision in these two rows is not measured
on the same candidate set** — see "Reading this honestly". Each row's threshold is
on its own scale: a cosine for the baseline, a calibrated probability for the model.

## Results — cost-derived bands

What the system actually does. Not an F1: three populations and a bill.

- **auto-merge** 3,549 pairs, 164 of them wrong (precision 0.9538, recall 0.5354 against all 6322 true pairs in the split)
- **review** 865 pairs (0.30% of candidates), 565 of them true
- **auto-reject** 285,102 pairs, losing 2026 true pairs outright, on top of 346 that blocking never emitted
- realized cost **8,197** review-equivalents

### Sensitivity to the cost ratio

| C_fm | C_fs | p_hi | p_lo | merge | bad | review | missed | auto-P | auto-R | cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2 | 0.900 | 0.500 | 3,799 | 205 | 615 | 2026 | 0.9460 | 0.5685 | 6,717 |
| 20 | 2 | 0.950 | 0.500 | 3,549 | 164 | 865 | 2026 | 0.9538 | 0.5354 | 8,197 |
| 50 | 2 | 0.980 | 0.500 | 273 | 44 | 4,141 | 2026 | 0.8388 | 0.0362 | 10,393 |
| 50 | 5 | 0.980 | 0.200 | 273 | 44 | 4,698 | 1783 | 0.8388 | 0.0362 | 15,813 |
| 100 | 2 | 0.990 | 0.500 | 0 | 0 | 4,414 | 2026 | 0.0000 | 0.0000 | 8,466 |

## Calibration

| scores | Brier | ECE | PR-AUC |
| --- | ---: | ---: | ---: |
| raw booster | 0.00709 | 0.00082 | 0.7544 |
| Platt (out-of-fold) | 0.00758 | 0.00374 | 0.7544 |

PR-AUC is identical in both rows and must be: Platt is monotone, so it cannot
reorder pairs. Calibration moves what the number *means*, never the ranking —
which is also why it cannot flatter a ranking metric.

Reliability of the calibrated probabilities:

| bin | predicted | observed | pairs | gap |
| --- | ---: | ---: | ---: | ---: |
| 0.00–0.07 | 0.0077 | 0.0053 | 283,700 | +0.0024 |
| 0.07–0.13 | 0.0944 | 0.2917 | 576 | -0.1972 |
| 0.13–0.20 | 0.1643 | 0.3755 | 269 | -0.2111 |
| 0.20–0.27 | 0.2338 | 0.3654 | 156 | -0.1316 |
| 0.27–0.33 | 0.3016 | 0.4545 | 143 | -0.1530 |
| 0.33–0.40 | 0.3650 | 0.4286 | 126 | -0.0636 |
| 0.40–0.47 | 0.4330 | 0.5000 | 84 | -0.0670 |
| 0.47–0.53 | 0.4989 | 0.5172 | 87 | -0.0183 |
| 0.53–0.60 | 0.5658 | 0.5641 | 78 | +0.0017 |
| 0.60–0.67 | 0.6313 | 0.4881 | 84 | +0.1432 |
| 0.67–0.73 | 0.6999 | 0.5361 | 97 | +0.1638 |
| 0.73–0.80 | 0.7644 | 0.6082 | 97 | +0.1561 |
| 0.80–0.87 | 0.8354 | 0.6165 | 133 | +0.2189 |
| 0.87–0.93 | 0.9051 | 0.7588 | 228 | +0.1463 |
| 0.93–1.00 | 0.9724 | 0.9511 | 3,658 | +0.0213 |

## Feature importance (LightGBM gain)

| # | feature | gain |
| ---: | --- | ---: |
| 1 | `desc_token_jaccard` | 909,689,400 |
| 2 | `title_tfidf_cosine` | 520,333,026 |
| 3 | `price_abs_log_ratio` | 510,254,410 |
| 4 | `code_best_ratio` | 453,769,632 |
| 5 | `title_ratio` | 246,689,524 |
| 6 | `title_token_sort_ratio` | 220,973,504 |
| 7 | `title_common_prefix_ratio` | 180,187,089 |
| 8 | `title_idf_overlap` | 46,678,466 |
| 9 | `cross_title_desc_cosine_min` | 45,797,674 |
| 10 | `title_len_ratio` | 39,141,938 |

## Reading this honestly

- **Precision is not comparable to the baseline's; recall is.** Blocking discarded 98.19% of the
  test triangle before the model saw anything, so the negatives this model is scored against are
  the hard ones that survived. The baseline scored the full N² triangle. Recall *is* like-for-like
  — both divide by every true pair in the split, 6322 of them — so the honest summary is that this
  beats the baseline as a **pipeline**, not that the classifier beats TF-IDF on equal footing.
- **The baseline row was scored above a similarity floor of 0.2.** Pairs at or below that cosine
  were never scored, so the baseline reaches recall 0.9813 at most rather than the full
  triangle's. Its F1, precision and recall are exact — the train-chosen threshold 0.7060 sits
  above the floor, so no discarded pair could have passed it — but its PR-AUC ends at that ceiling
  and is a lower bound.
- **The recall ceiling is inherited, and on the test split it binds.** Blocking reaches PC 0.9453
  on test, leaving 346 true pairs that no model can score, so every recall above is capped at
  0.9453. When recall disappoints, check blocking first.
- **The cost ratio is load-bearing on this dataset.** Across the sensitivity grid the review queue
  ranges from 615 to 4,698 pairs out of 289,516, so `C_fm` and `C_fs` should be set deliberately
  for this catalog rather than left at the recorded default.
- **Most auto-rejected true pairs are near misses.** 2026 true pairs fall below `p_lo`, but only
  445 score below 0.01; the rest sit between that and `p_lo` (0.50), where a higher `C_fs` would
  route them to review instead. A further 346 true pairs never reached the model at all, because
  no blocker emitted them — a blocking problem, not a model one.
- **No class reweighting, and hyperparameters are untuned.** Both are deliberate; see
  `model/train.py`. Tuning would need a third split, and tuning against test is the leak this
  protocol exists to prevent.

