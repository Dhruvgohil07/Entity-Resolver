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
| model | **0.7194** | 0.8157 | 0.6435 | 0.7608 | 1.000 | 1.000 | 0.702 | 0.1857 | 0.7210 |
| baseline (TF-IDF) | 0.2867 | 0.2262 | 0.3915 | 0.2302 | 0.789 | 0.810 | 0.286 | 0.7060 | 0.2897 |

F1 +0.4327 against the baseline. **Precision in these two rows is not measured
on the same candidate set** — see "Reading this honestly". Each row's threshold is
on its own scale: a cosine for the baseline, a calibrated probability for the model.

## Results — cost-derived bands

What the system actually does. Not an F1: three populations and a bill.

- **auto-merge** 3,429 pairs, 122 of them wrong (precision 0.9644, recall 0.5231 against all 6322 true pairs in the split)
- **review** 822 pairs (0.28% of candidates), 491 of them true
- **auto-reject** 285,265 pairs, losing 2178 true pairs outright, on top of 346 that blocking never emitted
- realized cost **7,618** review-equivalents

### Sensitivity to the cost ratio

| C_fm | C_fs | p_hi | p_lo | merge | bad | review | missed | auto-P | auto-R | cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2 | 0.900 | 0.500 | 3,636 | 171 | 615 | 2178 | 0.9530 | 0.5481 | 6,681 |
| 20 | 2 | 0.950 | 0.500 | 3,429 | 122 | 822 | 2178 | 0.9644 | 0.5231 | 7,618 |
| 50 | 2 | 0.980 | 0.500 | 1,381 | 21 | 2,870 | 2178 | 0.9848 | 0.2151 | 8,276 |
| 50 | 5 | 0.980 | 0.200 | 1,381 | 21 | 3,553 | 1921 | 0.9848 | 0.2151 | 14,208 |
| 100 | 2 | 0.990 | 0.500 | 0 | 0 | 4,251 | 2178 | 0.0000 | 0.0000 | 8,607 |

## Calibration

| scores | Brier | ECE | PR-AUC |
| --- | ---: | ---: | ---: |
| raw booster | 0.00757 | 0.00115 | 0.7608 |
| Platt (out-of-fold) | 0.00804 | 0.00422 | 0.7608 |

PR-AUC is identical in both rows and must be: Platt is monotone, so it cannot
reorder pairs. Calibration moves what the number *means*, never the ranking —
which is also why it cannot flatter a ranking metric.

Reliability of the calibrated probabilities:

| bin | predicted | observed | pairs | gap |
| --- | ---: | ---: | ---: | ---: |
| 0.00–0.07 | 0.0085 | 0.0057 | 283,533 | +0.0028 |
| 0.07–0.13 | 0.0944 | 0.2908 | 729 | -0.1964 |
| 0.13–0.20 | 0.1627 | 0.2938 | 320 | -0.1311 |
| 0.20–0.27 | 0.2312 | 0.3568 | 227 | -0.1256 |
| 0.27–0.33 | 0.2969 | 0.3941 | 170 | -0.0972 |
| 0.33–0.40 | 0.3638 | 0.3676 | 136 | -0.0038 |
| 0.40–0.47 | 0.4304 | 0.3505 | 97 | +0.0799 |
| 0.47–0.53 | 0.4998 | 0.4800 | 100 | +0.0198 |
| 0.53–0.60 | 0.5666 | 0.5222 | 90 | +0.0444 |
| 0.60–0.67 | 0.6358 | 0.3864 | 88 | +0.2494 |
| 0.67–0.73 | 0.7048 | 0.6667 | 99 | +0.0382 |
| 0.73–0.80 | 0.7657 | 0.4674 | 92 | +0.2983 |
| 0.80–0.87 | 0.8357 | 0.6016 | 123 | +0.2341 |
| 0.87–0.93 | 0.9024 | 0.6723 | 177 | +0.2301 |
| 0.93–1.00 | 0.9752 | 0.9595 | 3,535 | +0.0157 |

## Feature importance (LightGBM gain)

| # | feature | gain |
| ---: | --- | ---: |
| 1 | `code_token_jaccard` | 512,676 |
| 2 | `title_tfidf_cosine` | 204,532 |
| 3 | `model_number_prefix_ratio` | 183,045 |
| 4 | `desc_token_jaccard` | 68,988 |
| 5 | `cross_title_desc_cosine_max` | 58,245 |
| 6 | `desc_len_ratio` | 49,106 |
| 7 | `desc_tfidf_cosine` | 31,967 |
| 8 | `cross_title_desc_cosine_min` | 15,660 |
| 9 | `code_tokens_both_present` | 13,282 |
| 10 | `model_number_both_present` | 12,701 |

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
  ranges from 615 to 4,251 pairs out of 289,516, so `C_fm` and `C_fs` should be set deliberately
  for this catalog rather than left at the recorded default.
- **Most auto-rejected true pairs are near misses.** 2178 true pairs fall below `p_lo`, but only
  469 score below 0.01; the rest sit between that and `p_lo` (0.50), where a higher `C_fs` would
  route them to review instead. A further 346 true pairs never reached the model at all, because
  no blocker emitted them — a blocking problem, not a model one.
- **No class reweighting, and hyperparameters are untuned.** Both are deliberate; see
  `model/train.py`. Tuning would need a third split, and tuning against test is the leak this
  protocol exists to prevent.

