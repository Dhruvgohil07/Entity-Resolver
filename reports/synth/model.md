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
| model | **0.7288** | 0.8486 | 0.6386 | 0.7675 | 1.000 | 1.000 | 0.711 | 0.2552 | 0.7294 |
| baseline (TF-IDF) | 0.2867 | 0.2262 | 0.3915 | 0.2302 | 0.789 | 0.810 | 0.286 | 0.7060 | 0.2897 |

F1 +0.4421 against the baseline. **Precision in these two rows is not measured
on the same candidate set** — see "Reading this honestly". Each row's threshold is
on its own scale: a cosine for the baseline, a calibrated probability for the model.

## Results — cost-derived bands

What the system actually does. Not an F1: three populations and a bill.

- **auto-merge** 3,487 pairs, 130 of them wrong (precision 0.9627, recall 0.5310 against all 6322 true pairs in the split)
- **review** 823 pairs (0.28% of candidates), 501 of them true
- **auto-reject** 285,206 pairs, losing 2118 true pairs outright, on top of 346 that blocking never emitted
- realized cost **7,659** review-equivalents

### Sensitivity to the cost ratio

| C_fm | C_fs | p_hi | p_lo | merge | bad | review | missed | auto-P | auto-R | cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2 | 0.900 | 0.500 | 3,675 | 177 | 635 | 2118 | 0.9518 | 0.5533 | 6,641 |
| 20 | 2 | 0.950 | 0.500 | 3,487 | 130 | 823 | 2118 | 0.9627 | 0.5310 | 7,659 |
| 50 | 2 | 0.980 | 0.500 | 1,611 | 25 | 2,699 | 2118 | 0.9845 | 0.2509 | 8,185 |
| 50 | 5 | 0.980 | 0.200 | 1,611 | 25 | 3,330 | 1872 | 0.9845 | 0.2509 | 13,940 |
| 100 | 2 | 0.990 | 0.500 | 0 | 0 | 4,310 | 2118 | 0.0000 | 0.0000 | 8,546 |

## Calibration

| scores | Brier | ECE | PR-AUC |
| --- | ---: | ---: | ---: |
| raw booster | 0.00736 | 0.00104 | 0.7675 |
| Platt (out-of-fold) | 0.00784 | 0.00397 | 0.7675 |

PR-AUC is identical in both rows and must be: Platt is monotone, so it cannot
reorder pairs. Calibration moves what the number *means*, never the ranking —
which is also why it cannot flatter a ranking metric.

Reliability of the calibrated probabilities:

| bin | predicted | observed | pairs | gap |
| --- | ---: | ---: | ---: | ---: |
| 0.00–0.07 | 0.0083 | 0.0056 | 283,643 | +0.0026 |
| 0.07–0.13 | 0.0960 | 0.2876 | 619 | -0.1916 |
| 0.13–0.20 | 0.1633 | 0.3291 | 313 | -0.1658 |
| 0.20–0.27 | 0.2315 | 0.3623 | 207 | -0.1308 |
| 0.27–0.33 | 0.3020 | 0.3935 | 155 | -0.0916 |
| 0.33–0.40 | 0.3675 | 0.3893 | 131 | -0.0218 |
| 0.40–0.47 | 0.4311 | 0.4348 | 92 | -0.0037 |
| 0.47–0.53 | 0.5012 | 0.4227 | 97 | +0.0785 |
| 0.53–0.60 | 0.5664 | 0.4082 | 98 | +0.1582 |
| 0.60–0.67 | 0.6335 | 0.5513 | 78 | +0.0822 |
| 0.67–0.73 | 0.7023 | 0.5682 | 88 | +0.1342 |
| 0.73–0.80 | 0.7667 | 0.6053 | 114 | +0.1614 |
| 0.80–0.87 | 0.8306 | 0.6715 | 137 | +0.1590 |
| 0.87–0.93 | 0.9053 | 0.7102 | 176 | +0.1950 |
| 0.93–1.00 | 0.9767 | 0.9577 | 3,568 | +0.0190 |

## Feature importance (LightGBM gain)

| # | feature | gain |
| ---: | --- | ---: |
| 1 | `code_token_jaccard` | 507,955 |
| 2 | `model_number_prefix_ratio` | 224,510 |
| 3 | `title_tfidf_cosine` | 178,037 |
| 4 | `desc_token_jaccard` | 61,867 |
| 5 | `cross_title_desc_cosine_max` | 61,590 |
| 6 | `desc_len_ratio` | 48,623 |
| 7 | `desc_tfidf_cosine` | 31,726 |
| 8 | `model_number_both_present` | 18,652 |
| 9 | `cross_title_desc_cosine_min` | 15,114 |
| 10 | `code_tokens_both_present` | 10,497 |

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
  ranges from 635 to 4,310 pairs out of 289,516, so `C_fm` and `C_fs` should be set deliberately
  for this catalog rather than left at the recorded default.
- **Most auto-rejected true pairs are near misses.** 2118 true pairs fall below `p_lo`, but only
  513 score below 0.01; the rest sit between that and `p_lo` (0.50), where a higher `C_fs` would
  route them to review instead. A further 346 true pairs never reached the model at all, because
  no blocker emitted them — a blocking problem, not a model one.
- **No class reweighting, and hyperparameters are untuned.** Both are deliberate; see
  `model/train.py`. Tuning would need a third split, and tuning against test is the leak this
  protocol exists to prevent.

