# Model: LightGBM over the pair vector, calibrated, cost-banded

The first number in this project comparable to the baseline's **test F1 0.5204** (`reports/baseline_tfidf.md`). Per-column PR-AUCs in
`reports/features.md` are not that number and never were.

Regenerate with:

```bash
python -m dedup.model.evaluate --dataset abt-buy --out reports/model.md
```

## Setup

- Dataset: `abt-buy` — 2,173 records, 1,076 entities, 1,118 true pairs.
- Split: entity-grouped, `test_fraction=0.3`, `seed=0` → 1,521 train / 652 test records.
- Model: LightGBM over the 33-column pair vector, featurizer fit on train only.
- Calibration: Platt, fit on out-of-fold predictions over 5 entity-grouped folds of train — 42,218 pairs, 777 positives.
- Cost model: `C_fm=20 C_fs=2 C_review=1 -> p_hi=0.9500 p_lo=0.5000`.

| split | records | candidates | true pairs found | in split | PC | RR | neg/pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,521 | 55,021 | 773 | 777 | 0.9949 | 0.9524 | 70.2 |
| test | 652 | 18,819 | 341 | 341 | 1.0000 | 0.9113 | 54.2 |

## Results — threshold chosen on train, spent on test

The baseline's protocol, so this row is the like-for-like comparison. The
threshold is chosen on the **out-of-fold** train predictions, because in-sample
train scores from a fitted booster are near-perfect and a threshold picked on
them would mean nothing.

| | Test F1 | Test P | Test R | PR-AUC | P@10 | P@100 | R-prec | Threshold | Oracle F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model | **0.8813** | 0.8603 | 0.9032 | 0.9489 | 1.000 | 0.990 | 0.886 | 0.0093 | 0.8978 |
| baseline (TF-IDF) | 0.5204 | 0.4605 | 0.5982 | 0.4720 | 0.600 | 0.620 | 0.504 | 0.6243 | 0.5249 |

F1 +0.3609 against the baseline. **Precision in these two rows is not measured
on the same candidate set** — see "Reading this honestly". Each row's threshold is
on its own scale: a cosine for the baseline, a calibrated probability for the model.

## Results — cost-derived bands

What the system actually does. Not an F1: three populations and a bill.

- **auto-merge** 271 pairs, 7 of them wrong (precision 0.9742, recall 0.7742 against all 341 true pairs in the split)
- **review** 19 pairs (0.10% of candidates), 16 of them true
- **auto-reject** 18,529 pairs, losing 61 true pairs outright
- realized cost **281** review-equivalents

### Sensitivity to the cost ratio

| C_fm | C_fs | p_hi | p_lo | merge | bad | review | missed | auto-P | auto-R | cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2 | 0.900 | 0.500 | 275 | 8 | 15 | 61 | 0.9709 | 0.7830 | 217 |
| 20 | 2 | 0.950 | 0.500 | 271 | 7 | 19 | 61 | 0.9742 | 0.7742 | 281 |
| 50 | 2 | 0.980 | 0.500 | 258 | 3 | 32 | 61 | 0.9884 | 0.7478 | 304 |
| 50 | 5 | 0.980 | 0.200 | 258 | 3 | 45 | 53 | 0.9884 | 0.7478 | 460 |
| 100 | 2 | 0.990 | 0.500 | 244 | 3 | 46 | 61 | 0.9877 | 0.7067 | 468 |

## Calibration

| scores | Brier | ECE | PR-AUC |
| --- | ---: | ---: | ---: |
| raw booster | 0.00310 | 0.00245 | 0.9489 |
| Platt (out-of-fold) | 0.00346 | 0.00101 | 0.9489 |

PR-AUC is identical in both rows and must be: Platt is monotone, so it cannot
reorder pairs. Calibration moves what the number *means*, never the ranking —
which is also why it cannot flatter a ranking metric.

Reliability of the calibrated probabilities:

| bin | predicted | observed | pairs | gap |
| --- | ---: | ---: | ---: | ---: |
| 0.00–0.07 | 0.0026 | 0.0026 | 18,505 | -0.0001 |
| 0.07–0.13 | 0.0933 | 0.2000 | 5 | -0.1067 |
| 0.13–0.20 | 0.1544 | 0.5000 | 6 | -0.3456 |
| 0.20–0.27 | 0.2218 | 0.4286 | 7 | -0.2067 |
| 0.33–0.40 | 0.3512 | 0.7500 | 4 | -0.3988 |
| 0.40–0.47 | 0.4591 | 1.0000 | 1 | -0.5409 |
| 0.47–0.53 | 0.4939 | 1.0000 | 2 | -0.5061 |
| 0.53–0.60 | 0.5811 | 1.0000 | 2 | -0.4189 |
| 0.60–0.67 | 0.6351 | 1.0000 | 2 | -0.3649 |
| 0.73–0.80 | 0.7655 | 1.0000 | 3 | -0.2345 |
| 0.80–0.87 | 0.8308 | 0.3333 | 3 | +0.4975 |
| 0.87–0.93 | 0.8968 | 1.0000 | 6 | -0.1032 |
| 0.93–1.00 | 0.9933 | 0.9707 | 273 | +0.0226 |

## Feature importance (LightGBM gain)

| # | feature | gain |
| ---: | --- | ---: |
| 1 | `model_number_prefix_ratio` | 36,810 |
| 2 | `title_tfidf_cosine` | 14,163 |
| 3 | `cross_title_desc_cosine_max` | 12,288 |
| 4 | `desc_len_ratio` | 5,152 |
| 5 | `title_len_ratio` | 2,594 |
| 6 | `cross_title_desc_cosine_min` | 2,443 |
| 7 | `code_best_ratio` | 1,847 |
| 8 | `model_number_both_present` | 1,845 |
| 9 | `desc_token_jaccard` | 1,802 |
| 10 | `code_token_jaccard` | 1,150 |

## Reading this honestly

- **Precision is not comparable to the baseline's; recall is.** Blocking discarded 91.13% of the
  test triangle before the model saw anything, so the negatives this model is scored against are
  the hard ones that survived. The baseline scored the full N² triangle. Recall *is* like-for-like
  — both divide by every true pair in the split, 341 of them — so the honest summary is that this
  beats the baseline as a **pipeline**, not that the classifier beats TF-IDF on equal footing.
- **The recall ceiling is inherited, and on the test split it does not bind.** Blocking reaches PC
  1.0000 on test (0.9949 on train), so no test figure here is capped by the blocker — a property
  of this split, not a general result.
- **The cost ratio is not validated by this dataset.** Across the sensitivity grid `C_fm` spans
  10–100 and the review queue moves only from 15 to 46 pairs out of 18,819, because few pairs
  score anywhere near either threshold. The ratio is a recorded judgment call (CLAUDE.md), and it
  only becomes load-bearing once `synth/` populates the middle of the distribution.
- **The auto-rejected true pairs are the real loss, and they are not a threshold problem.** 61
  true pairs fall below `p_lo`, and 33 of them score below 0.01 — the model is confidently wrong,
  not undecided. Blocking emitted every one of them. No threshold recovers a pair the classifier
  buried; that is an `error-analyst` question for the next stage, not a tuning knob.
- **`desc_len_ratio` ranks #4 on gain, and that is expected.** CLAUDE.md records it pointing
  backwards univariately on Abt-Buy — a measured consequence of the deduplication framing — and a
  tree model uses a backwards column correctly where a human reading the column name does not.
- **No class reweighting, and hyperparameters are untuned.** Both are deliberate; see
  `model/train.py`. Tuning would need a third split, and tuning against test is the leak this
  protocol exists to prevent.

