# Baseline: TF-IDF char-3gram cosine

The number every later stage is justified against (CLAUDE.md, "Baseline to
beat"). No blocking, no features, no learned model: one vectorizer, one
cosine, one threshold.

Regenerate with:

```bash
python -m dedup.eval.baseline --dataset abt-buy --out reports/baseline_tfidf.md
```

## Setup

- Dataset: `abt-buy` — 2173 records, 1076 ground-truth entities.
- 1118 true pairs among 2,359,878 possible pairs.
- Vectorizer: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))`, L2-normalized, so the dot product is the cosine.
- Split: entity-grouped, `test_fraction=0.3`, `seed=0` → 1521 train / 652 test records.
- Test side: 341 true pairs in 212,226 pairs — 1 positive per 622 pairs.
- Similarity floor: none — every pair with a non-zero cosine is scored. Recall is divided by all 341 test true pairs
  either way, so a raised floor shows up as lost recall rather than as a smaller denominator.

The vectorizer is fit on the train split and the threshold is chosen on the
train split; both are then spent on test. Recall is divided by every true pair
in the test split, including any the similarity floor never scored.

## Results

| Text | Test F1 | Test P | Test R | PR-AUC | P@10 | P@100 | R-prec | Threshold | Oracle F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title | **0.5204** | 0.4605 | 0.5982 | 0.4720 | 0.600 | 0.620 | 0.504 | 0.6243 | 0.5249 |
| title + description | **0.4349** | 0.3265 | 0.6510 | 0.2898 | 0.000 | 0.250 | 0.337 | 0.4882 | 0.4418 |

**Threshold** is the best-F1 cut chosen on train. **Oracle F1** is the best F1
reachable on test, chosen with the test labels in hand — not a result, just the
size of what threshold selection left on the table.

Per-variant detail:

- **title** — 6,804 char n-gram features, 186,715 scored candidate pairs (reduction ratio 0.1202, recall ceiling 1.0000).
  - train: threshold=0.6243 P=0.3527 R=0.5701 F1=0.4358
  - test:  threshold=0.6243 P=0.4605 R=0.5982 F1=0.5204
- **title + description** — 9,674 char n-gram features, 205,484 scored candidate pairs (reduction ratio 0.0318, recall ceiling 1.0000).
  - train: threshold=0.4882 P=0.2065 R=0.6654 F1=0.3151
  - test:  threshold=0.4882 P=0.3265 R=0.6510 F1=0.4349

## Reading this honestly

- **This is the deduplication framing, not the record-linkage one.** Every pair of the
  2173 records is a candidate, and a pair is positive when the two records
  share an `entity_id`. Published Abt-Buy F1 figures are for the cross-source task — Abt
  row against Buy row only — over the 1097 shipped pairs. Transitivity through the size-3
  clusters makes 1118 pairs true here, and same-side pairs are in the
  candidate set. Close to the published task, not identical to it; compare accordingly.
- **PR-AUC, not ROC-AUC.** At 1 positive per 622 pairs, ROC-AUC reads ~0.99
  for a model with no practical value.
- **Train F1 comes out *below* test F1, and that is not a bug.** The train side holds
  1,155,960 pairs against test's 212,226 — 5.4x as many —
  while true pairs grow only linearly with records. At a fixed cosine threshold the false
  positives scale with the pair count and the true positives do not, so precision, and
  with it F1, falls as the catalog grows. A single global threshold therefore does not
  transfer across catalog sizes; the gap between the train-chosen threshold and the test
  oracle above shows how little that cost here, but on a 200k-record `synth/` catalog it
  is the whole problem.
- **Best-F1 thresholds are a baseline convention, not the plan.** The real thresholds come
  from expected cost, because a false merge corrupts the catalog and a false split merely
  leaves a duplicate (CLAUDE.md, Invariants).
- **No blocking.** Scoring is the full upper triangle, so this measures the similarity
  function with no recall ceiling above it. That is what makes it the right reference for
  `blocking/` — and why it will not scale to `synth/`.

