# Baseline: TF-IDF char-3gram cosine

The number every later stage is justified against (CLAUDE.md, "Baseline to
beat"). No blocking, no features, no learned model: one vectorizer, one
cosine, one threshold.

Regenerate with:

```bash
python -m dedup.eval.baseline --dataset synth-20k --min-similarity 0.2 --out reports/synth/baseline_tfidf.md
```

## Setup

- Dataset: `synth-20k` — 18829 records, 8401 ground-truth entities.
- 21284 true pairs among 177,256,206 possible pairs.
- Vectorizer: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))`, L2-normalized, so the dot product is the cosine.
- Split: entity-grouped, `test_fraction=0.3`, `seed=0` → 13166 train / 5663 test records.
- Test side: 6322 true pairs in 16,031,953 pairs — 1 positive per 2,536 pairs.
- Similarity floor: `0.2` — pairs at or below it are discarded unscored. Recall is divided by all 6322 test true pairs
  either way, so a raised floor shows up as lost recall rather than as a smaller denominator.

The vectorizer is fit on the train split and the threshold is chosen on the
train split; both are then spent on test. Recall is divided by every true pair
in the test split, including any the similarity floor never scored.

## Results

| Text | Test F1 | Test P | Test R | PR-AUC | P@10 | P@100 | R-prec | Threshold | Oracle F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title | **0.2867** | 0.2262 | 0.3915 | 0.2302 | 0.789 | 0.810 | 0.286 | 0.7060 | 0.2897 |
| title + description | **0.2387** | 0.2800 | 0.2080 | 0.2463 | 1.000 | 0.990 | 0.230 | 0.8968 | 0.2407 |

**Threshold** is the best-F1 cut chosen on train. **Oracle F1** is the best F1
reachable on test, chosen with the test labels in hand — not a result, just the
size of what threshold selection left on the table.

Per-variant detail:

- **title** — 18,874 char n-gram features, 268,868 scored candidate pairs (reduction ratio 0.9832, recall ceiling 0.9813).
  - train: threshold=0.7060 P=0.2833 R=0.3652 F1=0.3191
  - test:  threshold=0.7060 P=0.2262 R=0.3915 F1=0.2867
- **title + description** — 21,142 char n-gram features, 359,104 scored candidate pairs (reduction ratio 0.9776, recall ceiling 0.9967).
  - train: threshold=0.8968 P=0.3409 R=0.2046 F1=0.2557
  - test:  threshold=0.8968 P=0.2800 R=0.2080 F1=0.2387

## Reading this honestly

- **This is the deduplication framing.** Every pair of the 18829 records is a
  candidate, and a pair is positive when the two records share an `entity_id`. That makes
  21284 pairs true here, counted by transitivity: an entity of k records holds
  C(k, 2) of them.
- **PR-AUC, not ROC-AUC.** At 1 positive per 2,536 pairs, ROC-AUC reads ~0.99
  for a model with no practical value.
- **Train F1 comes out *above* test F1 on this catalog** (0.3191 against
  0.2867), although the train side holds 86,665,195 pairs against
  test's 16,031,953 — 5.4x as many. A fixed cosine threshold's false positives
  scale with the pair count and its true positives do not, which on its own pushes the larger
  side's precision down; here that size effect did not dominate. A single global threshold
  still does not transfer across catalog sizes, and the oracle column above measures what it
  left on the table on this split.
- **Best-F1 thresholds are a baseline convention, not the plan.** The real thresholds come
  from expected cost, because a false merge corrupts the catalog and a false split merely
  leaves a duplicate (CLAUDE.md, Invariants).
- **No blocking, but a similarity floor.** Scoring covers the full upper triangle, but pairs at
  or below cosine 0.2 were discarded to fit in memory, so the title variant is
  measured under a recall ceiling of 0.9813 rather than none. Recall still divides by every
  true pair, so the floor shows as lost recall — compare `blocking/` against this with that
  ceiling in mind.

