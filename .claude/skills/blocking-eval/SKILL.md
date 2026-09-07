---
name: blocking-eval
description: How to evaluate and report entity-resolution blockers. Use whenever writing, changing, tuning, or benchmarking anything under src/dedup/blocking/ (standard, sorted_neighborhood, lsh, ann, union), whenever computing pair completeness or reduction ratio, and whenever producing or updating the blocking table in reports/.
---

# Evaluating blockers

Blocking sets a hard recall ceiling that nothing downstream can lift. A true duplicate pair no
blocker emits is unrecoverable — the classifier never sees it, the clusterer never sees it, and no
amount of model work brings it back. So a blocker is not scored like a classifier.

## The two numbers

Report both, always, together. Either one alone is uninterpretable — dropping every pair gives a
perfect reduction ratio, and emitting all N² gives perfect completeness.

**Pair completeness (PC)** — the ceiling.

```
PC = |true duplicate pairs in candidate set| / |true duplicate pairs in ground truth|
```

The denominator is the *full* ground-truth pair set. Computing it over pairs that survived
blocking makes PC identically 1.0 — a self-fulfilling bug that has fooled published papers.
Assert `denominator == len(ground_truth_pairs)` in the evaluator.

**Reduction ratio (RR)** — the cost saving.

```
RR = 1 - |candidate pairs| / (N * (N - 1) / 2)
```

Report the absolute candidate count alongside RR. At N = 200k, RR = 0.999 still leaves 20M pairs,
and the feature stage has to compute all of them — the ratio flatters what the absolute number
tells you plainly.

## Never report these for a blocker

Accuracy, precision, recall-of-non-pairs, or F1 over blocker output. A blocker's negatives are
not errors; discarding non-duplicates is its job. If a table has a `precision` column for a
blocker, the column is wrong, not low.

## Table format

Write to `reports/blocking.md`. One row per blocker, plus a mandatory `union (all)` row —
the union's PC is the system's actual ceiling, and it is the only row the rest of the pipeline
inherits.

```
| blocker              | params        | candidates | PC     | RR     | build s | query s |
|----------------------|---------------|-----------:|-------:|-------:|--------:|--------:|
| standard (brand)     | key=brand     |  1,204,331 | 0.8412 | 0.9862 |     1.2 |     3.4 |
| sorted_neighborhood  | w=20          |    412,880 | 0.7733 | 0.9953 |     2.8 |     1.1 |
| lsh (minhash)        | 128p, t=0.5   |    883,102 | 0.9105 | 0.9899 |    14.6 |     8.9 |
| ann (faiss HNSW)     | M=32, ef=100  |    701,455 | 0.8967 | 0.9920 |    62.3 |    11.4 |
| union (all)          | —             |  2,088,940 | 0.9684 | 0.9761 |       — |       — |
```

Head the table with dataset name, N, ground-truth pair count, and random seed. A blocking table
without N is not reproducible and not defensible.

## Interpreting a result

- **Union PC is the number that matters.** Individual blockers are expected to be mediocre and to
  miss different failure modes — that is why they are unioned. A blocker with low standalone PC
  earns its place if it lifts the union.
- **A blocker that raises candidates without raising union PC is pure cost.** Drop it.
- **When system recall disappoints, read this table before touching the model.** If union PC is
  0.90, the pipeline's recall ceiling is 0.90 no matter how good the classifier gets.
- Report which true pairs the union misses, not just how many. The missed set is where the next
  blocker's design comes from — grouped by failure mode (brand alias, token drop, unit swap),
  five worked examples beat a count.
