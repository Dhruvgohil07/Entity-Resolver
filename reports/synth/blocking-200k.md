# Blocking: candidate generation and the recall ceiling

`synth-200k` — N = 165,714 records, 181,045 ground-truth pairs, 13,730,482,041 possible pairs.
Blockers are deterministic given the records, so no seed applies. Candidate counts and
completeness figures reproduce exactly between runs — the ANN index is built
single-threaded for that reason. Only the wall-clock columns vary.

Regenerate with:

```bash
python -m dedup.blocking.evaluate --dataset synth-200k --ann-components 128 --ann-neighbours 150 --lsh-max-neighbours 500 --out reports/synth/blocking-200k.md
```

## The table

| blocker | params | candidates | PC | RR | build s | query s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| standard (model number) | key=model_number_key, max_block=100 | 204,420 | 0.5355 | 1.0000 | 0.1 | 1.1 |
| standard (code tokens) | key=code-shaped title tokens, max_block=100 | 193,611 | 0.6076 | 1.0000 | 2.3 | 1.3 |
| standard (rare tokens) | key=title tokens with df<=30, max_block=100 | 309,409 | 0.4147 | 1.0000 | 1.5 | 1.4 |
| sorted_neighborhood | w=20 | 3,148,376 | 0.1247 | 0.9998 | 0.1 | 1.8 |
| lsh (minhash) | 128p, t=0.4, token, cap=500 | 13,667,938 | 0.4042 | 0.9990 | 63.0 | 95.7 |
| ann (faiss HNSW) | M=32, ef=100, k=150, svd=128 | 17,634,460 | 0.5000 | 0.9987 | 50.1 | 26.9 |
| union (all) | — | 27,150,108 | 0.8469 | 0.9980 | — | — |

**PC** is pair completeness — true pairs surviving, divided by all 181,045 ground-truth pairs (never by the survivors).
**RR** is reduction ratio — the fraction of the 13,730,482,041 possible pairs discarded.

There is deliberately no precision, F1 or accuracy column. A blocker's output is
overwhelmingly non-duplicates by construction; discarding non-duplicates is the job,
not an error.

### Ceiling caveats

- ⚠ standard (model number): 83 block(s) exceeded max_block_size=100 and were dropped, so any true pair held only by one of them is not in this candidate set
- ⚠ standard (code tokens): 131 block(s) exceeded max_block_size=100 and were dropped, so any true pair held only by one of them is not in this candidate set
- ⚠ lsh (minhash): 61611 record(s) had more than max_neighbours=500 LSH neighbours and were dropped, so any true pair depending on one of them is not in this candidate set

## What this means

**The union row is the only one the rest of the pipeline inherits.** Its pair
completeness of **0.8469** is a hard ceiling on system recall: the
27721 true pairs no blocker emitted are never scored by `features/`,
never seen by `model/`, and never reach `cluster/`. No amount of model work
recovers them.

Individual blockers are *expected* to be mediocre. They earn their place by failing
differently — a blocker with low standalone PC is worth keeping if it lifts the union,
and a blocker that raises the candidate count without lifting the union is pure cost.

## The pairs blocking missed

27721 of 181,045 true pairs never became candidates. This
set, not the count, is where the next blocker's design comes from:

1. `30WLBK/XAC Omnimount Mount WLBK`
   `Wall spkr Mount - 8885P321 Included Supports`
2. `30WLBK/XAC Omnimount Mount WLBK`
   `Omnimount Speaker Mount - 30WLBK`
3. `Wall spkr Mount - 8885P321 Included Supports`
   `Omnimount Wazl Mount - 30WLBK`
4. `Wall spkr Mount - 8885P321 Included Supports`
   `Omnimount Speaker Mount - 30WLBK`
5. `Omnimount Wall Mount - 86JUCS`
   `86JUCS-R Omnimount Wall Speaker`
6. `86JUCS-R Omnimount Wall Speaker`
   `Omnimount Wall spkr Mzount - 86-JUCS Necessary Hardware Included`
7. `Omnimount Wall spkr Mount`
   `Omnimount Wall Speaker Mont`
8. `Omnimount Wall spkr Mount`
   `Omnimount Wall Speaker Mont`
9. `Omnimount Wall - 8149X779 All`
   `Omnimount Wall Speaker Mount 03QCSC Mount QCSC`
10. `Wall Speaker Mount - 31CIPD Omnimount Wall Speaker Mount`
   `Omnimount Wall spkr Mount`

## Reading this honestly

- **This catalog is synthetic.** `synth/` derived it from Abt-Buy's train split, as its
  `manifest.json` records, so no record of Abt-Buy's test split reached it and its ground
  truth is exact by construction. How its duplicates compare with real ones is measured in
  `reports/synth/realism.md` — read that before trusting any number here.
- **At this N the reduction ratio is load-bearing.** 13,730,482,041 possible pairs is
  past the 44,739,242 an exhaustive baseline can score in 1 GiB, so the candidate
  count above is the difference between a catalog `features/` can vectorize and one it
  cannot. Read RR beside that count, never instead of it.
- **The blocker parameters were tuned on Abt-Buy, not on this catalog.** The window,
  neighbour count and document-frequency cutoff were swept over Abt-Buy and applied here
  unchanged, so these figures measure how those settings transfer, not what tuning on this
  catalog would reach.
- **Which ceiling applies depends on what is being compared.** The 0.8469 above is the
  batch-deduplication number over the whole catalog. When `features/` and `model/` report
  test-split recall against the baseline's test-split F1, the ceiling that binds them is
  the test-split one, not this.
