# Blocking: candidate generation and the recall ceiling

`synth-20k` — N = 18,829 records, 21,284 ground-truth pairs, 177,256,206 possible pairs.
Blockers are deterministic given the records, so no seed applies. Candidate counts and
completeness figures reproduce exactly between runs — the ANN index is built
single-threaded for that reason. Only the wall-clock columns vary.

Regenerate with:

```bash
python -m dedup.blocking.evaluate --dataset synth-20k --out reports/synth/blocking.md
```

## The table

| blocker | params | candidates | PC | RR | build s | query s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| standard (model number) | key=model_number_key, max_block=100 | 20,775 | 0.5672 | 0.9999 | 0.0 | 0.2 |
| standard (code tokens) | key=code-shaped title tokens, max_block=100 | 51,864 | 0.6506 | 0.9997 | 0.3 | 0.3 |
| standard (rare tokens) | key=title tokens with df<=30, max_block=100 | 130,552 | 0.6094 | 0.9993 | 0.3 | 0.5 |
| sorted_neighborhood | w=20 | 357,561 | 0.3110 | 0.9980 | 0.0 | 0.3 |
| lsh (minhash) | 128p, t=0.4, token | 1,382,151 | 0.6047 | 0.9922 | 12.6 | 4.4 |
| ann (faiss HNSW) | M=32, ef=100, k=10 | 141,178 | 0.7705 | 0.9992 | 107.4 | 118.2 |
| union (all) | — | 1,758,253 | 0.9238 | 0.9901 | — | — |

**PC** is pair completeness — true pairs surviving, divided by all 21,284 ground-truth pairs (never by the survivors).
**RR** is reduction ratio — the fraction of the 177,256,206 possible pairs discarded.

There is deliberately no precision, F1 or accuracy column. A blocker's output is
overwhelmingly non-duplicates by construction; discarding non-duplicates is the job,
not an error.

### Ceiling caveats

- ⚠ standard (code tokens): 1 block(s) exceeded max_block_size=100 and were dropped, so any true pair held only by one of them is not in this candidate set

## What this means

**The union row is the only one the rest of the pipeline inherits.** Its pair
completeness of **0.9238** is a hard ceiling on system recall: the
1621 true pairs no blocker emitted are never scored by `features/`,
never seen by `model/`, and never reach `cluster/`. No amount of model work
recovers them.

Individual blockers are *expected* to be mediocre. They earn their place by failing
differently — a blocker with low standalone PC is worth keeping if it lifts the union,
and a blocker that raises the candidate count without lifting the union is pure cost.

## The pairs blocking missed

1621 of 21,284 true pairs never became candidates. This
set, not the count, is where the next blocker's design comes from:

1. `Wall Speaker Mount - 22WLBK Supports Speakers lbs Sold`
   `Omnimount Wall Spefaker Mount`
2. `Omnimount spkr - 96RYTX/US`
   `Wall Speaker Mount - 96-RYTX`
3. `Wall Speaker Mount - 96-RYTX`
   `Omnimount Mount - 6972P302 Sold Single Black`
4. `Sony Compact Player/Recorder CD-R CD-RW MP3`
   `Compact High Spehd Finalizing`
5. `Solo S2 Detector - 405YD18E`
   `Radar/Laser Detector`
6. `Kenwood 6-Disc CD - KDCC-669`
   `Kenwood CD Changer`
7. `6-Dsic Changer KDCC069`
   `Kenwood 6-Disc CD With Changer Control`
8. `DTC-175FP Cuisinart Automatic Brew Serve - And Pour-Through`
   `Cuisinart And`
9. `Sharp Over The Counter Microwave - 5337R663 Stainless`
   `Ohver Microwave R1216-SS Settings Auto-Touch Control Panel`
10. `Maytag Bisque Over-The-Range Microwave - MMV-4200BT`
   `Maytag Over-The-Range Oven`

## Reading this honestly

- **This catalog is synthetic.** `synth/` derived it from Abt-Buy's train split, as its
  `manifest.json` records, so no record of Abt-Buy's test split reached it and its ground
  truth is exact by construction. How its duplicates compare with real ones is measured in
  `reports/synth/realism.md` — read that before trusting any number here.
- **At this N the reduction ratio is load-bearing.** 177,256,206 possible pairs is
  past the 44,739,242 an exhaustive baseline can score in 1 GiB, so the candidate
  count above is the difference between a catalog `features/` can vectorize and one it
  cannot. Read RR beside that count, never instead of it.
- **The blocker parameters were tuned on Abt-Buy, not on this catalog.** The window,
  neighbour count and document-frequency cutoff were swept over Abt-Buy and applied here
  unchanged, so these figures measure how those settings transfer, not what tuning on this
  catalog would reach.
- **Which ceiling applies depends on what is being compared.** The 0.9238 above is the
  batch-deduplication number over the whole catalog. When `features/` and `model/` report
  test-split recall against the baseline's test-split F1, the ceiling that binds them is
  the test-split one, not this.
