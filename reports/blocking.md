# Blocking: candidate generation and the recall ceiling

`abt-buy` — N = 2,173 records, 1,118 ground-truth pairs, 2,359,878 possible pairs.
Blockers are deterministic given the records, so no seed applies. Candidate counts and
completeness figures reproduce exactly between runs — the ANN index is built
single-threaded for that reason. Only the wall-clock columns vary.

**These parameters were chosen against these same numbers.** The window, neighbour
count and document-frequency cutoff were selected by sweeping them over this full
catalog, not over a held-out split, so the figures below are fit to this dataset and
are optimistic as an estimate of what these settings would do on an unseen one. See
*Reading this honestly* for the held-out ceilings.

Regenerate with:

```bash
python -m dedup.blocking.evaluate --dataset abt-buy --out reports/blocking.md
```

## The table

| blocker | params | candidates | PC | RR | build s | query s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| standard (model number) | key=model_number_key, max_block=100 | 628 | 0.5349 | 0.9997 | 0.0 | 0.0 |
| standard (code tokens) | key=code-shaped title tokens, max_block=100 | 2,072 | 0.6708 | 0.9991 | 0.0 | 0.0 |
| standard (rare tokens) | key=title tokens with df<=30, max_block=100 | 29,555 | 0.8694 | 0.9875 | 0.1 | 0.1 |
| sorted_neighborhood | w=20 | 41,097 | 0.6190 | 0.9826 | 0.0 | 0.0 |
| lsh (minhash) | 128p, t=0.4, token | 27,828 | 0.5894 | 0.9882 | 0.7 | 0.1 |
| ann (faiss HNSW) | M=32, ef=100, k=10 | 14,603 | 0.9562 | 0.9938 | 3.6 | 2.8 |
| union (all) | — | 84,117 | 0.9928 | 0.9644 | — | — |

**PC** is pair completeness — true pairs surviving, divided by all 1,118 ground-truth pairs (never by the survivors).
**RR** is reduction ratio — the fraction of the 2,359,878 possible pairs discarded.

There is deliberately no precision, F1 or accuracy column. A blocker's output is
overwhelmingly non-duplicates by construction; discarding non-duplicates is the job,
not an error.

### Ceiling caveats

- No blocker dropped a block or otherwise lowered its own ceiling on this run.

## What this means

**The union row is the only one the rest of the pipeline inherits.** Its pair
completeness of **0.9928** is a hard ceiling on system recall: the
8 true pairs no blocker emitted are never scored by `features/`,
never seen by `model/`, and never reach `cluster/`. No amount of model work
recovers them.

Individual blockers are *expected* to be mediocre. They earn their place by failing
differently — a blocker with low standalone PC is worth keeping if it lifts the union,
and a blocker that raises the candidate count without lifting the union is pure cost.

## The pairs blocking missed

8 of 1,118 true pairs never became candidates. This
set, not the count, is where the next blocker's design comes from:

1. `Canon Color Ink Tank - CL41CL`
   `Canon Ink Cartridge For PIXMA iP1600, iP6210D and iP6220D Printers - 0617B002`
2. `Tivo Wireless Adapter - AG0100`
   `Tvio Wireless USB Network Adpator`
3. `LG DLEX7177RM Cherry Red XL Capacity Electric SteamDryer - DLEX7177RD`
   `LG 27' Front-Load Electric Dryer with 7.3 cu. ft. Capacity`
4. `Canon Black Photo Ink Cartridge - CLI8B`
   `Canon Ink Cartridge For PIXMA iP4200, iP5200, iP5200R and iP6600D Printers - 0620B002`
5. `LG LFX25971SB 24.7 Cu. Ft. Smooth Black French Door Bottom Freezer Refrigerator - LFX25971BK`
   `LG 25 Cu. Ft. Ice & Water Dispensing`
6. `LG Over-The-Range White Microwave Oven - LMV1680WH`
   `LG 1.6 cu.ft. Over the Range`
7. `LG Over-The-Range Stainless Steel Microwave Oven  - LMV1680SS`
   `1.6 cu.ft. Over the Range Microwave`
8. `LG 24' LDS4821BB Semi Integrated Built In Black Dishwasher - LDS4821BK`
   `LG Semi-Integrated Electronic Panel with Digital Status Display`

## Reading this honestly

- **Abt-Buy is pre-blocked.** It ships as two curated catalogs of ~1,000 records each,
  already scoped to overlapping product ranges. A high union PC here says the benchmark
  is small and clean, not that blocking is solved. `synth/` — 200k to 1M records with
  known ground truth — is where this stage earns its keep, and where a dense ANN index
  stops fitting in memory.
- **RR is flattered by a small N.** 2,359,878 possible pairs is a number a
  laptop can brute-force; the baseline does exactly that. The reduction ratio only
  becomes load-bearing when N² stops being computable.
- **The parameters were tuned on the catalog they are scored on.** No held-out split was
  used to pick the window, neighbour count or df cutoff, so treat the union figure as a
  best-of-sweep number rather than a clean estimate. Measured for comparison on the
  entity-grouped split (`seed=0`), where the same settings give union pair completeness
  **0.9949 on train and 1.0000 on test** — so the selection bias here is small, but it is
  present and unmeasured until parameters are chosen on train alone.
- **Which ceiling applies depends on what is being compared.** The 0.9928 above is the
  batch-deduplication number over the whole catalog. When `features/` and `model/` report
  test-split recall against the baseline's test-split F1, the ceiling that binds them is
  the test-split one, not this.
