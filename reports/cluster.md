# Clusters: entities from the scored pair graph

Pairs become entities here, and the unit of quality changes with them: B-cubed over
records, reported separately from anything pairwise (CLAUDE.md, Invariants). The
pairwise figures for the same scorer are in `reports/model.md`.

Regenerate with:

```bash
python -m dedup.cluster.evaluate --dataset abt-buy --out reports/cluster.md
```

## Setup

- Dataset: `abt-buy`, test split — 652 records, 321 entities, 341 true pairs. Blocking emitted 18,819 candidate pairs holding 341 of them.
- Split: entity-grouped, `test_fraction=0.3`, `seed=0`.
- Scorer: the `reports/model.md` pipeline rerun in-process — LightGBM over the 33-column pair vector, Platt fit out of fold over 5 entity-grouped folds.
- Cost model: `C_fm=20 C_fs=2 C_review=1 -> p_hi=0.9500 p_lo=0.5000`. A pair a partition leaves apart falls back to its band — review at p ≥ `p_lo`, rejection below — so the cost-based methods merge only where that beats both, which for a lone pair is exactly `p_hi` (`cluster/base.py`).
- Correlation clustering: local search from components at `p_hi`, from average linkage and from 8 seeded pivot orders; the lowest expected cost wins.

## Results

| method | criterion | clusters | largest | fused | split entities | implied pairs | review | B³ P | B³ R | B³ F1 | pair P | pair R | E[cost] | cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all singletons | no merges — the floor | 652 | 1 | 0 | 321 | 0 | 291 | 1.0000 | 0.4923 | **0.6598** | 0.0000 | 0.0000 | 395.5 | 413 |
| connected components | p ≥ 0.9500, `p_hi` | 388 | 6 | 3 | 70 | 16 | 22 | 0.9877 | 0.8891 | **0.9358** | 0.9298 | 0.7771 | 479.7 | 538 |
| connected components | p ≥ 0.5000, `p_lo` — review band merged unreviewed | 366 | 6 | 6 | 52 | 25 | 0 | 0.9778 | 0.9167 | **0.9462** | 0.8956 | 0.8299 | 718.4 | 776 |
| connected components | p ≥ 0.0169, best pairwise F1 | 326 | 10 | 20 | 30 | 90 | 0 | 0.9278 | 0.9504 | **0.9390** | 0.7160 | 0.8944 | 2,799.3 | 2,492 |
| average linkage | mean merge gain > 0 — a lone pair at p > `p_hi` | 395 | 3 | 0 | 73 | 0 | 32 | 1.0000 | 0.8829 | **0.9378** | 1.0000 | 0.7595 | 171.0 | 154 |
| correlation clustering | lowest expected cost found | 395 | 3 | 1 | 74 | 0 | 31 | 0.9980 | 0.8814 | **0.9361** | 0.9923 | 0.7566 | 170.1 | 193 |
| connected components | true candidate edges — the ceiling | 321 | 3 | 0 | 0 | 0 | 11 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1,353.8 | 11 |

- **fused**: clusters holding records of more than one entity. **split entities**: entities spread over more than one cluster.
- **implied pairs**: merged pairs no edge at the row's criterion supports — what transitivity added. The cost-based rows are measured against edges at p ≥ `p_hi`, the pairs the bands auto-merge.
- **review**: pairs left in different clusters at p ≥ `p_lo`, queued for a reviewer.
- **pair P / R**: every pair the partition merges, candidate or not, against all 341 true pairs in the split.
- **E[cost]**: the objective under the model's probabilities, with unemitted pairs at p = 0. **cost**: what ground truth bills with the reviewer assumed correct — 20 per false merge, 1 per queued pair, 2 per true pair left apart outside the queue, the terms `reports/model.md` bills its bands on.

## Chaining, shown

Connected components at `p_hi` is the auto-merge band with its edges closed under transitivity.
These are all 3 of its clusters that hold more than one entity, and what each cost-based method
did with the same records: **separated**, **separated, but split an entity**, or **still fused**.

**1. 4 records, 2 entities** — 4 edges at `p_hi`, 2 implied pairs; cross-entity edges scored
0.9952, 0.9893. Average linkage: **separated**; correlation clustering: **separated**.

| entity | title |
| --- | --- |
| `abt_buy:e00509` | Sony DVP-FX820 Black 8' Portable DVD Player - DVPFX820 |
| `abt_buy:e00509` | Sony DVP-FX820 Portable DVD Player - DVPFX820 |
| `abt_buy:e00512` | Sony DVP-FX820 Red 8' Portable DVD Player - DVPFX820R |
| `abt_buy:e00512` | Sony DVP-FX820/R Portable DVD Player - DVPFX820/R |

**2. 4 records, 2 entities** — 5 edges at `p_hi`, 1 implied pair; cross-entity edges scored
0.9953, 0.9952, 0.9719. Average linkage: **separated**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `abt_buy:e00194` | Weber Stainless Steel Genesis S320 LP Grill - 3780001 |
| `abt_buy:e00194` | Weber Genesis S-320 3780001 60' Freestanding Gas Grill |
| `abt_buy:e00195` | Weber Stainless Steel Genesis S320 Natural Gas Grill - 3880001 |
| `abt_buy:e00195` | Weber Genesis S-320 3880001 60' Freestanding Gas Grill with 637 sq. in. Cooking Surface, 3 Stainless Steel Burners, Flush-Mounted Side Burner & Stainless Steel Shroud: Natural Gas |

**3. 6 records, 3 entities** — 5 edges at `p_hi`, 10 implied pairs; cross-entity edges scored
0.9886, 0.9886. Average linkage: **separated**; correlation clustering: **separated**.

| entity | title |
| --- | --- |
| `abt_buy:e00099` | Weber Q 300 Liquid Propane Outdoor Grill - 426001 |
| `abt_buy:e00099` | Weber Q 300 Gas Grill |
| `abt_buy:e00530` | Weber Summit E-620 Copper Liquid Propane Gas Outdoor Grill - 1752001 |
| `abt_buy:e00530` | Weber Summit E-620 Copper LP Gas Grill |
| `abt_buy:e00709` | Weber Q 320 Liquid Propane Table And Outdoor Grill - 586002 |
| `abt_buy:e00709` | Weber Q 320 Gas Grill |

## Reading this honestly

- **B-cubed flatters doing nothing on this split.** Leaving every record a singleton scores B³ F1
  0.6598: precision is 1 by construction, and recall is 0.4923 because entities average 2.03
  records. Read every row against that floor rather than against zero — connected components at
  `p_hi` is +0.2760 above it.
- **The threshold that maximizes pairwise F1 trades cluster precision for recall, and chaining is
  why.** At p ≥ 0.0169, chosen on out-of-fold train predictions, connected components merges 90
  pairs no edge supports, against 16 at `p_hi`. The largest cluster grows from 6 records to 10,
  fused clusters from 3 to 20, and B³ precision falls from 0.9877 to 0.9278. B³ F1 still rises,
  0.9358 to 0.9390, because recall gained more than precision lost — which is why no F1, pairwise
  or B-cubed, should pick the threshold: the precision lost here is fused products.
- **Chaining happens at `p_hi` as well: 3 of its clusters fuse more than one product.** Average
  linkage cleanly separates 3 of them and correlation clustering 2. Correlation clustering leaves
  1 of the 3 fused, and it keeps a record in a cluster only while no single move lowers expected
  cost under the model's own probabilities — so it is those probabilities holding them together,
  and the fix belongs to an `error-analyst` pass over `model/` and `features/`, not to this stage.
  Average linkage separates 1 of those anyway. That is its merge order, not the objective:
  correlation clustering reached a lower expected cost overall with them fused.
- **The ceiling does not bind on this split.** Blocking emitted all 341 true pairs, so components
  over the true candidate edges rebuilds every entity (B³ F1 1.0000) and nothing above is capped
  by blocking — a property of this split, not a general result.
- **The objective and ground truth disagree on the winner.** Correlation clustering has the lowest
  expected cost (170.1), but average linkage the lowest realized cost (154). The objective reads
  the model's probabilities, so where the two disagree it is those probabilities that are wrong
  about some pairs.
- **Under the model's probabilities the truth is not the cheapest partition.** The ceiling row
  costs 1,353.8 in expectation against 170.1 for the best partition found, because merging a true
  pair the model scores low is priced as a false merge. No search over this objective reaches the
  ceiling, however thorough; that gap is the scorer's to close.
- **Review is a third outcome here too, and B-cubed does not see it.** Every pair a partition
  leaves apart falls back to its band — queued for review at p ≥ `p_lo`, rejected below — so a
  lone pair merges only where the bands would auto-merge it, and both cost columns bill on the
  same terms as `reports/model.md`. The B-cubed figures score the partition before any review is
  resolved: connected components at `p_hi` leaves 22 pairs queued, average linkage 32 pairs.
- **Clustering lowers the bill, not only the entity count.** On those same terms the pairwise
  bands alone — every pair decided on its own, producing no entities at all — bill 284; average
  linkage bills 154.

