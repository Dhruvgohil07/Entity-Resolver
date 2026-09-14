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
| all singletons | no merges — the floor | 652 | 1 | 0 | 321 | 0 | 293 | 1.0000 | 0.4923 | **0.6598** | 0.0000 | 0.0000 | 391.0 | 405 |
| connected components | p ≥ 0.9500, `p_hi` | 376 | 4 | 4 | 60 | 11 | 12 | 0.9880 | 0.9054 | **0.9449** | 0.9485 | 0.8094 | 356.8 | 420 |
| connected components | p ≥ 0.5000, `p_lo` — review band merged unreviewed | 364 | 5 | 5 | 49 | 16 | 0 | 0.9838 | 0.9223 | **0.9521** | 0.9288 | 0.8416 | 504.3 | 548 |
| connected components | p ≥ 0.0099, best pairwise F1 | 316 | 11 | 19 | 25 | 159 | 0 | 0.9118 | 0.9606 | **0.9356** | 0.6243 | 0.9208 | 4,256.5 | 3,834 |
| average linkage | mean merge gain > 0 — a lone pair at p > `p_hi` | 384 | 3 | 1 | 64 | 0 | 23 | 0.9985 | 0.8978 | **0.9454** | 0.9963 | 0.7889 | 160.5 | 155 |
| correlation clustering | lowest expected cost found | 384 | 3 | 2 | 65 | 0 | 22 | 0.9964 | 0.8962 | **0.9437** | 0.9889 | 0.7859 | 159.8 | 194 |
| connected components | true candidate edges — the ceiling | 321 | 3 | 0 | 0 | 0 | 8 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1,235.1 | 8 |

- **fused**: clusters holding records of more than one entity. **split entities**: entities spread over more than one cluster.
- **implied pairs**: merged pairs no edge at the row's criterion supports — what transitivity added. The cost-based rows are measured against edges at p ≥ `p_hi`, the pairs the bands auto-merge.
- **review**: pairs left in different clusters at p ≥ `p_lo`, queued for a reviewer.
- **pair P / R**: every pair the partition merges, candidate or not, against all 341 true pairs in the split.
- **E[cost]**: the objective under the model's probabilities, with unemitted pairs at p = 0. **cost**: what ground truth bills with the reviewer assumed correct — 20 per false merge, 1 per queued pair, 2 per true pair left apart outside the queue, the terms `reports/model.md` bills its bands on.

## Chaining, shown

Connected components at `p_hi` is the auto-merge band with its edges closed under transitivity.
These are all 4 of its clusters that hold more than one entity, and what each cost-based method
did with the same records: **separated**, **separated, but split an entity**, or **still fused**.

**1. 4 records, 2 entities** — 3 edges at `p_hi`, 3 implied pairs; cross-entity edges scored
0.9944. Average linkage: **separated**; correlation clustering: **separated**.

| entity | title |
| --- | --- |
| `abt_buy:e00509` | Sony DVP-FX820 Black 8' Portable DVD Player - DVPFX820 |
| `abt_buy:e00509` | Sony DVP-FX820 Portable DVD Player - DVPFX820 |
| `abt_buy:e00512` | Sony DVP-FX820 Red 8' Portable DVD Player - DVPFX820R |
| `abt_buy:e00512` | Sony DVP-FX820/R Portable DVD Player - DVPFX820/R |

**2. 4 records, 2 entities** — 5 edges at `p_hi`, 1 implied pair; cross-entity edges scored
0.9944, 0.9924, 0.9810. Average linkage: **separated**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `abt_buy:e00194` | Weber Stainless Steel Genesis S320 LP Grill - 3780001 |
| `abt_buy:e00194` | Weber Genesis S-320 3780001 60' Freestanding Gas Grill |
| `abt_buy:e00195` | Weber Stainless Steel Genesis S320 Natural Gas Grill - 3880001 |
| `abt_buy:e00195` | Weber Genesis S-320 3880001 60' Freestanding Gas Grill with 637 sq. in. Cooking Surface, 3 Stainless Steel Burners, Flush-Mounted Side Burner & Stainless Steel Shroud: Natural Gas |

**3. 4 records, 3 entities** — 3 edges at `p_hi`, 3 implied pairs; cross-entity edges scored
0.9672, 0.9603. Average linkage: **separated**; correlation clustering: **separated**.

| entity | title |
| --- | --- |
| `abt_buy:e00024` | Maytag Over-The-Range Microwave Oven - MMV5207BK |
| `abt_buy:e00024` | Maytag 2.0 Cu. Ft. Over-the-Range Microwave Oven |
| `abt_buy:e00697` | LG Over-The-Range Black Microwave Oven - LMV1680BK |
| `abt_buy:e00699` | LG 2.0 Cu. Ft. Over-The-Range White Microwave Oven - LMVM2085WH |

**4. 3 records, 2 entities** — 2 edges at `p_hi`, 1 implied pair; cross-entity edges scored
0.9940. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `abt_buy:e00753` | Samsung YP-S2ZW 1GB Flash MP3 Player - YP-S2ZG/XAA |
| `abt_buy:e00754` | Samsung S2 White 1GB Flash MP3 Player - YPS2ZW |
| `abt_buy:e00754` | Samsung YP-S2ZW 1GB Flash MP3 Player - YP-S2ZW/XAA |

## Reading this honestly

- **B-cubed flatters doing nothing on this split.** Leaving every record a singleton scores B³ F1
  0.6598: precision is 1 by construction, and recall is 0.4923 because entities average 2.03
  records. Read every row against that floor rather than against zero — connected components at
  `p_hi` is +0.2851 above it.
- **The threshold that maximizes pairwise F1 makes worse clusters, and chaining is why.** At p ≥
  0.0099, chosen on out-of-fold train predictions, connected components merges 159 pairs no edge
  supports, against 11 at `p_hi`. The largest cluster grows from 4 records to 11, fused clusters
  from 4 to 19, and B³ precision falls from 0.9880 to 0.9118. B³ F1 falls with it, 0.9449 to
  0.9356.
- **Chaining happens at `p_hi` as well: 4 of its clusters fuse more than one product.** Average
  linkage cleanly separates 3 of them and correlation clustering 2. Correlation clustering leaves
  2 of the 4 fused, and it keeps a record in a cluster only while no single move lowers expected
  cost under the model's own probabilities — so it is those probabilities holding them together,
  and the fix belongs to an `error-analyst` pass over `model/` and `features/`, not to this stage.
  Average linkage separates 1 of those anyway. That is its merge order, not the objective:
  correlation clustering reached a lower expected cost overall with them fused.
- **The ceiling does not bind on this split.** Blocking emitted all 341 true pairs, so components
  over the true candidate edges rebuilds every entity (B³ F1 1.0000) and nothing above is capped
  by blocking — a property of this split, not a general result.
- **The objective and ground truth disagree on the winner.** Correlation clustering has the lowest
  expected cost (159.8), but average linkage the lowest realized cost (155). The objective reads
  the model's probabilities, so where the two disagree it is those probabilities that are wrong
  about some pairs.
- **Under the model's probabilities the truth is not the cheapest partition.** The ceiling row
  costs 1,235.1 in expectation against 159.8 for the best partition found, because merging a true
  pair the model scores low is priced as a false merge. No search over this objective reaches the
  ceiling, however thorough; that gap is the scorer's to close.
- **Review is a third outcome here too, and B-cubed does not see it.** Every pair a partition
  leaves apart falls back to its band — queued for review at p ≥ `p_lo`, rejected below — so a
  lone pair merges only where the bands would auto-merge it, and both cost columns bill on the
  same terms as `reports/model.md`. The B-cubed figures score the partition before any review is
  resolved: connected components at `p_hi` leaves 12 pairs queued, average linkage 23 pairs.
- **Clustering lowers the bill, not only the entity count.** On those same terms the pairwise
  bands alone — every pair decided on its own, producing no entities at all — bill 265; average
  linkage bills 155.

