# Clusters: entities from the scored pair graph

Pairs become entities here, and the unit of quality changes with them: B-cubed over
records, reported separately from anything pairwise (CLAUDE.md, Invariants). The
pairwise figures for the same scorer are in `reports/synth/model.md`.

Regenerate with:

```bash
python -m dedup.cluster.evaluate --dataset synth-20k --out reports/synth/cluster.md
```

## Setup

- Dataset: `synth-20k`, test split — 5,663 records, 2,541 entities, 6,322 true pairs. Blocking emitted 289,516 candidate pairs holding 5,976 of them.
- Split: entity-grouped, `test_fraction=0.3`, `seed=0`.
- Scorer: the `reports/synth/model.md` pipeline rerun in-process — LightGBM over the 33-column pair vector, Platt fit out of fold over 5 entity-grouped folds.
- Cost model: `C_fm=20 C_fs=2 C_review=1 -> p_hi=0.9500 p_lo=0.5000`. A pair a partition leaves apart falls back to its band — review at p ≥ `p_lo`, rejection below — so the cost-based methods merge only where that beats both, which for a lone pair is exactly `p_hi` (`cluster/base.py`).
- Correlation clustering: local search from components at `p_hi`, from average linkage and from 8 seeded pivot orders; the lowest expected cost wins.

## Results

| method | criterion | clusters | largest | fused | split entities | implied pairs | review | B³ P | B³ R | B³ F1 | pair P | pair R | E[cost] | cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all singletons | no merges — the floor | 5,663 | 1 | 0 | 1523 | 0 | 4,310 | 1.0000 | 0.4487 | **0.6195** | 0.0000 | 0.0000 | 9,624.2 | 9,238 |
| connected components | p ≥ 0.9500, `p_hi` | 3,619 | 14 | 31 | 869 | 512 | 639 | 0.9820 | 0.7548 | **0.8535** | 0.9040 | 0.5718 | 14,136.9 | 12,993 |
| connected components | p ≥ 0.5000, `p_lo` — review band merged unreviewed | 3,213 | 24 | 68 | 718 | 2,619 | 0 | 0.9329 | 0.8108 | **0.8676** | 0.6200 | 0.6795 | 59,779.4 | 56,712 |
| connected components | p ≥ 0.2552, best pairwise F1 | 3,029 | 33 | 99 | 654 | 4,079 | 0 | 0.9007 | 0.8318 | **0.8649** | 0.5152 | 0.7200 | 95,168.8 | 89,220 |
| average linkage | mean merge gain > 0 — a lone pair at p > `p_hi` | 3,756 | 8 | 32 | 922 | 35 | 1,011 | 0.9924 | 0.7287 | **0.8404** | 0.9797 | 0.5112 | 7,808.9 | 7,279 |
| correlation clustering | lowest expected cost found | 3,757 | 8 | 29 | 924 | 32 | 1,007 | 0.9926 | 0.7290 | **0.8406** | 0.9794 | 0.5117 | 7,802.0 | 7,295 |
| connected components | true candidate edges — the ceiling | 2,602 | 8 | 0 | 61 | 251 | 452 | 1.0000 | 0.9876 | **0.9938** | 1.0000 | 0.9850 | 53,577.3 | 642 |

- **fused**: clusters holding records of more than one entity. **split entities**: entities spread over more than one cluster.
- **implied pairs**: merged pairs no edge at the row's criterion supports — what transitivity added. The cost-based rows are measured against edges at p ≥ `p_hi`, the pairs the bands auto-merge.
- **review**: pairs left in different clusters at p ≥ `p_lo`, queued for a reviewer.
- **pair P / R**: every pair the partition merges, candidate or not, against all 6,322 true pairs in the split.
- **E[cost]**: the objective under the model's probabilities, with unemitted pairs at p = 0. **cost**: what ground truth bills with the reviewer assumed correct — 20 per false merge, 1 per queued pair, 2 per true pair left apart outside the queue, the terms `reports/synth/model.md` bills its bands on.

## Chaining, shown

Connected components at `p_hi` is the auto-merge band with its edges closed under transitivity.
These are the first 10 of the 31 of its clusters that hold more than one entity, and what each
cost-based method did with the same records: **separated**, **separated, but split an entity**, or
**still fused**.

**1. 2 records, 2 entities** — 1 edge at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9765. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f00ab052b2f11:e006` | Battery Charger Canon NB-5L Lithium |
| `synthetic:fc9a3c0683cda:e005` | Canon NB-5L Lithium Battery |

**2. 6 records, 3 entities** — 7 edges at `p_hi`, 8 implied pairs; cross-entity edges scored
0.9777, 0.9749, 0.9747, 0.9740. Average linkage: **still fused**; correlation clustering: **still
fused**.

| entity | title |
| --- | --- |
| `synthetic:f08acb65e3df5:e000` | Pioneer CD-I200 iBus Interface For - CDI200 Compatible Pioneer Units Black |
| `synthetic:f08acb65e3df5:e000` | CD-I200 Pioneer iBus Cable For iPod - CDI200 iPod Fast |
| `synthetic:f08acb65e3df5:e008` | Pioneer Interface Cable For iPod - CDI200 CDI200 Direct Connection |
| `synthetic:f08acb65e3df5:e011` | Pioneer CD-B694 iBus Interface Cable iPod - CDI200 |
| `synthetic:f08acb65e3df5:e011` | Pioneer iBus Interface For iPod - CDI200 Your iPod |
| `synthetic:f08acb65e3df5:e011` | Pioneer iBus Interface Cable iPod - CDI200 Connection For Your iPod |

**3. 2 records, 2 entities** — 1 edge at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9690. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:fe10e9fe3aeea:e004` | Klipsch PM20 Speakers - System-Specific Loudness Contour |
| `synthetic:fe10e9fe3aeea:e006` | PM20 Computer Speakers |

**4. 6 records, 4 entities** — 9 edges at `p_hi`, 6 implied pairs; cross-entity edges scored
0.9802, 0.9760, 0.9706, 0.9681, 0.9662, 0.9640. Average linkage: **still fused**; correlation
clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f7c2e414b2efe:e000` | Scientific AT18 Wearable Waterproof Camcorder |
| `synthetic:f7c2e414b2efe:e002` | Oregon AT18 Wearable Waterproof Action |
| `synthetic:f7c2e414b2efe:e009` | Oregon AT18 Wearable Waterproof Action Camcorder |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Wearable Action Camcorder |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Waterproof Action Camcorder Oregon Scientific AT18 |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Wearable Waterproof Action Camcorder Conditions Mounts |

**5. 9 records, 3 entities** — 31 edges at `p_hi`, 5 implied pairs; cross-entity edges scored
0.9831, 0.9830, 0.9827, 0.9827, 0.9827, 0.9826, 0.9824, 0.9823, 0.9819, 0.9817, 0.9800, 0.9799,
0.9798, 0.9791, 0.9769, 0.9681, 0.9587, 0.9540. Average linkage: **still fused**; correlation
clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f1be761646246:e000` | Panasonic LMAF30U3 Three Pack Of Single-Sided 30 DVD-RAM Discs - LMAF30U3 Camcorder DVD |
| `synthetic:f1be761646246:e000` | Panasonic LM-AF30U3 Three Pack Single-Sided 30 Minute DVD-RAM - LMAF30U3 Compatible With DVD |
| `synthetic:f1be761646246:e000` | Panasonic LM-AF30U3 Pack Of Minute DVD-RAM Discs - Panasonic |
| `synthetic:f1be761646246:e000` | LM-AF30U3 Three Pack Single-Sided 30 Minute DVD-RAM Discs LMAF30U3 |
| `synthetic:f1be761646246:e000` | Panasonic LM-AF30U3 Three Pack Of Single-Sided 30 Minute Discs |
| `synthetic:f1be761646246:e003` | Panasonic Three Pack Single-Sided 30 Minute DVD-RAM Discs - LMAF30U3 |
| `synthetic:f1be761646246:e008` | Panasonic Three Pack Of Single-Sided 30 Minute DVD-RAM Discs - LMAF30U3 And DVD-RAM |
| `synthetic:f1be761646246:e008` | Panasonic Pack 30 Minure DVD-RAM - LMAF30U3 Pack Single-Sided |
| `synthetic:f1be761646246:e008` | Panasonic Three Pack Of 30 Discs - LMAF30U3 |

**6. 3 records, 2 entities** — 3 edges at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9742, 0.9723. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f7fad4cfd6fe2:e000` | KXTG4500B 5.8 GHz Cordless Phone System - KXTG4500B |
| `synthetic:f7fad4cfd6fe2:e000` | Black 5.8 GHz Cordless Phone sys - KXTG4500B System Frequency-Hopping Spread Spectrum |
| `synthetic:f7fad4cfd6fe2:e006` | Panasonic Black GHz Phoine System - KXTG4500B Spectrum Technology |

**7. 7 records, 3 entities** — 17 edges at `p_hi`, 4 implied pairs; cross-entity edges scored
0.9823, 0.9812, 0.9743, 0.9699, 0.9680, 0.9665, 0.9661. Average linkage: **still fused**;
correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f94058cece539:e002` | LG 24" LDF6920BB Flly Integrated Built In blk Dishwasher |
| `synthetic:f94058cece539:e005` | LG 24 in. LDF6920BB Fully Integrated Built In Dishwasher |
| `synthetic:f94058cece539:e005` | LG 24' LDF6920BB Fully Integrated Built In Balck |
| `synthetic:f94058cece539:e005` | LG LDF6920BB Fully Built In Bwack Dishwasher |
| `synthetic:f94058cece539:e005` | LG 24' LDF6920BB Integrated Built Black Dishwasher System |
| `synthetic:f94058cece539:e005` | LG LDF6920BB Fully Integrated Built In blk Dishwasher Motor |
| `synthetic:f94058cece539:e008` | LG 12' LDF6920BB Fully Integrated Built In Blalck Dishwasher - LDF6920BB |

**8. 4 records, 3 entities** — 3 edges at `p_hi`, 3 implied pairs; cross-entity edges scored
0.9781, 0.9772. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f09766bde4b60:e002` | LFC25770ST 25.0 Cu. Steel Door Bottom fridge LFC25771SS |
| `synthetic:f09766bde4b60:e002` | LG LFC25770ST Cu. Steel French Door Freezer fridge LFC25771SS |
| `synthetic:f09766bde4b60:e004` | LG LFC25770ST 25.0 Stainless Steel French Door Bottom fridge |
| `synthetic:f09766bde4b60:e005` | LG LFC25770ST Cu. Ft. Stainless Steel French Door Bottom Freezer Ddawer Stainless |

**9. 11 records, 6 entities** — 23 edges at `p_hi`, 32 implied pairs; cross-entity edges scored
0.9792, 0.9791, 0.9783, 0.9768, 0.9762, 0.9757, 0.9748, 0.9747, 0.9708, 0.9688, 0.9681, 0.9668,
0.9645, 0.9572, 0.9506. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f7896de221773:e003` | Nikon D700 Digital SLR Digital - Matrix Metering Scene Recognition |
| `synthetic:f7896de221773:e004` | Nikon D700 Digital Camera Megapixels FX-Format CMOS |
| `synthetic:f7896de221773:e005` | Nikon D700 SLR Digital Camera |
| `synthetic:f7896de221773:e006` | Nikon D700 Digital SLR Digital Camera - D-216EDNH |
| `synthetic:f7896de221773:e006` | D700 SLR Camera |
| `synthetic:f7896de221773:e006` | D216EDNH Nikon D700 SLR - Magnesium-Alloy Construction Point With |
| `synthetic:f7896de221773:e006` | Nikon D700 Digital SLR Digital Camera D216EDNH |
| `synthetic:f7896de221773:e006` | Nikon D700 dig SLR Digital Camera |
| `synthetic:f7896de221773:e006` | Nikon D700 dig - D216EDNH |
| `synthetic:f7896de221773:e007` | D700 Digital SLR Digital Camera - Color Matrix |
| `synthetic:f7896de221773:e010` | Nikon D700 Digital SLR dig - Live View Shooting |

**10. 6 records, 5 entities** — 6 edges at `p_hi`, 9 implied pairs; cross-entity edges scored
0.9738, 0.9713, 0.9712, 0.9697, 0.9690. Average linkage: **still fused**; correlation clustering:
**still fused**.

| entity | title |
| --- | --- |
| `synthetic:fc028fb3b0530:e002` | LG DLE3733W Whilte XL Capacity Electric Dryer |
| `synthetic:fc028fb3b0530:e005` | LG DLE3733W wht XL Capacity Electric Dryer Panel |
| `synthetic:fc028fb3b0530:e005` | DLE3733W XL Capacity Electric Dial-A-Cycle Drum Lgiht Transparent |
| `synthetic:fc028fb3b0530:e007` | LG DLE3733W White XL Capacity Electric Dryer |
| `synthetic:fc028fb3b0530:e009` | DLE3733W White XL Electric Dryer Dial-A-Cycle Drum Light |
| `synthetic:fc028fb3b0530:e010` | DLE3733W XL Capacity Electric Dryer - Cntrol Panel |

## Reading this honestly

- **B-cubed flatters doing nothing on this split.** Leaving every record a singleton scores B³ F1
  0.6195: precision is 1 by construction, and recall is 0.4487 because entities average 2.23
  records. Read every row against that floor rather than against zero — connected components at
  `p_hi` is +0.2341 above it.
- **The threshold that maximizes pairwise F1 trades cluster precision for recall, and chaining is
  why.** At p ≥ 0.2552, chosen on out-of-fold train predictions, connected components merges 4,079
  pairs no edge supports, against 512 at `p_hi`. The largest cluster grows from 14 records to 33,
  fused clusters from 31 to 99, and B³ precision falls from 0.9820 to 0.9007. B³ F1 still rises,
  0.8535 to 0.8649, because recall gained more than precision lost — which is why no F1, pairwise
  or B-cubed, should pick the threshold: the precision lost here is fused products.
- **Chaining happens at `p_hi` as well: 31 of its clusters fuse more than one product.** Average
  linkage cleanly separates 2 of them and correlation clustering 2; separating them but splitting
  an entity: average linkage 1, correlation clustering 2. Correlation clustering leaves 27 of the
  31 fused, and it keeps a record in a cluster only while no single move lowers expected cost
  under the model's own probabilities — so it is those probabilities holding them together, and
  the fix belongs to an `error-analyst` pass over `model/` and `features/`, not to this stage.
- **The ceiling binds.** Blocking emitted 5,976 of 6,322 true pairs, and transitivity over the
  true ones recovers 251 more, so the best any clusterer here can reach is B³ recall 0.9876.
  Cluster recall can exceed blocking's pair completeness — an entity of three needs only two of
  its pairs emitted — but not by more than that.
- **The objective and ground truth disagree on the winner.** Correlation clustering has the lowest
  expected cost (7,802.0), but average linkage the lowest realized cost (7,279). The objective
  reads the model's probabilities, so where the two disagree it is those probabilities that are
  wrong about some pairs.
- **Under the model's probabilities the truth is not the cheapest partition.** The ceiling row
  costs 53,577.3 in expectation against 7,802.0 for the best partition found, because merging a
  true pair the model scores low is priced as a false merge. No search over this objective reaches
  the ceiling, however thorough; that gap is the scorer's to close.
- **Review is a third outcome here too, and B-cubed does not see it.** Every pair a partition
  leaves apart falls back to its band — queued for review at p ≥ `p_lo`, rejected below — so a
  lone pair merges only where the bands would auto-merge it, and both cost columns bill on the
  same terms as `reports/synth/model.md`. The B-cubed figures score the partition before any
  review is resolved: connected components at `p_hi` leaves 639 pairs queued, average linkage
  1,011 pairs.
- **Clustering lowers the bill, not only the entity count.** On those same terms the pairwise
  bands alone — every pair decided on its own, producing no entities at all — bill 8,351; average
  linkage bills 7,279.

