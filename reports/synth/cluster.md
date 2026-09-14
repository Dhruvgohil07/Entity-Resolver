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
| all singletons | no merges — the floor | 5,663 | 1 | 0 | 1523 | 0 | 4,414 | 1.0000 | 0.4487 | **0.6195** | 0.0000 | 0.0000 | 9,351.6 | 9,158 |
| connected components | p ≥ 0.9500, `p_hi` | 3,584 | 14 | 67 | 874 | 540 | 672 | 0.9735 | 0.7562 | **0.8512** | 0.8907 | 0.5761 | 14,562.7 | 14,156 |
| connected components | p ≥ 0.5000, `p_lo` — review band merged unreviewed | 3,134 | 23 | 110 | 692 | 2,908 | 0 | 0.9169 | 0.8193 | **0.8654** | 0.6016 | 0.6968 | 65,909.3 | 62,174 |
| connected components | p ≥ 0.3023, best pairwise F1 | 2,960 | 26 | 134 | 631 | 4,327 | 0 | 0.8842 | 0.8391 | **0.8611** | 0.5126 | 0.7355 | 98,422.4 | 91,784 |
| average linkage | mean merge gain > 0 — a lone pair at p > `p_hi` | 3,739 | 8 | 69 | 934 | 49 | 1,111 | 0.9859 | 0.7250 | **0.8356** | 0.9697 | 0.5066 | 7,815.1 | 7,855 |
| correlation clustering | lowest expected cost found | 3,733 | 8 | 54 | 925 | 46 | 1,040 | 0.9878 | 0.7298 | **0.8394** | 0.9692 | 0.5172 | 7,787.5 | 7,864 |
| connected components | true candidate edges — the ceiling | 2,602 | 8 | 0 | 61 | 251 | 464 | 1.0000 | 0.9876 | **0.9938** | 1.0000 | 0.9850 | 51,734.8 | 654 |

- **fused**: clusters holding records of more than one entity. **split entities**: entities spread over more than one cluster.
- **implied pairs**: merged pairs no edge at the row's criterion supports — what transitivity added. The cost-based rows are measured against edges at p ≥ `p_hi`, the pairs the bands auto-merge.
- **review**: pairs left in different clusters at p ≥ `p_lo`, queued for a reviewer.
- **pair P / R**: every pair the partition merges, candidate or not, against all 6,322 true pairs in the split.
- **E[cost]**: the objective under the model's probabilities, with unemitted pairs at p = 0. **cost**: what ground truth bills with the reviewer assumed correct — 20 per false merge, 1 per queued pair, 2 per true pair left apart outside the queue, the terms `reports/synth/model.md` bills its bands on.

## Chaining, shown

Connected components at `p_hi` is the auto-merge band with its edges closed under transitivity.
These are the first 10 of the 67 of its clusters that hold more than one entity, and what each
cost-based method did with the same records: **separated**, **separated, but split an entity**, or
**still fused**.

**1. 2 records, 2 entities** — 1 edge at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9811. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f00ab052b2f11:e001` | 4574K550 Canon Battery Charger |
| `synthetic:fd5e4c8696f3f:e008` | 4581V016 Polk Audio CSI A4 Cherry Center Channel Loudspeaker |

**2. 2 records, 2 entities** — 1 edge at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9725. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f00ab052b2f11:e006` | Battery Charger Canon NB-5L Lithium |
| `synthetic:fc9a3c0683cda:e005` | Canon NB-5L Lithium Battery |

**3. 6 records, 3 entities** — 7 edges at `p_hi`, 8 implied pairs; cross-entity edges scored
0.9764, 0.9714, 0.9712, 0.9706. Average linkage: **still fused**; correlation clustering: **still
fused**.

| entity | title |
| --- | --- |
| `synthetic:f08acb65e3df5:e000` | Pioneer CD-I200 iBus Interface For - CDI200 Compatible Pioneer Units Black |
| `synthetic:f08acb65e3df5:e000` | CD-I200 Pioneer iBus Cable For iPod - CDI200 iPod Fast |
| `synthetic:f08acb65e3df5:e008` | Pioneer Interface Cable For iPod - CDI200 CDI200 Direct Connection |
| `synthetic:f08acb65e3df5:e011` | Pioneer CD-B694 iBus Interface Cable iPod - CDI200 |
| `synthetic:f08acb65e3df5:e011` | Pioneer iBus Interface For iPod - CDI200 Your iPod |
| `synthetic:f08acb65e3df5:e011` | Pioneer iBus Interface Cable iPod - CDI200 Connection For Your iPod |

**4. 2 records, 2 entities** — 1 edge at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9793. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f45343b1b3d0d:e001` | Samsung |
| `synthetic:ff1a7e24fce77:e006` | Samsng 46-inch 6 LCD Flat HDTV - LN61Z513 |

**5. 2 records, 2 entities** — 1 edge at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9637. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:fe10e9fe3aeea:e004` | Klipsch PM20 Speakers - System-Specific Loudness Contour |
| `synthetic:fe10e9fe3aeea:e006` | PM20 Computer Speakers |

**6. 6 records, 4 entities** — 11 edges at `p_hi`, 4 implied pairs; cross-entity edges scored
0.9726, 0.9719, 0.9712, 0.9675, 0.9673, 0.9666, 0.9635, 0.9632. Average linkage: **still fused**;
correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f7c2e414b2efe:e000` | Scientific AT18 Wearable Waterproof Camcorder |
| `synthetic:f7c2e414b2efe:e002` | Oregon AT18 Wearable Waterproof Action |
| `synthetic:f7c2e414b2efe:e009` | Oregon AT18 Wearable Waterproof Action Camcorder |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Wearable Action Camcorder |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Waterproof Action Camcorder Oregon Scientific AT18 |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Wearable Waterproof Action Camcorder Conditions Mounts |

**7. 14 records, 4 entities** — 44 edges at `p_hi`, 47 implied pairs; cross-entity edges scored
0.9787, 0.9786, 0.9783, 0.9778, 0.9776, 0.9773, 0.9772, 0.9772, 0.9771, 0.9763, 0.9761, 0.9754,
0.9753, 0.9750, 0.9728, 0.9707, 0.9706, 0.9676, 0.9671, 0.9623, 0.9573. Average linkage: **still
fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f1be761646246:e000` | Panasonic LMAF30U3 Three Pack Of Single-Sided 30 DVD-RAM Discs - LMAF30U3 Camcorder DVD |
| `synthetic:f1be761646246:e000` | Panasonic LM-AF30U3 Three Pack Single-Sided 30 Minute DVD-RAM - LMAF30U3 Compatible With DVD |
| `synthetic:f1be761646246:e000` | Panasonic LM-AF30U3 Pack Of Minute DVD-RAM Discs - Panasonic |
| `synthetic:f1be761646246:e000` | LM-AF30U3 Three Pack Single-Sided 30 Minute DVD-RAM Discs LMAF30U3 |
| `synthetic:f1be761646246:e000` | Panasonic LM-AF30U3 Three Pack Of Single-Sided 30 Minute Discs |
| `synthetic:f1be761646246:e001` | Panasonic LMAF30U5 Three Pack Of Single-Sided 30 Minute DVD-RAM Discs - LMAF30U3 |
| `synthetic:f1be761646246:e001` | Panasonic LM-AF30U5 Three Pack 30 Minute DVD-RAM Discs - LMAF30U3 |
| `synthetic:f1be761646246:e001` | Panasonic LM-AF30U5 Three Pack Of 30 DVD-RAM Discs LMAF30U3 DVD-RAM Drives Rewritable |
| `synthetic:f1be761646246:e001` | LMAF30U5 Panasonic Three Pack Of Single-Sided 30 Minute DVD-RAM - LMAF30U3 |
| `synthetic:f1be761646246:e001` | Panasonic LM-AF30U5 Three Pack Of Single-Sided 30 DVD-RAM Discs - LMAF30U3 DVD-RAM Discs LMAF30U3 Compatible |
| `synthetic:f1be761646246:e003` | Panasonic Three Pack Single-Sided 30 Minute DVD-RAM Discs - LMAF30U3 |
| `synthetic:f1be761646246:e008` | Panasonic Three Pack Of Single-Sided 30 Minute DVD-RAM Discs - LMAF30U3 And DVD-RAM |
| `synthetic:f1be761646246:e008` | Panasonic Pack 30 Minure DVD-RAM - LMAF30U3 Pack Single-Sided |
| `synthetic:f1be761646246:e008` | Panasonic Three Pack Of 30 Discs - LMAF30U3 |

**8. 3 records, 2 entities** — 3 edges at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9718, 0.9694. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f7fad4cfd6fe2:e000` | KXTG4500B 5.8 GHz Cordless Phone System - KXTG4500B |
| `synthetic:f7fad4cfd6fe2:e000` | Black 5.8 GHz Cordless Phone sys - KXTG4500B System Frequency-Hopping Spread Spectrum |
| `synthetic:f7fad4cfd6fe2:e006` | Panasonic Black GHz Phoine System - KXTG4500B Spectrum Technology |

**9. 6 records, 2 entities** — 11 edges at `p_hi`, 4 implied pairs; cross-entity edges scored
0.9811. Average linkage: **still fused**; correlation clustering: **separated**.

| entity | title |
| --- | --- |
| `synthetic:f927395fb591a:e000` | Canon EOS Rebel XS Black Digittl SLR Camera - XSREB-1855B Black Finish |
| `synthetic:f927395fb591a:e000` | Canon EOS Rebel Black dig SLR XSREB1855B |
| `synthetic:f927395fb591a:e000` | Canon EOS Rebel XS Digital SLR Camera XSREB1855B |
| `synthetic:f927395fb591a:e000` | Canon EOS Rebel XS Black SLR Camera - XSREB1855B |
| `synthetic:f927395fb591a:e000` | Canon EOS Rebel Black Digital SLR Camera - XSREB1855B |
| `synthetic:fc0d4bf4ca83e:e003` | Canon Easy Photo Pack 1338B001 Pack |

**10. 7 records, 3 entities** — 14 edges at `p_hi`, 7 implied pairs; cross-entity edges scored
0.9765, 0.9752, 0.9695, 0.9683. Average linkage: **still fused**; correlation clustering:
**separated**.

| entity | title |
| --- | --- |
| `synthetic:f94058cece539:e002` | LG 24" LDF6920BB Flly Integrated Built In blk Dishwasher |
| `synthetic:f94058cece539:e005` | LG 24 in. LDF6920BB Fully Integrated Built In Dishwasher |
| `synthetic:f94058cece539:e005` | LG 24' LDF6920BB Fully Integrated Built In Balck |
| `synthetic:f94058cece539:e005` | LG LDF6920BB Fully Built In Bwack Dishwasher |
| `synthetic:f94058cece539:e005` | LG 24' LDF6920BB Integrated Built Black Dishwasher System |
| `synthetic:f94058cece539:e005` | LG LDF6920BB Fully Integrated Built In blk Dishwasher Motor |
| `synthetic:f94058cece539:e008` | LG 12' LDF6920BB Fully Integrated Built In Blalck Dishwasher - LDF6920BB |

## Reading this honestly

- **B-cubed flatters doing nothing on this split.** Leaving every record a singleton scores B³ F1
  0.6195: precision is 1 by construction, and recall is 0.4487 because entities average 2.23
  records. Read every row against that floor rather than against zero — connected components at
  `p_hi` is +0.2318 above it.
- **The threshold that maximizes pairwise F1 trades cluster precision for recall, and chaining is
  why.** At p ≥ 0.3023, chosen on out-of-fold train predictions, connected components merges 4,327
  pairs no edge supports, against 540 at `p_hi`. The largest cluster grows from 14 records to 26,
  fused clusters from 67 to 134, and B³ precision falls from 0.9735 to 0.8842. B³ F1 still rises,
  0.8512 to 0.8611, because recall gained more than precision lost — which is why no F1, pairwise
  or B-cubed, should pick the threshold: the precision lost here is fused products.
- **Chaining happens at `p_hi` as well: 67 of its clusters fuse more than one product.** Average
  linkage cleanly separates 2 of them and correlation clustering 17; separating them but splitting
  an entity: correlation clustering 2. Correlation clustering leaves 48 of the 67 fused, and it
  keeps a record in a cluster only while no single move lowers expected cost under the model's own
  probabilities — so it is those probabilities holding them together, and the fix belongs to an
  `error-analyst` pass over `model/` and `features/`, not to this stage.
- **The ceiling binds.** Blocking emitted 5,976 of 6,322 true pairs, and transitivity over the
  true ones recovers 251 more, so the best any clusterer here can reach is B³ recall 0.9876.
  Cluster recall can exceed blocking's pair completeness — an entity of three needs only two of
  its pairs emitted — but not by more than that.
- **The objective and ground truth disagree on the winner.** Correlation clustering has the lowest
  expected cost (7,787.5), but average linkage the lowest realized cost (7,855). The objective
  reads the model's probabilities, so where the two disagree it is those probabilities that are
  wrong about some pairs.
- **Under the model's probabilities the truth is not the cheapest partition.** The ceiling row
  costs 51,734.8 in expectation against 7,787.5 for the best partition found, because merging a
  true pair the model scores low is priced as a false merge. No search over this objective reaches
  the ceiling, however thorough; that gap is the scorer's to close.
- **Review is a third outcome here too, and B-cubed does not see it.** Every pair a partition
  leaves apart falls back to its band — queued for review at p ≥ `p_lo`, rejected below — so a
  lone pair merges only where the bands would auto-merge it, and both cost columns bill on the
  same terms as `reports/synth/model.md`. The B-cubed figures score the partition before any
  review is resolved: connected components at `p_hi` leaves 672 pairs queued, average linkage
  1,111 pairs.
- **Clustering lowers the bill, not only the entity count.** On those same terms the pairwise
  bands alone — every pair decided on its own, producing no entities at all — bill 8,889; average
  linkage bills 7,855.

