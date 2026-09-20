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
| all singletons | no merges — the floor | 5,663 | 1 | 0 | 1523 | 0 | 4,251 | 1.0000 | 0.4487 | **0.6195** | 0.0000 | 0.0000 | 9,759.2 | 9,299 |
| connected components | p ≥ 0.9500, `p_hi` | 3,650 | 13 | 31 | 889 | 481 | 629 | 0.9829 | 0.7502 | **0.8509** | 0.9143 | 0.5655 | 13,653.8 | 12,131 |
| connected components | p ≥ 0.5000, `p_lo` — review band merged unreviewed | 3,236 | 24 | 62 | 739 | 2,772 | 0 | 0.9305 | 0.8041 | **0.8627** | 0.6017 | 0.6685 | 62,761.7 | 60,132 |
| connected components | p ≥ 0.1857, best pairwise F1 | 2,909 | 30 | 127 | 638 | 5,242 | 0 | 0.8696 | 0.8376 | **0.8533** | 0.4525 | 0.7322 | 122,709.9 | 115,386 |
| average linkage | mean merge gain > 0 — a lone pair at p > `p_hi` | 3,781 | 8 | 29 | 938 | 42 | 1,004 | 0.9929 | 0.7243 | **0.8376** | 0.9815 | 0.5041 | 8,055.7 | 7,252 |
| correlation clustering | lowest expected cost found | 3,783 | 8 | 30 | 939 | 42 | 995 | 0.9928 | 0.7247 | **0.8378** | 0.9807 | 0.5051 | 8,047.2 | 7,303 |
| connected components | true candidate edges — the ceiling | 2,602 | 8 | 0 | 61 | 251 | 453 | 1.0000 | 0.9876 | **0.9938** | 1.0000 | 0.9850 | 54,866.1 | 643 |

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
0.9736. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f00ab052b2f11:e006` | Battery Charger Canon NB-5L Lithium |
| `synthetic:fc9a3c0683cda:e005` | Canon NB-5L Lithium Battery |

**2. 6 records, 3 entities** — 7 edges at `p_hi`, 8 implied pairs; cross-entity edges scored
0.9768, 0.9762, 0.9762, 0.9743. Average linkage: **still fused**; correlation clustering: **still
fused**.

| entity | title |
| --- | --- |
| `synthetic:f08acb65e3df5:e000` | Pioneer CD-I200 iBus Interface For - CDI200 Compatible Pioneer Units Black |
| `synthetic:f08acb65e3df5:e000` | CD-I200 Pioneer iBus Cable For iPod - CDI200 iPod Fast |
| `synthetic:f08acb65e3df5:e008` | Pioneer Interface Cable For iPod - CDI200 CDI200 Direct Connection |
| `synthetic:f08acb65e3df5:e011` | Pioneer CD-B694 iBus Interface Cable iPod - CDI200 |
| `synthetic:f08acb65e3df5:e011` | Pioneer iBus Interface For iPod - CDI200 Your iPod |
| `synthetic:f08acb65e3df5:e011` | Pioneer iBus Interface Cable iPod - CDI200 Connection For Your iPod |

**3. 4 records, 4 entities** — 3 edges at `p_hi`, 3 implied pairs; cross-entity edges scored
0.9772, 0.9703, 0.9616. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f45343b1b3d0d:e005` | Samsung 7.1-Channel Home Theater System - System Power |
| `synthetic:f45343b1b3d0d:e008` | 7.1-Channel Blu-ray Home Theater sys - Full Playback Black Finish |
| `synthetic:f45343b1b3d0d:e011` | Samsung 7.1-Channel Home System |
| `synthetic:ff37f41901015:e011` | Onkyo Black 7.1-Channel Home Theater sys |

**4. 2 records, 2 entities** — 1 edge at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9689. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:fe10e9fe3aeea:e004` | Klipsch PM20 Speakers - System-Specific Loudness Contour |
| `synthetic:fe10e9fe3aeea:e006` | PM20 Computer Speakers |

**5. 6 records, 4 entities** — 11 edges at `p_hi`, 4 implied pairs; cross-entity edges scored
0.9787, 0.9721, 0.9704, 0.9666, 0.9604, 0.9554, 0.9510, 0.9501. Average linkage: **still fused**;
correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f7c2e414b2efe:e000` | Scientific AT18 Wearable Waterproof Camcorder |
| `synthetic:f7c2e414b2efe:e002` | Oregon AT18 Wearable Waterproof Action |
| `synthetic:f7c2e414b2efe:e009` | Oregon AT18 Wearable Waterproof Action Camcorder |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Wearable Action Camcorder |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Waterproof Action Camcorder Oregon Scientific AT18 |
| `synthetic:f7c2e414b2efe:e011` | Oregon Scientific AT18 Wearable Waterproof Action Camcorder Conditions Mounts |

**6. 9 records, 3 entities** — 31 edges at `p_hi`, 5 implied pairs; cross-entity edges scored
0.9822, 0.9821, 0.9821, 0.9820, 0.9820, 0.9818, 0.9818, 0.9817, 0.9816, 0.9815, 0.9812, 0.9797,
0.9795, 0.9794, 0.9779, 0.9694, 0.9678, 0.9540. Average linkage: **still fused**; correlation
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

**7. 3 records, 2 entities** — 3 edges at `p_hi`, 0 implied pairs; cross-entity edges scored
0.9766, 0.9754. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f7fad4cfd6fe2:e000` | KXTG4500B 5.8 GHz Cordless Phone System - KXTG4500B |
| `synthetic:f7fad4cfd6fe2:e000` | Black 5.8 GHz Cordless Phone sys - KXTG4500B System Frequency-Hopping Spread Spectrum |
| `synthetic:f7fad4cfd6fe2:e006` | Panasonic Black GHz Phoine System - KXTG4500B Spectrum Technology |

**8. 7 records, 3 entities** — 17 edges at `p_hi`, 4 implied pairs; cross-entity edges scored
0.9817, 0.9803, 0.9702, 0.9682, 0.9676, 0.9644, 0.9508. Average linkage: **still fused**;
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

**9. 4 records, 3 entities** — 3 edges at `p_hi`, 3 implied pairs; cross-entity edges scored
0.9741, 0.9707. Average linkage: **still fused**; correlation clustering: **still fused**.

| entity | title |
| --- | --- |
| `synthetic:f09766bde4b60:e002` | LFC25770ST 25.0 Cu. Steel Door Bottom fridge LFC25771SS |
| `synthetic:f09766bde4b60:e002` | LG LFC25770ST Cu. Steel French Door Freezer fridge LFC25771SS |
| `synthetic:f09766bde4b60:e004` | LG LFC25770ST 25.0 Stainless Steel French Door Bottom fridge |
| `synthetic:f09766bde4b60:e005` | LG LFC25770ST Cu. Ft. Stainless Steel French Door Bottom Freezer Ddawer Stainless |

**10. 11 records, 6 entities** — 22 edges at `p_hi`, 33 implied pairs; cross-entity edges scored
0.9783, 0.9773, 0.9769, 0.9750, 0.9740, 0.9730, 0.9728, 0.9684, 0.9681, 0.9678, 0.9675, 0.9655,
0.9612, 0.9571. Average linkage: **still fused**; correlation clustering: **still fused**.

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

## Reading this honestly

- **B-cubed flatters doing nothing on this split.** Leaving every record a singleton scores B³ F1
  0.6195: precision is 1 by construction, and recall is 0.4487 because entities average 2.23
  records. Read every row against that floor rather than against zero — connected components at
  `p_hi` is +0.2315 above it.
- **The threshold that maximizes pairwise F1 trades cluster precision for recall, and chaining is
  why.** At p ≥ 0.1857, chosen on out-of-fold train predictions, connected components merges 5,242
  pairs no edge supports, against 481 at `p_hi`. The largest cluster grows from 13 records to 30,
  fused clusters from 31 to 127, and B³ precision falls from 0.9829 to 0.8696. B³ F1 still rises,
  0.8509 to 0.8533, because recall gained more than precision lost — which is why no F1, pairwise
  or B-cubed, should pick the threshold: the precision lost here is fused products.
- **Chaining happens at `p_hi` as well: 31 of its clusters fuse more than one product.** Average
  linkage cleanly separates 2 of them and correlation clustering 2; separating them but splitting
  an entity: average linkage 1, correlation clustering 1. Correlation clustering leaves 28 of the
  31 fused, and it keeps a record in a cluster only while no single move lowers expected cost
  under the model's own probabilities — so it is those probabilities holding them together, and
  the fix belongs to an `error-analyst` pass over `model/` and `features/`, not to this stage.
- **The ceiling binds.** Blocking emitted 5,976 of 6,322 true pairs, and transitivity over the
  true ones recovers 251 more, so the best any clusterer here can reach is B³ recall 0.9876.
  Cluster recall can exceed blocking's pair completeness — an entity of three needs only two of
  its pairs emitted — but not by more than that.
- **The objective and ground truth disagree on the winner.** Correlation clustering has the lowest
  expected cost (8,047.2), but average linkage the lowest realized cost (7,252). The objective
  reads the model's probabilities, so where the two disagree it is those probabilities that are
  wrong about some pairs.
- **Under the model's probabilities the truth is not the cheapest partition.** The ceiling row
  costs 54,866.1 in expectation against 8,047.2 for the best partition found, because merging a
  true pair the model scores low is priced as a false merge. No search over this objective reaches
  the ceiling, however thorough; that gap is the scorer's to close.
- **Review is a third outcome here too, and B-cubed does not see it.** Every pair a partition
  leaves apart falls back to its band — queued for review at p ≥ `p_lo`, rejected below — so a
  lone pair merges only where the bands would auto-merge it, and both cost columns bill on the
  same terms as `reports/synth/model.md`. The B-cubed figures score the partition before any
  review is resolved: connected components at `p_hi` leaves 629 pairs queued, average linkage
  1,004 pairs.
- **Clustering lowers the bill, not only the entity count.** On those same terms the pairwise
  bands alone — every pair decided on its own, producing no entities at all — bill 8,310; average
  linkage bills 7,252.

