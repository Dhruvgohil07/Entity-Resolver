# Decisions log

The project's record of judgment calls and measured decisions. Moved here from `CLAUDE.md`'s
"Open questions" section on 2026-10-06; earlier history is in that file's git log.

- **Open questions** are deliberately left undecided, to be settled against more data.
- **Settled by measurement** entries record what was measured and why, so they are not re-litigated.

When a change settles or revisits an entry, update it here in the same commit and say so in the
commit body. Figures quoted below are as measured at the time; several entries say explicitly which
of their numbers predate later pipeline changes -- re-derive those before quoting them.

## Open questions

Defects found so far in `normalize.py`, `schema.py` and `eval/metrics.py` are fixed, each pinned by
a regression test naming the failure mode. These judgment calls are deliberately left open, to be
settled against more data rather than treated as decided:

- **Evaluation uses the deduplication framing, not record linkage, and that costs comparability.**
  The baseline scores every pair of the combined catalog and calls a pair positive when the two
  records share an `entity_id` — so same-side pairs are candidates, and Abt-Buy has 1118 true pairs
  rather than the 1097 its mapping file ships. Published Abt-Buy figures instead score Abt rows
  against Buy rows only. The dedup framing is the right one for what this service actually does
  (CLAUDE.md's opening question is "given a catalog of N records", not "given two catalogs"), and it
  is what keeps every stage free of a per-source branch. But it means our numbers are near, not
  equal, to the literature's. Revisit when Amazon-Google lands: it has the same two-sided shape, so
  if the gap matters for comparison, the fix is a second reported number computed in `eval/`, never
  a side-aware branch in `blocking/` or `features/`.
- **The spec-suffix list in `_SPEC_UNIT_SUFFIXES` is conservative on purpose, and the conservatism
  is about *short* suffixes specifically.** It rejects `1200W` and `12MP` as model numbers while
  deliberately omitting `wh` and `a`, which collide with real vendor codes (Bose 161WH, HP Officejet
  8500A). A false model number fuses unrelated products into one block; a missing one only loses a
  signal — so the list should grow only against evidence. The spelled-out units added alongside
  them (`volt`, `channel`, `inch`, `megapixel`, `watt`, `hertz`, `pack`, …) are a deliberate
  exception rather than a loosening of that rule: nothing sells a vendor code reading `12-VOLT` or
  `7.1-CHANNEL`, so unlike `wh` and `a` they collide with nothing, and they became *reachable* only
  once the spec check started running per `/`-separated segment — `MOUNT/12-VOLT` is not a spec read
  whole. Both halves were measured on the real catalog, not reasoned about; see the shape-gate entry
  below. Where a spelled-out unit would collide with a real code, the short-suffix rule applies and
  it stays out.
- **A review is charged per pair, not per cluster merge.** Pricing every pair a partition leaves
  apart at min(p·C_fs, C_review) keeps the clustering objective additive and bills clusters on
  exactly the terms `reports/model.md` bills its bands on, which is what makes `reports/cluster.md`'s
  154 comparable with the bands' 284. But a reviewer shown a proposed merge of two clusters makes
  one decision, not |A|·|B| of them, so per-pair billing overstates review for large clusters and
  under-queues their merges. On Abt-Buy's twos and threes the two readings barely differ. The
  under-queueing is measured, not hypothetical: on a triangle with edges 1.0, 1.0 and 0.91, both
  cost-based clusterers merge the 0.91 pair the bands route to review, because splitting the third
  record off leaves both its pairs apart at one review each (2.0) against one merge at
  (1 - 0.91)·20 = 1.8, so any closing edge above 0.90 merges. Billed as one merge decision it
  would cost one review and be queued. Neither cost-based row merges such a pair on the Abt-Buy
  test split, and no test pins the case. Settle it with `service/`'s review queue: measured time
  per merge decision against cluster size. That measurement is now instrumentable, not answered:
  `service/store.py`'s `review_queue` table carries `left_cluster_id`/`right_cluster_id` and
  `created_at`/`decided_at` on every row, so "time per decision against the pair's cluster sizes"
  is a query away once real review decisions accumulate -- nobody has made one yet.
- **`synth/`'s entity sizes and structure are judgment calls, not measurements.** Sizes run 40%
  singletons, 30% pairs, 15% triples and 15% from four to eight records, because Abt-Buy has only
  pairs and triples and the p = 0 question below needs larger entities; nothing grounds the
  shares. Two structural deviations are reported, not tuned away (`reports/synth/realism.md`):
  brand is withheld per listing independently, so both listings carry one on 26.3% of synthetic
  pairs against 1.0% of Abt-Buy's, and `desc_len_ratio` no longer points backwards; and every
  sibling derives from one seed, so their *shape* differs from the seeds', whose siblings are
  independent products. The sibling *rate* used to match and no longer does: `realism.md` now reads
  0.498 against the seeds' 0.552 and calls siblings **sparser**, where it previously read 0.508 and
  called the rate a match. Nothing about the generator changed — the catalog on disk is untouched —
  the shape gate stopped taking apertures and prose as codes, which moved what counts as "a
  same-brand code within two edits" on both sides. Far siblings dilute the near ones, so a model
  scored here meets fewer hard negatives than the seeds hold. That row is *reported*, not calibrated
  (`realism.md` marks it `—`), so no ±0.05 tolerance is claimed for it and none is broken; every
  calibrated row is still within, and the widest gap actually narrowed, code keys equal -0.044 →
  -0.032. Revisit against a second real catalog before trusting a synthetic number that leans on
  either deviation.
- **Online lookup's recall against a real catalog is unmeasured.** `service/`'s lookup path only
  ever queries `ann` and `standard`'s `model_number`/`code_token` keys -- `lsh`,
  `sorted_neighborhood` and `standard`'s `rare_token_keys` are excluded because none of the three
  can be incrementally maintained without a full rebuild (LSH's bands, the sorted array, and rare
  tokens' corpus-wide document frequency all change on every insert). What that costs recall against
  the batch path's union PC 0.9928 has never been measured -- it could be small (per the
  leave-one-out table below, `rare_token_keys` alone was worth +0.0089 on Abt-Buy) or could be the
  dominant gap once `sorted_neighborhood`'s +0.0045 and Abt-Buy-specific effects are counted.
  Settle it by running real lookups over a held-out split and comparing against the batch union's PC
  on the same split, not by reasoning from the batch table alone -- online queries see one record's
  neighbourhood at a time, not the whole catalog's.
- **The `ann` vectorizer's vocabulary freezes at `build_index` time and never refits.** A lookup
  whose distinguishing tokens are genuinely new to the corpus (a brand or model-number pattern never
  seen when the run was indexed) gets no signal from that vocabulary, silently degrading to
  `standard`'s two key functions alone for that record. Re-running `build_index` periodically is the
  only mitigation shipped, and how much drift accumulates between re-runs, on a real catalog taking
  real lookups, is not measured.
- **Online lookup never asks whether two existing clusters should merge.** The decision logic
  (`service/lookup.py`) picks which one cluster a new record joins; it structurally cannot notice
  that clusters A and B have now accumulated several strong review-links against each other, which
  `cluster/agglomerative.py`'s objective would eventually resolve on a full batch re-run. No
  mechanism surfaces that signal or triggers a re-run -- an operational gap, not an algorithmic one.
- **The near-perfect top of the synthetic ranking is a scale effect, not a synthetic artifact --
  settled by an `error-analyst` pass, not fixed.** The earlier hypothesis (a shared `####L###`
  part-number shape fooling the model) does not hold: it appears in only 1 of the 43 highest-scored
  false pairs on `synth-20k`'s test split, against a 1.4% base rate -- no enrichment. What is true:
  ranks 1-44 are all wrong and the first true pair is at rank 45, while `raw >= 0.99` is 92% precise
  immediately below that. The 43 score *below* the all-candidate mean on every similarity column
  (`model_number_exact` 0.0000 against 0.0207, `title_tfidf_cosine` 0.0833 against 0.2167) --
  they are not confidently wrong because a feature misfires, they are unusually dissimilar. The
  cause: the generator's alternate-part-number corruption manufactures genuine zero-evidence
  duplicates (62% of the 95 zero-evidence true pairs in test carry that shape on one side, against a
  12% base rate -- the generator is well calibrated here), the booster learns a real high-score
  region for them, and that region also catches unrelated pairs. The zero-evidence-positive *rate*
  in train is nearly identical to Abt-Buy's (1.68% of positives against 1.81%) -- only the absolute
  count differs, 233 against 14, straddling `model/train.py`'s `min_child_samples=20`: a leaf
  isolating the pattern forms easily on `synth-20k`'s scale and cannot form on Abt-Buy's, which is
  why Abt-Buy shows no such inversion (P@100 1.0000). That implies the same failure mode should
  appear on a real catalog this size, the opposite conclusion from "the model learned a synthetic
  artifact." **The named fix is now applied and measured on both catalogs**, and the trade it was
  recorded as making ("an unmeasured PR-AUC cost") turned out to be backwards on the catalog that
  matters: `monotone_constraints` *raises* PR-AUC where the failure mode exists. Three arms, each
  a full evaluation, with fix 1 (the `code_key` gate) already in place so only the constraint
  varies:

  | arm | AB F1 | AB PR-AUC | AB false merges | AB cost | 20k F1 | 20k PR-AUC | 20k P@10 | 20k false merges | 20k cost |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | unconstrained | 0.8925 | 0.9561 | 6 | 246 | 0.7447 | 0.7033 | 0.000 | 290 | 10,581 |
  | code columns only | **0.8957** | 0.9557 | **5** | **232** | 0.7212 | 0.7131 | 0.000 | 220 | 9,607 |
  | code + title (shipped) | 0.8813 | 0.9489 | 7 | 281 | 0.7194 | **0.7608** | **1.000** | **122** | **7,618** |

  **All three rows predate the second shape-gate fix and are deliberately left as measured.** The
  table's only job is the three-way comparison, which is valid only while every arm shares one
  extraction — updating the shipped row alone would break exactly that. For reference, the shipped
  arm now reads AB F1 0.8892 / PR-AUC 0.9472 / 7 false merges / cost 284 and 20k F1 0.7288 / PR-AUC
  0.7675 / P@10 1.000 / 130 false merges / cost 7,659 (`reports/model.md`, `reports/synth/model.md`),
  so the fix improved F1 and PR-AUC on both catalogs while the bill rose slightly on both. Re-running
  the other two arms is the work required before any row here is quoted as current.

  The code-only arm was measured because it looked like it might dominate, and on Abt-Buy it does
  -- best F1, fewest false merges, cheapest bill of the three. It does not fix the thing the
  constraint exists for: `synth-20k`'s P@10 stays 0.000 and P@100 reaches only 0.220. The title
  columns are load-bearing, which follows from the error analysis above rather than contradicting
  it -- the 43 top-ranked false pairs score below the candidate mean on *every* similarity column,
  code and title alike, so it is "low title agreement, high score" that has to be forbidden.

  Shipped anyway on the full set, against a real cost on Abt-Buy (F1 -0.0112, PR-AUC -0.0072, bill
  246 -> 281), because the cost model is what this project optimizes and not F1: on `synth-20k`
  false merges fall 290 -> 122 and the realized bill falls 28%, while the ranking inversion the
  `error-analyst` pass found disappears outright (P@10 and P@100 both 0.000 -> 1.000). Abt-Buy's
  side of that trade is 1 extra false merge and 5 extra missed pairs on a 341-true-pair split --
  and by this entry's own analysis Abt-Buy is too small to *form* the leaf the constraint forbids,
  so it pays for the guard without being able to show the benefit. `MONOTONE_INCREASING_FEATURES`
  is derived against the live feature registry rather than written as a positional literal, and
  raises if a name stops resolving, so a renamed column cannot silently drop its constraint.
  `synth/`'s corruption rates stay as they are -- capping the alternate-part-number operator would
  suppress a symptom by making the catalog *less* like a real one on the one axis `realism.md`
  currently matches. Still open, and now cheap to measure: whether the title columns need the full
  constraint or only a subset, since code-only is strictly better on Abt-Buy.

## Settled by measurement

Recorded so they are not re-litigated:

- **Model-number blocking keys strip separators; the printed code is kept alongside.**
  `normalize.py` grew `model_number_key` next to `model_number` because Abt writes `KXTS208W` and
  Buy writes `KX-TS208W` for the same Panasonic phone. Both are correct as printed vendor codes, so
  neither field wins: `model_number` stays as the source wrote it for the review queue, and the
  stripped form is what blocks. Worth pair completeness 0.3336 → 0.5832 — about two fifths of
  achievable recall on the strongest key there is, previously lost to punctuation. The rule is
  `normalize.code_key`, public and shared: `blocking/standard.py` briefly carried its own copy built
  on `str.lower`, and two implementations that must agree is how train/serve skew starts. It
  decomposes NFKD internally rather than trusting the caller, because its callers disagree — the
  model-number path passes a code taken from the raw title, blocking passes tokens from the folded
  one, and without that a precomposed `Ü` is dropped whole while a decomposed one keeps its base
  letter, keying the same code two ways.
- **Extraction's shape gate reads the printed token *and* its `code_key`, each for what only it
  can see.** Two fixes in opposite directions, and the second is why the first is recorded here
  with its cost rather than as a clean win.

  `_qualifies_as_model_number` originally tested the raw candidate token against
  `^[A-Za-z0-9-]+$`, which rejects `/` -- so on
  `'Samsung YP-S2ZW 1GB Flash MP3 Player - YP-S2ZG/XAA'` the trailing, correct code failed the gate
  and extraction fell through to the leading token, which is a *different* listing's code entirely.
  Moving the gate onto `code_key(token)` fixed that, on the principle that a token is a model
  number when its *comparison form* is model-number-shaped -- the form `_model_number_key` blocks
  on and `blocking/` indexes. Two details settled then still hold: the 4-character floor stays on
  the token **as printed**, because moving it onto the key silently dropped real short codes
  (`XM-6`, `GR-4`, `IP-3`, whose keys are three characters); and the spec check runs on the key,
  since `18-200MM` and `1.5TB` reduce to the digits-then-unit form the spec list is written against
  and the raw-token gate never saw them.

  **That overshot, and `/code-review` caught it.** `code_key` strips exactly the separators that
  distinguish a code from a measurement, so a gate reading nothing else cannot tell `9N00.101` (a
  real Targus code) from `802.11n` (a WiFi standard) -- and it admitted `f/3.5-5.6G`, a lens
  aperture, as a model number. On Abt-Buy that gave the Nikon D60 kit (`e00465`) and the D90 kit
  (`e00883`), two different products, the single shared key `f3556g`, setting `model_number_exact`
  and `model_number_prefix_ratio` to 1 for their pair; both columns are monotone-constrained, so a
  false key there can *only* push a score up. That is precisely the "a false model number fuses
  unrelated products into one block" cost the `_SPEC_UNIT_SUFFIXES` entry above exists to avoid,
  reintroduced one layer above the list it is written on.

  The fix was measured against the real population rather than argued from the two examples. Of the
  90 Abt-Buy extractions the original `^[A-Za-z0-9-]+$` gate rejected, **83 are genuine vendor
  codes** (`MB226LL/A`, `EC-NV30ZSBA/US`, `2595B002(AA)`, `#EL012A`, `9N00.101`, `NIKO_215930348`)
  and **7 are specs or prose**. Reverting the char-class gate to kill the 7 would have cost the 83,
  so three rules on the printed form separate them instead: a single character before a `/` is an
  English abbreviation and not a code stem (`f/`, `w/`, `b/w` -- the shortest real slash-code
  measured is `CV/FO-10`); one spec segment condemns the whole token (`MOUNT/12-VOLT`,
  `1080p/60Hz`), which is why the spec check now runs per `/`-separated segment and why
  `_SPEC_UNIT_SUFFIXES` grew spelled-out units; and a `.` marks a decimal measurement when pure
  digits start the segment, which is the one thing separating `802.11n` and `2.7-Inch` from
  `9N00.101` and `1EG0.052.00` -- every dotted code on the real catalog carries a letter before its
  first `.` and every dotted spec does not.

  Blast radius of the second fix, measured before anything was regenerated: extraction changes on
  **8 of 2173** Abt-Buy records, all 8 from a false spec to nothing, and on **311 of 18,829**
  `synth-20k` records -- where it is better than neutral, because 200-odd of those were extracting
  a spec *ahead of* the real vendor code (`7.1-CHANNEL` instead of `TXSR798W`, `5-PACK` instead of
  `LMAF877RE6`) and now get the code. What it cost on Abt-Buy: the model-number blocker's
  standalone PC **0.5832 → 0.5823**, with the union unchanged -- it is entirely subsumed, so losing
  a key costs blocking nothing; `model_number_prefix_ratio` coverage 0.7422 → 0.7318 and univariate
  PR-AUC 0.6349 → 0.6320, `model_number_exact`'s 0.5269 → 0.5239. Both columns' class *separation*
  rose even as their PR-AUC fell (+0.7975 → +0.8031 and +0.6839 → +0.6879), which is what removing
  false positives from a column should look like. Downstream it is a clear gain: **test F1 0.8813 →
  0.8892** at precision 0.8603 → 0.8958, and Platt's ECE 0.00101 → 0.00087. The realized bill rises
  **281 → 284**, because 3 pairs moved from auto-merge into review; the cost model is what this
  project optimizes, so that is stated rather than buried under the F1.

  One correction the review forces, recorded rather than smoothed over: the first fix's blast radius
  ("66 codes previously missed entirely, 24 replaced, 5 correctly dropped as specs") counted some of
  these aperture and prose extractions among its improvements, so the gains it claimed
  (PC 0.5349 → 0.5832, `prefix_ratio` PR-AUC 0.5784 → 0.6349) were slightly inflated. The corrected
  figures are the ones above. No test caught this and no report disagreed with itself; it surfaced
  only because someone read the gate against real titles.

  `synth-20k` shows the failure removed in the form the review predicted, which Abt-Buy is too small
  to display. `reports/synth/cluster.md` used to carry a **4-record, 4-entity** fusion held together
  by nothing but `7.1-Channel` — cross-entity edges at 0.9772 / 0.9703 / 0.9616, *still fused* under
  both cost-based clusterers, and spanning two different seed families and two brands (`Samsung
  7.1-Channel Home Theater System`, `Onkyo Black 7.1-Channel Home Theater sys`). It is gone from the
  chaining section entirely. That is the review's "a false model number fuses unrelated products"
  claim observed end to end rather than argued: a spec became a shared key, the key became a
  perfect-match feature under a monotone constraint, and the constraint carried four distinct
  entities into one cluster. The honest counterweight is that average linkage's *total* fused count
  on that catalog rose 29 → 32 as every other probability moved, so this bought one named,
  understood fusion and not a lower count.

  Both of the first fix's predicted error flips survive the second: `reports/cluster.md`'s
  Green/White fusion is still gone from the chaining section, and average linkage still reaches B³
  precision **1.0000** with **0 fused clusters**. Pinned by eight tests in `tests/test_normalize.py`
  -- four from the first fix (the `YP-S2ZG/XAA` regression, the separator family, the
  spec-wearing-a-separator case, the short-code floor) and four from the second, naming the aperture
  fusion by entity id, the one-character-prefix abbreviation, the spec segment, and the
  decimal-against-dotted-code contrast.
- **The blocker sweep carries no selection optimism, because pair completeness is monotone in
  every parameter swept.** Recorded for a long time as a caveat on the 0.9928 ceiling ("a
  best-of-sweep number, not a clean estimate — small bias here, but unquantified"), and the worry
  does not survive measurement. Each blocker's parameter was selected on the **train** split alone
  and spent on **test**, against the value test itself would have chosen:

  | blocker | train pick | its test PC | test pick | oracle test PC | gap |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `standard (rare tokens)` df | 100 | 1.0000 | 100 | 1.0000 | +0.0000 |
  | `sorted_neighborhood` w | 80 | 0.9560 | 80 | 0.9560 | +0.0000 |
  | `lsh` threshold | 0.2 | 0.8739 | 0.2 | 0.8739 | +0.0000 |
  | `ann` k | 50 | 1.0000 | 10 | 1.0000 | +0.0000 |

  The gap is zero everywhere, and not by luck: PC rises monotonically with a looser window, a
  higher df cutoff, a lower LSH threshold and more neighbours, so the argmax is always the loose
  end of the sweep and train and test agree on it by construction. There is no noise for a sweep
  to overfit. What the sweep actually picks is a point on a recall-against-candidates curve, which
  is a budget decision, not an estimate that can be optimistically biased — and every committed
  parameter sits deliberately *below* the PC-maximizing end, measured on the test split:

  | blocker | committed | PC | candidates | loosest swept | PC | candidates |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `standard (rare tokens)` | df=30 | 0.9883 | 7,801 | df=100 | 1.0000 | 23,244 |
  | `sorted_neighborhood` | w=20 | 0.8094 | 12,198 | w=80 | 0.9560 | 48,348 |
  | `lsh (minhash)` | t=0.4 | 0.6070 | 2,161 | t=0.2 | 0.8739 | 13,999 |
  | `ann (faiss HNSW)` | k=10 | 1.0000 | 4,319 | k=50 | 1.0000 | 20,968 |

  So if anything the committed union understates what these blockers could reach, and `ann` alone
  already reaches 1.0000 on the test split at the committed `k`, buying nothing from k=20 or k=50
  but 2-5x the candidates. A one-off measurement, not a committed report.
- **The leave-one-out tables are now committed reports, not one-off measurements — because this
  session demonstrated exactly how an uncommitted figure fails.** Both tables below were carried in
  CLAUDE.md for months, re-verified by hand on 2026-09-14, and then silently invalidated when
  `_qualifies_as_model_number` changed what extraction returns. Nothing failed. No test noticed. The
  only reason they are right today is that someone remembered to re-measure them, which is precisely
  the property "a negative result someone can re-run is evidence" was supposed to guarantee and does
  not: *re-runnable* is not *re-run*. `blocking/evaluate.py` grew `--leave-one-out`, which drops each
  blocker, re-unions the rest and renders the marginal table into the report, so `reports/blocking.md`
  and `reports/synth/blocking.md` now carry their own marginals and regenerate with everything else.
  Opt-in rather than always-on: it costs one extra union per blocker over the whole candidate set,
  which is not free at `synth-200k`, so `blocking-200k.md` is still measured without it.

  This settles the promotion question for the figure that most needed it and leaves the rest
  deliberately uncommitted, but for a narrower reason than before: the character-shingle LSH
  comparison and the `ann` SVD/`k` sweeps each need a blocker configuration the CLI cannot express,
  so committing them means adding flags for configurations nothing else uses. They stay one-off, and
  the honest reading is that they will go stale the same way — the `ann` sweep figures happen to be
  unaffected by this change because `ann` reads titles only, which is luck rather than design.
- **Which blockers earn their candidates depends on the catalog.** On Abt-Buy three of the six add
  no completeness the others do not already have. Leave-one-out marginals against the committed
  six-blocker union, measured, not estimated:

  | blocker | marginal candidates | marginal PC |
  | --- | ---: | ---: |
  | `standard (model number)` | +0 | +0.0000 |
  | `standard (code tokens)` | +737 | +0.0000 |
  | `standard (rare tokens)` | +15,244 | +0.0089 |
  | `sorted_neighborhood` | +28,474 | +0.0045 |
  | `lsh (minhash)` | +18,317 | +0.0000 |
  | `ann (faiss HNSW)` | +2,345 | +0.0215 |

  The model-number blocker is *entirely* subsumed — every pair it finds, something else finds too,
  which follows from `code_token_keys` indexing every code-shaped token while extraction commits to
  one. LSH earns nothing for 18,317 candidates; `sorted_neighborhood` buys 0.0045 for 28,474.
  Only `ann`, `rare tokens` and `sorted_neighborhood` move the ceiling at all.

  None are deleted. A negative result someone can re-run is evidence; the same claim asserted from
  a deleted experiment is not, and all three are configuration- and dataset-specific. LSH in
  particular was measured with *token* shingles on six-to-ten-token titles, which is close to the
  worst case for Jaccard. Re-checked on `synth-20k`, where token-drop corruption exists, against its
  own six-blocker union (PC 0.9238), and the verdict does not transfer:

  | blocker | marginal candidates | marginal PC |
  | --- | ---: | ---: |
  | `standard (model number)` | +0 | +0.0000 |
  | `standard (code tokens)` | +7,741 | +0.0022 |
  | `standard (rare tokens)` | +46,051 | +0.0157 |
  | `sorted_neighborhood` | +257,234 | +0.0147 |
  | `lsh (minhash)` | +1,233,795 | +0.0343 |
  | `ann (faiss HNSW)` | +25,242 | +0.0314 |

  Token LSH adds the most completeness of any blocker there, for 70% of the candidates. Character
  shingles trade differently: in its place they cut the union to 665,261 candidates at PC 0.9078,
  and added alongside it they lift the union only to 0.9319. At `synth-200k` token LSH does not run —
  222M raw pairs and a failed 1.66 GiB allocation — so `reports/synth/blocking-200k.md` is measured
  `--without lsh` and says so. Only the model-number blocker is subsumed on both catalogs. Still
  none are deleted: the LSH verdict turns on the catalog, which is what keeping it re-runnable was for.
  Both tables above are now committed reports rather than one-off measurements (see the entry
  above): `reports/blocking.md` and `reports/synth/blocking.md` carry their own marginals via
  `--leave-one-out`. The character-shingle figures are still a one-off, needing a blocker
  configuration the CLI cannot express. The memory failure
  reproduces by running `blocking.evaluate --dataset synth-200k --ann-components 128` without
  `--without`. Re-verified 2026-09-14: every marginal-candidate, marginal-PC and character-shingle
  figure in both tables above reproduced exactly.
- **`ann` is the load-bearing blocker, and raising its neighbour count `k` at scale is a real,
  swept-not-tuned lever, now committed.** faiss HNSW over char-3gram TF-IDF reaches PC 0.9562 alone
  on Abt-Buy, beating every exact-key blocker combined, because it needs no shared token at all —
  it is what catches `Bose 161WH` against `Boss 161 Speaker`, a source typo in the brand. Its index
  is built **single-threaded on purpose**: parallel HNSW construction gave 14,608 / 14,603 / 14,602
  candidates across three runs of identical input, and a committed report whose numbers drift is not
  reproducible. Its dense index does not scale, and SVD alone is not the fix. On `synth-20k`,
  projecting to 128 components at the same `k=10` halves its completeness, 0.7705 to 0.3897, at the
  same candidate count; raising `k` to 50 recovers it to 0.8187 for 631,673 candidates (re-verified
  2026-09-14, reproduced exactly; these two remain one-off measurements, not a committed report).
  At `synth-200k`, SVD-128 with `k=10` reaches only 0.0597, with ten neighbours crowded out by
  sibling families about 99 entities wide — so scaling `ann` means raising `k` with the projection.
  `default_blocker_set(ann_neighbours=...)` now exposes `k` as a parameter, mirroring
  `ann_components`, and a full sweep at `synth-200k` is committed: `k`=10/50/100/150 give PC
  0.0597/0.2225/0.3805/0.5000, still climbing at `k`=150 — not a plateau, a practical stopping
  point (`reports/synth/blocking-200k.md`). `default_blocker_set`'s literal default stays `k=10`,
  what every Abt-Buy/`synth-20k` report was measured with; a larger catalog spends a higher `k`
  explicitly via `--ann-neighbours`.
- **A memory bound on `lsh`'s candidate generation is the other lever, and it also works.**
  Unbounded token LSH cannot run at all on `synth-200k`: 222M raw candidate pairs failed a 1.66 GiB
  allocation, because nothing capped how many candidates one record's query could contribute before
  they were all accumulated into one array. `MinHashLSHBlocker(max_neighbours=...)` bounds it the
  way `standard.py`'s `max_block_size` bounds a runaway exact-key block — a record whose query
  returns more than the cap contributes zero candidates, counted and surfaced as a warning, same
  shape as a dropped block. Swept at `synth-200k`: cap=100/300/500 give PC 0.0871/0.3201/0.4042,
  dropping 138,364/85,041/61,611 of 165,714 records respectively — real completeness, real cost,
  still climbing at cap=500, not exhausted. `default_blocker_set`'s literal default stays `None`
  (unbounded), what every Abt-Buy/`synth-20k` report — including the one recording the OOM — was
  measured with; a catalog past that budget spends a cap explicitly via `--lsh-max-neighbours`.
  Combined with `ann` at `k`=150, every default blocker now runs on `synth-200k` for the first
  time — no `--without`, no OOM — reaching union pair completeness **0.8469**
  (`reports/synth/blocking-200k.md`), against 0.6712 with `lsh` omitted entirely -- that last figure
  is a one-off from a `--without lsh` run, not the committed command, so it predates the shape-gate
  fix and is the one number in this entry that has not been re-measured. Neither lever was
  pushed to its limit; both are available, measured, swept-not-tuned levers for whoever revisits
  this ceiling.
- **What blocking still misses is a different identifier system, not a near-miss.** Of 1,118 true
  pairs, 8 survive nothing. They are two failure modes, and neither is fixable by tuning a window or
  a threshold: (a) vendor SKU against distributor part number — `Canon Color Ink Tank - CL41CL` vs
  `Canon Ink Cartridge For PIXMA iP1600 ... - 0617B002`, two disjoint numbering schemes for one
  product; and (b) a truncated marketplace title carrying no code at all — `LG Over-The-Range White
  Microwave Oven - LMV1680WH` vs `LG 1.6 cu.ft. Over the Range`. Closing (a) needs a
  manufacturer-part-number cross-reference, which is data this project does not have; closing (b)
  needs the description, which `features/` will have and blocking does not.
- **The baseline compares normalized title only; adding description makes it worse.** Measured on
  Abt-Buy: title alone gives test F1 0.5204 / PR-AUC 0.4720, title + description gives 0.4349 /
  0.2898 — and P@10 collapses from 0.600 to 0.000, so the very top of the ranking is what breaks.
  The cause is a measured length asymmetry, not prose quality: Abt descriptions average 249
  characters and are never empty, Buy's average 34 (median 14) and are empty on 40% of rows. So
  concatenation makes an Abt vector that is mostly description face a Buy vector that is mostly
  title, diluting exactly the true pairs it was meant to help. What the two sides do share is
  category vocabulary — `finish` appears in 759 descriptions, `black` in 622, `digital` in 386 —
  which lifts *unrelated* pairs instead. Description is not worthless; it belongs in `features/`
  as its own signal with a missingness indicator, not glued onto the title string.
- **`'` folds to inches, not feet.** Typographically `'` is the foot mark, and an earlier pass
  changed it on that basis. Measuring the actual CSVs overturned it: all 249 digit+`'` occurrences
  across `Abt.csv` and `Buy.csv` are screen and driver sizes (`3.0' LCD Display`, `32' to 50' LCD`,
  `4' x 6' Print Paper`, `1-1/8' Dome Tweeter`) and none is a length in feet. Normalize follows the
  source convention, not the typographic one. Revisit only if a source that genuinely sells by the
  foot is added.
- **The missingness invariant binds the reporting path too, not just the vector.** The first
  version of `features/evaluate.py` averaged `FILL_VALUE` into its per-column class means, which
  reads a null as a mismatch — the exact error invariant I5 exists to prevent, committed one
  layer above the vector where I5 was being checked. It was not a rounding difference. It
  reported `brand_equal` as pointing *backwards* (0.0176 on true pairs against 0.0890 on false)
  because the column is covered on 2.3% of positives against 24.3% of negatives — Abt has no
  brand column, so true pairs are almost all cross-source. On the 4,506 pairs where a brand
  actually exists it points forwards and strongly: **0.7500 against 0.3655, separation +0.3845**.
  The same restriction matters for ranking: a fill of 0.0 sorts last for a similarity column but,
  once negated, sorts *first* for a distance column, ranking an absent price as a perfect price
  match. Every figure in the report is now computed over covered pairs only, pinned by
  `test_class_means_ignore_the_fill` and
  `test_a_distance_columns_fill_is_not_ranked_as_perfect_agreement`.
  Caught by the `er-invariants` agent, not by the test suite — the suite verified the invariant
  inside `vectorize.py` and never asked whether the report obeyed it.
- **One feature column genuinely points the wrong way on Abt-Buy, and it is not a bug.**
  `desc_len_ratio` reads 0.2213 on true pairs against 0.4763 on false, over 12,133 covered pairs
  (200 positive) — measured on covered pairs only, so this one survives the correction above.
  Flipping its sign would be the wrong fix. It is the deduplication framing showing through:
  same-side pairs (Buy against Buy) are candidates, and Abt descriptions average 249 characters
  against Buy's 34, so two descriptions of *similar* length are evidence of being same-side,
  which on this dataset means evidence of being a non-match. A tree model uses that correctly; a
  human reading the column name does not, which is why `reports/features.md` calls it out. This
  is also the most concrete cost yet measured for the framing question left open above.
- **Vendor-code columns dominate the feature ranking, as predicted.** Top five by univariate
  PR-AUC on test: `model_number_prefix_ratio` 0.6320, `code_token_jaccard` 0.6274,
  `model_number_exact` 0.5239, `title_tfidf_cosine` 0.5222, `code_best_ratio` 0.4646. The two
  `model_number_*` columns moved up when extraction was fixed to gate on `code_key` -- coverage
  0.6917 → 0.7422 carried `prefix_ratio` past `code_token_jaccard` into first place -- and then
  part way back down when the same gate was taught to reject apertures and prose again (coverage
  0.7318, PR-AUC 0.6349 → 0.6320), which leaves `prefix_ratio` first by 0.0046 rather than 0.0075.
  Both columns' class separation rose across that second change even as their PR-AUC fell; see the
  shape-gate entry above. A test
  asserts the code columns stay at the top — if they ever do not, the feature is broken, not the
  claim. Worth noting `cross_title_desc_cosine_max` reaches 0.4473: the column added for the
  truncated-title failure reads 0.7056 on the `LMV1680WH` / `1.6 cu.ft.` pair where every string
  column reads under 0.32.
- **`price_abs_log_ratio` uses `log1p`, which costs exact scale-freeness at the bottom of the
  range.** A 2x gap reads 0.647 at $10–$20 against log(2) = 0.693 at $1000–$2000. Accepted
  rather than worked around: `schema.py` permits a price of 0.0 and `log(0)` is -inf, which would
  propagate through the whole vector with nothing pointing back at the cause. Abt-Buy's lowest
  price is $1.75 with two rows under $5, so the distortion is confined to a corner of the range
  that barely exists, and a tree model splits on thresholds rather than reading the value as a
  ratio. Pinned by a test so it stays a known property.
- **Calibration is fit out-of-fold on entity-grouped folds of train, with Platt, not isotonic.**
  The obvious design was a third held-out split calibrated with isotonic, which the invariant names
  first. Isotonic lost on resolution exactly where the cost model reads: fit on a held-out split's
  230 positives it produces **one** distinct output level at or above 0.9, and that level is 1.0, so
  `p_hi = 0.95` cannot separate anything inside a 253-pair atom. Out-of-fold isotonic reaches 5
  distinct outputs above 0.9; Platt keeps 285. Five entity-grouped folds also give the calibrator
  777 positives against 230 and leave all 1521 train records for the booster, lifting its raw
  precision at score 0.95 from 0.9669 to 0.9845 over one trained on a held-out split's remainder;
  OOF Platt is the best calibrated of six variants measured (ECE 0.00085 against raw 0.00244 and
  OOF isotonic 0.00318), with recall at 0.95 of 0.8006 against isotonic's 0.6510. On
  `synth-20k`, with 14,441 out-of-fold positives against 777, isotonic's shape advantage arrives
  and its resolution problem stays. It is the better calibrated — ECE 0.00084 against Platt's
  0.00374 and Brier 0.00707 against 0.00758, while Platt is *worse* calibrated than the raw
  booster's 0.00082 — but it still collapses where the cost model reads: 6 distinct outputs at or
  above 0.9 against Platt's 9,091, and recall at `p_hi` 0.95 of 0.3969 against Platt's 0.5354, at
  near-equal precision (0.9613 against 0.9538). Neither dominates. Isotonic states probabilities
  honestly and cannot separate confident pairs; Platt separates them and misstates the review band
  by up to 0.22 (the reliability table in `reports/synth/model.md`). Platt stays the default
  because auto-merge is the irreversible decision and resolution matters most there, but the
  out-of-fold half of this entry is settled and the Platt half is reopened, not decided: a
  calibrator keeping both properties is the next thing to measure. The `synth-20k` isotonic figures
  come from a one-off comparison, not a committed report: `reports/synth/model.md` measures Platt
  against the raw booster only. Re-verified 2026-09-14 against two independent retrains (identical
  both times, confirming `model/train.py`'s determinism claim): every figure in this entry
  reproduced exactly except Platt's distinct-output count, previously recorded as 3,761 -- corrected
  here to 9,091, the value both retrains gave. Ruled out as the cause: rounding the calibrated
  probabilities before counting (checked at 2-6 decimal places; none landed near 3,761 either).

  **Every number in this entry predates `monotone_constraints` and the `code_key` extraction fix,
  and is therefore stale as a description of the current booster.** It is kept rather than deleted
  because what it establishes is a *shape* — isotonic states probabilities honestly and collapses
  where the cost model reads, Platt separates and misstates the band — and that shape is a property
  of the two calibrators, not of one booster. The specific counts are not: a constrained booster
  produces a different score distribution, so the 6-against-9,091 distinct-output figures and the
  recall-at-`p_hi` pair both need re-deriving before being quoted again. That re-derivation is the
  same run that would answer the open half of this entry, so it is one piece of work, not two.

  **The open half is now measured: `BetaCalibrator` keeps both properties, and it is the first
  thing tried that does.** Beta calibration is `sigmoid(a*log(s) - b*log(1-s) + c)`, fit by logistic
  regression on those two transformed features — still parametric and strictly monotone, so it can
  never reorder pairs and never collapses a range to one level, but with two shape parameters
  instead of Platt's one slope. Platt is its `a == b` special case. `fit` refuses a negative
  coefficient for the same reason `PlattCalibrator` refuses a non-positive slope. Four calibrators
  on the same out-of-fold scores, `synth-20k` test split (289,516 candidates, 6,322 true pairs):

  | calibrator | ECE | Brier | PR-AUC | distinct levels >= 0.9 | P @ `p_hi` | R @ `p_hi` | n >= `p_hi` |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | raw booster | 0.00115 | 0.00757 | 0.7608 | 3,262 | 0.9797 | 0.3584 | 2,313 |
  | Platt (default) | 0.00422 | 0.00804 | 0.7608 | 3,598 | 0.9644 | 0.5231 | 3,429 |
  | isotonic | **0.00059** | 0.00753 | 0.7547 | **21** | 0.9778 | 0.3967 | 2,565 |
  | beta | 0.00097 | **0.00753** | 0.7608 | 3,085 | **0.9808** | 0.3159 | 2,036 |

  Beta reaches isotonic-class calibration (ECE 0.00097 against 0.00059, against Platt's 0.00422 —
  Platt is 4x worse than the raw booster here) while keeping Platt-class resolution (3,085 distinct
  levels above 0.9 against isotonic's 21). It is also the most precise at the merge bar. That is
  both properties at once, which is what this entry asked for.

  Two things keep it from being an automatic switch. **It is worse than Platt on Abt-Buy** (ECE
  0.00286 against 0.00101), the same small-catalog/large-catalog split `monotone_constraints`
  shows, and for the same structural reason: two extra shape parameters need positives to fit.
  And **it auto-merges far less**: recall at `p_hi` 0.3159 against Platt's 0.5231, 2,036 pairs
  against 3,429. That is not a defect — it is beta declining to put pairs above 0.95 that Platt
  was overconfident about, which is visible in precision at the bar rising to 0.9808 — but it
  moves real volume out of auto-merge and into review, and the realized bill of that trade has
  not been measured. Measure the bill before switching the default; the shipped default is still
  out-of-fold Platt.

  One sharper observation, which supersedes the distinct-level count as the way to state isotonic's
  problem: **isotonic is the only map here whose PR-AUC falls** (0.7547 against 0.7608 for every
  other, and 0.9403 against 0.9489 on Abt-Buy). It does not reorder pairs — it is monotone — but it
  is only *weakly* monotone, and the ties it creates destroy ranking information that the strictly
  monotone maps preserve. "Calibration must never reorder pairs" was the right rule and an
  incomplete one: it must not flatten them either.
- **The two thresholds are closed-form in the cost ratios, the default is 20 : 2 : 1, and Abt-Buy
  cannot validate it.** Minimizing per-pair expected cost — `(1-p) * C_fm` to merge, `p * C_fs` to
  reject, `C_review` to review — gives `p_hi = 1 - C_review/C_fm` and `p_lo = C_review/C_fs`, so the
  absolute scale cancels. The default (`p_hi` 0.95, `p_lo` 0.50) prices a false merge at 20 reviews
  because it is irreversible and because connected-components chaining fuses two clusters from one
  bad edge, which a pairwise cost structurally understates. It cannot be fitted here: the calibrated
  distribution is bimodal — 18,475 of 18,819 test candidates below 0.01, 280 above 0.95, 64 in the
  whole middle — so `C_fm` anywhere in 10–100 moves the review queue only from 8 pairs to 35. It
  stays a parameter, and `threshold.py` rejects an empty band: 20 : 1 : 1 gives `p_lo` 1.0 against
  `p_hi` 0.95. `synth-20k` populates the middle — Platt puts 14,435 of its 289,516
  test candidates between 0.01 and 0.95, against 64 of 18,819 on Abt-Buy — and there the ratio is
  load-bearing: across the same grid the review queue moves from 615 to 4,698 pairs. It still cannot
  be *fitted*, since the costs are business prices rather than quantities data estimates, but two
  measured facts now bound it. At `C_fm` 100 (`p_hi` 0.99) nothing merges at all, because
  calibrated probabilities barely reach that high; and the cheapest ratio in the grid on realized
  cost is 10 : 2 (6,641 against the default's 7,659), which bills the synthetic catalog's own error
  mix and so is no reason to move the default. It stays 20 : 2 : 1.

  One of the two bounds previously recorded here has been **overturned, and by something worth
  noticing**: at `C_fm` 50 auto-merge used to collapse from 3,549 pairs to 273 while its precision
  *fell* to 0.8388 — a tighter merge bar admitting a *worse* set, which is not how a threshold is
  supposed to behave. Under `monotone_constraints` that row reads 1,611 pairs at precision 0.9845,
  rising with the bar as it should. The anomaly was the same pathological high-score region the
  `error-analyst` pass found at the top of the synthetic ranking, seen from the cost model's side
  instead of the ranking's; constraining the booster removed both. It is recorded rather than
  quietly deleted because it is the clearest evidence that the inversion was a real modelling
  defect and not a quirk of precision@k. The 14,435 mid-band count comes from the
  same one-off calibration comparison; the sensitivity grid and its review-queue range are in
  `reports/synth/model.md`. Re-verified 2026-09-14: the mid-band count and every row of the
  sensitivity grid reproduced exactly, independently of `reports/synth/model.md`'s own committed
  table. **The grid rows are regenerated with `reports/synth/model.md` and so stay current; the
  14,435 mid-band count is not, and predates `monotone_constraints`.** Re-derive it from the
  committed report before quoting it. The entry's *conclusion* is unaffected either way, because it
  rests on the ratio being a business price rather than a quantity data can estimate, and on the
  Abt-Buy distribution being too bimodal to constrain it -- neither of which a booster change
  touches.
- **Clusters keep the bands' three outcomes: a partition merges only where merging beats both
  review and rejection.** The obvious clusterer is connected components over `p_hi` edges, and on
  the Abt-Buy test split it chains: 3 fused clusters through 16 implied pairs, realized cost 538.
  The first fix weighed merge against split alone, which breaks even at τ = C_fm / (C_fm + C_fs),
  0.9091 — but τ always lies inside the review band, so it auto-merged pairs the bands price as
  cheaper to review; the `er-invariants` audit caught it before commit. Pricing each pair a
  partition leaves apart at min(p·C_fs, C_review) instead makes a lone pair merge exactly at
  `p_hi`, and average linkage on that objective bills 154 against the bands' own 284, with B³ P
  0.9877 → 1.0000 for R 0.8891 → 0.8829 and 32 pairs queued. On `synth-20k`, with entities up to
  eight records, average linkage reaches B³ recall 0.7287 against components-at-`p_hi`'s 0.7548
  (−0.026, against −0.006 on Abt-Buy) and splits 922 entities against 869, and the true pairs
  blocking never emitted — 251 of which transitivity recovers — are exactly the pairs the objective
  prices at p = 0. Its bill still beats the bands' on the same terms, 7,279 against 7,659, and
  fixing extraction plus constraining the booster cut its fused clusters from 69 to 29, then the
  second shape-gate fix moved them back to 32 — that count is not monotone in scorer quality, and
  one specific fusion it *did* remove is named in the shape-gate entry above.

  **The entity-size breakdown this entry set as a precondition is now measured, and it confirms
  p = 0 under-merging with a clean dose-response.** B³ recall on `synth-20k`'s test split, average
  linkage against components-at-`p_hi`, by true entity size:

  | entity size | records | average linkage | components @ `p_hi` | delta |
  | ---: | ---: | ---: | ---: | ---: |
  | 1 | 1,018 | 1.0000 | 1.0000 | +0.0000 |
  | 2 | 1,522 | 0.7503 | 0.7516 | -0.0013 |
  | 3 | 1,146 | 0.6545 | 0.6719 | -0.0175 |
  | 4 | 528 | 0.6278 | 0.6818 | -0.0540 |
  | 5 | 570 | 0.6042 | 0.6618 | -0.0575 |
  | 6 | 492 | 0.5854 | 0.6497 | -0.0644 |
  | 7 | 203 | 0.5552 | 0.6087 | -0.0535 |
  | 8 | 184 | 0.6250 | 0.7391 | -0.1141 |
  | overall | 5,663 | 0.7243 | 0.7502 | -0.0259 |

  **The per-size rows predate the second shape-gate fix and are a one-off measurement, not a
  committed report — re-derive them before quoting a single row.** The overall row does re-derive
  from `reports/synth/cluster.md` as it stands: 0.7287 against 0.7548, a **-0.0261** deficit where
  this table recorded -0.0259. So the aggregate the conclusion rests on is unchanged to within
  0.0002 across a scorer change that moved every probability, which is the reason the dose-response
  reading below is left standing rather than re-measured: the mechanism is structural (unemitted
  pairs priced at p = 0), not a property of one booster.

  The deficit is ~0 at size 2 and -0.1141 at size 8, rising with size almost monotonically. That
  is the predicted signature and the mechanism is not subtle: an entity of *k* records implies
  k(k-1)/2 pairs, blocking emits only some of them, and every pair it never emitted enters the
  objective at p = 0 — priced not as *unknown* but as a confident non-duplicate. Average linkage
  averages over all of them, so the larger the entity the more zero-evidence pairs drag its mean
  merge gain below the bar. Components at `p_hi` never averages — one surviving edge merges — so
  unemitted pairs cost it nothing, which is exactly why it wins on recall here while losing badly
  on precision and bill.

  It also explains why Abt-Buy could never have shown this: its entities are only sizes 2 and 3,
  the two rows where the deficit is -0.0013 and -0.0175. The -0.006 overall gap recorded there was
  not a weaker version of the effect, it was the effect measured where it cannot appear.

  What this does *not* settle is the fix, and the entry deliberately gated the fix behind this
  measurement rather than the other way round. Three candidates, none yet measured: price unemitted
  pairs at the catalog's base rate instead of 0; exclude them from average linkage's mean entirely,
  so absence of evidence stops being evidence of absence; or price them at `p_lo`. The second is the
  most principled and the most invasive — it breaks the objective's current claim to bill every pair
  a partition decides, which is what makes `reports/cluster.md`'s 154 comparable with the bands' 284,
  and `cluster/base.py`'s objective is the one an `er-invariants` audit has already caught once.
  Worth doing, worth doing deliberately.
- **A lower expected cost did not buy a better partition, because the scorer bounds the
  objective — and this now survives the scorer change that was meant to revisit it.** Correlation
  clustering finds expected cost 170.1 against average linkage's 171.0, yet bills 193 against 154:
  it re-fuses the Weber 3780001 / 3880001 grills, whose cross-entity pairs the model scores
  0.97–0.99, so under the model fused *is* cheaper — average linkage separates them by merge order,
  not by the objective. It is now the *only* fusion either cost-based clusterer makes on this
  split, average linkage having dropped to zero once extraction was fixed. The true partition costs
  1,353.8 in expectation, because merging the true pairs the model rejects is priced as false
  merges, so no search over this objective reaches the ceiling.

  This entry used to end "Revisit when an `error-analyst` pass changes the scorer." That trigger
  has now fired twice over — the `code_key` extraction fix and `monotone_constraints` together
  moved every probability in the pipeline, and on `synth-20k` they cut average linkage's fused
  clusters from 69 to 29 — and the relationship is unchanged on both catalogs: correlation
  clustering reaches the lower expected cost (7,802.0 against 7,808.9 on `synth-20k`) and the
  higher realized bill (7,295 against 7,279). Three scorers now, across two catalogs, same
  direction — the second shape-gate fix moved every probability again and did not touch it. That is
  no longer a curiosity about one model's errors; it is a property of optimizing a proxy whose
  weights are themselves estimates.

  **So the choice is settled: average linkage is the default, and `service/batch.py` is right to
  hardcode it** (`CLUSTERER = "average_linkage"`). CLAUDE.md previously said "`service/` has not
  picked one", which was simply wrong — it picked one at v1 — and the measurement now justifies the
  pick rather than merely tolerating it. Correlation clustering stays, undeleted and still reported
  every run, because it is what makes the proxy's limit visible: delete it and the next reader has
  no way to see that the objective's own winner loses.
- **What a report may say about a dataset lives with the dataset, and measured claims stay
  measured.** The renderers printed Abt-Buy's facts for any `--dataset` — "pre-blocked", a 0.5204
  baseline, Abt-against-Buy description lengths — and two more were false the first time another
  catalog ran: `synth-20k` has train F1 *above* test F1, and a raised similarity floor does impose a
  recall ceiling. Dataset passages now live in `DatasetNotes`, and anything a report can measure is
  a branch on the measurement. A test used to assert that a report on `dataset="synthetic"` says
  "pre-blocked"; it now asserts the opposite.
