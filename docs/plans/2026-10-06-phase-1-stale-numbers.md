# Phase 1: fix the stale numbers and stop them going stale

- **Date:** 2026-10-06
- **Status:** in progress -- 1.1 under way; `abt-buy` runs by default (1.1e decided)
- **Source:** Phase 1 of `docs/plans/2026-10-05-pending-work-roadmap.md`, decomposed into tasks
  after measuring the ground it stands on. The roadmap's ordering (1.1 first) is kept; its effort
  estimates are revised upward for 1.2, 1.3 and 1.4, and 1.6 is resequenced ahead of 1.3. Both
  changes are justified under "What measurement already settled" and "Revisions to the roadmap".

Any task that settles an entry in `docs/DECISIONS.md` updates that entry in the same commit.

## What measurement already settled

Before decomposing 1.1 I re-ran every committed report and diffed it against the committed file.
This de-risks the whole task and changes its design, so it is recorded here rather than
rediscovered during implementation.

**Every committed report reproduces exactly, except the wall-clock columns.** Measured by
re-running each report's own recorded command and diffing:

| report | runtime | differing lines | cause |
| --- | ---: | ---: | --- |
| `reports/baseline_tfidf.md` | 3 s | 0 | -- |
| `reports/blocking.md` | 9 s | 2 | `ann` build/query seconds |
| `reports/features.md` | 6 s | 0 | -- |
| `reports/model.md` | 46 s | 0 | -- |
| `reports/cluster.md` | 41 s | 0 | -- |
| `reports/synth/baseline_tfidf.md` | 21 s | 0 | -- |
| `reports/synth/blocking.md` | 149 s | 10 | five blockers' build/query seconds |
| `reports/synth/features.md` | 124 s | 0 | -- |
| `reports/synth/realism.md` | 1 s | 0 | -- |
| `reports/synth/model.md` | 660 s | 0 | -- |
| `reports/synth/cluster.md` | 596 s | 0 | -- |
| `reports/synth/blocking-200k.md` | 269 s | 12 | six blockers' build/query seconds |

**All twelve reproduce.** Across the whole committed set, the blocking reports' wall-clock columns
are the only cells that move -- every candidate count, pair completeness and reduction ratio
reproduced to the digit, on all three catalogs, including the 17.6M-candidate `ann` row at 200k.
That is the single fact 1.1's design rests on, and it is measured rather than assumed.

The cost split that decides 1.1e is stark: the five `abt-buy` reports total **105 s**, while the
seven synthetic ones total **1,820 s (30 minutes)** -- `synth/model.md` alone is 660 s and
`synth/cluster.md` 596 s.

So the premise behind Phase 1 needs restating precisely: **the committed reports are not stale.**
What is stale is the set of figures in `docs/DECISIONS.md` that were produced by one-off scripts
and never became reports -- the calibration table, the three-arm monotone table, the
per-entity-size B-cubed table. That distinction is what drives the revisions below: 1.2, 1.3 and
1.4 are not "re-run a command", they are "build the command that was never built, then run it".

Four findings that constrain 1.1's design, each found by running it rather than reading the code:

1. **Wall-clock columns are the only non-determinism**, and `reports/blocking.md` already declares
   this in its own header -- *"Candidate counts and completeness figures reproduce exactly between
   runs -- the ANN index is built single-threaded for that reason. Only the wall-clock columns
   vary."* Masking those two columns is therefore implementing the report's stated contract, not
   inventing an exception to suit a test. Every candidate count, pair completeness and reduction
   ratio reproduced to the digit.
2. **The renderers read sibling reports off disk.** `eval/baseline.registered_baseline` opens the
   path in `DatasetNotes.baseline_report`, parses a row out of it, and *warns to stderr* rather
   than raising when it is absent. Re-rendering `features.md` with an empty `reports/` silently
   produces different prose ("no baseline report is registered for this dataset"). A reproduction
   test must therefore run with the committed `reports/` tree present.
3. **The `--out` path leaks into derived prose, and re-wraps it.** `cluster/evaluate.py` derives
   its cross-references to `reports/model.md` from its own `--out` directory, so re-rendering to a
   temporary path changes four lines of prose and re-wraps a paragraph. Substituting the path back
   afterwards does not recover the wrapping. The fix that works: re-run in a sandbox working
   directory that mirrors the repo layout, passing the committed *relative* `--out` and an absolute
   `--root`. Validated -- `features.md`, `cluster.md` and `model.md` then reproduce byte for byte.
4. **`realism.md` re-derives its own provenance block.** Narrowed from how this was first written,
   after checking it directly: the historical "it was written by" command was built from both the
   current `--out` and the current `--report`. The `--out` half is harmless -- it names the
   catalog's real location, just spelled absolutely instead of relatively. The `--report` half was
   a false claim about history: re-rendering to a scratch path produced a block reading
   `--records 20000 --seed 0 --out ... --report scratch.md`, an invocation that never ran. Fixed in
   1.1f by dropping `--report` from the historical command entirely, since a report path is no
   part of writing a catalog.

Baseline for the cost conversation: the suite is **714 passed, 1 skipped in 258 s** today, with the
benchmark present. The one skip is the semantic test of 1.5.

## Revisions to the roadmap

Stated rather than applied silently.

- **1.2, 1.3 and 1.4 are each M, not S.** All three read as "re-run", but none of the figures they
  re-run is reachable from committed code. `BetaCalibrator` exists and is unit-tested but is
  unreachable from `fit_calibrated_scorer`, which hardcodes Platt; **no isotonic calibrator class
  exists at all**, only prose about one. `monotone_constraints()` takes the featurizer's specs but
  no feature-set argument, so there is no code-only or title-subset arm to select, and
  `model.evaluate.evaluate` does not accept a `params` override either. `bcubed()` returns one
  score with no per-size breakdown. Each task therefore needs its knob built before it can be
  turned. That is the roadmap's own Optional item -- "commit the one-off figures by adding the CLI
  flags they need" -- arriving as a prerequisite rather than a nice-to-have, and it is the right
  trade: a flag makes the answer reproducible, a script makes it stale again on the next
  extraction change.
- **1.6 moves ahead of 1.3.** 1.3 has to choose *which* title columns go in its subset arm, and
  1.6's error analysis over `features/` is what produces the evidence for that choice. Done in the
  roadmap's order, 1.3's subset is a guess; done in this order, it is derived. 1.6 is S either way.

## Tasks

### 1.1 Report-reproduction test -- M, no dependencies, do first

Lands before 1.2-1.4 so their regenerated reports are guarded rather than hand-checked.

- [x] **1.1a Discover reports and parse their recorded commands.** Glob `reports/**/*.md` so a new
  report is covered without editing a registry, and parse the command from the first fenced `bash`
  block after the provenance marker. Note the marker text differs: eleven reports say
  `Regenerate with:`, `realism.md` says `Rendered against the catalog already on disk. Re-render
  with:`. Assert each parsed command's `--out` (or `--report`) equals that report's own
  repo-relative path. *Done when* a report whose block is missing, malformed, or points at the
  wrong path fails a test that names the report.
- [x] **1.1b Sandbox fixture.** A temporary working directory holding a copy of the committed
  `reports/` tree, with the dataset supplied as an absolute `--root` and the committed relative
  `--out` passed through unchanged. `realism.md` also needs the catalog it names reachable at the
  committed relative path, since its `--out` is a data path. *Done when* re-running any committed
  command inside the sandbox writes to the same relative path the committed report records, and
  `features.md` -- the report whose prose depends on a sibling -- reproduces byte for byte.
- [x] **1.1c The comparison.** Byte-exact, with the blocker tables' two wall-clock columns masked
  on both sides. On failure print a unified diff and the exact command to regenerate. *Done when*
  changing one digit of a committed figure fails with a diff pointing at that line, and the
  message is enough to act on without reading the test.
- [x] **1.1d Pin the mask.** The mask is the one place this test can go blind, so test it: on the
  committed blocking reports it must alter exactly the wall-clock cells and nothing else. *Done
  when* widening it to touch a pair-completeness or candidate-count cell fails.
- [x] **1.1e Gating, visible in `pytest -rs`.** Use `skipif`, never `-m` deselection: CLAUDE.md's
  testing contract is that `pytest -rs` tells you what did not run, and a deselected test is
  invisible. Recommended split -- the five `abt-buy` reports (105 s) run whenever
  `data/raw/abt-buy` exists; the synthetic reports skip with a reason unless opted in by an
  environment variable. *Done when* `pytest -rs` names every report that was not reproduced.
  **This is the one open decision in Phase 1 -- see "Decision needed".**
- [x] **1.1f Make `realism.md`'s provenance block a literal.** Finding 4 above: the historical
  command was re-derived from the current `--report`, so it was not a record. *Done when*
  re-rendering to a different path leaves the "it was written by" block unchanged, pinned by a test.

**1.1 as shipped.** `tests/test_report_reproduction.py`, 27 cases over the 12 reports discovered by
glob:

- 15 run with no data at all -- the provenance blocks parse, each command writes the report it is
  recorded in, the mask alters exactly the seven blocker rows' wall-clock cells and nothing else,
  and it leaves every non-blocking report untouched.
- 12 reproduction cases: the five `abt-buy` reports pass; the seven that touch a synthetic catalog
  skip under `DEDUP_REPRODUCE_SLOW`, each naming itself and the variable in `pytest -rs`.
- **Verified cost: the suite goes 258 s -> 383 s, 735 passed and 8 skipped.** Per case, from
  `--durations`: `model.md` 61.8 s, `cluster.md` 51.0 s, `features.md` 8.7 s, `blocking.md` 7.4 s,
  `baseline_tfidf.md` under 6.5 s. That is +48% against the +40% estimated when 1.1e was decided.
  One earlier timing of 1:43:58 was discarded as confounded -- it overlapped with other work on
  the same machine, and LightGBM and faiss both multi-thread by default, so two concurrent suites
  oversubscribe the CPU badly. Measure this one alone or not at all.
- Verified by regression, not by reading: a one-digit edit to `reports/baseline_tfidf.md`'s
  headline F1 failed with a three-line diff pointing at the cell and the command to regenerate.
  `realism.md` was regenerated for 1.1f and reproduces.

Two things worth knowing before 1.2-1.4 lean on this:

- The guard compares each report against the *committed* state of its siblings, because the
  renderers read them off disk. That is the right semantics -- a stale sibling fails its own case --
  but it means a change rippling through two reports needs both regenerated in the same commit, or
  the second one's case fails on the first one's new figure.
- Every CLI prints its report to stdout as well as writing `--out`, so the test drains captured
  output; without that the diff is followed by the whole report again.
- **The suite now trains the Abt-Buy model twice and clusters it twice**, ~113 s of genuinely
  duplicated computation: `test_model_evaluate.py`'s setup is 56.3 s against this file's 61.8 s,
  and `test_cluster_evaluate.py`'s 52.3 s against 51.0 s. They cannot share a fit -- the per-stage
  tests call `evaluate()` in-process while this one must run the recorded command in a sandbox, and
  that difference is the point of it. Left as it is deliberately, because the two check different
  things: the per-stage tests assert *claims* ("the model beats the published baseline") and fail
  with the reason attached, this one asserts the *artifact* and fails with a line number. Logically
  the artifact check subsumes the figures; diagnostically it does not. Worth revisiting if the
  suite's runtime becomes the binding constraint -- it is a recorded trade, not an oversight.

### 1.6 Error analysis over `features/` -- S, no dependencies, informs 1.3

Moved ahead of 1.3. The roadmap asks for "the second feature gap that was never written down": the
first was the truncated-title case, which `cross_title_desc_cosine_max` was added for and which
`docs/DECISIONS.md` records; the second was found in the same pass and never recorded.

- [ ] **1.6a Re-derive the error population under today's extraction.** `reports/model.md` already
  localises it: 61 true pairs fall below `p_lo`, 34 of them score below 0.01, and blocking emitted
  every one of them -- so this is a feature gap, not a recall gap. Start there and at the
  top-ranked false pairs. *Done when* both populations are dumped with their feature vectors.
- [ ] **1.6b Characterise, compare against base rates, and record.** The standard the existing
  synthetic-ranking entry set: a hypothesis is accepted only against an enrichment over the
  candidate base rate, not because the examples look convincing. *Done when* the gap is recorded in
  `docs/DECISIONS.md`, or recorded as confirmed absent with the base-rate comparison that rules it
  out -- a negative result is a result here, since the entry claims a gap exists.
- [ ] **1.6c Report which title columns carry "low title agreement, high score".** This is what
  1.3's subset arm needs. *Done when* 1.3a can pick its subset from a measurement.

### 1.2 Calibration comparison -- M, depends on 1.1

- [ ] **1.2a Add `IsotonicCalibrator`.** It must declare, in code, that it breaks the project's
  strict-monotonicity rule: `docs/DECISIONS.md` records that isotonic is only *weakly* monotone and
  is the only map measured whose PR-AUC falls (0.7547 against 0.7608), because the ties it creates
  destroy ranking information. That makes it a measurable arm but never a legal default, and the
  class should say so where `PairScorer.probabilities` can be read next to it. *Done when* tests
  pin both the flattening and the PR-AUC fall -- the documented failure mode becomes a guard
  instead of prose.
- [ ] **1.2b Thread `--calibrator {platt,beta,isotonic,none}`** through `fit_calibrated_scorer` ->
  `model.evaluate.evaluate` -> CLI, and the same through `cluster/evaluate.py`, which is where the
  realized bill of a calibrator change becomes visible. *Done when* each arm is reproducible from a
  command that can be recorded in a report's "Regenerate with" block.
- [ ] **1.2c Run four arms on both catalogs** -- ECE, Brier, PR-AUC, distinct levels >= 0.9,
  precision and recall at `p_hi`, and n >= `p_hi`. **All four arms share one booster and one set of
  out-of-fold scores** -- only the calibrator applied to them differs, which is how the committed
  table was produced ("four calibrators on the same out-of-fold scores"). So this is one fit with
  four maps applied, not four runs, and it should be built that way: a single pass that emits the
  whole table. At `synth-20k`'s 660 s per fit that is the difference between 11 minutes and 44.
  *Done when* the decisions-log table is current and its "every number in this entry predates
  `monotone_constraints`" warning can be deleted.
- [ ] **1.2d Price the switch to beta.** The log's explicit blocker: beta auto-merges far less
  (recall at `p_hi` 0.3159 against Platt's 0.5231) at higher precision, and "the realized bill of
  that trade has not been measured". *Done when* the bill is measured on both catalogs and the
  default is either changed or kept with the number stated.
- [ ] **1.2e If the default changes, regenerate what depends on it.** `model.md`, `cluster.md` and
  both synthetic counterparts, plus any saved `artifacts/scorer` and the service index built from
  it. Note the manifest does *not* protect against this: it hashes the pickle against its own
  recorded digest and checks `feature_names`, so a scorer saved with the old calibrator loads
  cleanly and keeps serving the old probabilities -- nothing ties a persisted artifact to the
  current default. Worth deciding whether the manifest should record the calibrator type, which
  would turn this from a thing to remember into a thing that refuses. *Done when* 1.1's test is
  green on the regenerated reports and the commit body says what moved and why.

### 1.3 Monotone-constraint comparison -- M, depends on 1.1 and 1.6c

- [ ] **1.3a Make the constraint set selectable.** Give `monotone_constraints()` a feature-set
  argument and define the named sets -- code columns, title columns, and the subset 1.6c points at
  -- then thread `--monotone {full,code-only,title-subset,none}` through to the CLI. Keep the
  existing guard that raises when a listed name no longer resolves against the live registry, for
  every set rather than just the full one. *Done when* all four arms are reachable from a recorded
  command.
- [ ] **1.3b Re-run every arm under today's extraction.** All four arms, both catalogs, one
  extraction -- the existing table's rows are deliberately frozen because a three-way comparison is
  valid only when every arm shares one extraction, so the replacement table must be generated the
  same way, in one pass. **Unlike 1.2, these arms cannot share a fit**: a monotone constraint
  changes what the booster learns, so each arm is a full retrain. Budget accordingly -- four arms
  at `synth-20k`'s 660 s is ~44 minutes of compute on top of ~3 minutes for `abt-buy`, and that is
  the real reason this task is M. *Done when* the table is current and includes the title-subset
  row.
- [ ] **1.3c Answer the title-subset question and update the entry.** The log's open half: whether
  the title columns need the full constraint or only a subset, given code-only is strictly better
  on Abt-Buy and strictly worse on `synth-20k`'s P@10. *Done when* the entry states the answer and
  which arm ships.

### 1.4 Per-entity-size B-cubed recall -- M, depends on 1.1; unblocks 3.1

- [ ] **1.4a Add `bcubed_by_entity_size()`** to `cluster/bcubed.py`. *Done when* a unit test pins
  it against a hand-built partition and its overall row equals `bcubed()`'s recall -- the identity
  that makes the per-size rows trustworthy.
- [ ] **1.4b Render it in `reports/synth/cluster.md`**, as a section conditional on the catalog
  actually having entities above size 3. Abt-Buy has only sizes 2 and 3, and the entry's own
  conclusion is that this is "the effect measured where it cannot appear" -- so printing the table
  for Abt-Buy would invite exactly the misreading the convention against unconditional prose exists
  to prevent. *Done when* the table is a committed report under 1.1's guard rather than a one-off,
  and the Abt-Buy report does not grow a meaningless section.
- [ ] **1.4c Update the entry** -- replace the table and drop its "re-derive before quoting a
  single row" warning. *Done when* every row is current and 3.1 has a baseline to measure against.

### 1.5 Verify the semantic feature block -- S, independent

- [ ] **1.5a Download the weights and run the skipped test.**
  `sentence-transformers/all-MiniLM-L6-v2`, ~90 MB; `model_is_cached()` currently returns False and
  `test_semantic_cosine_is_high_for_a_paraphrase` is the suite's one skip. *Done when* the suite
  reports 715 passed, 0 skipped.
- [ ] **1.5b Exercise the real path, not just the unit test.** `features.evaluate --semantic` on
  `abt-buy`, which runs `SemanticBlock.fit`/`.transform` through `PairFeaturizer` at 34 columns
  rather than 33. *Done when* both methods have run once in the pipeline that would use them.
- [ ] **1.5c Check whether `title_embedding_cosine` belongs in `MONOTONE_INCREASING_FEATURES`.**
  It is a similarity in [0, 1] that rises with agreement, which is exactly the stated membership
  rule, and it is absent -- plausibly because the set predates anyone running `--semantic`. The
  column count also changes with `--semantic`, and `monotone_constraints` is derived from the live
  featurizer layout for that reason, so the two interact. *Done when* the omission is either
  corrected or recorded as deliberate.
- [ ] **1.5d Decide whether a `--semantic` report gets committed.** *Done when* either a committed
  report exists under 1.1's guard, or the decision not to commit one is recorded with the reason.

## Decision needed

**1.1e: does the `abt-buy` reproduction run in the default `pytest`?** The five `abt-buy` reports
cost 105 s against a suite that takes 258 s today -- roughly a 40% increase. The recommendation is
yes: `abt-buy` is the catalog every invariant and headline figure is quoted from, 105 s is
proportionate to a guard against publishing a stale number, and the synthetic tier -- 30 minutes,
of which one report is 11 -- stays opt-in. The alternative is to gate all twelve behind
the environment variable, which keeps the suite at 258 s but means the guard fires only when
someone remembers to ask for it; a guard nobody runs is the state this task exists to leave
behind.

## Out of scope

Deliberately not in Phase 1, and not silently dropped:

- Phase 2's Amazon-Google loader, though its raw CSVs are already downloaded under
  `data/raw/amazon-googleproducts/`.
- Phase 3's three candidate fixes for p = 0 under-merging. 1.4 builds their measuring stick; 3.1
  spends it.
- The `synth-200k` blocking sweeps (3.2) and the online-lookup replay harness (4.1).
