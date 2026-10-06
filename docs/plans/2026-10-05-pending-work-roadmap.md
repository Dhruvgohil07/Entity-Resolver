# Pending work roadmap

- **Date:** 2026-10-05
- **Status:** proposed -- nothing started
- **Source:** the pending items in `CLAUDE.md` and the open questions in `docs/DECISIONS.md`

Ordered by what each task unblocks and how cheap it is. Phase 1 refreshes stale numbers and closes
the gap that keeps making them stale; Phase 2 adds the second real dataset several open questions
are waiting on; later phases need measurement harnesses or real usage. Effort is relative:
**S** is under a session, **M** about a session, **L** several.

Any task that settles an entry in `docs/DECISIONS.md` updates that entry in the same commit.

## Phase 1: Fix the stale numbers and stop them going stale

Do 1.1 first, so 1.2–1.4 regenerate under a guard rather than by hand.

**Decomposed into tasks in `docs/plans/2026-10-06-phase-1-stale-numbers.md`**, which measured the
ground first and revised three of these estimates: the committed reports all reproduce today, so
what is actually stale is the one-off figures in `docs/DECISIONS.md` that never became reports —
making 1.2, 1.3 and 1.4 each M rather than S, because each needs its CLI flag built before the
measurement can be run reproducibly. 1.6 also moves ahead of 1.3, which depends on its output.

- [ ] **1.1 Report-reproduction test** (M). Regenerate each committed report's key figures and
  compare them with the committed file. *Done when* a change to extraction or the model fails a
  test instead of silently leaving a report stale.
- [ ] **1.2 Re-run the calibration comparison** (S). Raw, Platt, isotonic and beta on both
  catalogs, plus the realized cost of switching to beta. *Done when* the calibration entry has
  current numbers and a decision on whether beta becomes the default.
- [ ] **1.3 Re-run the monotone-constraint comparison with a title-subset arm** (S). All arms under
  today's extraction, plus one constraining only some title columns. *Done when* the three-arm
  table is current and the title-subset question is answered.
- [ ] **1.4 Re-run the per-entity-size B-cubed recall table** on `synth-20k` (S). *Done when* every
  row is current; it is the baseline for 3.1.
- [ ] **1.5 Verify the semantic feature block** (S). Download the sentence-transformers weights
  (~90 MB) and run the normally skipped test. *Done when* `SemanticBlock.fit`/`.transform` have run
  at least once.
- [ ] **1.6 Re-run an error analysis over `features/`** (S) to recover the second feature gap that
  was never written down. *Done when* it is recorded in `docs/DECISIONS.md` or confirmed absent.

## Phase 2: Add a second real dataset

- [ ] **2.1 Amazon-Google loader** (M). Loader module, `DatasetNotes`, cp1252 fixtures, registry
  entry. *Done when* `load_dataset('amazon-google')` works and the dataset-name import test passes.
- [ ] **2.2 Run all five report CLIs on Amazon-Google** (M). *Done when* `reports/amazon-google/`
  exists, with every interpretive sentence chosen from measured values.
- [ ] **2.3 Cross-source number** (S). The published-benchmark framing as a second number in
  `eval/`, with no per-source branch downstream. *Done when* both numbers are reported side by side
  and the evaluation-framing question is resolved or narrowed.
- [ ] **2.4 Re-check `synth/` realism against two real catalogs** (S). *Done when* the two
  structural deviations are settled or confirmed.

## Phase 3: Fix measured but unfixed problems

- [ ] **3.1 Large clusters split too often** (M). Measure the three candidate fixes for pairs
  blocking never emitted -- price them at the base rate, exclude them from the mean, or price them
  at `p_lo` -- against 1.4's per-size table. *Depends on* 1.4, ideally also 2.2. *Done when* a fix
  ships, or all three are rejected with measurements.
- [ ] **3.2 Push 200k blocking further** (S–M). Continue the `k` and LSH-cap sweeps, which were
  still improving at the last values tried. *Done when* a plateau or a stated time-budget stopping
  point is reached.

## Phase 4: Measure online lookup

- [ ] **4.1 Lookup replay harness** (M). Index a train split, look up held-out records one at a
  time, compare recall against batch blocking on the same split. *Done when* lookup's recall cost is
  a measured number.
- [ ] **4.2 Vocabulary drift** (S), with the same harness: index an early slice, look up a later
  one. *Done when* you know how often `build_index` needs re-running. *Depends on* 4.1.
- [ ] **4.3 Flag clusters that should merge** (M). Surface pairs of existing clusters that pile up
  strong review links to each other, for a batch re-run. *Done when* the gap is surfaced.

## Phase 5: Product features

Sits last because 5.2 needs real reviewers. Move 5.1 up to straight after Phase 1 if a usable
review tool matters more than settling measurements.

- [ ] **5.1 htmx review UI** (L) over the existing API, no Node toolchain. *Done when* a reviewer
  can work through the queue in a browser.
- [ ] **5.2 Per-pair vs per-cluster-merge review cost** (S, once data exists). Time each decision
  against cluster size. *Depends on* 5.1 and real review decisions.
- [ ] **5.3 Persistent entity identity** (L). *Done when* a cluster keeps its ID across batch runs.

## Optional

- [ ] Commit the one-off figures (character-shingle LSH, `ann` SVD sweeps) by adding the CLI flags
  they need.
- [ ] Compare against a fine-tuned cross-encoder as an upper bound on model quality.
