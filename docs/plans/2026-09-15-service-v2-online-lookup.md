# service/ v2: online lookup

- **Date:** 2026-09-15
- **Status:** done
- **Implemented in:** `276638d` feat(service): add online lookup -- index build/query split, decide(), /lookup route
- **Source:** plan mode (`splendid-doodling-hippo.md`), copied unchanged below this header.
  It records the plan as written before the work began; what shipped can differ -- see the
  commit and `docs/DECISIONS.md`.

## Context

`service/` v1 (batch dedup + review queue) shipped and was verified against real Abt-Buy data.
CLAUDE.md's second opening question — "given one new record, which existing records does it likely
duplicate" — is still unanswered: no blocker exposes a build/query split (all four rebuild their
index from scratch per call), and `cluster/` has no incremental-partition path. This plan closes
that gap, deliberately narrowed in scope (confirmed, not re-asked, given the pattern of this
session): only `ann` and `standard`'s two corpus-independent key functions become genuinely
online-queryable; a lookup extends one existing batch run in place rather than inventing a separate
"live catalog" concept. Both are real trade-offs, stated plainly below and in what ships, not
swept under "future work."

## Design

**Persisted, incrementally-queryable `ann` index** (`AnnIndex`, added to `blocking/ann.py`,
sharing `AnnBlocker`'s vectorizer/SVD fit-transform logic via two extracted helpers so there is
never a second, hand-copied "vector for a record" definition — exactly the train/serve skew
CLAUDE.md's `code_key` anecdote warns about). Splits into `build()` (fit vectorizer [+ SVD],
construct the FAISS `IndexHNSWFlat`, from a batch), `query_one()` (transform one new record through
the already-fitted vectorizer/SVD, search), `add()` (transform + `index.add()`, so a later lookup
sees it too). Persistence mirrors `PairScorer.save`/`.load` exactly — a directory, `manifest.json`
with an `artifact_sha256`, refuse on hash mismatch — with `faiss.write_index`/`read_index` for the
index itself (confirmed round-trips HNSW params correctly) and `pickle` for the fitted
vectorizer/SVD (the same mechanism `PairScorer` already uses for its own sklearn objects).
Confirmed directly: `IndexHNSWFlat.add()` genuinely works incrementally after construction (built
20 vectors, added 1, `ntotal` went 20→21, found it on search). The single-threaded determinism
guard (`omp_set_num_threads(1)`) stays on `build()` — it exists because *parallel construction of a
batch* races when multiple new nodes link concurrently — but does not apply to `add()`, which
inserts one vector at a time with nothing else to race against.

**Stated cost, not hidden:** the vectorizer is fit once at `build()` and never refit — refitting
after insertion would silently remap every already-indexed vector's dimensions. A lookup whose
distinguishing vocabulary is genuinely new to the corpus gets no signal from it. Unmeasured how
much this costs over many lookups — an Open Questions entry, not asserted away.

**Persisted, incrementally-queryable `standard` index** (`InvertedIndex`, new module
`blocking/standard_index.py`) — `model_number_keys` and `code_token_keys` only.
**`rare_token_keys` is excluded, not deferred quietly**: it needs corpus-wide document frequency,
which every insert changes, so it cannot be incrementally maintained without a full rebuild — the
exact thing this index exists to avoid. Plain JSON persistence (no pickle needed, no sklearn
object, human-inspectable), same directory+manifest+hash-refusal shape.

**The decision logic** (new `service/lookup.py`, pure functions, mirroring `batch.py`'s
"no I/O" split) — no `ScoredGraph`/`average_linkage` involved; every artifact gets its own local
positional numbering for one request, with `record_id` the only identifier threaded between them.
Scores the new record against every candidate from `AnnIndex.query_one` + `InvertedIndex.query_one`
via the existing `PairScorer.probabilities` (works unchanged for any record list shape), groups
candidates by their *current* `cluster_id`, and applies `CostModel`'s `p_hi`/`p_lo` thresholds
directly — the closed-form logic `assign_bands` already embodies, not reinvented:

- No cluster scores ≥ `p_lo` for anything → **new singleton cluster**, nothing queued.
- Some cluster(s) score in `[p_lo, p_hi)`, none ≥ `p_hi` → **new singleton cluster**, one review row
  per such cluster (not just the top one — collapsing several plausible matches into one
  understates the ambiguity a reviewer needs to see).
- One or more clusters score ≥ `p_hi` → merge into the **highest-scoring** one (ties broken on the
  smaller `cluster_id`, matching `agglomerative.py`'s own convention); **every other qualifying
  cluster — both the rest of the `p_hi` set and the whole `[p_lo, p_hi)` set — gets one review row**
  against the new record's cluster. This is `cluster/base.py`'s own invariant ("every pair a
  partition leaves apart falls back to its band") applied to a partition of size one, not an ad hoc
  rule.

**Stated limitation, not hidden:** this never asks whether two *existing* clusters should merge —
only which one cluster the new record joins. A repeated pattern of strong review-links between the
same two clusters is a signal `average_linkage` would eventually resolve on a full batch re-run,
and this design structurally cannot surface or act on it. Open Questions entry, not solved here.

**Schema** — a new table, `run_indexes` (`run_id`, `ann_root`, `standard_root`, `built_at`), not
new columns on `runs`: `CREATE TABLE IF NOT EXISTS` cannot retroactively add a column to an
existing table, so extending `runs` directly would silently break every DB file created before this
change. No row exists until lookup is explicitly enabled for a run (see below) — a plain batch run
stays exactly as immutable-looking as it is today until someone opts in. `runs`' live counts
(`n_records`, `n_clusters`, band counts) update on every lookup — the alternative (freeze at batch
time) would make `GET /runs/{id}` silently wrong for any run that has accepted a lookup, worse than
the `UPDATE` cost. **Stated plainly:** a run that has accepted a lookup is no longer reproducible
from `service/batch.py`'s CLI — its `clusters` table now holds insertion-order labels for at least
one member, not `canonical_labels` over a full graph — and `store.py`'s docstring must say so, not
keep implying every run is a frozen snapshot.

**Two operations, deliberately split** (mirroring why batch execution stayed CLI-only in v1):

1. **`python -m dedup.service.build_index --db ... --run-id ... --index-root ...`** — a new CLI,
   not a route: building `AnnIndex` refits a vectorizer over the whole run's records, the same
   unbounded, request-shaped-wrong operation that kept batch execution out of `app.py` in the first
   place. The explicit "this run now accepts lookups" transition.
2. **`POST /runs/{run_id}/lookup`** — body reuses `schema.Record`'s fields (minus `record_id`/
   `entity_id`, assigned by the service); response names the decision (`auto_merge`/`review`/
   `new_cluster`), the record as stored, its landing cluster, and any secondary review rows.
   404 unknown run, 409 no index built yet (points at `build_index`), 409 duplicate `record_id`.
   Writes via a new `store.apply_lookup()` (one transaction, mirroring `write_run`'s
   `BEGIN`/`COMMIT`/`ROLLBACK`), then **always** re-saves both indexes with the new record added —
   not batched, because a crash between requests must not silently reopen a recall gap on restart.
   **Stated cost:** `faiss.write_index` re-serializes the whole graph on every insert, so lookup
   latency grows with catalog size — a real scaling limit this pass doesn't solve, named in
   CLAUDE.md rather than hidden.

**A real, previously-unstated concurrency hazard, checked not assumed:** every route in `app.py` is
`def`, not `async def`, so FastAPI/Starlette executes them in a thread-pool — concurrent requests
genuinely run in parallel threads today. A single DuckDB connection and, worse, an in-memory
mutating FAISS index are both unsafe under that. Fix: a per-`run_id` `threading.Lock` in
`app.state`, held for the duration of any route touching the connection or a mutable index
(the lookup route and, in fairness, the existing review-decision route too) — throughput-limiting
by design, the same "out of scope on purpose" posture the docstring already takes for batch
execution.

## What's explicitly not in this pass

- `lsh`, `sorted_neighborhood`, `rare_token_keys` are not online-queryable — a real, **unmeasured**
  recall gap against the batch path's PC 0.9928. Open Questions entry, per CLAUDE.md's own rule
  that a claim like this gets a measured number or an explicit "not yet measured," never neither.
- No cross-run entity identity — a lookup-minted `cluster_id` is still `f"{run_id}:{label}"`.
- No cross-cluster reconciliation within one run (the decision-logic limitation above).
- Vectorizer vocabulary drift under sustained lookup — unmeasured.

## Implementation sequence

1. `blocking/ann.py`: extract shared fit/transform helpers from `AnnBlocker._vectors`; add
   `AnnIndex` (build/query_one/add/save/load). Tests: `tests/test_blocking_ann_index.py` —
   round-trip, a known near-duplicate found by `query_one`, `add()` then found, save→load parity,
   tamper-refusal, the `n_components` (SVD) path.
2. `blocking/standard_index.py`: `InvertedIndex` (build/query_one/add/save/load), JSON-backed.
   Tests: `tests/test_blocking_standard_index.py` — same shape.
3. `service/store.py`: `run_indexes` table; `get_records_by_ids`, `get_run_indexes`,
   `set_run_indexes`, `apply_lookup` (one transaction: new `records` row with the next `position`,
   new `clusters` row or an existing one's `size` bumped, 0+ `review_queue` rows, `runs` counts
   updated); reject a duplicate `record_id` before writing. Extend `tests/test_service_store.py`.
4. `service/lookup.py`: the pure scoring+decision function from Design, above. Tests:
   `tests/test_service_lookup.py`'s logic-only cases (no store, no app) mirroring how
   `service/batch.py`'s `run_batch()` is tested standalone.
5. `service/build_index.py`: the CLI. Extend `tests/test_service_batch.py`-style end-to-end
   coverage or a new file — build indexes for a real written run, confirm `run_indexes` populated.
6. `service/app.py`: `POST /runs/{run_id}/lookup`, `service/schemas.py`'s `LookupResult`/
   `LookupRequest`, the per-`run_id` lock (applied to this route and the existing decision route).
   `tests/test_service_lookup.py`'s `TestClient` cases: auto-merge, review (single and the
   multi-candidate-different-clusters case explicitly), new-cluster, duplicate-`record_id` 409,
   no-index-built 409.
7. CLAUDE.md: the two-question paragraph, the `service/` bullet, Layout's `service/` line,
   Commands (the `build_index` and `curl .../lookup` examples), three new Open Questions entries
   (the recall gap, vocabulary drift, no-cross-cluster-reconciliation) — each stated as measured or
   explicitly unmeasured, never neither.
8. One commit, scope `service`.

## Critical files

- `src/dedup/blocking/ann.py` — `AnnIndex`, shared fit/transform helpers
- `src/dedup/blocking/standard_index.py` — new, `InvertedIndex`
- `src/dedup/service/store.py` — `run_indexes` table, `apply_lookup`
- `src/dedup/service/lookup.py` — new, the decision logic
- `src/dedup/service/build_index.py` — new, the CLI
- `src/dedup/service/app.py`, `src/dedup/service/schemas.py` — the route
- `src/dedup/model/threshold.py`, `src/dedup/cluster/base.py` — reused, not modified
  (`CostModel`, `assign_bands`'s thresholds, the leave-apart-pricing invariant)

## Verification

- `pytest` green after each numbered step.
- `ruff check src tests` clean.
- End-to-end smoke test against real Abt-Buy data (same shape as v1's): batch-run it, `build_index`,
  then look up (a) a genuine near-duplicate of an existing record — confirm auto-merge into the
  right cluster; (b) an ambiguous variant — confirm a review row against the right cluster;
  (c) an unrelated product — confirm a new singleton; (d) a record scoring ≥ `p_hi` against two
  different existing clusters — confirm exactly one merge and one review row, not both merged and
  not both queued.
- Confirm a second `build_index` + lookup cycle still round-trips the persisted indexes correctly
  (save→load parity holds across a real run, not just the unit-test fixture).
