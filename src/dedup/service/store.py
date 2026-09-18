"""The DuckDB record/cluster/review store a batch run writes and the API reads.

No pydantic, no FastAPI here -- `service/batch.py` and `service/app.py` both
depend on this module without depending on each other, which is what keeps the
CLI that populates the store and the app that serves it from needing to agree
on anything beyond this file's row shapes.

Five tables, one batch run per `runs` row:

  * **`runs`** -- one row per invocation: dataset, the scorer artifact's hash
    (traceable back to the report that produced it, per CLAUDE.md's standard),
    cost model, clusterer name, and candidate-level band counts. No status
    column: a row is inserted once, atomically, after the whole pipeline
    (blocking -> scoring -> clustering -> writes) succeeds, so a crash mid-run
    leaves no partial row and no dangling foreign keys.
  * **`clusters`** -- `cluster_id` is `f"{run_id}:{label}"`, stable *within*
    one run (labels come from `cluster.base.canonical_labels`, deterministic
    given the same graph) but not a persistent cross-run entity identity: two
    runs over similar catalogs mint unrelated ids for "the same" product.
    Reconciling identity across re-runs is online-lookup territory, deferred
    with it.
  * **`records`** -- first-class columns for what the API filters/sorts/
    displays on, plus a `raw_json` column holding `record.model_dump_json()`
    as the authoritative row (the same reasoning `data/synthetic.py` gives for
    JSONL over CSV: a `Record` field added later needs no migration here).
  * **`review_queue`** -- populated from `cluster.base.review_mask`, **not**
    from `assign_bands(...).review`. This is the one correctness-critical
    choice in the schema: a partition can merge a pair the pairwise bands
    would have queued, or leave apart one they would have auto-merged
    (`cluster/base.py`'s own docstring), so `assign_bands` alone would produce
    a queue inconsistent with the clusters the same run returns. A decided
    row is `(left_record_id, right_record_id, is_match)` -- already the shape
    `pair_labels()` wants, which is the "human decisions flow back as
    training labels" pipeline CLAUDE.md names. `left_cluster_id`/
    `right_cluster_id` plus `created_at`/`decided_at` make the open question
    "measured time per merge decision against cluster size" instrumentable --
    collected here, not analyzed. Auto-merge/auto-reject pairs are not
    itemized (100x+ the row count for no v1 use case); `runs` only counts
    them. A decision is write-once: `UNIQUE (run_id, left_record_id,
    right_record_id)` plus `review.py`'s already-decided check keep a row
    that may already have been read as a training label from being silently
    overwritten.
  * **`run_indexes`** -- one row per run that has opted into online lookup
    (`service/build_index.py`), naming the persisted `AnnIndex`/`InvertedIndex`
    directories the same way `runs.scorer_root` already names a `PairScorer`'s:
    a caller-chosen filesystem path, independent of where the DuckDB file
    itself lives. `apply_lookup` (below) is what a run accepting a lookup
    actually changes: `runs`' live counts update on every call (the
    alternative -- freezing them at batch time -- would make `GET
    /runs/{id}` silently wrong for any run that has ever accepted one), and
    `clusters`/`records` gain rows outside `write_run`'s one-shot insert.
    **A run that has accepted a lookup is no longer reproducible from
    `service/batch.py`'s CLI**: at least one of its cluster labels was minted
    by insertion order (`service/lookup.py`'s decision logic), not by
    `cluster.base.canonical_labels` over a full graph.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np

from dedup.blocking.pairs import unpack
from dedup.model.threshold import BandAssignment, CostModel
from dedup.normalize import NormalizedRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            VARCHAR PRIMARY KEY,
    dataset           VARCHAR NOT NULL,
    scorer_root       VARCHAR NOT NULL,
    scorer_sha256     VARCHAR NOT NULL,
    cost_false_merge  DOUBLE NOT NULL,
    cost_false_split  DOUBLE NOT NULL,
    cost_review       DOUBLE NOT NULL,
    clusterer         VARCHAR NOT NULL,
    created_at        TIMESTAMP NOT NULL,
    n_records         INTEGER NOT NULL,
    n_candidates      INTEGER NOT NULL,
    n_clusters        INTEGER NOT NULL,
    n_auto_merge      INTEGER NOT NULL,
    n_review          INTEGER NOT NULL,
    n_auto_reject     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id  VARCHAR PRIMARY KEY,
    run_id      VARCHAR NOT NULL REFERENCES runs(run_id),
    label       INTEGER NOT NULL,
    size        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS records (
    run_id      VARCHAR NOT NULL REFERENCES runs(run_id),
    record_id   VARCHAR NOT NULL,
    position    INTEGER NOT NULL,
    cluster_id  VARCHAR NOT NULL REFERENCES clusters(cluster_id),
    source      VARCHAR NOT NULL,
    title       VARCHAR NOT NULL,
    brand       VARCHAR,
    category    VARCHAR,
    price       DOUBLE,
    raw_json    VARCHAR NOT NULL,
    PRIMARY KEY (run_id, record_id)
);

CREATE TABLE IF NOT EXISTS review_queue (
    review_id         VARCHAR PRIMARY KEY,
    run_id            VARCHAR NOT NULL REFERENCES runs(run_id),
    left_record_id    VARCHAR NOT NULL,
    right_record_id   VARCHAR NOT NULL,
    probability       DOUBLE NOT NULL,
    band              VARCHAR NOT NULL,
    left_cluster_id   VARCHAR NOT NULL REFERENCES clusters(cluster_id),
    right_cluster_id  VARCHAR NOT NULL REFERENCES clusters(cluster_id),
    created_at        TIMESTAMP NOT NULL,
    is_match          BOOLEAN,
    reviewer_id       VARCHAR,
    decided_at        TIMESTAMP,
    UNIQUE (run_id, left_record_id, right_record_id)
);

-- Online lookup (service/lookup.py, service/build_index.py): a separate table,
-- not new columns on `runs` -- `CREATE TABLE IF NOT EXISTS` cannot retroactively
-- add a column to an existing table, so extending `runs` directly would silently
-- break every DB file created before this change. No row exists for a run until
-- `build_index` explicitly enables lookup for it; a plain batch run stays exactly
-- as immutable-looking as before until someone opts in.
CREATE TABLE IF NOT EXISTS run_indexes (
    run_id         VARCHAR PRIMARY KEY REFERENCES runs(run_id),
    ann_root       VARCHAR NOT NULL,
    standard_root  VARCHAR NOT NULL,
    built_at       TIMESTAMP NOT NULL
);
"""


def connect(path: Path | str) -> duckdb.DuckDBPyConnection:
    """Open (creating if new) the DuckDB file at `path`, with the schema applied.

    Idempotent -- every statement is `CREATE TABLE IF NOT EXISTS`, so calling
    this against an existing store is just a connection, not a migration.
    """
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(_SCHEMA)
    return conn


@dataclass(frozen=True)
class RunRow:
    run_id: str
    dataset: str
    scorer_root: str
    scorer_sha256: str
    cost: CostModel
    clusterer: str
    created_at: datetime
    n_records: int
    n_candidates: int
    n_clusters: int
    n_auto_merge: int
    n_review: int
    n_auto_reject: int


@dataclass(frozen=True)
class ClusterRow:
    cluster_id: str
    run_id: str
    label: int
    size: int


@dataclass(frozen=True)
class RecordRow:
    run_id: str
    record_id: str
    position: int
    cluster_id: str
    source: str
    title: str
    brand: str | None
    category: str | None
    price: float | None
    raw_json: str


@dataclass(frozen=True)
class ReviewQueueRow:
    review_id: str
    run_id: str
    left_record_id: str
    right_record_id: str
    probability: float
    band: str
    left_cluster_id: str
    right_cluster_id: str
    created_at: datetime
    is_match: bool | None
    reviewer_id: str | None
    decided_at: datetime | None


def write_run(
    conn: duckdb.DuckDBPyConnection,
    *,
    dataset: str,
    scorer_root: str,
    scorer_sha256: str,
    cost: CostModel,
    clusterer: str,
    records: Sequence[NormalizedRecord],
    pair_keys: np.ndarray,
    probabilities: np.ndarray,
    cluster_labels: np.ndarray,
    review_pairs: np.ndarray,
    bands: BandAssignment,
) -> str:
    """Write one batch run's `runs`/`clusters`/`records`/`review_queue` rows.

    One transaction: every table either gets this run's rows or none of them
    do. `review_pairs` (a boolean mask, e.g. `cluster.base.review_mask`'s
    output) must align with `pair_keys`/`probabilities` position for position
    -- exactly what `ScoredGraph.edges()` and `review_mask` already produce
    together. Returns the new `run_id`.
    """
    run_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    n = len(records)

    labels = np.asarray(cluster_labels, dtype=np.int64)
    if labels.shape != (n,):
        raise ValueError(f"cluster_labels shape {labels.shape} does not match {n} records")

    cluster_ids = np.array([f"{run_id}:{label}" for label in labels], dtype=object)
    sizes = np.bincount(labels)

    review_pairs = np.asarray(review_pairs, dtype=bool)
    if review_pairs.shape != pair_keys.shape:
        raise ValueError(
            f"review_pairs shape {review_pairs.shape} does not match pair_keys {pair_keys.shape}"
        )
    left, right = unpack(pair_keys, n)

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                dataset,
                scorer_root,
                scorer_sha256,
                cost.false_merge,
                cost.false_split,
                cost.review,
                clusterer,
                now,
                n,
                int(pair_keys.size),
                int(labels.max()) + 1 if n else 0,
                int(bands.auto_merge.sum()),
                int(bands.review.sum()),
                int(bands.auto_reject.sum()),
            ],
        )

        # duckdb's executemany refuses an empty parameter list, and an empty
        # review queue (or, for a degenerate empty catalog, empty clusters/
        # records) is a legitimate result, not a bug -- so every batch is
        # built first and only sent if it is non-empty.
        cluster_params = [
            (f"{run_id}:{label}", run_id, int(label), int(sizes[label]))
            for label in range(len(sizes))
        ]
        if cluster_params:
            conn.executemany("INSERT INTO clusters VALUES (?, ?, ?, ?)", cluster_params)

        record_params = [
            (
                run_id,
                record.raw.record_id,
                position,
                cluster_ids[position],
                record.raw.source,
                record.raw.title,
                record.raw.brand,
                record.raw.category,
                record.raw.price,
                record.raw.model_dump_json(),
            )
            for position, record in enumerate(records)
        ]
        if record_params:
            conn.executemany(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", record_params
            )

        review_positions = np.flatnonzero(review_pairs)
        review_params = [
            (
                uuid.uuid4().hex,
                run_id,
                records[left[i]].raw.record_id,
                records[right[i]].raw.record_id,
                float(probabilities[i]),
                "review",
                cluster_ids[left[i]],
                cluster_ids[right[i]],
                now,
                None,
                None,
                None,
            )
            for i in review_positions
        ]
        if review_params:
            conn.executemany(
                "INSERT INTO review_queue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                review_params,
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return run_id


def _run_row(row: tuple) -> RunRow:
    return RunRow(
        run_id=row[0],
        dataset=row[1],
        scorer_root=row[2],
        scorer_sha256=row[3],
        cost=CostModel(false_merge=row[4], false_split=row[5], review=row[6]),
        clusterer=row[7],
        created_at=row[8],
        n_records=row[9],
        n_candidates=row[10],
        n_clusters=row[11],
        n_auto_merge=row[12],
        n_review=row[13],
        n_auto_reject=row[14],
    )


def get_run(conn: duckdb.DuckDBPyConnection, run_id: str) -> RunRow | None:
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).fetchone()
    return _run_row(row) if row is not None else None


def list_runs(
    conn: duckdb.DuckDBPyConnection, *, limit: int = 50, offset: int = 0
) -> tuple[list[RunRow], int]:
    """Newest first, like every other listing here."""
    total = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?", [limit, offset]
    ).fetchall()
    return [_run_row(r) for r in rows], total


def _record_row(row: tuple) -> RecordRow:
    return RecordRow(
        run_id=row[0],
        record_id=row[1],
        position=row[2],
        cluster_id=row[3],
        source=row[4],
        title=row[5],
        brand=row[6],
        category=row[7],
        price=row[8],
        raw_json=row[9],
    )


def get_record(
    conn: duckdb.DuckDBPyConnection, run_id: str, record_id: str
) -> RecordRow | None:
    row = conn.execute(
        "SELECT * FROM records WHERE run_id = ? AND record_id = ?", [run_id, record_id]
    ).fetchone()
    return _record_row(row) if row is not None else None


def list_records(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    *,
    cluster_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[RecordRow], int]:
    where = "run_id = ?" + (" AND cluster_id = ?" if cluster_id is not None else "")
    params = [run_id] + ([cluster_id] if cluster_id is not None else [])
    total = conn.execute(f"SELECT count(*) FROM records WHERE {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM records WHERE {where} ORDER BY position LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return [_record_row(r) for r in rows], total


def _cluster_row(row: tuple) -> ClusterRow:
    return ClusterRow(cluster_id=row[0], run_id=row[1], label=row[2], size=row[3])


def get_cluster(
    conn: duckdb.DuckDBPyConnection, run_id: str, cluster_id: str
) -> ClusterRow | None:
    row = conn.execute(
        "SELECT * FROM clusters WHERE run_id = ? AND cluster_id = ?", [run_id, cluster_id]
    ).fetchone()
    return _cluster_row(row) if row is not None else None


def list_clusters(
    conn: duckdb.DuckDBPyConnection, run_id: str, *, limit: int = 50, offset: int = 0
) -> tuple[list[ClusterRow], int]:
    total = conn.execute(
        "SELECT count(*) FROM clusters WHERE run_id = ?", [run_id]
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM clusters WHERE run_id = ? ORDER BY label LIMIT ? OFFSET ?",
        [run_id, limit, offset],
    ).fetchall()
    return [_cluster_row(r) for r in rows], total


def get_clusters_by_ids(
    conn: duckdb.DuckDBPyConnection, run_id: str, cluster_ids: Sequence[str]
) -> list[ClusterRow]:
    """Batch fetch -- `service/lookup.py` needs every candidate cluster's
    *total* size (not just how many of its members blocking surfaced as
    candidates) to price a merge against `cluster/base.merge_credit`'s real
    objective, mirroring `get_records_by_ids`'s shape."""
    if not cluster_ids:
        return []
    placeholders = ", ".join("?" for _ in cluster_ids)
    rows = conn.execute(
        f"SELECT * FROM clusters WHERE run_id = ? AND cluster_id IN ({placeholders})",
        [run_id, *cluster_ids],
    ).fetchall()
    return [_cluster_row(r) for r in rows]


def _review_row(row: tuple) -> ReviewQueueRow:
    return ReviewQueueRow(
        review_id=row[0],
        run_id=row[1],
        left_record_id=row[2],
        right_record_id=row[3],
        probability=row[4],
        band=row[5],
        left_cluster_id=row[6],
        right_cluster_id=row[7],
        created_at=row[8],
        is_match=row[9],
        reviewer_id=row[10],
        decided_at=row[11],
    )


def get_review_item(
    conn: duckdb.DuckDBPyConnection, run_id: str, review_id: str
) -> ReviewQueueRow | None:
    row = conn.execute(
        "SELECT * FROM review_queue WHERE run_id = ? AND review_id = ?", [run_id, review_id]
    ).fetchone()
    return _review_row(row) if row is not None else None


def list_review_queue(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    *,
    decided: bool | None = False,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ReviewQueueRow], int]:
    """`decided=False` (the default) is the working queue -- pending items only."""
    where = "run_id = ?"
    params: list[object] = [run_id]
    if decided is True:
        where += " AND is_match IS NOT NULL"
    elif decided is False:
        where += " AND is_match IS NULL"
    total = conn.execute(f"SELECT count(*) FROM review_queue WHERE {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM review_queue WHERE {where} ORDER BY created_at LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return [_review_row(r) for r in rows], total


def decide_review(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    review_id: str,
    *,
    is_match: bool,
    reviewer_id: str | None,
) -> None:
    """The raw write. Not-found and already-decided are `service/review.py`'s
    job, not this module's -- store.py stays the same "no business logic,
    just rows" layer for writes that it already is for reads."""
    conn.execute(
        "UPDATE review_queue SET is_match = ?, reviewer_id = ?, decided_at = ? "
        "WHERE run_id = ? AND review_id = ?",
        [is_match, reviewer_id, datetime.now(UTC), run_id, review_id],
    )


def iter_records(conn: duckdb.DuckDBPyConnection, run_id: str) -> list[RecordRow]:
    """Every record in `run_id`, in `position` order -- `service/build_index.py`'s
    reader. Unpaginated on purpose: building an index needs the whole run's
    catalog, not a page of it, and it runs as a CLI, not a request handler."""
    rows = conn.execute(
        "SELECT * FROM records WHERE run_id = ? ORDER BY position", [run_id]
    ).fetchall()
    return [_record_row(r) for r in rows]


def get_records_by_ids(
    conn: duckdb.DuckDBPyConnection, run_id: str, record_ids: Sequence[str]
) -> list[RecordRow]:
    """Batch fetch -- `service/lookup.py` scoring a new record against a
    handful of ann/inverted-index candidates, not one `get_record` per id."""
    if not record_ids:
        return []
    placeholders = ", ".join("?" for _ in record_ids)
    rows = conn.execute(
        f"SELECT * FROM records WHERE run_id = ? AND record_id IN ({placeholders})",
        [run_id, *record_ids],
    ).fetchall()
    return [_record_row(r) for r in rows]


@dataclass(frozen=True)
class RunIndexesRow:
    run_id: str
    ann_root: str
    standard_root: str
    built_at: datetime


def get_run_indexes(conn: duckdb.DuckDBPyConnection, run_id: str) -> RunIndexesRow | None:
    row = conn.execute(
        "SELECT * FROM run_indexes WHERE run_id = ?", [run_id]
    ).fetchone()
    if row is None:
        return None
    return RunIndexesRow(run_id=row[0], ann_root=row[1], standard_root=row[2], built_at=row[3])


def set_run_indexes(
    conn: duckdb.DuckDBPyConnection, run_id: str, *, ann_root: str, standard_root: str
) -> None:
    """`service/build_index.py`'s one write -- upsert, since rebuilding an
    already-enabled run's indexes (after a lookup, or from scratch) is a
    legitimate re-run, not an error."""
    conn.execute(
        "INSERT INTO run_indexes VALUES (?, ?, ?, ?) "
        "ON CONFLICT (run_id) DO UPDATE SET "
        "ann_root = excluded.ann_root, standard_root = excluded.standard_root, "
        "built_at = excluded.built_at",
        [run_id, ann_root, standard_root, datetime.now(UTC)],
    )


@dataclass(frozen=True)
class ReviewTarget:
    """One existing cluster a looked-up record should be queued against --
    the specific existing record whose score decided it (`review_queue`
    always names real records, never bare cluster ids), and that score."""

    cluster_id: str
    record_id: str
    probability: float


@dataclass(frozen=True)
class LookupWriteResult:
    """What `apply_lookup` wrote: the new record's own row, and every
    `review_queue` row created for it -- returned directly rather than
    making the caller re-query for "the reviews this lookup just created,"
    which would otherwise mean fetching a whole page of the run's pending
    queue and filtering it in memory, and could miss rows past that page."""

    record: RecordRow
    review_rows: list[ReviewQueueRow]


def apply_lookup(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    *,
    record: NormalizedRecord,
    record_id: str,
    merge_cluster_id: str | None,
    review_targets: Sequence[ReviewTarget],
    n_auto_merge_candidates: int,
    n_review_candidates: int,
    n_auto_reject_candidates: int,
) -> LookupWriteResult:
    """Insert one looked-up record into an existing run: joins
    `merge_cluster_id` if given (bumping that cluster's `size`), else starts
    a new singleton cluster; writes one `review_queue` row per
    `review_targets` entry against the record's final cluster -- `cluster/
    base.py`'s own invariant, "every pair a partition leaves apart falls
    back to its band," applied to a partition of size one, not an ad hoc
    rule; updates `runs`' live counts (`n_records` always, `n_clusters` iff
    a new singleton was created, the three band counts by
    `n_*_candidates` -- the same candidate-level accounting `write_run`
    already does via `assign_bands`, just for this one lookup's candidates
    rather than a whole batch). One transaction, mirroring `write_run`'s
    BEGIN/COMMIT/ROLLBACK shape.

    Raises `ValueError` if `record_id` already exists in this run --
    `write_run` never needed this check, since a batch run always starts
    from an empty table; a lookup extends one that might already hold it.
    """
    conn.execute("BEGIN TRANSACTION")
    try:
        exists = conn.execute(
            "SELECT 1 FROM records WHERE run_id = ? AND record_id = ?", [run_id, record_id]
        ).fetchone()
        if exists is not None:
            raise ValueError(f"record {record_id!r} already exists in run {run_id!r}")

        position = conn.execute(
            "SELECT COALESCE(max(position), -1) + 1 FROM records WHERE run_id = ?", [run_id]
        ).fetchone()[0]

        if merge_cluster_id is not None:
            cluster_id = merge_cluster_id
            conn.execute(
                "UPDATE clusters SET size = size + 1 WHERE cluster_id = ?", [cluster_id]
            )
            new_cluster = False
        else:
            next_label = conn.execute(
                "SELECT COALESCE(max(label), -1) + 1 FROM clusters WHERE run_id = ?", [run_id]
            ).fetchone()[0]
            cluster_id = f"{run_id}:{next_label}"
            conn.execute(
                "INSERT INTO clusters VALUES (?, ?, ?, ?)",
                [cluster_id, run_id, next_label, 1],
            )
            new_cluster = True

        raw = record.raw
        conn.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                record_id,
                position,
                cluster_id,
                raw.source,
                raw.title,
                raw.brand,
                raw.category,
                raw.price,
                raw.model_dump_json(),
            ],
        )

        now = datetime.now(UTC)
        review_rows = [
            ReviewQueueRow(
                review_id=uuid.uuid4().hex,
                run_id=run_id,
                left_record_id=target.record_id,
                right_record_id=record_id,
                probability=target.probability,
                band="review",
                left_cluster_id=target.cluster_id,
                right_cluster_id=cluster_id,
                created_at=now,
                is_match=None,
                reviewer_id=None,
                decided_at=None,
            )
            for target in review_targets
        ]
        if review_rows:
            conn.executemany(
                "INSERT INTO review_queue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        r.review_id,
                        r.run_id,
                        r.left_record_id,
                        r.right_record_id,
                        r.probability,
                        r.band,
                        r.left_cluster_id,
                        r.right_cluster_id,
                        r.created_at,
                        r.is_match,
                        r.reviewer_id,
                        r.decided_at,
                    )
                    for r in review_rows
                ],
            )

        conn.execute(
            "UPDATE runs SET "
            "n_records = n_records + 1, "
            "n_clusters = n_clusters + ?, "
            "n_auto_merge = n_auto_merge + ?, "
            "n_review = n_review + ?, "
            "n_auto_reject = n_auto_reject + ? "
            "WHERE run_id = ?",
            [
                1 if new_cluster else 0,
                n_auto_merge_candidates,
                n_review_candidates,
                n_auto_reject_candidates,
                run_id,
            ],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return LookupWriteResult(
        record=RecordRow(
            run_id=run_id,
            record_id=record_id,
            position=position,
            cluster_id=cluster_id,
            source=raw.source,
            title=raw.title,
            brand=raw.brand,
            category=raw.category,
            price=raw.price,
            raw_json=raw.model_dump_json(),
        ),
        review_rows=review_rows,
    )
