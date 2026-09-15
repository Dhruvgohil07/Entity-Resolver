"""The FastAPI app: read/review endpoints over what `service/batch.py` already wrote.

Batch execution stays CLI-only in v1, deliberately. Blocking -> features ->
model -> cluster can take minutes at scale; running it synchronously inside a
request handler blocks a worker for the duration with no way to report
progress or a partial failure, and background-job infrastructure (a queue, a
worker process, a job-status table) is out of scope here. Every route below
only ever reads what `batch.py` wrote, or records a review decision.

`create_app(db_path=...)` connects once at startup (the lifespan handler) and
closes once at shutdown -- one connection per process, not one per request,
since DuckDB's file-based mode expects a single read-write connection. Tests
pass `db_path` explicitly; `uvicorn dedup.service.app:app` reads it from
`$DEDUP_DB_PATH`, resolved lazily inside the lifespan handler so importing
this module never requires the variable to be set.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query, Request

from dedup.blocking.ann import AnnIndex
from dedup.blocking.standard_index import InvertedIndex
from dedup.model.train import PairScorer
from dedup.normalize import normalize
from dedup.schema import Record
from dedup.service import lookup as lookup_logic
from dedup.service import review, schemas, store

DB_PATH_ENV = "DEDUP_DB_PATH"


def _record_out(row: store.RecordRow) -> schemas.RecordOut:
    return schemas.RecordOut(
        record_id=row.record_id,
        cluster_id=row.cluster_id,
        source=row.source,
        title=row.title,
        brand=row.brand,
        category=row.category,
        price=row.price,
    )


def _run_summary(row: store.RunRow) -> schemas.RunSummary:
    return schemas.RunSummary(
        run_id=row.run_id,
        dataset=row.dataset,
        created_at=row.created_at,
        n_records=row.n_records,
        n_clusters=row.n_clusters,
        n_auto_merge=row.n_auto_merge,
        n_review=row.n_review,
        n_auto_reject=row.n_auto_reject,
    )


def _run_detail(row: store.RunRow) -> schemas.RunDetail:
    return schemas.RunDetail(
        **_run_summary(row).model_dump(),
        scorer_root=row.scorer_root,
        scorer_sha256=row.scorer_sha256,
        clusterer=row.clusterer,
        cost_false_merge=row.cost.false_merge,
        cost_false_split=row.cost.false_split,
        cost_review=row.cost.review,
        n_candidates=row.n_candidates,
    )


def _review_item(
    conn: duckdb.DuckDBPyConnection, run_id: str, row: store.ReviewQueueRow
) -> schemas.ReviewQueueItem:
    left = store.get_record(conn, run_id, row.left_record_id)
    right = store.get_record(conn, run_id, row.right_record_id)
    assert left is not None and right is not None  # a queued pair always names real records
    return schemas.ReviewQueueItem(
        review_id=row.review_id,
        left_record=_record_out(left),
        right_record=_record_out(right),
        probability=row.probability,
        band=row.band,
        created_at=row.created_at,
        is_match=row.is_match,
        reviewer_id=row.reviewer_id,
        decided_at=row.decided_at,
    )


def get_conn(request: Request) -> duckdb.DuckDBPyConnection:
    """`request.app.state.conn` -- correctly scoped per app instance without
    needing to close over anything from `create_app`, which matters here:
    with `from __future__ import annotations` in effect, FastAPI resolves a
    route's `Annotated[..., Depends(...)]` hints against the function's
    *module* globals, not an enclosing closure. A `get_conn` (or a `ConnDep`
    alias pointing at one) defined *inside* `create_app` is invisible to that
    resolution and silently falls back to treating `conn` as a query
    parameter -- caught by every route in this file returning 422 until this
    moved out to module scope."""
    return request.app.state.conn


ConnDep = Annotated[duckdb.DuckDBPyConnection, Depends(get_conn)]


def _lock_for(app: FastAPI, run_id: str) -> threading.Lock:
    """One `threading.Lock` per `run_id`, created on first use.

    Every route in this file is `def`, not `async def`, so FastAPI/Starlette
    runs them in a thread-pool -- concurrent requests genuinely execute in
    parallel threads today, not sequentially. A single DuckDB connection and,
    worse, the in-memory mutating `AnnIndex`/`InvertedIndex` objects the
    lookup route caches below are both unsafe under that: two threads
    calling `index.add()`/`.search()` on the same FAISS `IndexHNSWFlat`
    concurrently is a real hazard, since `search()` actively traverses a
    graph another thread's `add()` may be rewriting. Held for the duration
    of any route touching the connection or a mutable index -- both the
    lookup route and, in fairness, the existing review-decision route --
    throughput-limiting by design, the same "out of scope on purpose"
    posture the module docstring already takes for batch execution.

    Guarded by one global lock over the dict itself: two threads racing to
    create the *first* lock for the same unseen `run_id` is the one unsafe
    window a per-run_id lock cannot protect itself.
    """
    with app.state.locks_guard:
        if run_id not in app.state.locks:
            app.state.locks[run_id] = threading.Lock()
        return app.state.locks[run_id]


def create_app(db_path: Path | str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = Path(db_path) if db_path is not None else Path(os.environ[DB_PATH_ENV])
        app.state.conn = store.connect(resolved)
        app.state.locks = {}
        app.state.locks_guard = threading.Lock()
        # Loaded ann/inverted indexes and scorers are cached per run_id --
        # otherwise every lookup would reread the whole persisted FAISS graph
        # and re-unpickle a booster from disk. Safe to mutate in place only
        # because every access is made under _lock_for(app, run_id).
        app.state.ann_cache = {}
        app.state.inverted_cache = {}
        app.state.scorer_cache = {}
        try:
            yield
        finally:
            app.state.conn.close()

    app = FastAPI(title="dedup service", lifespan=lifespan)

    def page(items: Sequence, total: int, limit: int, offset: int) -> schemas.Page:
        return schemas.Page(items=list(items), total=total, limit=limit, offset=offset)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runs", response_model=schemas.Page[schemas.RunSummary])
    def list_runs(
        conn: ConnDep,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        rows, total = store.list_runs(conn, limit=limit, offset=offset)
        return page([_run_summary(r) for r in rows], total, limit, offset)

    @app.get("/runs/{run_id}", response_model=schemas.RunDetail)
    def get_run(run_id: str, conn: ConnDep):
        row = store.get_run(conn, run_id)
        if row is None:
            raise HTTPException(404, f"no run {run_id!r}")
        return _run_detail(row)

    @app.get("/runs/{run_id}/records", response_model=schemas.Page[schemas.RecordOut])
    def list_records(
        run_id: str,
        conn: ConnDep,
        cluster_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        rows, total = store.list_records(
            conn, run_id, cluster_id=cluster_id, limit=limit, offset=offset
        )
        return page([_record_out(r) for r in rows], total, limit, offset)

    @app.get("/runs/{run_id}/records/{record_id}", response_model=schemas.RecordOut)
    def get_record(
        run_id: str, record_id: str, conn: ConnDep
    ):
        row = store.get_record(conn, run_id, record_id)
        if row is None:
            raise HTTPException(404, f"no record {record_id!r} in run {run_id!r}")
        return _record_out(row)

    @app.get("/runs/{run_id}/clusters", response_model=schemas.Page[schemas.ClusterSummary])
    def list_clusters(
        run_id: str,
        conn: ConnDep,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        rows, total = store.list_clusters(conn, run_id, limit=limit, offset=offset)
        items = [schemas.ClusterSummary(cluster_id=r.cluster_id, size=r.size) for r in rows]
        return page(items, total, limit, offset)

    @app.get("/runs/{run_id}/clusters/{cluster_id}", response_model=schemas.ClusterDetail)
    def get_cluster(
        run_id: str, cluster_id: str, conn: ConnDep
    ):
        row = store.get_cluster(conn, run_id, cluster_id)
        if row is None:
            raise HTTPException(404, f"no cluster {cluster_id!r} in run {run_id!r}")
        members, _ = store.list_records(
            conn, run_id, cluster_id=cluster_id, limit=max(row.size, 1)
        )
        return schemas.ClusterDetail(
            cluster_id=row.cluster_id, size=row.size, members=[_record_out(m) for m in members]
        )

    @app.get(
        "/runs/{run_id}/review-queue", response_model=schemas.Page[schemas.ReviewQueueItem]
    )
    def list_review_queue(
        run_id: str,
        conn: ConnDep,
        decided: bool = False,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        rows, total = store.list_review_queue(
            conn, run_id, decided=decided, limit=limit, offset=offset
        )
        return page([_review_item(conn, run_id, r) for r in rows], total, limit, offset)

    @app.get(
        "/runs/{run_id}/review-queue/{review_id}", response_model=schemas.ReviewQueueItem
    )
    def get_review_item(
        run_id: str, review_id: str, conn: ConnDep
    ):
        row = store.get_review_item(conn, run_id, review_id)
        if row is None:
            raise HTTPException(404, f"no review item {review_id!r} in run {run_id!r}")
        return _review_item(conn, run_id, row)

    @app.post(
        "/runs/{run_id}/review-queue/{review_id}/decision",
        response_model=schemas.ReviewQueueItem,
    )
    def decide(
        run_id: str,
        review_id: str,
        body: schemas.ReviewDecisionRequest,
        request: Request,
        conn: ConnDep,
    ):
        with _lock_for(request.app, run_id):
            try:
                decided = review.record_decision(
                    conn,
                    run_id,
                    review_id,
                    is_match=body.is_match,
                    reviewer_id=body.reviewer_id,
                )
            except LookupError as exc:
                raise HTTPException(404, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return _review_item(conn, run_id, decided)

    @app.post("/runs/{run_id}/lookup", response_model=schemas.LookupResult)
    def lookup(
        run_id: str,
        body: schemas.LookupRequest,
        request: Request,
        conn: ConnDep,
    ):
        run = store.get_run(conn, run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id!r}")

        with _lock_for(request.app, run_id):
            indexes_row = store.get_run_indexes(conn, run_id)
            if indexes_row is None:
                raise HTTPException(
                    409,
                    f"run {run_id!r} has no index built yet -- run "
                    f"`python -m dedup.service.build_index --db ... --run-id {run_id} "
                    f"--index-root ...` first",
                )

            record_id = body.record_id or uuid.uuid4().hex
            if store.get_record(conn, run_id, record_id) is not None:
                raise HTTPException(
                    409, f"record {record_id!r} already exists in run {run_id!r}"
                )

            new_record = normalize(
                Record(
                    record_id=record_id,
                    source=body.source,
                    title=body.title,
                    description=body.description,
                    brand=body.brand,
                    category=body.category,
                    price=body.price,
                )
            )

            cache = request.app.state
            if run_id not in cache.ann_cache:
                cache.ann_cache[run_id] = AnnIndex.load(Path(indexes_row.ann_root))
                cache.inverted_cache[run_id] = InvertedIndex.load(
                    Path(indexes_row.standard_root)
                )
            if run_id not in cache.scorer_cache:
                cache.scorer_cache[run_id] = PairScorer.load(Path(run.scorer_root))
            ann_index: AnnIndex = cache.ann_cache[run_id]
            inverted_index: InvertedIndex = cache.inverted_cache[run_id]
            scorer: PairScorer = cache.scorer_cache[run_id]

            candidate_ids = {cid for cid, _ in ann_index.query_one(new_record)}
            candidate_ids |= inverted_index.query_one(new_record)
            candidate_rows = store.get_records_by_ids(conn, run_id, sorted(candidate_ids))
            # lookup.decide() prices a merge against each candidate cluster's
            # *total* size (cluster/base.merge_credit's objective), not just
            # how many of its members blocking surfaced here.
            cluster_sizes = {
                c.cluster_id: c.size
                for c in store.get_clusters_by_ids(
                    conn, run_id, sorted({row.cluster_id for row in candidate_rows})
                )
            }
            candidates = [
                lookup_logic.CandidateRecord(
                    record_id=row.record_id,
                    cluster_id=row.cluster_id,
                    record=normalize(Record.model_validate_json(row.raw_json)),
                    cluster_size=cluster_sizes[row.cluster_id],
                )
                for row in candidate_rows
            ]

            decision = lookup_logic.decide(new_record, candidates, scorer=scorer, cost=run.cost)

            review_targets = [
                store.ReviewTarget(
                    cluster_id=target.cluster_id,
                    record_id=target.record_id,
                    probability=target.probability,
                )
                for target in decision.review_targets
            ]
            written = store.apply_lookup(
                conn,
                run_id,
                record=new_record,
                record_id=record_id,
                merge_cluster_id=decision.merge_cluster_id,
                review_targets=review_targets,
                n_auto_merge_candidates=decision.n_auto_merge_candidates,
                n_review_candidates=decision.n_review_candidates,
                n_auto_reject_candidates=decision.n_auto_reject_candidates,
            )

            # Always re-save, in-memory and on disk, with the new record
            # added -- a crash between requests must not silently reopen a
            # recall gap on restart. faiss.write_index re-serializes the
            # whole graph on every call, so lookup latency grows with
            # catalog size; a real scaling limit this pass doesn't solve.
            ann_index.add(new_record, record_id)
            ann_index.save(Path(indexes_row.ann_root))
            inverted_index.add(new_record, record_id)
            inverted_index.save(Path(indexes_row.standard_root))

            cluster_row = store.get_cluster(conn, run_id, written.record.cluster_id)
            assert cluster_row is not None  # apply_lookup just wrote or grew it
            if decision.merge_cluster_id is not None:
                decision_label = "auto_merge"
            elif review_targets:
                decision_label = "review"
            else:
                decision_label = "new_cluster"

            outcome = schemas.LookupResult(
                decision=decision_label,
                record=_record_out(written.record),
                cluster=schemas.ClusterSummary(
                    cluster_id=cluster_row.cluster_id, size=cluster_row.size
                ),
                secondary_reviews=[
                    _review_item(conn, run_id, r) for r in written.review_rows
                ],
            )
        return outcome

    return app


app = create_app()
