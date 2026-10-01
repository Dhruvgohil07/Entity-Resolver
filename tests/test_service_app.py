"""Tests for the FastAPI app: `TestClient` against a `store.py`-populated DB.

No real batch run and no scorer here -- these exercise routing, serialization,
pagination and `review.py`'s error mapping, independent of model fixtures
(that end-to-end path is `tests/test_service_batch.py`'s job).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dedup.blocking.pairs import pack
from dedup.model.threshold import BandAssignment, CostModel
from dedup.normalize import normalize
from dedup.schema import Record
from dedup.service import store
from dedup.service.app import create_app, get_conn


def rec(record_id: str, title: str):
    return normalize(Record(record_id=record_id, source="synthetic", title=title))


def seed(conn) -> str:
    """r1/r2 merge into one cluster, r3 stays its own; (r1, r3) is the one
    pair left apart above p_lo -- the review queue's one item."""
    records = [rec("r1", "widget deluxe xy123z"), rec("r2", "widget deluxe xy123z v2"), rec("r3", "unrelated gadget")]
    n = len(records)
    pair_keys = pack(np.array([0, 0, 1]), np.array([1, 2, 2]), n)
    probabilities = np.array([0.99, 0.6, 0.01])
    cost = CostModel()
    bands = BandAssignment(
        auto_merge=np.array([True, False, False]),
        review=np.array([False, True, False]),
        auto_reject=np.array([False, False, True]),
        cost=cost,
    )
    return store.write_run(
        conn,
        dataset="synthetic",
        scorer_root="artifacts/scorer",
        scorer_sha256="deadbeef",
        cost=cost,
        clusterer="average_linkage",
        records=records,
        pair_keys=pair_keys,
        probabilities=probabilities,
        cluster_labels=np.array([0, 0, 1]),
        review_pairs=np.array([False, True, False]),
        bands=bands,
    )


@pytest.fixture
def seeded(tmp_path):
    db = tmp_path / "x.duckdb"
    conn = store.connect(db)
    run_id = seed(conn)
    conn.close()
    app = create_app(db_path=db)
    with TestClient(app) as client:
        yield client, run_id


def test_health(seeded):
    client, _ = seeded
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_runs(seeded):
    client, run_id = seeded
    body = client.get("/runs").json()
    assert body["total"] == 1
    assert body["items"][0]["run_id"] == run_id


def test_get_run_detail(seeded):
    client, run_id = seeded
    body = client.get(f"/runs/{run_id}").json()
    assert body["clusterer"] == "average_linkage"
    assert (body["n_auto_merge"], body["n_review"], body["n_auto_reject"]) == (1, 1, 1)


def test_unknown_run_is_404(seeded):
    client, _ = seeded
    assert client.get("/runs/not-a-real-run").status_code == 404


def test_list_records_and_filter_by_cluster(seeded):
    client, run_id = seeded
    body = client.get(f"/runs/{run_id}/records").json()
    assert body["total"] == 3

    r1 = next(r for r in body["items"] if r["record_id"] == "r1")
    filtered = client.get(f"/runs/{run_id}/records", params={"cluster_id": r1["cluster_id"]}).json()
    assert filtered["total"] == 2
    assert {r["record_id"] for r in filtered["items"]} == {"r1", "r2"}


def test_get_cluster_detail(seeded):
    client, run_id = seeded
    clusters = client.get(f"/runs/{run_id}/clusters").json()["items"]
    two_member = next(c for c in clusters if c["size"] == 2)
    body = client.get(f"/runs/{run_id}/clusters/{two_member['cluster_id']}").json()
    assert body["size"] == 2
    assert {m["record_id"] for m in body["members"]} == {"r1", "r2"}


def test_review_queue_defaults_to_pending(seeded):
    client, run_id = seeded
    body = client.get(f"/runs/{run_id}/review-queue").json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["is_match"] is None
    assert {item["left_record"]["record_id"], item["right_record"]["record_id"]} == {"r1", "r3"}


def test_a_decision_moves_the_item_out_of_the_pending_queue(seeded):
    client, run_id = seeded
    review_id = client.get(f"/runs/{run_id}/review-queue").json()["items"][0]["review_id"]

    resp = client.post(
        f"/runs/{run_id}/review-queue/{review_id}/decision",
        json={"is_match": True, "reviewer_id": "alice"},
    )
    assert resp.status_code == 200
    assert resp.json()["is_match"] is True
    assert resp.json()["reviewer_id"] == "alice"

    assert client.get(f"/runs/{run_id}/review-queue").json()["total"] == 0
    decided = client.get(f"/runs/{run_id}/review-queue", params={"decided": True}).json()
    assert decided["total"] == 1


def test_a_second_decision_is_refused_with_409(seeded):
    client, run_id = seeded
    review_id = client.get(f"/runs/{run_id}/review-queue").json()["items"][0]["review_id"]
    client.post(f"/runs/{run_id}/review-queue/{review_id}/decision", json={"is_match": True})

    resp = client.post(
        f"/runs/{run_id}/review-queue/{review_id}/decision", json={"is_match": False}
    )
    assert resp.status_code == 409


def test_deciding_an_unknown_item_is_404(seeded):
    client, run_id = seeded
    resp = client.post(
        f"/runs/{run_id}/review-queue/not-a-real-id/decision", json={"is_match": True}
    )
    assert resp.status_code == 404


def test_pagination_round_trip(seeded):
    client, run_id = seeded
    body = client.get(f"/runs/{run_id}/records", params={"limit": 1, "offset": 1}).json()
    assert (body["limit"], body["offset"], body["total"]) == (1, 1, 3)
    assert len(body["items"]) == 1


# ---------------------------------------------------------------------------
# Concurrency: what each request may and may not share


def test_a_request_gets_a_private_cursor_not_the_apps_connection(seeded):
    """Every route here is `def`, so Starlette runs them in a thread-pool and
    two requests really do overlap. Handing them one connection object is not
    a throughput question but a correctness one: a DuckDB transaction belongs
    to the connection, so one request's BEGIN encloses another's reads, and a
    second concurrent writer raises "cannot start a transaction within a
    transaction" and leaves the connection wedged in "Current transaction is
    aborted" -- where every later query on it, from any route, fails until
    the process restarts.

    Both halves are asserted, because either alone can pass while the bug is
    live: distinct objects, *and* genuinely independent transaction contexts.
    """
    client, _ = seeded
    app = client.app
    request = SimpleNamespace(app=app)

    first, second = get_conn(request), get_conn(request)
    conn_a, conn_b = next(first), next(second)
    try:
        assert conn_a is not app.state.conn
        assert conn_b is not app.state.conn
        assert conn_a is not conn_b

        # The wedging scenario, in miniature: on one shared connection the
        # second BEGIN raises. On two cursors both stand.
        conn_a.execute("BEGIN TRANSACTION")
        conn_b.execute("BEGIN TRANSACTION")
        conn_a.execute("ROLLBACK")
        conn_b.execute("ROLLBACK")
    finally:
        for generator in (first, second):
            with pytest.raises(StopIteration):
                next(generator)


def test_an_uncommitted_write_is_invisible_to_another_request(seeded):
    """The dirty-read half of the same bug. A GET arriving mid-`apply_lookup`
    must see the store as it was, not a half-written run -- which is what
    sharing the connection would give it, since the two requests would be
    inside one transaction."""
    client, run_id = seeded
    app = client.app
    # Held, not inlined: the dependency is a generator, and dropping it lets
    # the GC run its `finally` and close the cursor out from under the test.
    dependency = get_conn(SimpleNamespace(app=app))
    writer = next(dependency)

    before = client.get(f"/runs/{run_id}").json()["n_records"]
    writer.execute("BEGIN TRANSACTION")
    writer.execute("UPDATE runs SET n_records = n_records + 99 WHERE run_id = ?", [run_id])
    try:
        assert client.get(f"/runs/{run_id}").json()["n_records"] == before
    finally:
        writer.execute("ROLLBACK")
        dependency.close()
    assert client.get(f"/runs/{run_id}").json()["n_records"] == before
