"""Tests for the DuckDB store: schema, round-trips, and the write-once guard.

Storage mechanics only -- these hand-construct `review_pairs`/`cluster_labels`
rather than running real blocking/clustering, the same way `test_cluster_base.py`
hand-builds a `ScoredGraph` rather than scoring one.
"""

import duckdb
import numpy as np
import pytest

from dedup.blocking.pairs import pack
from dedup.model.threshold import BandAssignment, CostModel
from dedup.normalize import normalize
from dedup.schema import Record
from dedup.service import store


def rec(record_id: str, title: str):
    return normalize(Record(record_id=record_id, source="synthetic", title=title))


def small_run():
    """3 records: r1/r2 merge into one cluster, r3 stays its own; (r1, r3) is
    the one pair still apart and above p_lo -- the review queue's one item."""
    records = [rec("r1", "widget deluxe xy123z"), rec("r2", "widget deluxe xy123z v2"), rec("r3", "unrelated gadget")]
    n = len(records)
    pair_keys = pack(np.array([0, 0, 1]), np.array([1, 2, 2]), n)  # (r1,r2) (r1,r3) (r2,r3)
    probabilities = np.array([0.99, 0.6, 0.01])
    cluster_labels = np.array([0, 0, 1])
    review_pairs = np.array([False, True, False])
    cost = CostModel()
    bands = BandAssignment(
        auto_merge=np.array([True, False, False]),
        review=np.array([False, True, False]),
        auto_reject=np.array([False, False, True]),
        cost=cost,
    )
    return records, pair_keys, probabilities, cluster_labels, review_pairs, bands


def write_small_run(conn):
    records, pair_keys, probabilities, cluster_labels, review_pairs, bands = small_run()
    return store.write_run(
        conn,
        dataset="synthetic",
        scorer_root="artifacts/scorer",
        scorer_sha256="deadbeef",
        cost=bands.cost,
        clusterer="average_linkage",
        records=records,
        pair_keys=pair_keys,
        probabilities=probabilities,
        cluster_labels=cluster_labels,
        review_pairs=review_pairs,
        bands=bands,
    )


def test_connect_creates_every_table(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    tables = {
        row[0]
        for row in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    assert {"runs", "clusters", "records", "review_queue"} <= tables


def test_connect_is_idempotent(tmp_path):
    path = tmp_path / "x.duckdb"
    store.connect(path).close()
    conn = store.connect(path)  # must not raise on an existing schema
    assert store.list_runs(conn) == ([], 0)


def test_a_batch_run_round_trips_every_table(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)

    run = store.get_run(conn, run_id)
    assert run.dataset == "synthetic"
    assert run.n_records == 3
    assert run.n_clusters == 2
    assert (run.n_auto_merge, run.n_review, run.n_auto_reject) == (1, 1, 1)

    clusters, total = store.list_clusters(conn, run_id)
    assert total == 2
    assert {c.size for c in clusters} == {2, 1}

    records, total = store.list_records(conn, run_id)
    assert total == 3
    assert {r.record_id for r in records} == {"r1", "r2", "r3"}
    r1 = store.get_record(conn, run_id, "r1")
    assert r1.title == "widget deluxe xy123z"

    r2 = store.get_record(conn, run_id, "r2")
    assert r1.cluster_id == r2.cluster_id
    r3 = store.get_record(conn, run_id, "r3")
    assert r3.cluster_id != r1.cluster_id

    queue, total = store.list_review_queue(conn, run_id)
    assert total == 1
    item = queue[0]
    assert {item.left_record_id, item.right_record_id} == {"r1", "r3"}
    assert item.is_match is None
    assert item.left_cluster_id != item.right_cluster_id


def test_filtering_records_by_cluster(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)
    r1 = store.get_record(conn, run_id, "r1")

    same_cluster, total = store.list_records(conn, run_id, cluster_id=r1.cluster_id)
    assert total == 2
    assert {r.record_id for r in same_cluster} == {"r1", "r2"}


def test_decided_filter_on_the_review_queue(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)
    item = store.list_review_queue(conn, run_id)[0][0]

    pending, _ = store.list_review_queue(conn, run_id, decided=False)
    assert len(pending) == 1

    store.decide_review(conn, run_id, item.review_id, is_match=True, reviewer_id="alice")

    pending, _ = store.list_review_queue(conn, run_id, decided=False)
    assert pending == []
    decided, _ = store.list_review_queue(conn, run_id, decided=True)
    assert decided[0].is_match is True
    assert decided[0].reviewer_id == "alice"
    assert decided[0].decided_at is not None


def test_the_same_pair_cannot_be_queued_twice_for_one_run(tmp_path):
    # UNIQUE (run_id, left_record_id, right_record_id) is what makes a decided
    # row trustworthy as a training label -- two rows for the same pair would
    # leave it ambiguous which decision is the real one.
    conn = store.connect(tmp_path / "x.duckdb")
    records, pair_keys, probabilities, cluster_labels, review_pairs, bands = small_run()
    run_id = store.write_run(
        conn,
        dataset="synthetic",
        scorer_root="artifacts/scorer",
        scorer_sha256="deadbeef",
        cost=bands.cost,
        clusterer="average_linkage",
        records=records,
        pair_keys=pair_keys,
        probabilities=probabilities,
        cluster_labels=cluster_labels,
        review_pairs=review_pairs,
        bands=bands,
    )
    # Insert a duplicate review row for the same run/pair directly -- store.write_run
    # itself never does this (one run writes each pair once), so the constraint is
    # exercised at the table level, the same layer that would catch a future bug.
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO review_queue VALUES "
            "('dup', ?, 'r1', 'r3', 0.6, 'review', ?, ?, now(), NULL, NULL, NULL)",
            [run_id, f"{run_id}:0", f"{run_id}:1"],
        )


def test_get_records_by_ids_batch_fetches(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)

    rows = store.get_records_by_ids(conn, run_id, ["r1", "r3"])
    assert {r.record_id for r in rows} == {"r1", "r3"}
    assert store.get_records_by_ids(conn, run_id, []) == []


# ---------------------------------------------------------------------------
# Online lookup: run_indexes, apply_lookup
# ---------------------------------------------------------------------------


def test_run_indexes_round_trip(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)

    assert store.get_run_indexes(conn, run_id) is None

    store.set_run_indexes(conn, run_id, ann_root="idx/ann", standard_root="idx/standard")
    row = store.get_run_indexes(conn, run_id)
    assert (row.ann_root, row.standard_root) == ("idx/ann", "idx/standard")


def test_set_run_indexes_is_an_upsert(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)

    store.set_run_indexes(conn, run_id, ann_root="idx/ann", standard_root="idx/standard")
    store.set_run_indexes(conn, run_id, ann_root="idx/ann2", standard_root="idx/standard2")

    row = store.get_run_indexes(conn, run_id)
    assert (row.ann_root, row.standard_root) == ("idx/ann2", "idx/standard2")


def test_apply_lookup_merges_into_an_existing_cluster(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)
    r1 = store.get_record(conn, run_id, "r1")
    before = store.get_run(conn, run_id)

    new_record = rec("r4", "widget deluxe xy123z v3")
    written = store.apply_lookup(
        conn,
        run_id,
        record=new_record,
        record_id="r4",
        merge_cluster_id=r1.cluster_id,
        review_targets=[],
        n_auto_merge_candidates=1,
        n_review_candidates=0,
        n_auto_reject_candidates=0,
    )
    assert written.record.cluster_id == r1.cluster_id
    assert written.review_rows == []

    after = store.get_run(conn, run_id)
    assert after.n_records == before.n_records + 1
    assert after.n_clusters == before.n_clusters  # merged, no new cluster
    assert after.n_auto_merge == before.n_auto_merge + 1

    merged_cluster = store.get_cluster(conn, run_id, r1.cluster_id)
    assert merged_cluster.size == 3  # r1, r2, and now r4

    records, total = store.list_records(conn, run_id, cluster_id=r1.cluster_id)
    assert total == 3
    assert {r.record_id for r in records} == {"r1", "r2", "r4"}


def test_apply_lookup_starts_a_new_singleton_and_queues_reviews(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)
    r1 = store.get_record(conn, run_id, "r1")
    r3 = store.get_record(conn, run_id, "r3")
    before = store.get_run(conn, run_id)

    new_record = rec("r4", "ambiguous item")
    written = store.apply_lookup(
        conn,
        run_id,
        record=new_record,
        record_id="r4",
        merge_cluster_id=None,
        review_targets=[
            store.ReviewTarget(cluster_id=r1.cluster_id, record_id="r1", probability=0.7),
            store.ReviewTarget(cluster_id=r3.cluster_id, record_id="r3", probability=0.6),
        ],
        n_auto_merge_candidates=0,
        n_review_candidates=2,
        n_auto_reject_candidates=0,
    )
    assert written.record.cluster_id not in (r1.cluster_id, r3.cluster_id)
    assert len(written.review_rows) == 2
    assert {r.left_cluster_id for r in written.review_rows} == {r1.cluster_id, r3.cluster_id}

    after = store.get_run(conn, run_id)
    assert after.n_records == before.n_records + 1
    assert after.n_clusters == before.n_clusters + 1
    assert after.n_review == before.n_review + 2

    new_cluster = store.get_cluster(conn, run_id, written.record.cluster_id)
    assert new_cluster.size == 1

    queue, _ = store.list_review_queue(conn, run_id, decided=False)
    against_r4 = [q for q in queue if "r4" in (q.left_record_id, q.right_record_id)]
    assert len(against_r4) == 2
    assert {q.left_cluster_id for q in against_r4} == {r1.cluster_id, r3.cluster_id}


def test_apply_lookup_refuses_a_duplicate_record_id(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
    run_id = write_small_run(conn)

    with pytest.raises(ValueError, match="already exists"):
        store.apply_lookup(
            conn,
            run_id,
            record=rec("r1", "widget deluxe xy123z"),
            record_id="r1",
            merge_cluster_id=None,
            review_targets=[],
            n_auto_merge_candidates=0,
            n_review_candidates=0,
            n_auto_reject_candidates=0,
        )
    # And the refused write left no trace -- still 3 records, run counts unmoved.
    _, total = store.list_records(conn, run_id)
    assert total == 3
