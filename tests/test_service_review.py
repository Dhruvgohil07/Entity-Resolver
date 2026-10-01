"""Tests for the review-decision business logic, against a bare connection --
no TestClient, no app.py.
"""

import numpy as np
import pytest

from dedup.blocking.pairs import pack
from dedup.model.threshold import BandAssignment, CostModel
from dedup.normalize import normalize
from dedup.schema import Record
from dedup.service import review, store


def rec(record_id: str, title: str):
    return normalize(Record(record_id=record_id, source="synthetic", title=title))


@pytest.fixture
def run(tmp_path):
    conn = store.connect(tmp_path / "x.duckdb")
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
    run_id = store.write_run(
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
    review_id = store.list_review_queue(conn, run_id)[0][0].review_id
    return conn, run_id, review_id


def test_a_decision_is_recorded(run):
    conn, run_id, review_id = run
    decided = review.record_decision(conn, run_id, review_id, is_match=True, reviewer_id="alice")
    assert decided.is_match is True
    assert decided.reviewer_id == "alice"
    assert decided.decided_at is not None

    # And it is the row the store now holds, not just the return value.
    stored = store.get_review_item(conn, run_id, review_id)
    assert stored.is_match is True


def test_an_unknown_review_id_is_refused(run):
    conn, run_id, _ = run
    with pytest.raises(LookupError, match="no review item"):
        review.record_decision(conn, run_id, "not-a-real-id", is_match=True, reviewer_id=None)


def test_a_second_decision_on_the_same_item_is_refused(run):
    # A decided row may already have been read as a training label -- silently
    # overwriting it would mutate that label out from under whatever read it.
    conn, run_id, review_id = run
    review.record_decision(conn, run_id, review_id, is_match=True, reviewer_id="alice")
    with pytest.raises(ValueError, match="already decided"):
        review.record_decision(conn, run_id, review_id, is_match=False, reviewer_id="bob")

    # The first decision must survive the refused second attempt untouched.
    stored = store.get_review_item(conn, run_id, review_id)
    assert stored.is_match is True
    assert stored.reviewer_id == "alice"
