"""Tests for `service/lookup.py`'s `decide()`: pure decision logic, no store,
no scorer training -- a stub scorer returns fixed probabilities so behavior
is checked against known scores, not real model quality (that end-to-end
path is `tests/test_service_batch.py`'s job for batch, and the `TestClient`
route tests for lookup).
"""

import numpy as np

from dedup.model.threshold import CostModel
from dedup.normalize import normalize
from dedup.schema import Record
from dedup.service.lookup import CandidateRecord, decide


def rec(record_id: str, title: str):
    return normalize(Record(record_id=record_id, source="synthetic", title=title))


class _StubScorer:
    """Returns fixed probabilities in candidate order -- `decide()` is what's
    under test here, not real model scoring."""

    def __init__(self, probabilities):
        self._probabilities = probabilities

    def probabilities(self, records, pair_keys):
        assert len(records) == len(self._probabilities) + 1
        return np.array(self._probabilities, dtype=np.float64)


COST = CostModel()  # default 20:2:1 -> p_hi=0.95, p_lo=0.5, C_fm=20


def candidates(*specs: tuple[str, str, int]) -> list[CandidateRecord]:
    """`specs`: `(record_id, cluster_id, cluster_size)` triples.
    `cluster_size` is that cluster's *total* membership, which may exceed
    how many of its members appear in `specs` -- exactly the case a merge
    member blocking never surfaced as a candidate needs to be tested."""
    return [
        CandidateRecord(record_id=rid, cluster_id=cid, record=rec(rid, "candidate"), cluster_size=size)
        for rid, cid, size in specs
    ]


def test_no_candidates_is_a_new_singleton_with_no_reviews():
    decision = decide(rec("new", "x"), [], scorer=_StubScorer([]), cost=COST)
    assert decision.merge_cluster_id is None
    assert decision.review_targets == []
    assert (
        decision.n_auto_merge_candidates,
        decision.n_review_candidates,
        decision.n_auto_reject_candidates,
    ) == (0, 0, 0)


def test_a_confident_match_against_a_singleton_cluster_merges():
    decision = decide(
        rec("new", "x"), candidates(("c1", "run:0", 1)), scorer=_StubScorer([0.99]), cost=COST
    )
    assert decision.merge_cluster_id == "run:0"
    assert decision.review_targets == []
    assert decision.n_auto_merge_candidates == 1


def test_an_ambiguous_match_queues_review_and_stays_a_singleton():
    decision = decide(
        rec("new", "x"), candidates(("c1", "run:0", 1)), scorer=_StubScorer([0.7]), cost=COST
    )
    assert decision.merge_cluster_id is None
    assert len(decision.review_targets) == 1
    assert decision.review_targets[0].cluster_id == "run:0"
    assert decision.review_targets[0].record_id == "c1"
    assert decision.n_review_candidates == 1


def test_a_clear_non_match_is_a_new_singleton_with_no_reviews():
    decision = decide(
        rec("new", "x"), candidates(("c1", "run:0", 1)), scorer=_StubScorer([0.1]), cost=COST
    )
    assert decision.merge_cluster_id is None
    assert decision.review_targets == []
    assert decision.n_auto_reject_candidates == 1


def test_a_strong_edge_against_one_member_does_not_merge_a_whole_weakly_matched_cluster():
    """The chaining bug an `er-invariants` audit caught: an earlier version
    merged into `run:0` here because its best candidate (c1, p=0.99) alone
    cleared p_hi -- fusing the new record to c2 (p=0.2) too, on no evidence
    about c2 at all. `cluster/base.merge_credit`'s real objective prices the
    whole cluster: credit(0.99) + credit(0.2) = 20.8 + 4.4 = 25.2, against
    C_fm * cluster_size = 20 * 2 = 40 -- not enough to merge. The strong edge
    still surfaces as a review row instead of being silently dropped."""
    decision = decide(
        rec("new", "x"),
        candidates(("c1", "run:0", 2), ("c2", "run:0", 2)),
        scorer=_StubScorer([0.99, 0.2]),
        cost=COST,
    )
    assert decision.merge_cluster_id is None
    assert len(decision.review_targets) == 1
    assert decision.review_targets[0].cluster_id == "run:0"
    assert decision.review_targets[0].record_id == "c1"  # the best-scoring member represents it
    assert decision.review_targets[0].probability == 0.99


def test_every_member_matching_strongly_does_merge_the_cluster():
    """The other side of the same rule: when the aggregate credit across
    every candidate found for a cluster does clear C_fm * cluster_size, the
    merge goes through -- this is not a blanket "never merge a multi-member
    cluster" change, just no longer decided by one edge alone."""
    decision = decide(
        rec("new", "x"),
        candidates(("c1", "run:0", 2), ("c2", "run:0", 2)),
        scorer=_StubScorer([0.99, 0.97]),
        cost=COST,
    )
    assert decision.merge_cluster_id == "run:0"
    assert decision.review_targets == []


def test_an_unscored_cluster_member_earns_no_credit_and_can_block_a_merge():
    """`cluster/base.py`'s "a pair with no edge earns no credit" rule: a
    3-member cluster where blocking surfaced only one candidate (p=0.99)
    prices the merge against the cluster's *total* size, not just the one
    candidate found -- credit(0.99) = 20.8 against C_fm * 3 = 60, so the
    two unscored members' absence of evidence blocks the merge."""
    decision = decide(
        rec("new", "x"), candidates(("c1", "run:0", 3)), scorer=_StubScorer([0.99]), cost=COST
    )
    assert decision.merge_cluster_id is None
    assert len(decision.review_targets) == 1
    assert decision.review_targets[0].cluster_id == "run:0"


def test_scoring_above_p_hi_against_two_different_singleton_clusters_merges_exactly_one():
    # The multi-candidate-different-clusters case: both clear p_hi against a
    # singleton cluster, so one merge happens (the higher aggregate credit)
    # and the other is not silently dropped -- it is queued for review,
    # exactly as cluster/base.py's leave-apart-pricing invariant ("every
    # pair a partition leaves apart falls back to its band") would price it.
    decision = decide(
        rec("new", "x"),
        candidates(("c1", "run:0", 1), ("c2", "run:1", 1)),
        scorer=_StubScorer([0.99, 0.97]),
        cost=COST,
    )
    assert decision.merge_cluster_id == "run:0"
    assert len(decision.review_targets) == 1
    assert decision.review_targets[0].cluster_id == "run:1"
    assert decision.review_targets[0].record_id == "c2"
    assert decision.n_auto_merge_candidates == 2


def test_ties_break_on_the_smaller_cluster_id():
    # Matches agglomerative.py's own tie convention.
    decision = decide(
        rec("new", "x"),
        candidates(("c1", "run:5", 1), ("c2", "run:2", 1)),
        scorer=_StubScorer([0.99, 0.99]),
        cost=COST,
    )
    assert decision.merge_cluster_id == "run:2"
    assert decision.review_targets[0].cluster_id == "run:5"


def test_a_mix_of_merge_and_review_targets_across_three_singleton_clusters():
    decision = decide(
        rec("new", "x"),
        candidates(("c1", "run:0", 1), ("c2", "run:1", 1), ("c3", "run:2", 1)),
        scorer=_StubScorer([0.99, 0.7, 0.1]),
        cost=COST,
    )
    assert decision.merge_cluster_id == "run:0"
    assert {t.cluster_id for t in decision.review_targets} == {"run:1"}
    assert decision.n_auto_reject_candidates == 1
