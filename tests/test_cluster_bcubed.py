"""Tests for B-cubed and the pair counts a partition implies.

The hand-worked cases are small enough to check on paper, which is the point: a
metric is only trustworthy against numbers someone derived without it.
"""

import numpy as np
import pytest

from dedup.cluster.bcubed import bcubed, closure_pair_counts, cluster_errors
from dedup.model.threshold import CostModel

TRUTH = ["a", "a", "b", "c", "c", "c"]


def test_a_perfect_partition_scores_one():
    score = bcubed([5, 5, 1, 2, 2, 2], TRUTH)
    assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)
    assert score.n_records == 6


def test_all_singletons_have_precision_one_and_recall_one_over_entity_size():
    score = bcubed(np.arange(6), TRUTH)
    assert score.precision == 1.0
    assert score.recall == pytest.approx((2 * (1 / 2) + 1 * 1 + 3 * (1 / 3)) / 6)


def test_one_cluster_has_recall_one():
    score = bcubed([0] * 6, TRUTH)
    assert score.recall == 1.0
    # a-records score 2/6 each, the b-record 1/6, c-records 3/6 each.
    assert score.precision == pytest.approx((2 * (2 / 6) + 1 / 6 + 3 * (3 / 6)) / 6)


def test_a_chained_fusion_worked_by_hand():
    """Records 0, 1 of entity a fused with record 2 of entity b; record 3 of b alone."""
    score = bcubed([0, 0, 0, 1], ["a", "a", "b", "b"])
    assert score.precision == pytest.approx((2 / 3 + 2 / 3 + 1 / 3 + 1) / 4)
    assert score.recall == pytest.approx((1 + 1 + 1 / 2 + 1 / 2) / 4)


def test_relabelling_clusters_or_entities_changes_nothing():
    predicted = [0, 0, 0, 1, 2, 2]
    renamed = [9, 9, 9, 4, 7, 7]
    other_truth = ["x", "x", "y", "z", "z", "z"]
    assert bcubed(predicted, TRUTH) == bcubed(renamed, other_truth)


def test_every_record_is_scored_including_isolated_ones():
    """Dropping the records no edge touched would move the number, so lengths must match."""
    truth = ["a", "a", "b", "b"]
    with pytest.raises(ValueError, match="one predicted label per record"):
        bcubed([0, 0, 1], truth)
    # And refusing is not pedantry: leaving out record 3, a singleton, changes recall.
    assert bcubed([0, 0, 1, 2], truth).recall != bcubed([0, 0, 1], truth[:3]).recall


def test_an_unlabelled_record_is_refused():
    with pytest.raises(ValueError, match="entity_id"):
        bcubed([0, 1], np.array(["a", None], dtype=object))


def test_zero_records_is_refused():
    with pytest.raises(ValueError, match="zero records"):
        bcubed([], [])


def test_closure_pairs_count_every_merged_pair_not_only_candidates():
    counts = closure_pair_counts([0, 0, 0, 1], ["a", "a", "b", "b"])
    # Cluster {0, 1, 2} merges three pairs, one of them true; the split holds two true pairs.
    assert (counts.n_merged_pairs, counts.n_true_merged_pairs, counts.n_true_pairs) == (3, 1, 2)
    assert (counts.n_false_merges, counts.n_false_splits) == (2, 1)
    assert counts.precision == pytest.approx(1 / 3)
    assert counts.recall == pytest.approx(1 / 2)


def test_realized_cost_bills_the_bands_three_outcomes():
    """Two false merges and one true pair left apart, billed with and without a review."""
    counts = closure_pair_counts([0, 0, 0, 1], ["a", "a", "b", "b"])
    cost = CostModel()
    assert counts.realized_cost(cost, n_review=0, n_review_true=0) == pytest.approx(2 * 20 + 1 * 2)
    # The true pair queued for a (correct) reviewer costs a review instead of a miss;
    # a queued false pair costs a review on top.
    assert counts.realized_cost(cost, n_review=1, n_review_true=1) == pytest.approx(2 * 20 + 1)
    assert counts.realized_cost(cost, n_review=2, n_review_true=1) == pytest.approx(2 * 20 + 2)


def test_realized_cost_refuses_more_queued_duplicates_than_exist():
    counts = closure_pair_counts([0, 0, 0, 1], ["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="n_review_true"):
        counts.realized_cost(CostModel(), n_review=3, n_review_true=2)


def test_closure_pairs_of_singletons_have_no_merges():
    counts = closure_pair_counts(np.arange(6), TRUTH)
    assert counts.n_merged_pairs == 0
    assert counts.precision == 0.0
    assert counts.recall == 0.0
    assert counts.n_true_pairs == 1 + 3


def test_cluster_errors_count_fusions_and_splits():
    errors = cluster_errors([0, 0, 0, 1], ["a", "a", "b", "b"])
    assert (errors.n_clusters, errors.n_fused_clusters, errors.n_split_entities) == (2, 1, 1)
