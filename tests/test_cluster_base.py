"""Tests for the scored-graph contract and the partition objective.

Two carry the weight, and both check the objective against a definition written
somewhere else rather than against itself: `test_expected_cost_matches_the_sum_over_every_pair`
spells the cost out pair by pair, and
`test_an_unmerged_pair_falls_back_to_the_cheaper_of_review_and_rejection` reads it
off `model.threshold.expected_costs`, the bands' own definition.
"""

import itertools

import numpy as np
import pytest

from dedup.cluster.base import (
    ScoredGraph,
    canonical_labels,
    check_partition,
    expected_partition_cost,
    merge_credit,
    review_mask,
    unmerged_pair_cost,
)
from dedup.model.threshold import Band, CostModel, expected_costs

COST = CostModel()


def brute_force_cost(labels, n, pairs, cost):
    total = 0.0
    for i, j in itertools.combinations(range(n), 2):
        p = pairs.get((i, j), 0.0)
        if labels[i] == labels[j]:
            total += (1 - p) * cost.false_merge
        else:
            total += min(p * cost.false_split, cost.review)
    return total


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def test_from_pairs_accepts_either_index_order():
    graph = ScoredGraph.from_pairs(3, {(2, 0): 0.5, (0, 1): 0.9})
    left, right, p = graph.edges()
    assert left.tolist() == [0, 0]
    assert right.tolist() == [1, 2]
    assert p.tolist() == [0.9, 0.5]


def test_a_raw_score_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="calibrated probability"):
        ScoredGraph.from_pairs(2, {(0, 1): 1.3})


def test_a_nan_probability_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        ScoredGraph.from_pairs(2, {(0, 1): float("nan")})


def test_unsorted_or_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="sorted and unique"):
        ScoredGraph(3, np.array([2, 1]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="sorted and unique"):
        ScoredGraph.from_pairs(3, {(0, 1): 0.5, (1, 0): 0.6})


def test_self_pairs_and_out_of_range_indices_are_rejected():
    with pytest.raises(ValueError, match="i < j"):
        ScoredGraph(3, np.array([4]), np.array([0.5]))  # 4 = 1 * 3 + 1, the pair (1, 1)
    with pytest.raises(ValueError, match="out of range"):
        ScoredGraph.from_pairs(2, {(0, 2): 0.5})
    with pytest.raises(ValueError, match="out of range"):
        ScoredGraph(2, np.array([4]), np.array([0.5]))


def test_keys_and_probabilities_must_align():
    with pytest.raises(ValueError, match="equal shape"):
        ScoredGraph(3, np.array([1, 2]), np.array([0.5]))


def test_the_edge_threshold_is_inclusive():
    """`>=`, the convention the bands use, so a threshold means one thing everywhere."""
    graph = ScoredGraph.from_pairs(3, {(0, 1): 0.95, (1, 2): 0.9499})
    left, right, _ = graph.edges(min_p=0.95)
    assert list(zip(left.tolist(), right.tolist())) == [(0, 1)]


# ---------------------------------------------------------------------------
# partitions
# ---------------------------------------------------------------------------


def test_canonical_labels_depend_only_on_the_partition():
    assert canonical_labels([7, 7, 3, 9, 3]).tolist() == [0, 0, 1, 2, 1]
    assert canonical_labels(["b", "b", "a", "c", "a"]).tolist() == [0, 0, 1, 2, 1]
    assert canonical_labels([]).tolist() == []


def test_a_partition_must_label_every_record():
    with pytest.raises(ValueError, match="one label per record"):
        check_partition([0, 0], 3)


# ---------------------------------------------------------------------------
# the objective keeps the bands' three outcomes
# ---------------------------------------------------------------------------


def test_an_unmerged_pair_falls_back_to_the_cheaper_of_review_and_rejection():
    p = np.linspace(0.0, 1.0, 1001)
    columns = expected_costs(p, COST)
    review, reject = list(Band).index(Band.REVIEW), list(Band).index(Band.AUTO_REJECT)
    assert np.allclose(unmerged_pair_cost(p, COST), np.minimum(columns[:, review], columns[:, reject]))


def test_a_lone_pair_is_worth_merging_exactly_where_the_bands_auto_merge_it():
    p = np.linspace(0.0, 1.0, 100_001)
    away_from_the_boundary = np.abs(p - COST.auto_merge_threshold) > 1e-9
    worth_merging = merge_credit(p, COST) > COST.false_merge
    assert np.array_equal(
        worth_merging[away_from_the_boundary],
        (p >= COST.auto_merge_threshold)[away_from_the_boundary],
    )


def test_the_merge_versus_split_break_even_lies_inside_the_review_band():
    """Why no clusterer here merges at C_fm / (C_fm + C_fs): the bands price those pairs as reviews.

    The first version of this package did, and auto-merged review-band pairs.
    """
    tau = COST.false_merge / (COST.false_merge + COST.false_split)
    assert COST.auto_reject_threshold < tau < COST.auto_merge_threshold
    between = (tau + COST.auto_merge_threshold) / 2
    merged = (1 - between) * COST.false_merge
    assert merged > unmerged_pair_cost(between, COST)  # a review is cheaper than merging
    assert merge_credit(between, COST) < COST.false_merge


def test_the_review_mask_marks_only_pairs_left_apart_at_or_above_p_lo():
    graph = ScoredGraph.from_pairs(4, {(0, 1): 0.99, (0, 3): 0.49, (1, 2): 0.7, (2, 3): 0.5})
    # (0, 1) is merged; (0, 3) is apart below p_lo; (1, 2) and (2, 3) are apart at or above it.
    assert review_mask([0, 0, 1, 2], graph, COST).tolist() == [False, False, True, True]


@pytest.mark.parametrize("seed", range(6))
def test_expected_cost_matches_the_sum_over_every_pair(seed):
    rng = np.random.default_rng(seed)
    n = 9
    all_pairs = list(itertools.combinations(range(n), 2))
    chosen = rng.choice(len(all_pairs), size=14, replace=False)
    pairs = {all_pairs[k]: float(rng.random()) for k in chosen}
    labels = rng.integers(0, 4, size=n)

    graph = ScoredGraph.from_pairs(n, pairs)
    assert expected_partition_cost(labels, graph, COST) == pytest.approx(
        brute_force_cost(labels, n, pairs, COST)
    )


def test_an_unemitted_pair_inside_a_cluster_costs_a_full_false_merge():
    """The price of chaining, in the objective: A~B and B~C, A and C never compared."""
    graph = ScoredGraph.from_pairs(3, {(0, 1): 1.0, (1, 2): 1.0})
    assert expected_partition_cost([0, 0, 0], graph, COST) == pytest.approx(COST.false_merge)


def test_an_empty_catalog_costs_nothing():
    assert expected_partition_cost([], ScoredGraph.from_pairs(0, {}), COST) == 0.0
