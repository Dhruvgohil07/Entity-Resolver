"""Tests for average linkage stopped where a merge stops paying.

The one carrying the weight is `test_merge_gain_is_exactly_the_fall_in_expected_cost`:
it checks the quantity the algorithm ranks merges by against the objective it
claims to lower, the way `test_bands_are_the_argmin_of_expected_cost` checks the
band thresholds.
"""

import itertools

import numpy as np
import pytest

from dedup.cluster.agglomerative import average_linkage
from dedup.cluster.base import ScoredGraph, expected_partition_cost, merge_credit
from dedup.model.threshold import CostModel

COST = CostModel()


def random_graph(seed, n=8, n_edges=16, low=0.3):
    rng = np.random.default_rng(seed)
    all_pairs = list(itertools.combinations(range(n), 2))
    chosen = rng.choice(len(all_pairs), size=n_edges, replace=False)
    return ScoredGraph.from_pairs(n, {all_pairs[k]: float(rng.uniform(low, 1.0)) for k in chosen})


def test_a_bridge_weaker_than_the_pairs_it_joins_cannot_fuse_them():
    """Once both pairs form, the bridge's credit is far short of four pairs' worth of C_fm."""
    graph = ScoredGraph.from_pairs(4, {(0, 1): 0.99, (2, 3): 0.98, (1, 2): 0.97})
    assert average_linkage(graph, COST).tolist() == [0, 0, 1, 1]


def test_a_bridge_stronger_than_both_pairs_is_merged_first_and_never_undone():
    """The greedy blind spot the module docstring states; `correlation.py` recovers from it."""
    graph = ScoredGraph.from_pairs(4, {(0, 1): 0.99, (2, 3): 0.98, (1, 2): 0.995})
    assert average_linkage(graph, COST).tolist() == [0, 1, 1, 2]


def test_a_true_triangle_merges():
    graph = ScoredGraph.from_pairs(3, {(0, 1): 0.99, (0, 2): 0.97, (1, 2): 0.96})
    assert average_linkage(graph, COST).tolist() == [0, 0, 0]


def test_a_pair_missing_from_a_triangle_earns_no_credit():
    """The path A~B~C: C joining {A, B} buys one edge's credit for two pairs' worth of risk."""
    graph = ScoredGraph.from_pairs(3, {(0, 1): 0.99, (1, 2): 0.99})
    assert average_linkage(graph, COST).tolist() == [0, 0, 1]


def test_a_lone_pair_above_p_hi_merges():
    graph = ScoredGraph.from_pairs(2, {(0, 1): COST.auto_merge_threshold + 1e-6})
    assert average_linkage(graph, COST).tolist() == [0, 0]


def test_a_review_band_pair_stays_apart_even_above_the_merge_split_break_even():
    """The audit's case: at 0.93, merging beats splitting but a review beats both."""
    tau = COST.false_merge / (COST.false_merge + COST.false_split)
    p = (tau + COST.auto_merge_threshold) / 2
    graph = ScoredGraph.from_pairs(2, {(0, 1): p})
    assert average_linkage(graph, COST).tolist() == [0, 1]


def test_an_empty_graph_is_all_singletons():
    assert average_linkage(ScoredGraph.from_pairs(3, {}), COST).tolist() == [0, 1, 2]
    assert average_linkage(ScoredGraph.from_pairs(0, {}), COST).tolist() == []


@pytest.mark.parametrize("seed", range(10))
def test_merge_gain_is_exactly_the_fall_in_expected_cost(seed):
    graph = random_graph(seed)
    labels = np.random.default_rng(seed + 100).integers(0, 4, size=graph.n_records)
    left, right, p = graph.edges()
    credit = merge_credit(p, COST)
    before = expected_partition_cost(labels, graph, COST)

    for a, b in itertools.combinations(np.unique(labels).tolist(), 2):
        cross = ((labels[left] == a) & (labels[right] == b)) | (
            (labels[left] == b) & (labels[right] == a)
        )
        n_pairs = np.count_nonzero(labels == a) * np.count_nonzero(labels == b)
        gain = credit[cross].sum() - COST.false_merge * n_pairs
        merged = np.where(labels == b, a, labels)
        assert before - expected_partition_cost(merged, graph, COST) == pytest.approx(gain)


@pytest.mark.parametrize("seed", range(10))
def test_no_merge_of_the_final_clusters_would_lower_expected_cost(seed):
    graph = random_graph(seed, low=0.7)
    labels = average_linkage(graph, COST)
    final = expected_partition_cost(labels, graph, COST)
    for a, b in itertools.combinations(np.unique(labels).tolist(), 2):
        merged = np.where(labels == b, a, labels)
        assert expected_partition_cost(merged, graph, COST) >= final - 1e-9
