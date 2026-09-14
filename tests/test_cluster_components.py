"""Tests for connected components, and the chaining failure it must demonstrate.

CLAUDE.md: connected components chains, and the failure is demonstrated rather
than designed around silently. The first two tests are that demonstration in its
smallest form; `reports/cluster.md` measures it on real data.
"""

import numpy as np
import pytest

from dedup.cluster.base import ScoredGraph
from dedup.cluster.bcubed import bcubed
from dedup.cluster.components import cluster_shapes, connected_components, implied_pairs


def test_chaining_merges_two_records_that_were_never_compared():
    """A~B and B~C put A and C in one cluster on no evidence about A and C."""
    graph = ScoredGraph.from_pairs(3, {(0, 1): 0.99, (1, 2): 0.99})
    labels = connected_components(graph, 0.95)
    assert labels.tolist() == [0, 0, 0]
    shapes = cluster_shapes(labels, graph, 0.95)
    assert [(s.size, s.n_pairs, s.n_edges, s.n_implied_pairs) for s in shapes] == [(3, 3, 2, 1)]


def test_one_bad_edge_fuses_two_whole_clusters():
    triangle = {(0, 1): 0.99, (0, 2): 0.99, (1, 2): 0.99}
    other = {(i + 3, j + 3): p for (i, j), p in triangle.items()}
    graph = ScoredGraph.from_pairs(6, {**triangle, **other, (2, 3): 0.96})
    truth = ["a"] * 3 + ["b"] * 3

    labels = connected_components(graph, 0.95)
    assert len(set(labels.tolist())) == 1
    # One edge bought nine merged pairs: C(6, 2) = 15 against 7 edges.
    assert implied_pairs(labels, graph, 0.95) == 15 - 7
    assert bcubed(labels, truth).precision == pytest.approx(0.5)


def test_the_threshold_is_inclusive():
    graph = ScoredGraph.from_pairs(3, {(0, 1): 0.95, (1, 2): 0.9499})
    assert connected_components(graph, 0.95).tolist() == [0, 0, 1]


def test_records_without_an_edge_are_singletons_not_omitted():
    graph = ScoredGraph.from_pairs(4, {(0, 1): 0.99})
    assert connected_components(graph, 0.5).tolist() == [0, 0, 1, 2]


def test_an_empty_graph_is_all_singletons():
    assert connected_components(ScoredGraph.from_pairs(3, {}), 0.5).tolist() == [0, 1, 2]
    assert connected_components(ScoredGraph.from_pairs(0, {}), 0.5).tolist() == []


def test_an_unreachable_threshold_merges_nothing():
    """`best_f1` returns an infinite threshold on an empty curve; that must not crash."""
    graph = ScoredGraph.from_pairs(3, {(0, 1): 1.0})
    assert connected_components(graph, float("inf")).tolist() == [0, 1, 2]


def test_shapes_skip_singletons_and_need_no_labels():
    graph = ScoredGraph.from_pairs(5, {(0, 1): 0.99, (2, 3): 0.99, (3, 4): 0.99})
    labels = connected_components(graph, 0.95)
    shapes = cluster_shapes(labels, graph, 0.95)
    assert [(s.size, s.n_implied_pairs) for s in shapes] == [(2, 0), (3, 1)]
    assert implied_pairs(np.arange(5), graph, 0.95) == 0
