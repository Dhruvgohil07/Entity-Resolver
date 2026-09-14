"""Tests for correlation clustering.

It is a heuristic for an NP-hard objective, so the tests pin what it does
guarantee -- a local optimum under single-record moves, never worse than the two
partitions it starts from -- and check it against exhaustive search on the small
shapes chaining produces.
"""

import itertools

import numpy as np
import pytest

from dedup.cluster.agglomerative import average_linkage
from dedup.cluster.base import ScoredGraph, canonical_labels, expected_partition_cost
from dedup.cluster.components import connected_components
from dedup.cluster.correlation import correlation_clusters
from dedup.model.threshold import CostModel

COST = CostModel()


def set_partitions(n):
    """Every partition of n records, as restricted growth strings."""

    def grow(prefix, highest):
        if len(prefix) == n:
            yield list(prefix)
            return
        for label in range(highest + 2):
            yield from grow([*prefix, label], max(highest, label))

    if n == 0:
        yield []
        return
    yield from grow([0], 0)


def exhaustive_optimum(graph):
    return min(
        expected_partition_cost(np.array(partition), graph, COST)
        for partition in set_partitions(graph.n_records)
    )


def random_graph(seed, n=9, n_edges=18, low=0.6):
    rng = np.random.default_rng(seed)
    all_pairs = list(itertools.combinations(range(n), 2))
    chosen = rng.choice(len(all_pairs), size=n_edges, replace=False)
    return ScoredGraph.from_pairs(n, {all_pairs[k]: float(rng.uniform(low, 1.0)) for k in chosen})


BRIDGE = (4, {(0, 1): 0.99, (2, 3): 0.98, (1, 2): 0.995})
SHAPES = {
    "bridge stronger than the pairs it joins": BRIDGE,
    "path through a hub": (3, {(0, 1): 0.96, (1, 2): 0.97}),
    "two triangles and a bridge": (
        6,
        {
            (0, 1): 0.99, (0, 2): 0.98, (1, 2): 0.97,
            (3, 4): 0.99, (3, 5): 0.98, (4, 5): 0.97,
            (2, 3): 0.999,
        },
    ),
    "clique": (
        4,
        {(0, 1): 0.99, (0, 2): 0.98, (0, 3): 0.97, (1, 2): 0.99, (1, 3): 0.96, (2, 3): 0.955},
    ),
    "review-band pairs around a confident one": (
        4,
        {(0, 1): 0.99, (1, 2): 0.93, (2, 3): 0.7, (0, 2): 0.92},
    ),
}


def test_set_partitions_enumerates_the_bell_numbers():
    assert [sum(1 for _ in set_partitions(n)) for n in range(1, 7)] == [1, 2, 5, 15, 52, 203]


@pytest.mark.parametrize(("n", "pairs"), SHAPES.values(), ids=list(SHAPES))
def test_reaches_the_exhaustive_optimum_on_chaining_shapes(n, pairs):
    graph = ScoredGraph.from_pairs(n, pairs)
    labels = correlation_clusters(graph, COST)
    assert expected_partition_cost(labels, graph, COST) == pytest.approx(exhaustive_optimum(graph))


def test_undoes_the_bridge_average_linkage_merges_first():
    graph = ScoredGraph.from_pairs(*BRIDGE)
    assert average_linkage(graph, COST).tolist() == [0, 1, 1, 2]
    assert correlation_clusters(graph, COST).tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("seed", range(8))
def test_no_single_record_move_lowers_expected_cost(seed):
    graph = random_graph(seed)
    labels = correlation_clusters(graph, COST)
    found = expected_partition_cost(labels, graph, COST)
    fresh = graph.n_records  # an id no cluster uses: a move to a new singleton
    for record in range(graph.n_records):
        for target in [*np.unique(labels).tolist(), fresh]:
            moved = labels.copy()
            moved[record] = target
            assert expected_partition_cost(moved, graph, COST) >= found - 1e-9


@pytest.mark.parametrize("n_restarts", [0, 8])
@pytest.mark.parametrize("seed", range(8))
def test_never_worse_than_the_partitions_it_starts_from(seed, n_restarts):
    """Structural, not lucky: with no pivot restarts at all it still holds."""
    graph = random_graph(seed)
    found = expected_partition_cost(
        correlation_clusters(graph, COST, n_restarts=n_restarts, seed=seed), graph, COST
    )
    starts = (
        connected_components(graph, COST.auto_merge_threshold),
        average_linkage(graph, COST),
    )
    for start in starts:
        assert found <= expected_partition_cost(start, graph, COST) + 1e-9


def test_the_same_seed_gives_the_same_partition():
    graph = random_graph(3)
    first = correlation_clusters(graph, COST, seed=7)
    assert correlation_clusters(graph, COST, seed=7).tolist() == first.tolist()


def test_labels_come_back_canonical():
    labels = correlation_clusters(random_graph(5), COST)
    assert labels.tolist() == canonical_labels(labels).tolist()


def test_empty_graphs():
    assert correlation_clusters(ScoredGraph.from_pairs(3, {}), COST).tolist() == [0, 1, 2]
    assert correlation_clusters(ScoredGraph.from_pairs(0, {}), COST).tolist() == []


def test_invalid_search_parameters_are_rejected():
    graph = ScoredGraph.from_pairs(2, {(0, 1): 0.99})
    with pytest.raises(ValueError, match="n_restarts"):
        correlation_clusters(graph, COST, n_restarts=-1)
    with pytest.raises(ValueError, match="max_sweeps"):
        correlation_clusters(graph, COST, max_sweeps=0)
