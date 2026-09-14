"""Connected components over thresholded edges -- the obvious clusterer, and its failure.

Take every pair at or above a threshold as an edge and call each connected
component an entity. It is linear, needs nothing beyond the threshold, and is what
"transitivity" means in a pairwise mapping file. It is also the failure CLAUDE.md
requires to be demonstrated rather than designed around silently: **it chains.**
A~B and B~C merge A, B and C even when A and C were scored as different products
or never compared at all, so one bad edge fuses two whole clusters and every pair
between them.

The damage is countable without labels, which is what `cluster_shapes` reports.
A cluster of k records merges C(k, 2) pairs, and every one of them that is not
itself an edge was merged on no direct evidence -- an *implied* pair. Lowering the
threshold adds edges linearly and implied pairs much faster, which is why the
threshold that maximizes pairwise F1 is not a safe cluster threshold;
`reports/cluster.md` measures by how much.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _csgraph_components

from dedup.cluster.base import ScoredGraph, canonical_labels, check_partition


def connected_components(graph: ScoredGraph, threshold: float) -> np.ndarray:
    """One label per record: the components of the edges with `p >= threshold`.

    Records with no such edge come back as singletons rather than being left out,
    because B-cubed must score every record in the split.
    """
    n = graph.n_records
    if n == 0:
        return np.empty(0, dtype=np.int64)
    left, right, _ = graph.edges(min_p=threshold)
    adjacency = coo_matrix((np.ones(left.size, dtype=np.int8), (left, right)), shape=(n, n))
    _, labels = _csgraph_components(adjacency, directed=False)
    return canonical_labels(labels)


@dataclass(frozen=True)
class ClusterShape:
    """One multi-record cluster, described without ground truth."""

    cluster: int
    size: int
    n_edges: int  # in-cluster pairs with p >= the threshold the shape was measured at

    @property
    def n_pairs(self) -> int:
        return self.size * (self.size - 1) // 2

    @property
    def n_implied_pairs(self) -> int:
        """In-cluster pairs no edge supports: merged by transitivity alone."""
        return self.n_pairs - self.n_edges


def cluster_shapes(labels: np.ndarray, graph: ScoredGraph, threshold: float) -> list[ClusterShape]:
    """Size, edges and implied pairs of every cluster of two or more records.

    Reads no labels, so the serving path can use it: a cluster whose closure adds
    pairs no edge supports is where a chaining fusion hides.
    """
    labels = check_partition(labels, graph.n_records)
    if labels.size == 0:
        return []
    n_clusters = int(labels.max()) + 1
    left, right, _ = graph.edges(min_p=threshold)
    same = labels[left] == labels[right]
    sizes = np.bincount(labels, minlength=n_clusters)
    edges = np.bincount(labels[left[same]], minlength=n_clusters)
    return [
        ClusterShape(cluster=int(cluster), size=int(sizes[cluster]), n_edges=int(edges[cluster]))
        for cluster in np.flatnonzero(sizes >= 2)
    ]


def implied_pairs(labels: np.ndarray, graph: ScoredGraph, threshold: float) -> int:
    """Total implied pairs of a partition, against edges with `p >= threshold`."""
    return sum(shape.n_implied_pairs for shape in cluster_shapes(labels, graph, threshold))
