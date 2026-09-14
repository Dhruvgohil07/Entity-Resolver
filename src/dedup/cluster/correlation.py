"""Correlation clustering: search the expected-cost objective directly.

`base.py` defines the objective -- the bands' cost model applied to every pair,
merged pairs at `(1 - p) * C_fm` and pairs left apart falling back to review or
rejection, with a pair blocking never emitted scored p = 0 -- and
`agglomerative.py` descends it one merge at a time without ever undoing one.
Minimizing it exactly is a weighted correlation clustering problem, which is
NP-hard, so this module does what the literature does: a randomized construction
followed by local search.

  1. **Pivot** (KwikCluster, Ailon, Charikar & Newman). Visit records in a seeded
     random order; each unclustered record becomes a pivot and claims every
     unclustered neighbour whose pair is worth merging on its own (`p > p_hi`).
  2. **Local search.** Sweep the records, moving each to whichever cluster -- or a
     new singleton -- lowers expected cost most, until a sweep moves nothing. A
     move is taken only when it lowers the objective, so this terminates.

**The starts include both simpler clusterers.** Local search runs from connected
components at `p_hi`, from average linkage, and from `n_restarts` pivot orders, and
the lowest expected cost wins. On its own objective this is therefore never worse
than either -- by construction, not by luck -- which leaves the report free to ask
the question that matters: whether a lower expected cost buys better clusters
against ground truth. Choosing among starts reads no labels, so it is not a leak.

It is a heuristic, and its blind spot is stated rather than hidden: it moves one
record at a time. Where only moving two records *together* would help, a start
that lands there stays there; the pivot restarts exist to supply other starts.
"""

from __future__ import annotations

import numpy as np

from dedup.cluster.agglomerative import average_linkage
from dedup.cluster.base import (
    ScoredGraph,
    canonical_labels,
    expected_partition_cost,
    merge_credit,
)
from dedup.cluster.components import connected_components
from dedup.model.threshold import CostModel

DEFAULT_RESTARTS = 8
DEFAULT_MAX_SWEEPS = 50

# A move must lower expected cost by more than float noise, or two partitions of
# equal cost could trade records forever.
_EPSILON = 1e-9


def correlation_clusters(
    graph: ScoredGraph,
    cost: CostModel,
    *,
    n_restarts: int = DEFAULT_RESTARTS,
    seed: int = 0,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
) -> np.ndarray:
    """One label per record: the lowest-expected-cost partition local search found."""
    if n_restarts < 0:
        raise ValueError(f"n_restarts must be non-negative, got {n_restarts}")
    if max_sweeps < 1:
        raise ValueError(f"max_sweeps must be at least 1, got {max_sweeps}")

    neighbours = _neighbours(graph, cost)
    rng = np.random.default_rng(seed)
    starts = [
        connected_components(graph, cost.auto_merge_threshold),
        average_linkage(graph, cost),
    ]
    starts.extend(
        _pivot(neighbours, rng.permutation(graph.n_records), cost.false_merge)
        for _ in range(n_restarts)
    )

    best_labels: np.ndarray | None = None
    best_cost = 0.0
    for start in starts:
        labels = canonical_labels(_local_search(neighbours, start, cost, max_sweeps))
        value = expected_partition_cost(labels, graph, cost)
        # Ties keep the earlier start, so the simpler partitions win them.
        if best_labels is None or value < best_cost - _EPSILON * max(1.0, best_cost):
            best_labels, best_cost = labels, value
    assert best_labels is not None  # two starts always exist
    return best_labels


def _neighbours(graph: ScoredGraph, cost: CostModel) -> list[dict[int, float]]:
    """Merge credit per edge, by record. An edge with no credit is the same as none."""
    neighbours: list[dict[int, float]] = [{} for _ in range(graph.n_records)]
    left, right, p = graph.edges()
    for a, b, credit in zip(left.tolist(), right.tolist(), merge_credit(p, cost).tolist()):
        if credit > 0.0:
            neighbours[a][b] = credit
            neighbours[b][a] = credit
    return neighbours


def _pivot(
    neighbours: list[dict[int, float]], order: np.ndarray, false_merge: float
) -> np.ndarray:
    labels = [-1] * len(neighbours)
    for pivot in order.tolist():
        if labels[pivot] >= 0:
            continue
        labels[pivot] = pivot
        for other, credit in neighbours[pivot].items():
            if credit > false_merge and labels[other] < 0:  # worth merging on its own
                labels[other] = pivot
    return np.array(labels, dtype=np.int64)


def _local_search(
    neighbours: list[dict[int, float]],
    labels: np.ndarray,
    cost: CostModel,
    max_sweeps: int,
) -> np.ndarray:
    """Single-record moves until none lowers expected cost.

    Placing a record in cluster X changes the cost of its pair with each member u
    by `merge_credit(p_u) - C_fm`, with no credit for a member it has no edge to.
    So the gain of X is `(credit to X) - C_fm * |X|`, a new singleton gains 0, and
    moving from home to X lowers the objective exactly when X gains more.
    """
    n = len(neighbours)
    label = canonical_labels(labels).tolist()
    size = [0] * n  # cluster ids never exceed n - 1
    for cluster in label:
        size[cluster] += 1
    free = [cluster for cluster in range(n - 1, -1, -1) if size[cluster] == 0]
    movable = [record for record in range(n) if neighbours[record]]

    for _ in range(max_sweeps):
        moved = False
        for record in movable:
            home = label[record]
            pull: dict[int, float] = {}
            for other, credit in neighbours[record].items():
                pull[label[other]] = pull.get(label[other], 0.0) + credit

            best = pull.get(home, 0.0) - cost.false_merge * (size[home] - 1)
            target = home
            for cluster in sorted(pull):
                if cluster == home:
                    continue
                gain = pull[cluster] - cost.false_merge * size[cluster]
                if gain > best + _EPSILON:
                    target, best = cluster, gain
            if size[home] > 1 and 0.0 > best + _EPSILON:
                target = free.pop()
            if target == home:
                continue

            size[home] -= 1
            if size[home] == 0:
                free.append(home)
            label[record] = target
            size[target] += 1
            moved = True
        if not moved:
            break
    return np.array(label, dtype=np.int64)
