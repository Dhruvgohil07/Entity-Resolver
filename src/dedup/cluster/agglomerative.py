"""Average linkage, stopped where merging stops paying for itself.

The greedy answer to chaining. Connected components merges two clusters if *any*
pair between them clears the threshold; average linkage merges them only if the
pairs between them justify it *on average*, and a pair blocking never emitted
counts toward that average with no credit at all. A single bridge between two true
pairs therefore cannot fuse them once both have formed, however confident the
bridge is.

The stopping rule is not a tuned cut. By `base.py`, merging A and B lowers expected
cost exactly when the merge credit on the edges between them exceeds
`C_fm * |A||B|`, so "merge while the best average gain is positive" is "merge while
some merge still beats leaving its pairs to review or rejection". For two single
records that is `p > p_hi`: a lone pair merges where the bands would auto-merge it
and nowhere else, and one in the review band stays apart and is queued. The result
is a partition no single further merge improves.

That is not the best partition, and the gap is structural rather than numerical:
average linkage never undoes a merge. If a bridge scores higher than the pairs it
joins, it is merged *first*, while both sides are still singletons and its gain is
the bridge's own -- and the two records it joined stay joined. `correlation.py` can
move records back out, and starts from this partition among others.

Clusters merge highest average gain first, ties broken on the smaller cluster ids,
so the result depends only on the graph. Everything is sparse: a pair of clusters
is considered only when an edge joins them, since with none there is no credit and
the merge can only cost.
"""

from __future__ import annotations

import heapq

import numpy as np

from dedup.cluster.base import ScoredGraph, canonical_labels, merge_credit
from dedup.model.threshold import CostModel


def average_linkage(graph: ScoredGraph, cost: CostModel) -> np.ndarray:
    """One label per record: average-linkage clusters, merged while a merge lowers cost."""
    n = graph.n_records
    left, right, p = graph.edges()
    credit = merge_credit(p, cost)

    size = [1] * n
    version = [0] * n  # bumped on every merge a cluster survives; stale heap entries skip
    alive = [True] * n
    parent = list(range(n))  # a merged-away cluster points at the cluster it joined
    # linked[c][d] is the merge credit summed over edges between clusters c and d.
    linked: list[dict[int, float]] = [{} for _ in range(n)]

    heap: list[tuple[float, int, int, int, int]] = []
    for a, b, weight in zip(left.tolist(), right.tolist(), credit.tolist()):
        if weight <= 0.0:
            continue
        linked[a][b] = weight
        linked[b][a] = weight
        gain = weight - cost.false_merge
        if gain > 0.0:
            heap.append((-gain, a, b, 0, 0))
    heapq.heapify(heap)

    while heap:
        _, a, b, version_a, version_b = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or version[a] != version_a or version[b] != version_b:
            continue

        # b joins a. Entries are pushed as (smaller id, larger id), so the
        # surviving id is always the smaller one.
        alive[b] = False
        parent[b] = a
        size[a] += size[b]
        version[a] += 1

        joined = linked[a]
        joined.pop(b, None)
        for c, weight in linked[b].items():
            if c == a:
                continue
            joined[c] = joined.get(c, 0.0) + weight
            del linked[c][b]
            linked[c][a] = joined[c]
        linked[b] = {}

        # Only gains involving the merged cluster changed, so only those are
        # re-pushed; one that is not positive can only become so through a later
        # merge, which re-pushes it then.
        for c, weight in joined.items():
            n_pairs = size[a] * size[c]
            gain = weight - cost.false_merge * n_pairs
            if gain > 0.0:
                low, high = (a, c) if a < c else (c, a)
                heapq.heappush(heap, (-gain / n_pairs, low, high, version[low], version[high]))

    labels = np.empty(n, dtype=np.int64)
    for record in range(n):
        root = record
        while parent[root] != root:
            root = parent[root]
        node = record
        while parent[node] != root:
            parent[node], node = root, parent[node]
        labels[record] = root
    return canonical_labels(labels)
