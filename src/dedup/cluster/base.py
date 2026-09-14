"""The contract every clusterer satisfies: a scored pair graph in, a partition out.

`model/` stops at pairs. The service answers in entities, so this package turns
scored candidate pairs into a partition of the records -- one label per record --
and every module in it agrees on three things this one pins down.

**A partition merges every pair inside a cluster, not just the candidates.** Two
records in one cluster are merged whether or not any blocker ever emitted them
together. That is the whole difference between a clusterer and the pairwise
bands, and it is exactly where connected components goes wrong: A~B and B~C put A
and C together on no evidence about A and C at all. So a pair blocking never
emitted is not "unknown" here. It is scored `p = 0`, the judgment blocking already
made by discarding it.

**Every pair a partition leaves apart falls back to its band.** The three outcomes
of `model/threshold.py` survive clustering: a pair in two different clusters at
`p >= p_lo` is queued for review, one below it is rejected. So a partition is priced
by the bands' own cost model, pair by pair,

    in one cluster      (1 - p) * C_fm                  merged, right or not
    apart               min(p * C_fs, C_review)         reviewed, or rejected

and the review queue is a by-product of the partition rather than something it
abolishes. The two terms cross exactly at `p_hi`: a lone pair is worth merging
precisely where the bands would auto-merge it, and nowhere else.

That rule replaced an earlier one which looks right and is not. Weighing merge
against split alone breaks even at tau = C_fm / (C_fm + C_fs) -- 0.9091 at
20 : 2 : 1 -- but tau always lies *inside* the review band, so a clusterer built on
it auto-merges exactly the pairs the bands price as cheaper to send to a reviewer.
The `er-invariants` audit caught it before it was committed.

Merging clusters A and B lowers the cost by `sum(merge_credit(p)) - C_fm * |A||B|`
over the edges between them; a pair with no edge earns no credit. Like the band
thresholds this is closed-form in the cost ratios and fit to nothing.
`agglomerative.py` merges while it is positive, `correlation.py` searches the same
objective, and `components.py` ignores it -- which is the reason to report all
three side by side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from dedup.blocking.pairs import unpack
from dedup.model.threshold import CostModel


@dataclass(frozen=True)
class ScoredGraph:
    """Records as nodes, scored candidate pairs as edges.

    Indices and probabilities only -- no records, no dataset name, no model -- so a
    clusterer is written once against this and runs on Abt-Buy, `synth/` output or
    a serve-time neighbourhood alike. `pair_keys` use the packing in
    `blocking/pairs.py`, which is what `model.train.prepare` already hands out.

    `probabilities` must be **calibrated**. Every threshold this package applies is
    derived from the cost model, and a cost model reading a raw booster score
    produces thresholds that are arithmetically valid and mean nothing.
    """

    n_records: int
    pair_keys: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        n = int(self.n_records)
        keys = np.asarray(self.pair_keys, dtype=np.int64).ravel()
        p = np.asarray(self.probabilities, dtype=np.float64).ravel()

        if n < 0:
            raise ValueError(f"n_records must be non-negative, got {n}")
        if keys.shape != p.shape:
            raise ValueError(
                f"pair_keys {keys.shape} and probabilities {p.shape} must have equal shape"
            )
        if keys.size:
            if np.any(np.diff(keys) <= 0):
                raise ValueError(
                    "pair_keys must be sorted and unique, as blocking/pairs.py packs them"
                )
            if keys[0] < 0 or keys[-1] >= n * n:
                raise ValueError(f"pair_keys out of range for n_records={n}")
            left, right = unpack(keys, n)
            if np.any(left >= right):
                raise ValueError(
                    "every pair key must pack (i, j) with i < j; a record is not an edge to itself"
                )
        if not np.isfinite(p).all():
            raise ValueError(
                f"{int((~np.isfinite(p)).sum())} probability(ies) are NaN or infinite; a pair "
                f"that cannot be scored must be left out of the graph, not scored NaN"
            )
        if p.size and (p.min() < 0.0 or p.max() > 1.0):
            raise ValueError(
                f"probabilities must lie in [0, 1], got [{p.min():.4f}, {p.max():.4f}] -- "
                f"this looks like a raw score rather than a calibrated probability"
            )

        object.__setattr__(self, "n_records", n)
        object.__setattr__(self, "pair_keys", keys)
        object.__setattr__(self, "probabilities", p)

    @classmethod
    def from_pairs(cls, n_records: int, pairs: Mapping[tuple[int, int], float]) -> ScoredGraph:
        """Build from `{(i, j): p}` with indices in either order."""
        if not pairs:
            return cls(n_records, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
        index = np.array(list(pairs), dtype=np.int64)
        probabilities = np.fromiter(pairs.values(), dtype=np.float64, count=len(pairs))
        if index.min() < 0 or index.max() >= n_records:
            raise ValueError(f"record index out of range for n_records={n_records}")
        keys = index.min(axis=1) * n_records + index.max(axis=1)
        order = np.argsort(keys, kind="stable")
        return cls(n_records, keys[order], probabilities[order])

    @property
    def n_edges(self) -> int:
        return int(self.pair_keys.size)

    def edges(self, min_p: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(left, right, p)` for every edge with `p >= min_p`, `left < right`.

        `>=` is the convention `model/threshold.assign_bands` and
        `eval/metrics.evaluate_at_threshold` use, so a threshold means the same
        thing applied to an edge as applied to a band.
        """
        if np.isnan(min_p):
            raise ValueError("min_p is NaN")
        keep = self.probabilities >= min_p
        left, right = unpack(self.pair_keys[keep], self.n_records)
        return left, right, self.probabilities[keep]


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel a partition 0, 1, 2, ... in order of each cluster's first record.

    Two arrays describing the same partition become identical, which is what lets
    a report reproduce byte for byte and lets a test assert on a partition without
    caring which ids an algorithm happened to use.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-d, got shape {labels.shape}")
    if labels.size == 0:
        return np.empty(0, dtype=np.int64)
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    return rank[inverse.ravel()]


def check_partition(labels: np.ndarray, n_records: int) -> np.ndarray:
    """Canonical labels, refusing any array that does not label every record.

    A clusterer's output that skips the records no edge touched is not a smaller
    partition, it is a wrong one: those records are singletons, and every metric
    and cost here has to see them.
    """
    labels = np.asarray(labels)
    if labels.shape != (n_records,):
        raise ValueError(
            f"a partition needs one label per record: got shape {labels.shape} for "
            f"{n_records} records"
        )
    return canonical_labels(labels)


def unmerged_pair_cost(probabilities: np.ndarray, cost: CostModel) -> np.ndarray:
    """Expected cost of a pair left in two clusters: reviewed at `p >= p_lo`, else rejected.

    The cheaper of the review and auto-reject columns of
    `model.threshold.expected_costs` -- the band a pair falls back to when the
    partition declines to merge it. A pair with no edge costs nothing apart.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    return np.minimum(p * cost.false_split, cost.review)


def merge_credit(probabilities: np.ndarray, cost: CostModel) -> np.ndarray:
    """What one edge contributes toward merging the two clusters it joins.

    Putting a pair in one cluster costs `(1 - p) * C_fm` instead of its unmerged
    cost, a change of `C_fm - merge_credit(p)`. So merging two clusters pays when the
    credit on the edges between them exceeds `C_fm` per pair between them, with a
    pair that has no edge earning none. Credit exceeds `C_fm` on its own exactly
    when `p > p_hi`.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    return unmerged_pair_cost(p, cost) + p * cost.false_merge


def review_mask(labels: np.ndarray, graph: ScoredGraph, cost: CostModel) -> np.ndarray:
    """Over `graph.edges()`: the pairs a partition leaves apart that fall to review.

    `>= p_lo`, the boundary `model.threshold.assign_bands` uses, so a pair the
    partition declines to merge lands in the same band it would have without it.
    """
    labels = check_partition(labels, graph.n_records)
    left, right, p = graph.edges()
    return (labels[left] != labels[right]) & (p >= cost.auto_reject_threshold)


def expected_partition_cost(labels: np.ndarray, graph: ScoredGraph, cost: CostModel) -> float:
    """The objective in the module docstring, under the model's probabilities.

    Computed without materializing a pair: in-cluster pairs that are not edges are
    counted as `C(k, 2)` minus the in-cluster edges, and each costs a full `C_fm`
    because blocking scored it `p = 0`. That term is the price of chaining.
    """
    labels = check_partition(labels, graph.n_records)
    if labels.size == 0:
        return 0.0
    left, right, p = graph.edges()
    same = labels[left] == labels[right]
    sizes = np.bincount(labels)
    n_in_cluster_non_edges = int(np.sum(sizes * (sizes - 1) // 2)) - int(same.sum())
    return float(
        np.sum((1.0 - p[same]) * cost.false_merge)
        + n_in_cluster_non_edges * cost.false_merge
        + np.sum(unmerged_pair_cost(p[~same], cost))
    )
