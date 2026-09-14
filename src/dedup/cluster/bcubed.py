"""B-cubed precision and recall, and the pair counts a partition implies.

CLAUDE.md: **cluster quality is measured with B-cubed, separately from pairwise
metrics.** Pairwise F1 over candidate edges cannot see what a clusterer adds -- the
pairs it merges by transitivity -- so it cannot see chaining. B-cubed scores each
*record* instead: of the records sharing its cluster, the fraction sharing its
entity (precision), and of the records sharing its entity, the fraction sharing
its cluster (recall), each averaged over records (Bagga & Baldwin 1998; Amigó et
al. 2009 for why it is the extrinsic clustering metric that satisfies the formal
constraints the others break).

**Every record in the split is scored, isolated ones included.** A record no edge
touched is a singleton cluster and it counts: precision 1, recall 1/|entity|.
Scoring only the records that appear in some edge moves the number -- the same
class of trap as a recall divided by the pairs blocking kept, one level up -- so
every function here takes one label per record and refuses arrays of different
lengths rather than aligning them.

`closure_pair_counts` sits beside B-cubed rather than replacing it. It is the
pairwise view of the *partition* -- every pair it merges, emitted by blocking or
not -- and it is what the cost model bills. `reports/cluster.md` prints both.

All three functions read one cluster x entity contingency table built in
O(n log n); no pair is ever materialized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dedup.cluster.base import canonical_labels
from dedup.model.threshold import CostModel


@dataclass(frozen=True)
class _Contingency:
    """Non-empty cells of the cluster x entity table."""

    cell_counts: np.ndarray  # records in each (cluster, entity) cell
    cell_cluster: np.ndarray
    cell_entity: np.ndarray
    cluster_sizes: np.ndarray
    entity_sizes: np.ndarray

    @property
    def n_records(self) -> int:
        return int(self.cluster_sizes.sum())


def _contingency(predicted: np.ndarray, truth: np.ndarray) -> _Contingency:
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    if predicted.ndim != 1 or truth.ndim != 1:
        raise ValueError(
            f"labels must be 1-d, got predicted {predicted.shape} and truth {truth.shape}"
        )
    if predicted.shape != truth.shape:
        raise ValueError(
            f"one predicted label per record is required: got {predicted.size} predicted "
            f"labels for {truth.size} records. Every record in the split is scored, including "
            f"those no edge touched -- they are singletons, not absent."
        )
    if truth.size == 0:
        raise ValueError("cannot score a partition of zero records")
    if truth.dtype == object and any(label is None for label in truth.tolist()):
        raise ValueError(
            "a record has no entity_id, so it belongs to an unknown entity and cannot be "
            "scored; this looks like serve-time data passed into an evaluation path"
        )

    clusters = canonical_labels(predicted)
    entities = canonical_labels(truth)
    n_entities = int(entities.max()) + 1
    codes, counts = np.unique(clusters * n_entities + entities, return_counts=True)
    return _Contingency(
        cell_counts=counts,
        cell_cluster=codes // n_entities,
        cell_entity=codes % n_entities,
        cluster_sizes=np.bincount(clusters),
        entity_sizes=np.bincount(entities),
    )


def _n_pairs(sizes: np.ndarray) -> int:
    sizes = np.asarray(sizes, dtype=np.int64)
    return int(np.sum(sizes * (sizes - 1) // 2))


@dataclass(frozen=True)
class BCubedScore:
    precision: float
    recall: float
    n_records: int

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 0.0 if total == 0 else 2 * self.precision * self.recall / total


def bcubed(predicted: np.ndarray, truth: np.ndarray) -> BCubedScore:
    """B-cubed precision and recall of `predicted` against `truth`, one label per record.

    A cell of `c` records inside a cluster of size `s` contributes `c` records each
    scoring `c / s`, so precision is `sum(c^2 / s) / n`; recall is the same with
    entity sizes.
    """
    table = _contingency(predicted, truth)
    counts = table.cell_counts.astype(np.float64)
    return BCubedScore(
        precision=float(np.sum(counts**2 / table.cluster_sizes[table.cell_cluster]))
        / table.n_records,
        recall=float(np.sum(counts**2 / table.entity_sizes[table.cell_entity]))
        / table.n_records,
        n_records=table.n_records,
    )


@dataclass(frozen=True)
class ClusterPairCounts:
    """The pairs a partition merges, against every true pair in the split."""

    n_merged_pairs: int  # every pair placed in one cluster, candidate or not
    n_true_merged_pairs: int
    n_true_pairs: int  # every same-entity pair in the split -- the recall denominator

    @property
    def n_false_merges(self) -> int:
        return self.n_merged_pairs - self.n_true_merged_pairs

    @property
    def n_false_splits(self) -> int:
        return self.n_true_pairs - self.n_true_merged_pairs

    @property
    def precision(self) -> float:
        return self.n_true_merged_pairs / self.n_merged_pairs if self.n_merged_pairs else 0.0

    @property
    def recall(self) -> float:
        """Against every true pair in the split, including those blocking never emitted."""
        return self.n_true_merged_pairs / self.n_true_pairs if self.n_true_pairs else 0.0

    def realized_cost(self, cost: CostModel, *, n_review: int, n_review_true: int) -> float:
        """What ground truth says the partition cost, in review-equivalents.

        Billed on the terms of `model.threshold.BandSummary.realized_cost`, with the
        reviewer assumed correct: `C_fm` per false merge, `C_review` per pair left
        apart and queued (`n_review`, of which `n_review_true` are duplicates), and
        `C_fs` per true pair left apart *outside* the queue. Like the band version it
        keeps the cost model falsifiable: the objective only ever saw `p`.
        """
        if not 0 <= n_review_true <= min(n_review, self.n_false_splits):
            raise ValueError(
                f"n_review_true={n_review_true} must lie in [0, min(n_review={n_review}, "
                f"true pairs left apart={self.n_false_splits})]"
            )
        return (
            self.n_false_merges * cost.false_merge
            + n_review * cost.review
            + (self.n_false_splits - n_review_true) * cost.false_split
        )


def closure_pair_counts(predicted: np.ndarray, truth: np.ndarray) -> ClusterPairCounts:
    table = _contingency(predicted, truth)
    return ClusterPairCounts(
        n_merged_pairs=_n_pairs(table.cluster_sizes),
        n_true_merged_pairs=_n_pairs(table.cell_counts),
        n_true_pairs=_n_pairs(table.entity_sizes),
    )


@dataclass(frozen=True)
class ClusterErrors:
    n_clusters: int
    n_fused_clusters: int  # clusters holding records of more than one entity
    n_split_entities: int  # entities whose records landed in more than one cluster


def cluster_errors(predicted: np.ndarray, truth: np.ndarray) -> ClusterErrors:
    """Counts of the two ways a partition goes wrong, by cluster and by entity."""
    table = _contingency(predicted, truth)
    return ClusterErrors(
        n_clusters=int(table.cluster_sizes.size),
        n_fused_clusters=int(np.count_nonzero(np.bincount(table.cell_cluster) > 1)),
        n_split_entities=int(np.count_nonzero(np.bincount(table.cell_entity) > 1)),
    )
