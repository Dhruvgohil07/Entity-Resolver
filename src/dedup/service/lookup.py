"""The scoring + decision logic for one new record against an existing run's
catalog -- `service/`'s online half.

Pure, no I/O, mirroring `batch.py`'s split: `service/app.py`'s lookup route
gathers candidates (via `blocking.ann.AnnIndex`/`blocking.standard_index.
InvertedIndex` + `store.get_records_by_ids`/`store.get_clusters_by_ids`) and
calls `decide()` here, then writes the result via `store.apply_lookup`.

No `ScoredGraph`, no `average_linkage`: every artifact gets its own local
positional numbering for one request -- `[new_record, *candidates]` -- with
`record_id` the only identifier threaded between this module, the persisted
indexes, and the DuckDB store. `average_linkage` needs a whole graph;
deciding where one new record goes does not, so this reuses `cluster/base.py`'s
own closed-form objective directly -- `merge_credit` and the `p_hi`/`p_lo`
band thresholds -- rather than adding an incremental-partition path to
`cluster/` itself, which still has none, and this does not give it one.

Merging is priced the way `cluster/base.py` prices merging *two clusters*,
not by one best edge: an earlier version merged a new record into a whole
existing cluster whenever its single highest-scoring member cleared `p_hi`,
which is exactly the chaining `cluster/base.py`'s own docstring names as the
difference between a clusterer and the pairwise bands (a caught, not
theoretical, mistake -- an `er-invariants` audit found it before this
shipped). `merge_credit(p) = unmerged_pair_cost(p) + p * C_fm` per candidate
edge, summed per cluster and compared against `C_fm * cluster_size` -- a
cluster member blocking never surfaced as a candidate contributes no edge
and so earns no credit, exactly `cluster/base.py`'s "a pair that has no edge
earns no credit" rule, priced here with `|new record| = 1` rather than
materializing every member's record. A cluster whose best edge clears
`p_hi` but whose aggregate credit does not clear the merge threshold still
gets a review row -- a strong single match is worth a human's attention even
when it should not auto-merge a whole cluster on that one edge alone.

`decide()` never asks whether two *existing* clusters should merge, only
which one cluster the new record joins; a repeated pattern of strong
review-links between the same two clusters is a signal a full batch re-run
would eventually resolve and this cannot (CLAUDE.md, Open questions).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dedup.blocking.pairs import pack
from dedup.cluster.base import merge_credit
from dedup.model.threshold import CostModel, assign_bands
from dedup.model.train import PairScorer
from dedup.normalize import NormalizedRecord


@dataclass(frozen=True)
class CandidateRecord:
    """One existing record the new record might match: its `record_id`, its
    *current* `cluster_id` (fetched fresh from the store -- it may have moved
    since the ann/inverted index was last built, though nothing in v2 can
    actually move a record between clusters after the fact, so this is a
    forward-looking safeguard more than a live concern today), the
    normalized record itself for scoring, and `cluster_size` -- that
    cluster's *total* membership, not how many of its members blocking
    surfaced as candidates, which is what `merge_credit` needs to price a
    merge the way `cluster/base.py` prices one. Every candidate sharing a
    `cluster_id` must carry the same `cluster_size`; `decide()` trusts its
    caller for this rather than re-deriving it.
    """

    record_id: str
    cluster_id: str
    record: NormalizedRecord
    cluster_size: int


@dataclass(frozen=True)
class ReviewTarget:
    cluster_id: str
    record_id: str
    probability: float


@dataclass(frozen=True)
class Decision:
    """What `decide()` concludes: exactly one of `merge_cluster_id` (join an
    existing cluster) or a new singleton (`merge_cluster_id` is `None`);
    zero or more `review_targets` against every *other* cluster the record
    scored at least `p_lo` against -- `cluster/base.py`'s own invariant,
    "every pair a partition leaves apart falls back to its band," applied to
    a partition of size one, not an ad hoc rule. The three candidate-level
    band counts mirror exactly what `write_run` already computes via
    `assign_bands` for a whole batch, scoped to this one lookup's candidates,
    for `store.apply_lookup` to add onto `runs`.
    """

    merge_cluster_id: str | None
    review_targets: list[ReviewTarget]
    n_auto_merge_candidates: int
    n_review_candidates: int
    n_auto_reject_candidates: int


_NO_CANDIDATES = Decision(
    merge_cluster_id=None,
    review_targets=[],
    n_auto_merge_candidates=0,
    n_review_candidates=0,
    n_auto_reject_candidates=0,
)


def decide(
    new_record: NormalizedRecord,
    candidates: Sequence[CandidateRecord],
    *,
    scorer: PairScorer,
    cost: CostModel,
) -> Decision:
    """Score `new_record` against every candidate, then route to the action
    that costs least.

    Candidates are grouped by their current cluster. Merge eligibility is
    the real `cluster/base.py` objective, not one best edge: the credit
    summed over every candidate edge into a cluster against `C_fm *
    cluster_size` (module docstring). A cluster's best edge still decides
    what a review row reports -- the representative pair a reviewer would
    look at first -- but no longer decides whether the *whole cluster*
    merges.
    """
    if not candidates:
        return _NO_CANDIDATES

    records: list[NormalizedRecord] = [new_record, *(c.record for c in candidates)]
    n = len(records)
    pair_keys = pack(
        np.zeros(len(candidates), dtype=np.int64), np.arange(1, n, dtype=np.int64), n
    )
    probabilities = scorer.probabilities(records, pair_keys)
    bands = assign_bands(probabilities, cost)
    credits = merge_credit(probabilities, cost)

    best_candidate: dict[str, CandidateRecord] = {}
    best_p: dict[str, float] = {}
    credit_sum: dict[str, float] = {}
    for candidate, p, credit in zip(candidates, probabilities, credits):
        p = float(p)
        credit_sum[candidate.cluster_id] = credit_sum.get(candidate.cluster_id, 0.0) + float(
            credit
        )
        if candidate.cluster_id not in best_p or p > best_p[candidate.cluster_id]:
            best_p[candidate.cluster_id] = p
            best_candidate[candidate.cluster_id] = candidate

    # A cluster member blocking never surfaced as a candidate contributes no
    # edge, hence no credit -- exactly cluster/base.py's "a pair with no edge
    # earns no credit" rule, priced against that cluster's *total* size
    # (cluster_size), not the count of candidates found for it.
    merge_eligible = {
        cluster_id: credit_sum[cluster_id]
        for cluster_id, candidate in best_candidate.items()
        if credit_sum[cluster_id] > cost.false_merge * candidate.cluster_size
    }
    review_worthy = {
        cluster_id: p for cluster_id, p in best_p.items() if p >= cost.auto_reject_threshold
    }

    merge_cluster_id: str | None = None
    review_targets: list[ReviewTarget] = []

    if merge_eligible:
        # Highest aggregate credit wins -- the quantity the merge decision is
        # actually made on; ties broken on the smaller cluster_id, matching
        # agglomerative.py's own tie convention.
        merge_cluster_id = min(merge_eligible, key=lambda c: (-merge_eligible[c], c))

    for cluster_id, p in review_worthy.items():
        if cluster_id == merge_cluster_id:
            continue
        candidate = best_candidate[cluster_id]
        review_targets.append(
            ReviewTarget(cluster_id=cluster_id, record_id=candidate.record_id, probability=p)
        )

    return Decision(
        merge_cluster_id=merge_cluster_id,
        review_targets=review_targets,
        n_auto_merge_candidates=int(bands.auto_merge.sum()),
        n_review_candidates=int(bands.review.sum()),
        n_auto_reject_candidates=int(bands.auto_reject.sum()),
    )
