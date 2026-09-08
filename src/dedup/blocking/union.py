"""Union of several blockers, and the two numbers a blocker is judged on.

Blockers are unioned because each is expected to miss a *different* set of
true pairs. On Abt-Buy the best single blocker reaches pair completeness
0.9562 and the union of five reaches 0.9928 -- and the union's number is the
only one the rest of the pipeline inherits, because the union is what
`features/` actually receives.

Scoring lives here rather than in `eval/metrics.py` because a blocker is not
scored like a classifier. There is deliberately no precision, F1 or accuracy
in this module: a blocker's output is ~99% non-duplicates by construction,
and discarding non-duplicates is its job.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from dedup.blocking.base import BlockerRun
from dedup.blocking.pairs import pack, total_pairs, union
from dedup.eval.splits import count_true_pairs, group_by_entity
from dedup.normalize import NormalizedRecord


def true_pair_keys(records: Sequence[NormalizedRecord]) -> np.ndarray:
    """Packed keys of every same-entity pair, from ground truth.

    Includes the pairs transitivity implies: a size-3 cluster is three pairs,
    not the two a pairwise mapping file lists. On Abt-Buy that is 1,118 pairs
    against the 1,097 shipped.
    """
    n = len(records)
    position = {record.raw.record_id: index for index, record in enumerate(records)}
    grouped = group_by_entity([record.raw for record in records])

    lefts, rights = [], []
    for members in grouped.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(position[m.record_id] for m in members), 2):
            lefts.append(a)
            rights.append(b)
    if not lefts:
        return np.empty(0, dtype=np.int64)
    return pack(np.array(lefts, dtype=np.int64), np.array(rights, dtype=np.int64), n)


@dataclass(frozen=True)
class BlockerScore:
    """One row of the blocking table."""

    name: str
    params: str
    n_candidates: int
    pair_completeness: float
    reduction_ratio: float
    build_seconds: float | None
    query_seconds: float | None
    n_true_pairs_found: int
    n_true_pairs_total: int


def score(
    run: BlockerRun,
    truth: np.ndarray,
    n_records: int,
    *,
    n_true_pairs_total: int,
) -> BlockerScore:
    """Pair completeness and reduction ratio for one blocker.

    `n_true_pairs_total` is passed in and checked rather than derived from
    `truth`, because that is the exact place this measurement goes wrong.
    Computing pair completeness over the pairs that survived blocking makes it
    identically 1.0 -- a self-fulfilling result that has reached publication.
    The denominator is the full ground-truth pair set or the number is
    meaningless.
    """
    if truth.size != n_true_pairs_total:
        raise ValueError(
            f"ground-truth pair set has {truth.size} pairs but n_true_pairs_total is "
            f"{n_true_pairs_total}; pair completeness must divide by every true pair, "
            f"never by the survivors"
        )

    found = int(np.isin(truth, run.keys, assume_unique=True).sum())
    all_pairs = total_pairs(n_records)
    return BlockerScore(
        name=run.name,
        params=run.params,
        n_candidates=run.n_candidates,
        pair_completeness=found / n_true_pairs_total if n_true_pairs_total else 0.0,
        reduction_ratio=1 - run.n_candidates / all_pairs if all_pairs else 0.0,
        build_seconds=run.build_seconds,
        query_seconds=run.query_seconds,
        n_true_pairs_found=found,
        n_true_pairs_total=n_true_pairs_total,
    )


def union_run(runs: Sequence[BlockerRun], name: str = "union (all)") -> BlockerRun:
    """Combine every blocker's candidate set into the one the pipeline sees.

    Build and query seconds are deliberately None on the union row: they are
    not additive in any way a reader could act on, and a summed number would
    invite comparing it against a single blocker's.
    """
    return BlockerRun(
        name=name,
        params="—",
        keys=union(*[run.keys for run in runs]),
        build_seconds=None,
        query_seconds=None,
    )


def missed_pairs(candidate_keys: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """True pairs no blocker emitted -- unrecoverable, and the next design input.

    Reporting how *many* were missed says the ceiling; reporting *which* says
    what to build next, which is why `evaluate.py` prints worked examples.
    """
    return truth[~np.isin(truth, candidate_keys, assume_unique=True)]


def ground_truth(records: Sequence[NormalizedRecord]) -> tuple[np.ndarray, int]:
    """The ground-truth pair set and its size, cross-checked two ways.

    `count_true_pairs` sums C(size, 2) over entity groups; `true_pair_keys`
    enumerates the pairs. They must agree -- if they do not, one of them is
    wrong and every completeness number downstream is wrong with it.
    """
    truth = true_pair_keys(records)
    expected = count_true_pairs([record.raw for record in records])
    if truth.size != expected:
        raise AssertionError(
            f"ground-truth pair set has {truth.size} pairs but count_true_pairs says {expected}"
        )
    return truth, expected
