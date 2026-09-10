"""The blocker set every stage after `blocking/` runs, and how to run it.

Both helpers here were originally defined inside CLI modules -- the set in
`blocking/evaluate.py`, the runner in `features/evaluate.py` -- and imported
back out by the next stage that needed them. That works until the third
consumer, at which point "which CLI owns the shared helper" stops having an
answer. `model/` is the third, `service/` will be the fourth, so they move to
a library module that has no argparse in it and no report to render.

The set itself is load-bearing rather than incidental: a stage evaluated
against a different blocker set is measured against a different recall
ceiling, and the ceiling is the thing every downstream number inherits.
"""

from __future__ import annotations

from collections.abc import Sequence

from dedup.blocking.ann import AnnBlocker
from dedup.blocking.base import Blocker, BlockerRun
from dedup.blocking.lsh import MinHashLSHBlocker
from dedup.blocking.sorted_neighborhood import SortedNeighborhoodBlocker
from dedup.blocking.standard import default_blockers
from dedup.blocking.union import BlockerScore, ground_truth, score, union_run
from dedup.normalize import NormalizedRecord


def default_blocker_set() -> list[Blocker]:
    """Every blocker, including the ones measurement says do not earn their place.

    `lsh` contributes zero marginal completeness on this benchmark and
    `sorted_neighborhood` buys 0.0044 for 30k candidates. They stay in the
    table because a negative result someone can re-run is evidence, and the
    same result asserted from a deleted experiment is not.
    """
    return [
        *default_blockers(),
        SortedNeighborhoodBlocker(window=20),
        MinHashLSHBlocker(threshold=0.4, shingles="token"),
        AnnBlocker(neighbours=10),
    ]


def block_split(records: Sequence[NormalizedRecord]) -> tuple[BlockerRun, BlockerScore]:
    """Run every blocker over one split and union the result.

    Blocking runs *inside* a split, never across the catalog. Cross-split
    pairs are all negatives by construction -- the entities were assigned to
    one side or the other -- so this loses no positives, and it keeps the
    packed pair indices meaningful as positions in this record list.
    """
    runs = [blocker.run(records) for blocker in default_blocker_set()]
    combined = union_run(runs)
    truth, n_true = ground_truth(records)
    return combined, score(combined, truth, len(records), n_true_pairs_total=n_true)
