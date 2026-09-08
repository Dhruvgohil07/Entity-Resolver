"""Entity-grouped train/test splitting.

CLAUDE.md's first invariant: **split by entity, never by pair.** A pair-level
split puts (A1, A2) in train and (A1, A3) in test, so the model has already
seen A1's exact title string at test time. Reported F1 climbs and none of it
transfers. Grouping on `Record.entity_id` -- which the loaders in `data/`
populate from ground truth for exactly this reason -- keeps every record of a
product on one side of the split.

The consequence downstream: a pair can only be formed inside one split.
Cross-split pairs are all negatives by construction (their entities differ
because the entities were assigned to different sides), so dropping them
loses no positives and costs only a small, uniform number of easy negatives.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from dedup.schema import Record


def group_by_entity(records: list[Record]) -> dict[str, list[Record]]:
    """Bucket records by `entity_id`, raising if any is unlabeled.

    An unlabeled record cannot be split or scored -- it belongs to an unknown
    cluster, so we can neither keep its siblings together nor say whether a
    pair containing it is a match. Loaders assign a singleton entity to
    genuinely unmatched benchmark records, so `None` here means the caller
    passed serve-time data into an evaluation path, which is a bug worth
    stopping on rather than silently dropping rows.
    """
    grouped: dict[str, list[Record]] = defaultdict(list)
    unlabeled = []
    for record in records:
        if record.entity_id is None:
            unlabeled.append(record.record_id)
            continue
        grouped[record.entity_id].append(record)
    if unlabeled:
        raise ValueError(
            f"{len(unlabeled)} record(s) have no entity_id and cannot be split or "
            f"scored, e.g. {sorted(unlabeled)[:3]}"
        )
    return dict(grouped)


def count_true_pairs(records: list[Record]) -> int:
    """Number of same-entity pairs among these records: sum of C(size, 2).

    This is the recall denominator every metric in `metrics.py` wants. It is
    computed from the records, not from the candidate set, precisely so that
    pruning cannot flatter recall.

    Note it exceeds the count of ground-truth pairs the benchmark ships when
    any cluster is larger than 2: Abt-Buy publishes 1097 matched pairs, but
    its 21 size-3 clusters each imply a third pair by transitivity, so a
    deduplication system is actually being asked for 1118.
    """
    return sum(len(group) * (len(group) - 1) // 2 for group in group_by_entity(records).values())


def split_by_entity(
    records: list[Record],
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[list[Record], list[Record]]:
    """Partition records into (train, test) so no entity spans both sides.

    Entities are shuffled and taken into test until the test side holds at
    least `test_fraction` of the records; whole entities move together, so
    the realized fraction lands near the target rather than on it. Entity ids
    are sorted before shuffling so the split depends only on `seed` and the
    set of entities -- not on the order the loader happened to return rows in.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    grouped = group_by_entity(records)
    entity_ids = sorted(grouped)
    np.random.default_rng(seed).shuffle(entity_ids)

    target = test_fraction * len(records)
    test: list[Record] = []
    train: list[Record] = []
    for entity_id in entity_ids:
        destination = test if len(test) < target else train
        destination.extend(grouped[entity_id])

    if not train or not test:
        raise ValueError(
            f"split produced an empty side ({len(train)} train, {len(test)} test) from "
            f"{len(grouped)} entities -- too few entities for test_fraction={test_fraction}"
        )
    return train, test


def true_pair_ids(records: list[Record]) -> set[frozenset[str]]:
    """Every same-entity pair, as frozensets of `record_id`.

    For tests and debugging -- materializing pairs is O(N^2) in the worst
    case and has no place in a scoring path.
    """
    return {
        frozenset(pair)
        for group in group_by_entity(records).values()
        for pair in combinations((record.record_id for record in group), 2)
    }
