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

The same leak has a second level, and `Record.split_group` closes it. Two
*distinct* entities can share source text -- `synth/` derives sibling products
and their duplicates from one seed listing -- and splitting those siblings
across train and test scores a test entity against training records built from
the same title. Records sharing a `split_group` therefore move as one unit.
Every benchmark loader leaves the field unset, and for such records the units
are exactly the entities, in exactly the order they had before the field
existed, so no committed split moves.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from dedup.schema import Record

# The split every report CLI scores on when no flag says otherwise. Named because
# more than the splitter depends on it: synthetic catalogs are seeded from its
# train side, and a catalog seeded from any other split could hold records these
# reports test on.
DEFAULT_TEST_FRACTION = 0.3
DEFAULT_SEED = 0


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


def group_for_split(records: list[Record]) -> dict[str, list[Record]]:
    """Bucket records into the units a split may not cut.

    A unit is a whole entity, or -- when its records carry a `split_group` --
    every entity in that group. An entity whose records disagree about their
    group is refused: there is no way to keep both the entity and the groups
    whole, and silently picking one would cut the other.

    Keys are prefixed by kind, so a `split_group` spelled like some unrelated
    `entity_id` cannot merge with it. Every entity key shares one prefix, so
    entity keys sort exactly as the bare entity ids do -- which is what keeps a
    split over ungrouped records identical to the entity-grouped split this
    function generalizes.
    """
    grouped: dict[str, list[Record]] = defaultdict(list)
    for entity_id, members in group_by_entity(records).items():
        groups = {member.split_group for member in members}
        if len(groups) > 1:
            raise ValueError(
                f"entity {entity_id!r} spans more than one split_group "
                f"({sorted(str(group) for group in groups)}); an entity must sit whole inside "
                f"one group, or splitting on groups would cut it"
            )
        (group,) = groups
        key = f"entity:{entity_id}" if group is None else f"group:{group}"
        grouped[key].extend(members)
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
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Record], list[Record]]:
    """Partition records into (train, test) so no entity spans both sides.

    Units -- entities, or split groups where records carry one -- are shuffled
    and taken into test until the test side holds at least `test_fraction` of
    the records; whole units move together, so the realized fraction lands
    near the target rather than on it. Unit keys are sorted before shuffling so
    the split depends only on `seed` and the set of units -- not on the order
    the loader happened to return rows in.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    grouped = group_for_split(records)
    unit_keys = sorted(grouped)
    np.random.default_rng(seed).shuffle(unit_keys)

    target = test_fraction * len(records)
    test: list[Record] = []
    train: list[Record] = []
    for key in unit_keys:
        destination = test if len(test) < target else train
        destination.extend(grouped[key])

    if not train or not test:
        raise ValueError(
            f"split produced an empty side ({len(train)} train, {len(test)} test) from "
            f"{len(grouped)} units -- too few entities or split groups for "
            f"test_fraction={test_fraction}"
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


def kfold_by_entity(
    records: list[Record],
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> list[list[Record]]:
    """Partition records into `n_folds` parts so no entity spans two of them.

    The same rule as `split_by_entity`, applied K ways: whole units (entities,
    or split groups) move together, so a fold's records can be blocked and
    scored as a self-contained catalog. That is what makes out-of-fold
    calibration honest -- a model scoring fold k has seen no record of any
    entity in fold k, so its predictions there carry the same optimism as
    predictions on unseen data.

    Units are dealt round-robin over the folds after shuffling rather than
    sliced into K contiguous runs, which keeps the fold sizes close even when
    cluster sizes vary. As in `split_by_entity`, keys are sorted before
    shuffling so the result depends only on `seed` and the set of units, never
    on the order the loader returned rows in.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    grouped = group_for_split(records)
    unit_keys = sorted(grouped)
    if len(unit_keys) < n_folds:
        raise ValueError(
            f"cannot build {n_folds} folds from {len(unit_keys)} entities or split groups -- "
            f"every fold must hold at least one entity"
        )
    np.random.default_rng(seed).shuffle(unit_keys)

    folds: list[list[Record]] = [[] for _ in range(n_folds)]
    for position, key in enumerate(unit_keys):
        folds[position % n_folds].extend(grouped[key])
    return folds
