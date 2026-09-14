"""Tests for entity-grouped splitting.

The leak these guard against is invisible in metrics -- a pair-level split
just reports a better number. So the tests assert the structural property
(no entity on both sides) rather than any downstream score.
"""

import numpy as np
import pytest

from dedup.eval.splits import (
    count_true_pairs,
    group_by_entity,
    group_for_split,
    kfold_by_entity,
    split_by_entity,
    true_pair_ids,
)
from dedup.schema import Record


def make_records(cluster_sizes: list[int]) -> list[Record]:
    """One entity per size, numbered so record ids stay readable in failures."""
    records = []
    for entity_index, size in enumerate(cluster_sizes):
        for member in range(size):
            records.append(
                Record(
                    record_id=f"synthetic:e{entity_index:03d}:{member}",
                    source="synthetic",
                    entity_id=f"synthetic:e{entity_index:03d}",
                    title=f"product {entity_index} listing {member}",
                )
            )
    return records


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


def test_no_entity_appears_on_both_sides_of_the_split():
    records = make_records([2] * 40 + [3] * 10)
    train, test = split_by_entity(records, test_fraction=0.3, seed=0)

    train_entities = {record.entity_id for record in train}
    test_entities = {record.entity_id for record in test}
    assert train_entities & test_entities == set()


def test_every_record_lands_on_exactly_one_side():
    records = make_records([2] * 40 + [3] * 10)
    train, test = split_by_entity(records, test_fraction=0.3, seed=0)

    ids = [record.record_id for record in train + test]
    assert sorted(ids) == sorted(record.record_id for record in records)
    assert len(ids) == len(set(ids))


def test_the_test_side_lands_near_the_requested_fraction():
    # Whole entities move together, so the realized fraction overshoots the
    # target by at most one entity rather than hitting it exactly.
    records = make_records([2] * 100)
    _, test = split_by_entity(records, test_fraction=0.3, seed=0)
    assert 0.30 <= len(test) / len(records) < 0.32


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_split():
    records = make_records([2] * 30)
    first = split_by_entity(records, seed=11)
    second = split_by_entity(records, seed=11)
    assert [r.record_id for r in first[1]] == [r.record_id for r in second[1]]


def test_the_split_does_not_depend_on_the_order_records_arrive_in():
    # Entity ids are sorted before shuffling for exactly this reason: a
    # cached split must not silently reshuffle because a loader changed the
    # order it emits rows in.
    records = make_records([2] * 30)
    _, test = split_by_entity(records, seed=5)
    _, shuffled_test = split_by_entity(list(reversed(records)), seed=5)

    assert {r.record_id for r in test} == {r.record_id for r in shuffled_test}


def test_different_seeds_give_different_splits():
    records = make_records([2] * 30)
    _, first = split_by_entity(records, seed=0)
    _, second = split_by_entity(records, seed=1)
    assert {r.record_id for r in first} != {r.record_id for r in second}


# ---------------------------------------------------------------------------
# Pair counting
# ---------------------------------------------------------------------------


def test_count_true_pairs_includes_the_pairs_transitivity_implies():
    # A size-3 cluster is 3 pairs, not the 2 a pairwise ground-truth file
    # lists. Getting this wrong understates the recall denominator, which
    # inflates every recall number downstream.
    assert count_true_pairs(make_records([3])) == 3
    assert count_true_pairs(make_records([2, 2, 3])) == 5


def test_a_singleton_entity_contributes_no_pairs():
    assert count_true_pairs(make_records([1, 1, 1])) == 0


def test_true_pair_ids_agrees_with_the_count():
    records = make_records([2, 3, 1, 4])
    assert len(true_pair_ids(records)) == count_true_pairs(records)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_an_unlabeled_record_is_rejected_rather_than_dropped():
    records = make_records([2])
    records.append(
        Record(record_id="synthetic:new", source="synthetic", title="a record with no entity")
    )
    with pytest.raises(ValueError, match="no entity_id"):
        split_by_entity(records)


def test_too_few_entities_to_fill_both_sides_is_an_error():
    # One entity cannot be split without breaking the invariant, so this
    # fails loudly instead of returning an empty train set.
    with pytest.raises(ValueError, match="empty side"):
        split_by_entity(make_records([4]), test_fraction=0.3)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_a_test_fraction_outside_the_open_unit_interval_is_rejected(fraction):
    with pytest.raises(ValueError, match="test_fraction"):
        split_by_entity(make_records([2] * 10), test_fraction=fraction)


def test_group_by_entity_keeps_every_member_together():
    records = make_records([2, 3])
    grouped = group_by_entity(records)
    assert sorted(len(group) for group in grouped.values()) == [2, 3]


# ---------------------------------------------------------------------------
# kfold_by_entity -- the same invariant, applied K ways for out-of-fold
# calibration in model/. A fold that shared an entity with the folds used to
# train would make the calibration set optimistic, which is the leak the whole
# out-of-fold arrangement exists to avoid.
# ---------------------------------------------------------------------------


def test_no_entity_spans_two_folds():
    folds = kfold_by_entity(make_records([2] * 30), n_folds=5, seed=0)
    seen: dict[str, int] = {}
    for index, fold in enumerate(folds):
        for record in fold:
            assert seen.setdefault(record.entity_id, index) == index


def test_the_folds_partition_the_catalog_exactly():
    records = make_records([2, 3, 1] * 10)
    folds = kfold_by_entity(records, n_folds=5, seed=0)

    ids = [record.record_id for fold in folds for record in fold]
    assert len(ids) == len(records)
    assert set(ids) == {record.record_id for record in records}


def test_every_fold_holds_at_least_one_entity():
    folds = kfold_by_entity(make_records([2] * 7), n_folds=5, seed=0)
    assert len(folds) == 5
    assert all(fold for fold in folds)


def test_folds_depend_only_on_the_seed_not_on_row_order():
    records = make_records([2, 3] * 12)
    shuffled = list(reversed(records))

    def signature(folds):
        return [sorted(record.record_id for record in fold) for fold in folds]

    assert signature(kfold_by_entity(records, seed=7)) == signature(
        kfold_by_entity(shuffled, seed=7)
    )


def test_a_different_seed_gives_a_different_partition():
    records = make_records([2] * 30)

    def signature(folds):
        return [sorted(record.record_id for record in fold) for fold in folds]

    assert signature(kfold_by_entity(records, seed=0)) != signature(
        kfold_by_entity(records, seed=1)
    )


def test_more_folds_than_entities_is_an_error():
    with pytest.raises(ValueError, match="every fold must hold at least one entity"):
        kfold_by_entity(make_records([2] * 3), n_folds=5)


@pytest.mark.parametrize("n_folds", [0, 1, -2])
def test_fewer_than_two_folds_is_an_error(n_folds):
    with pytest.raises(ValueError, match="n_folds must be at least 2"):
        kfold_by_entity(make_records([2] * 10), n_folds=n_folds)


def test_an_unlabeled_record_is_rejected_here_too():
    records = make_records([2] * 10)
    records.append(
        Record(record_id="synthetic:new", source="synthetic", title="a record with no entity")
    )
    with pytest.raises(ValueError, match="no entity_id"):
        kfold_by_entity(records)


# ---------------------------------------------------------------------------
# split_group -- distinct entities bound together because they share source
# text. synth/ derives siblings and their duplicates from one seed listing, so
# a sibling on each side of a split leaks the seed title the way a shared
# entity would.
# ---------------------------------------------------------------------------


def make_family_records(families: list[list[int]]) -> list[Record]:
    """One split group per family; each inner list is its entities' sizes."""
    records = []
    for family, entity_sizes in enumerate(families):
        for entity, size in enumerate(entity_sizes):
            for member in range(size):
                records.append(
                    Record(
                        record_id=f"synthetic:f{family:03d}:e{entity}:{member}",
                        source="synthetic",
                        entity_id=f"synthetic:f{family:03d}:e{entity}",
                        split_group=f"synthetic:f{family:03d}",
                        title=f"family {family} product {entity} listing {member}",
                    )
                )
    return records


def test_a_split_group_never_straddles_train_and_test():
    train, test = split_by_entity(make_family_records([[2, 1, 3]] * 30), seed=0)
    assert {r.split_group for r in train} & {r.split_group for r in test} == set()


def test_a_split_group_never_spans_two_folds():
    folds = kfold_by_entity(make_family_records([[2, 2]] * 20), n_folds=5, seed=0)
    seen: dict[str, int] = {}
    for index, fold in enumerate(folds):
        for record in fold:
            assert seen.setdefault(record.split_group, index) == index


def test_records_without_a_split_group_split_exactly_as_entities_do():
    """The field must be invisible to every loader that never sets it.

    Re-derived from the entity-grouped algorithm itself rather than from a saved
    answer, so this pins the partition every committed report was produced on.
    """
    records = make_records([2, 3, 1] * 20)
    entity_ids = sorted({r.entity_id for r in records})
    np.random.default_rng(4).shuffle(entity_ids)
    expected, filled = set(), 0
    for entity_id in entity_ids:
        if filled < 0.3 * len(records):
            members = [r.record_id for r in records if r.entity_id == entity_id]
            expected.update(members)
            filled += len(members)

    _, test = split_by_entity(records, test_fraction=0.3, seed=4)
    assert {r.record_id for r in test} == expected


def test_folds_without_a_split_group_are_exactly_the_entity_folds():
    records = make_records([2, 3] * 12)
    entity_ids = sorted({r.entity_id for r in records})
    np.random.default_rng(7).shuffle(entity_ids)
    expected: list[set[str]] = [set() for _ in range(5)]
    for position, entity_id in enumerate(entity_ids):
        expected[position % 5].update(r.record_id for r in records if r.entity_id == entity_id)

    folds = kfold_by_entity(records, n_folds=5, seed=7)
    assert [{r.record_id for r in fold} for fold in folds] == expected


def test_an_entity_spanning_two_split_groups_is_rejected():
    records = make_family_records([[2]] * 5)
    stray = records[0].model_copy(
        update={"record_id": "synthetic:stray", "split_group": "synthetic:elsewhere"}
    )
    with pytest.raises(ValueError, match="more than one split_group"):
        split_by_entity([*records, stray])


def test_an_entity_only_partly_grouped_is_rejected():
    """None and a group are two different answers, not a default and a value."""
    records = make_records([2] * 10)
    records[0] = records[0].model_copy(update={"split_group": "synthetic:g1"})
    with pytest.raises(ValueError, match="more than one split_group"):
        kfold_by_entity(records)


def test_a_split_group_named_like_an_entity_does_not_merge_with_it():
    lone = Record(
        record_id="synthetic:a:0", source="synthetic", entity_id="synthetic:a", title="lone"
    )
    grouped = [
        Record(
            record_id=f"synthetic:b:{i}",
            source="synthetic",
            entity_id="synthetic:b",
            split_group="synthetic:a",
            title=f"grouped {i}",
        )
        for i in range(2)
    ]
    units = group_for_split([lone, *grouped])
    assert sorted(len(members) for members in units.values()) == [1, 2]
