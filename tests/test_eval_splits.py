"""Tests for entity-grouped splitting.

The leak these guard against is invisible in metrics -- a pair-level split
just reports a better number. So the tests assert the structural property
(no entity on both sides) rather than any downstream score.
"""

import pytest

from dedup.eval.splits import (
    count_true_pairs,
    group_by_entity,
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
