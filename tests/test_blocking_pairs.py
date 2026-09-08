"""Tests for the packed-pair representation.

Everything in blocking/ speaks this format, so a defect here corrupts every
candidate set and every number computed from one. The ordering convention
(`i < j`) is the load-bearing part: without it the union of two blockers
double-counts every pair they agree on, which inflates the candidate count
and deflates the reduction ratio.
"""

import numpy as np
import pytest

from dedup.blocking.pairs import pack, pack_block, total_pairs, union, unpack


def test_pair_order_does_not_matter():
    # A blocker should never have to remember which way round it emitted a
    # pair -- (3, 7) and (7, 3) are the same unordered candidate.
    assert pack([3], [7], 10).tolist() == pack([7], [3], 10).tolist()


def test_self_pairs_are_dropped():
    # A record is not a candidate duplicate of itself. ANN returns each point
    # as its own nearest neighbour, so this case is reached on every run.
    assert pack([4, 4, 4], [4, 5, 4], 10).tolist() == pack([4], [5], 10).tolist()


def test_duplicates_collapse():
    keys = pack([1, 1, 2], [2, 2, 1], 10)
    assert keys.size == 1


def test_pack_unpack_round_trips():
    left = np.array([0, 3, 7, 2])
    right = np.array([5, 1, 9, 8])
    keys = pack(left, right, 12)
    a, b = unpack(keys, 12)

    assert bool((a < b).all())
    recovered = {(int(x), int(y)) for x, y in zip(a, b)}
    expected = {(min(int(x), int(y)), max(int(x), int(y))) for x, y in zip(left, right)}
    assert recovered == expected


def test_keys_are_sorted_so_union_can_assume_uniqueness():
    keys = pack([9, 0, 5], [1, 7, 2], 12)
    assert keys.tolist() == sorted(keys.tolist())
    assert len(set(keys.tolist())) == keys.size


def test_pack_block_is_every_pair_in_the_block():
    keys = pack_block(np.array([2, 5, 9, 11]), 20)
    assert keys.size == 4 * 3 // 2


def test_a_block_of_one_yields_nothing():
    assert pack_block(np.array([7]), 20).size == 0


def test_union_deduplicates_across_blockers():
    a = pack([0], [1], 10)
    b = pack([0, 2], [1, 3], 10)  # overlaps a on (0,1)
    merged = union(a, b)

    assert merged.size == 2
    assert merged.tolist() == sorted(set(a.tolist()) | set(b.tolist()))


def test_union_of_nothing_is_empty():
    assert union().size == 0
    assert union(np.empty(0, dtype=np.int64)).size == 0


def test_total_pairs_matches_the_closed_form():
    assert total_pairs(2173) == 2_359_878
    assert total_pairs(1) == 0
    assert total_pairs(0) == 0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_an_index_beyond_n_is_rejected():
    # Silently wrapping would produce a key that unpacks to a different pair,
    # which is far worse than an exception -- the candidate set would look
    # fine and point at the wrong records.
    with pytest.raises(ValueError, match="out of range"):
        pack([0], [10], 10)


def test_a_negative_index_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        pack([-1], [3], 10)


def test_mismatched_left_and_right_are_rejected():
    with pytest.raises(ValueError, match="equal shape"):
        pack([1, 2], [3], 10)
