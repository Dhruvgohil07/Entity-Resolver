"""Candidate pairs as packed integers.

Every blocker in this package emits the same thing: a sorted, deduplicated
`np.ndarray` of int64 keys, where a pair of record indices `(i, j)` with
`i < j` is packed as `i * n + j`.

Why not a set of `(i, j)` tuples, which is the obvious thing:

  * **Size.** A Python tuple-of-two-ints inside a set costs well over 100
    bytes once the tuple, its two int objects and the set's slot are counted.
    At the 20M candidate pairs a 200k-record `synth/` catalog produces that is
    multiple gigabytes; packed int64 is 8 bytes each, so the same candidate
    set is ~160 MB.
  * **Union.** Unioning several blockers is the whole point of this package,
    and `np.unique(np.concatenate(...))` over int64 is a sort, not millions of
    Python-level hash lookups.

The ordering convention (`i < j`) is what makes a pair unordered: without it
the union of two blockers double-counts every pair they agree on, which
inflates the candidate count and deflates the reduction ratio.
"""

from __future__ import annotations

import numpy as np

# int64 holds i*n+j while i,j < n as long as n^2 fits, i.e. n < ~3.04e9. Far
# beyond any catalog this pipeline targets, but the guard is cheap and the
# failure would be a silently wrong pair rather than an exception.
_MAX_RECORDS = 3_000_000_000


def _check_n(n: int) -> None:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > _MAX_RECORDS:
        raise ValueError(f"n={n} exceeds the {_MAX_RECORDS} records int64 packing supports")


def pack(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    """Pack index pairs into sorted, deduplicated int64 keys.

    Accepts pairs in either order and normalizes each to `(min, max)`, so a
    blocker never has to remember which way round it emitted them. Self-pairs
    are dropped: a record is not a candidate duplicate of itself.
    """
    _check_n(n)
    left = np.asarray(left, dtype=np.int64).ravel()
    right = np.asarray(right, dtype=np.int64).ravel()
    if left.shape != right.shape:
        raise ValueError(f"left {left.shape} and right {right.shape} must have equal shape")
    if left.size == 0:
        return np.empty(0, dtype=np.int64)

    if left.min(initial=0) < 0 or right.min(initial=0) < 0:
        raise ValueError("record indices must be non-negative")
    if left.max(initial=-1) >= n or right.max(initial=-1) >= n:
        raise ValueError(f"record index out of range for n={n}")

    low = np.minimum(left, right)
    high = np.maximum(left, right)
    keep = low != high
    return np.unique(low[keep] * n + high[keep])


def pack_block(members: np.ndarray, n: int) -> np.ndarray:
    """Every pair within one block, packed.

    A block of size `m` yields `m*(m-1)/2` pairs, which is why blockers cap
    block size rather than letting one runaway bucket dominate the candidate
    set -- a single block of 5,000 records is 12.5M pairs on its own.
    """
    members = np.asarray(members, dtype=np.int64).ravel()
    if members.size < 2:
        return np.empty(0, dtype=np.int64)
    left, right = np.triu_indices(members.size, k=1)
    return pack(members[left], members[right], n)


def unpack(keys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `pack`: keys -> (left, right) index arrays with left < right."""
    _check_n(n)
    keys = np.asarray(keys, dtype=np.int64).ravel()
    if n == 0:
        if keys.size:
            raise ValueError("cannot unpack a non-empty key array with n=0")
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return keys // n, keys % n


def union(*key_arrays: np.ndarray) -> np.ndarray:
    """Merge candidate sets from several blockers, deduplicated.

    This is the operation the whole package exists for: blockers are expected
    to be individually mediocre and to miss *different* true pairs, so the
    union's pair completeness is the number the rest of the pipeline inherits.
    """
    arrays = [np.asarray(a, dtype=np.int64).ravel() for a in key_arrays if np.size(a)]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays))


def total_pairs(n: int) -> int:
    """N*(N-1)/2 -- the denominator of the reduction ratio."""
    _check_n(n)
    return n * (n - 1) // 2
