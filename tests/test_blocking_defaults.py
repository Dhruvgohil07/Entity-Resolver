"""Tests for the shared blocker set and runner every later stage inherits.

`default_blocker_set` is the only place `ann_neighbours` and `lsh_max_neighbours`
get threaded into the blockers CLAUDE.md's committed reports were measured with.
A scale-driven parameter changing its literal default would move every one of
those reports silently -- exactly the failure mode `ann_components` already
guards against, and the same guard this file pins for its two new siblings.
"""

import numpy as np

from dedup.blocking.ann import DEFAULT_NEIGHBOURS as ANN_DEFAULT_NEIGHBOURS
from dedup.blocking.ann import AnnBlocker
from dedup.blocking.defaults import block_split, default_blocker_set
from dedup.blocking.lsh import MinHashLSHBlocker
from dedup.blocking.union import union_run
from dedup.normalize import normalize
from dedup.schema import Record


def rec(record_id: str, title: str, entity_id: str = "synthetic:e1"):
    return normalize(
        Record(record_id=record_id, source="synthetic", entity_id=entity_id, title=title)
    )


def _one(blockers, blocker_type):
    matches = [b for b in blockers if isinstance(b, blocker_type)]
    assert len(matches) == 1, f"expected exactly one {blocker_type.__name__}, got {len(matches)}"
    return matches[0]


def test_the_literal_defaults_reproduce_every_committed_report():
    # No committed report was ever measured with a projected ann k or a
    # bounded LSH -- both must stay unset unless a caller asks explicitly.
    blockers = default_blocker_set()
    ann = _one(blockers, AnnBlocker)
    lsh = _one(blockers, MinHashLSHBlocker)
    assert f"k={ANN_DEFAULT_NEIGHBOURS}" in ann.params
    assert "cap=" not in lsh.params


def test_ann_neighbours_threads_through():
    blockers = default_blocker_set(ann_neighbours=50)
    ann = _one(blockers, AnnBlocker)
    assert "k=50" in ann.params


def test_lsh_max_neighbours_threads_through():
    blockers = default_blocker_set(lsh_max_neighbours=100)
    lsh = _one(blockers, MinHashLSHBlocker)
    assert "cap=100" in lsh.params


def test_block_split_runs_the_same_set_default_blocker_set_returns():
    # block_split has no override parameters of its own -- it exists to keep
    # every downstream stage (features/, model/) on the literal defaults, not
    # a second, independently maintained blocker list that could drift from
    # default_blocker_set's.
    records = [
        rec("s:1", "widget deluxe model xy123z stainless", "e1"),
        rec("s:2", "widget deluxe xy123z model stainless v2", "e1"),
        rec("s:3", "unrelated gadget item", "e2"),
    ]
    run, score = block_split(records)
    manual = union_run([blocker.run(records) for blocker in default_blocker_set()])
    assert np.array_equal(run.keys, manual.keys)
    assert score.n_candidates == run.n_candidates
