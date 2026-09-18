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
from dedup.blocking.defaults import block_split, block_unlabeled, default_blocker_set
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


def test_block_unlabeled_runs_on_records_with_no_entity_id():
    # A genuine batch catalog -- entity_id=None on every record, per schema.py's
    # own docstring, because nobody knows the answer yet. block_split cannot
    # score this: ground_truth raises on an unlabeled record. This is the
    # entry point that actually can.
    records = [
        rec("s:1", "widget deluxe model xy123z stainless", entity_id=None),
        rec("s:2", "widget deluxe xy123z model stainless v2", entity_id=None),
        rec("s:3", "unrelated gadget item", entity_id=None),
    ]
    run = block_unlabeled(records)
    assert run.n_candidates > 0


def test_block_unlabeled_is_block_splits_labels_free_twin():
    # Same records, once with entity_id set (so block_split can run) and once
    # without (the shape block_unlabeled actually sees) -- the candidate set
    # must be identical, or the two blocker lists have silently diverged.
    labelled = [
        rec("s:1", "widget deluxe model xy123z stainless", "e1"),
        rec("s:2", "widget deluxe xy123z model stainless v2", "e1"),
        rec("s:3", "unrelated gadget item", "e2"),
    ]
    unlabelled = [normalize(r.raw.model_copy(update={"entity_id": None})) for r in labelled]

    split_run, _ = block_split(labelled)
    unlabeled_run = block_unlabeled(unlabelled)
    assert np.array_equal(split_run.keys, unlabeled_run.keys)
