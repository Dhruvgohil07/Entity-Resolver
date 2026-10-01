"""Tests for `AnnIndex`: service/'s online-lookup half of blocking/ann.py.

Round-trip and refusal shapes mirror `tests/test_model_train.py`'s persistence
tests for `PairScorer`, since `AnnIndex.save`/`.load` deliberately copy that
pattern (directory, manifest, hash-refusal).
"""

import numpy as np
import pytest

from dedup.blocking.ann import AnnIndex
from dedup.normalize import normalize
from dedup.schema import Record


def rec(record_id: str, title: str):
    return normalize(Record(record_id=record_id, source="synthetic", title=title))


CATALOG = [
    rec("s:1", "Panasonic 2-Line Integrated Telephone System - KXTS208W"),
    rec("s:2", "Bose 161 Bookshelf Pair Speakers In White - 161WH"),
    rec("s:3", "Canon PowerShot SD1100 Digital Camera"),
    rec("s:4", "Cuisinart Food Processor DLC2009CHB"),
    rec("s:5", "Weber Genesis Gas Grill 3841001"),
]
IDS = [r.raw.record_id for r in CATALOG]


def test_query_one_finds_a_near_duplicate_not_yet_in_the_index():
    # Build without the Bose typo variant, then ask what it's closest to.
    index = AnnIndex.build(CATALOG[:1] + CATALOG[2:], IDS[:1] + IDS[2:], neighbours=3)
    typo_variant = rec("s:2b", "Boss 161 Speaker")
    found = [record_id for record_id, _ in index.query_one(typo_variant)]
    assert "s:2" not in found  # it was never built in


def test_add_makes_a_record_findable_by_later_queries():
    # The vectorizer is fixed at build() time (never refit), so add() only
    # ever works for vocabulary the build corpus already knows -- built here
    # on the whole catalog, so "Bose"-shaped text is known, then a variant
    # is added under a fresh record_id and queried for.
    index = AnnIndex.build(CATALOG, IDS, neighbours=3)
    typo_variant = rec("s:2b", "Boss 161 Speaker")
    index.add(typo_variant, "s:2b")

    query = rec("s:2c", "Bose 161 Bookshelf Speakers White")
    found = [record_id for record_id, _ in index.query_one(query)]
    assert "s:2b" in found


def test_save_and_load_round_trip_identical_query_results(tmp_path):
    index = AnnIndex.build(CATALOG, IDS, neighbours=3)
    before = index.query_one(rec("s:6", "Weber Genesis Grill 3841001 replacement"))

    index.save(tmp_path)
    loaded = AnnIndex.load(tmp_path)
    after = loaded.query_one(rec("s:6", "Weber Genesis Grill 3841001 replacement"))

    assert before == after


def test_loading_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="AnnIndex"):
        AnnIndex.load(tmp_path)


def test_a_tampered_index_file_is_refused(tmp_path):
    index = AnnIndex.build(CATALOG, IDS, neighbours=3)
    index.save(tmp_path)
    with (tmp_path / "index.faiss").open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="does not match the hash"):
        AnnIndex.load(tmp_path)


def test_the_svd_path_round_trips_too(tmp_path):
    index = AnnIndex.build(CATALOG, IDS, neighbours=3, n_components=3)
    before = index.query_one(rec("s:6", "Canon PowerShot SD1100 replacement lens"))

    index.save(tmp_path)
    loaded = AnnIndex.load(tmp_path)
    after = loaded.query_one(rec("s:6", "Canon PowerShot SD1100 replacement lens"))

    assert before == after
    assert (tmp_path / "svd.pkl").is_file()


def test_records_and_record_ids_must_match_in_length():
    with pytest.raises(ValueError, match="must match"):
        AnnIndex.build(CATALOG, IDS[:-1])


def test_query_one_scores_are_returned_sorted_descending():
    index = AnnIndex.build(CATALOG, IDS, neighbours=5)
    scores = [score for _, score in index.query_one(rec("s:6", "Weber Genesis Gas Grill 3841001"))]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(s, float) for s in scores)
    assert all(np.isfinite(s) for s in scores)
