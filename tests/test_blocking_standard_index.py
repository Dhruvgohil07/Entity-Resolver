"""Tests for `InvertedIndex`: service/'s online-lookup half of the two
corpus-independent `standard.py` key functions.
"""

import json

import pytest

from dedup.blocking.standard_index import InvertedIndex
from dedup.normalize import normalize
from dedup.schema import Record


def rec(record_id: str, title: str):
    return normalize(Record(record_id=record_id, source="synthetic", title=title))


CATALOG = [
    rec("s:1", "Panasonic 2-Line Integrated Telephone System - KXTS208W"),
    rec("s:2", "Canon PowerShot SD1100 Digital Camera"),
    rec("s:3", "Cuisinart Food Processor DLC2009CHB"),
]
IDS = [r.raw.record_id for r in CATALOG]


def test_query_one_finds_a_shared_model_number_key():
    index = InvertedIndex.build(CATALOG, IDS)
    variant = rec("s:1b", "Panasonic KX-TS208W Corded Phone")
    assert "s:1" in index.query_one(variant)


def test_query_one_finds_a_shared_code_token_not_the_extracted_model_number():
    # code_token_keys indexes every code-shaped token, not just the one
    # extraction commits to -- the reason it exists at all (CLAUDE.md).
    index = InvertedIndex.build(CATALOG, IDS)
    variant = rec("s:2b", "Canon SD1100 IS PowerShot Camera Silver")
    assert "s:2" in index.query_one(variant)


def test_query_one_on_something_sharing_nothing_finds_nothing():
    index = InvertedIndex.build(CATALOG, IDS)
    assert index.query_one(rec("s:9", "unrelated gadget item")) == set()


def test_add_makes_a_record_findable_by_later_queries():
    # A code-shaped token needs both a digit and a letter (standard.py's own
    # rule) -- a pure-digit SKU like "3841001" qualifies for neither key
    # function, so the fixture uses an alphanumeric code instead.
    index = InvertedIndex.build(CATALOG, IDS)
    new_record = rec("s:4", "Weber Genesis Gas Grill - GNS320LP")
    index.add(new_record, "s:4")

    variant = rec("s:4b", "Weber GNS320LP Genesis Grill")
    assert "s:4" in index.query_one(variant)


def test_records_and_record_ids_must_match_in_length():
    with pytest.raises(ValueError, match="must match"):
        InvertedIndex.build(CATALOG, IDS[:-1])


def test_save_and_load_round_trip_identical_query_results(tmp_path):
    index = InvertedIndex.build(CATALOG, IDS)
    variant = rec("s:1b", "Panasonic KX-TS208W Corded Phone")
    before = index.query_one(variant)

    index.save(tmp_path)
    loaded = InvertedIndex.load(tmp_path)
    after = loaded.query_one(variant)

    assert before == after == {"s:1"}


def test_loading_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="InvertedIndex"):
        InvertedIndex.load(tmp_path)


def test_a_tampered_index_file_is_refused(tmp_path):
    # The hash is content-based (json.dumps of the parsed dict), so tampering
    # has to change the *parsed* structure, not just add bytes a JSON parser
    # would ignore (e.g. trailing whitespace) -- inject a bogus key/value.
    index = InvertedIndex.build(CATALOG, IDS)
    index.save(tmp_path)
    path = tmp_path / "model_number.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["not-a-real-key"] = ["s:99"]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the hash"):
        InvertedIndex.load(tmp_path)
