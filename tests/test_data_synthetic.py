"""Tests for synthetic catalogs on disk, and the CLI that writes them.

The loader's job is to refuse. A catalog that no longer matches its manifest, or
that carries a record without the split group keeping a seed's siblings together,
would load silently and produce reports on a dataset nobody measured.
"""

import json
from pathlib import Path

import pytest

from dedup.data import DATASETS, load_dataset
from dedup.data.synthetic import (
    MANIFEST_FILE,
    RECORDS_FILE,
    load_synthetic,
    read_manifest,
    write_catalog,
)
from dedup.schema import Record
from dedup.synth.generate import main

FIXTURES = Path(__file__).parent / "fixtures" / "abt-buy"
# A manifest seeded from the split every report is scored on -- the only one the
# loader accepts.
SEEDED = {"seed_split": {"side": "train", "test_fraction": 0.3, "seed": 0}}


def synthetic(record_id, **fields):
    base = {
        "record_id": record_id,
        "source": "synthetic",
        "entity_id": "synthetic:f1:e000",
        "split_group": "synthetic:f1",
        "title": "Sony Turntable - PSLX350H",
    }
    return Record(**{**base, **fields})


def test_none_and_empty_survive_the_round_trip(tmp_path):
    """The distinction CSV would collapse and features/missingness.py depends on."""
    records = [
        synthetic("synthetic:f1:e000:r0", description=None, brand=None),
        synthetic("synthetic:f1:e000:r1", description="", brand=""),
    ]
    write_catalog(tmp_path, records, {**SEEDED, "generator": "test"})
    loaded = load_synthetic(tmp_path)
    assert loaded == records
    assert loaded[0].description is None and loaded[1].description == ""


def test_the_same_records_write_the_same_bytes(tmp_path):
    records = [synthetic("synthetic:f1:e000:r0", price=9.99)]
    assert write_catalog(tmp_path / "a", records, SEEDED) == write_catalog(tmp_path / "b", records, SEEDED)
    assert (tmp_path / "a" / RECORDS_FILE).read_bytes() == (tmp_path / "b" / RECORDS_FILE).read_bytes()


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    (tmp_path / RECORDS_FILE).write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="dedup.synth.generate"):
        load_synthetic(tmp_path)


def test_records_that_no_longer_match_the_manifest_are_refused(tmp_path):
    write_catalog(tmp_path, [synthetic("synthetic:f1:e000:r0")], SEEDED)
    extra = synthetic("synthetic:f1:e000:r1").model_dump_json() + "\n"
    with (tmp_path / RECORDS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(extra)
    with pytest.raises(ValueError, match="does not match the hash"):
        load_synthetic(tmp_path)


def test_a_record_from_another_source_is_refused(tmp_path):
    write_catalog(tmp_path, [synthetic("abt_buy:abt:1", source="abt_buy")], SEEDED)
    with pytest.raises(ValueError, match="abt_buy"):
        load_synthetic(tmp_path)


def test_a_record_without_a_split_group_is_refused(tmp_path):
    write_catalog(tmp_path, [synthetic("synthetic:f1:e000:r0", split_group=None)], SEEDED)
    with pytest.raises(ValueError, match="split_group"):
        load_synthetic(tmp_path)


def test_a_catalog_seeded_from_another_split_is_refused(tmp_path):
    """At test_fraction 0.2 its seeds would include 216 of the 652 records reports test on."""
    other = {"seed_split": {"side": "train", "test_fraction": 0.2, "seed": 0}}
    write_catalog(tmp_path, [synthetic("synthetic:f1:e000:r0")], other)
    with pytest.raises(ValueError, match="split reports are scored on"):
        load_synthetic(tmp_path)


def test_both_catalogs_are_registered_as_synthetic():
    for name in ("synth-20k", "synth-200k"):
        spec = DATASETS[name]
        assert spec.load is load_synthetic
        assert spec.default_root.parts[:2] == ("data", "synth")
    # The 200k catalog is past the exhaustive baseline, so it claims none.
    assert DATASETS["synth-200k"].notes.baseline_report is None


def test_the_cli_seeds_from_the_train_split_and_writes_a_loadable_catalog(tmp_path):
    """The one report-producing CLI here with a test that calls its main()."""
    out = tmp_path / "catalog"
    assert main(["--root", str(FIXTURES), "--records", "40", "--out", str(out)]) == 0

    manifest = read_manifest(out)
    assert manifest["seed_dataset"] == "abt-buy"
    assert manifest["seed_split"] == {"side": "train", "test_fraction": 0.3, "seed": 0}
    records = load_synthetic(out)
    assert records and len(records) == manifest["n_records"]
    assert {r.raw_attributes["seed_entity_id"] for r in records} <= {
        r.entity_id for r in load_dataset("abt-buy", FIXTURES)
    }
    assert json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))["config"]["seed"] == 0
