"""Tests for synthetic catalogs on disk, and the CLI that writes them.

The loader's job is to refuse. A catalog that no longer matches its manifest, or
that carries a record without the split group keeping a seed's siblings together,
would load silently and produce reports on a dataset nobody measured.
"""

import json
from pathlib import Path

import pytest

from dedup.data import DATASETS, load_dataset
from dedup.data.abt_buy import load_abt_buy
from dedup.data.synthetic import (
    MANIFEST_FILE,
    RECORDS_FILE,
    load_synthetic,
    read_manifest,
    write_catalog,
)
from dedup.schema import Record
from dedup.synth.generate import main, verify_seed_provenance

FIXTURES = Path(__file__).parent / "fixtures" / "abt-buy"
REAL_ABT_BUY = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"
SYNTH_20K = Path(__file__).parent.parent / "data" / "synth" / "abt-buy-train-20k"
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
        assert spec.load.func is load_synthetic
        assert spec.load.keywords == {"seed_dataset": "abt-buy"}
        assert spec.default_root.parts[:2] == ("data", "synth")
    # The 200k catalog is past the exhaustive baseline, so it claims none.
    assert DATASETS["synth-200k"].notes.baseline_report is None


def test_a_catalog_seeded_from_another_dataset_is_refused(tmp_path):
    """Nothing tied a registry entry to the benchmark it claims to be seeded from --
    a catalog seeded from a different dataset than the registry expects would have
    loaded silently. Registered entries bind seed_dataset="abt-buy" via functools.partial."""
    write_catalog(tmp_path, [synthetic("synthetic:f1:e000:r0")], {**SEEDED, "seed_dataset": "amazon-google"})
    with pytest.raises(ValueError, match="not 'abt-buy'"):
        load_synthetic(tmp_path, seed_dataset="abt-buy")

    # Without an expectation to check against, the same catalog loads -- the check
    # is opt-in per caller, not a property of the file itself.
    assert load_synthetic(tmp_path)


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


@pytest.mark.skipif(
    not (REAL_ABT_BUY.is_dir() and SYNTH_20K.is_dir()),
    reason="Abt-Buy or the synth-20k catalog is not present; see CLAUDE.md > Data / Commands",
)
def test_the_committed_synth_20k_catalog_really_is_seeded_from_train():
    """The non-tautological check: independent of any one `generate.py` run, reload
    Abt-Buy fresh, re-derive its split with the code as it exists right now, and
    check the *committed* catalog's embedded seed entities against that -- not
    against what its own manifest claims. This is what closes the audit gap for the
    catalogs actually shipped in reports/synth/, not just for a freshly-generated one."""
    seed_records = load_abt_buy(REAL_ABT_BUY)
    catalog_records = load_dataset("synth-20k")
    verify_seed_provenance(catalog_records, seed_records)  # must not raise
