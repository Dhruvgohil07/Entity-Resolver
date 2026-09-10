"""Tests for the name -> loader registry.

This code had no test until `model/` became its fourth CLI consumer. Nothing
under `tests/` imported it — every other test calls `dedup.data.abt_buy`
directly — so the registry was exercised only by the CLIs and would not have
failed a test run if it broke. Both error paths below had been checked by hand
and never pinned.

The registry is what keeps CLAUDE.md's rule that **`data/` is the only place
that may know a dataset's name**: a CLI resolves `--dataset abt-buy` here and
passes `Record` objects onward, so no later stage imports a loader module.
`test_the_registry_is_the_only_place_a_dataset_name_appears` is the one that
actually guards the rule rather than the plumbing.
"""

from pathlib import Path

import pytest

from dedup.data import DATASETS, DatasetSpec, load_dataset

FIXTURES = Path(__file__).parent / "fixtures" / "abt-buy"
SOURCE_ROOT = Path(__file__).parent.parent / "src" / "dedup"


def test_every_registered_spec_is_self_consistent():
    for name, spec in DATASETS.items():
        assert isinstance(spec, DatasetSpec)
        assert spec.name == name, "the key and the spec's own name must agree"
        assert callable(spec.load)
        assert isinstance(spec.default_root, Path)


def test_abt_buy_is_registered():
    assert "abt-buy" in DATASETS


def test_an_unknown_name_lists_what_is_known():
    """The error has to be actionable -- a bare KeyError names nothing."""
    with pytest.raises(KeyError) as excinfo:
        load_dataset("amazon-google")
    message = str(excinfo.value)
    assert "amazon-google" in message
    assert "abt-buy" in message


def test_a_missing_directory_explains_that_data_is_gitignored():
    """The first thing a fresh clone hits, so it must not be a bare OSError."""
    with pytest.raises(FileNotFoundError) as excinfo:
        load_dataset("abt-buy", Path("data/raw/does-not-exist"))
    message = str(excinfo.value)
    assert "gitignored" in message
    assert "CLAUDE.md" in message


def test_an_explicit_root_overrides_the_default():
    """How every test and CLI reaches the committed fixtures."""
    records = load_dataset("abt-buy", FIXTURES)
    assert records
    assert all(record.entity_id for record in records), "ground truth must be populated"
    assert all(record.record_id.startswith("abt_buy:") for record in records)


def test_the_loader_returns_records_carrying_no_dataset_name():
    """Downstream stages see `Record` objects and nothing else."""
    records = load_dataset("abt-buy", FIXTURES)
    assert {record.source for record in records} == {"abt_buy"}
    # `source` is a record property, not a routing key: no stage after data/
    # is allowed to branch on it, which is what makes one blocker or feature
    # work unchanged across benchmarks.
    assert all(record.title for record in records)


def test_the_registry_is_the_only_place_a_dataset_name_appears():
    """CLAUDE.md: a per-source branch downstream is a bug, not a shortcut.

    Walks the real tree rather than a hand-listed set of modules, so a stage
    added later inherits the check by existing.
    """
    offenders = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path.parent.name == "data" or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "abt" not in line.lower():
                continue
            # Docstrings, comments and argparse defaults may name a benchmark;
            # importing its loader is what the rule forbids.
            if "import" in line and "abt" in line.lower():
                offenders.append(f"{path.relative_to(SOURCE_ROOT)}: {stripped}")
    assert not offenders, "stages after data/ must not import a dataset loader:\n" + "\n".join(
        offenders
    )
