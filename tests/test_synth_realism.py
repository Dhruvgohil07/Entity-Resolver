"""Tests for the difficulty statistics synthetic output is calibrated against.

Checked on catalogs small enough to compute by hand, because a wrong statistic
here would calibrate the generator to the wrong target and the realism report
would agree with itself.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from dedup.data.abt_buy import load_abt_buy
from dedup.eval.splits import split_by_entity
from dedup.schema import Record
from dedup.synth.generate import SynthConfig, generate
from dedup.synth.realism import (
    CALIBRATION_TOLERANCE,
    calibration_gaps,
    pair_statistics,
    render_realism,
)

REAL_DATA = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"


def rec(record_id, entity, title, **fields):
    return Record(record_id=record_id, source="synthetic", entity_id=entity, title=title, **fields)


def test_pair_statistics_on_a_hand_computed_catalog():
    stats = pair_statistics(
        [
            rec("a1", "e1", "Sony Turntable - PSLX350H", price=100.0, brand="Sony"),
            rec("a2", "e1", "Sony PSLX350H Belt Turntable", price=80.0, brand="Sony"),
            rec("b1", "e2", "Canon Camera - SD1100"),
            rec("b2", "e2", "Canon Camera Silver"),
            rec("c1", "e3", "Netgear Router"),
        ]
    )
    assert (stats.n_records, stats.n_entities, stats.n_true_pairs) == (5, 3, 2)
    assert stats.entity_sizes == {1: 1, 2: 2}
    # {sony, turntable, -, pslx350h} against {sony, pslx350h, belt, turntable}: 3 of 5.
    # {canon, camera, -, sd1100} against {canon, camera, silver}: 2 of 5.
    assert stats.title_jaccard.mean == pytest.approx(0.5)
    assert (stats.code_equal, stats.code_one_missing) == (0.5, 0.5)
    assert (stats.code_both_missing, stats.code_differ) == (0.0, 0.0)
    assert stats.price_both_present == 0.5
    assert stats.price_gap.p50 == pytest.approx(0.2)  # |100 - 80| / 100
    assert stats.brand_both_present == 0.5
    assert stats.description_empty == 1.0


def test_two_listings_printing_different_codes_count_as_differ_not_missing():
    stats = pair_statistics(
        [rec("a", "e1", "Canon Ink Tank - CL41CL"), rec("b", "e1", "Canon Ink Cartridge - 0617B002")]
    )
    assert (stats.code_differ, stats.code_equal, stats.code_one_missing) == (1.0, 0.0, 0.0)


def test_sibling_rate_counts_same_brand_codes_within_two_edits():
    stats = pair_statistics(
        [
            rec("a", "e1", "Nikon Flash - SB600"),
            rec("b", "e2", "Nikon Flash - SB900"),  # one edit from e1
            rec("c", "e3", "Nikon Lens - AF50MM"),  # same brand, far away
            rec("d", "e4", "Canon Flash - SB700"),  # close code, different brand
        ]
    )
    assert stats.sibling_rate == pytest.approx(2 / 4)


def test_a_catalog_without_codes_has_no_sibling_rate_and_it_can_be_skipped():
    records = [rec("a", "e1", "Netgear Router"), rec("b", "e1", "Netgear Wireless Router")]
    assert pair_statistics(records).sibling_rate is None
    coded = [rec("a", "e1", "Nikon Flash - SB600"), rec("b", "e2", "Nikon Flash - SB900")]
    assert pair_statistics(coded, siblings=False).sibling_rate is None


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

MANIFEST = {
    "seed_dataset": "abt-buy",
    "seed_split": {"side": "train", "test_fraction": 0.3, "seed": 0},
    "n_families": 2,
    "n_dropped_siblings": 0,
    "records_sha256": "0" * 64,
    "config": {"target_records": 4, "seed": 0},
}


def flat(markdown):
    return " ".join(markdown.split())


def small_catalog():
    return pair_statistics(
        [
            rec("a1", "e1", "Sony Turntable - PSLX350H", price=100.0),
            rec("a2", "e1", "Sony PSLX350H Belt Turntable", price=80.0),
            rec("b1", "e2", "Canon Camera - SD1100"),
            rec("b2", "e2", "Canon Camera Silver"),
        ]
    )


def test_a_catalog_matching_its_seeds_is_called_within_tolerance():
    stats = small_catalog()
    text = flat(render_realism(stats, stats, MANIFEST, catalog="data/synth/x", out="reports/synth/r.md"))
    assert "Every calibrated statistic is within ±0.05 of the seeds" in text
    assert "--out data/synth/x --report reports/synth/r.md" in text
    assert "their test split never reached this catalog" in text
    assert "outside" not in text


def test_the_provenance_block_does_not_track_the_report_path_it_is_rendered_to():
    """The regression: under `--report-only` the "it was written by" block was
    built from the *current* `--report`, so re-rendering to a scratch path made
    the report claim the catalog had been generated with a `--report` that never
    ran. Provenance that changes when you read it is not provenance. The
    historical command names no report at all now -- a report path is no part of
    writing a catalog -- so the block is identical whatever it is rendered to.

    The re-render command above it is the opposite case and *must* track the
    path, since it is a command to run that writes this report."""
    stats = small_catalog()
    kwargs = {"catalog": "data/synth/x", "report_only": True}
    committed = flat(render_realism(stats, stats, MANIFEST, out="reports/synth/r.md", **kwargs))
    scratch = flat(render_realism(stats, stats, MANIFEST, out="scratch.md", **kwargs))

    # `flat` collapses the fence onto the command, so the historical block reads
    # "... --out data/synth/x ```" -- the backticks are what pin that the command
    # ends at the catalog and never grew a --report.
    historical = "--records 4 --seed 0 --out data/synth/x ```"
    assert historical in committed
    assert historical in scratch

    # The re-render command is the one thing that may differ between the two.
    assert "--out data/synth/x --report reports/synth/r.md --report-only" in committed
    assert "--out data/synth/x --report scratch.md --report-only" in scratch


def test_a_catalog_off_target_names_what_is_outside_tolerance():
    seeds = small_catalog()
    synthetic = replace(seeds, code_equal=seeds.code_equal - 0.2)
    text = flat(render_realism(seeds, synthetic, MANIFEST, catalog="c", out="o"))
    assert "Every calibrated statistic is within" not in text
    assert "1 calibrated statistic(s) fall outside ±0.05 of the seeds" in text
    assert "code keys equal (-0.200)" in text


def test_a_catalog_seeded_from_another_split_is_not_said_to_have_missed_its_test_records():
    stats = small_catalog()
    other = {**MANIFEST, "seed_split": {"side": "train", "test_fraction": 0.2, "seed": 0}}
    text = flat(render_realism(stats, stats, other, catalog="c", out="o"))
    assert "never reached this catalog" not in text


def test_structural_deviations_are_stated_only_when_measured():
    seeds = small_catalog()
    synthetic = replace(seeds, brand_both_present=0.4, entity_sizes={2: 1, 5: 1})
    text = flat(render_realism(seeds, synthetic, MANIFEST, catalog="c", out="o"))
    assert "by design" in text
    assert "Entities are larger than any the seeds have" in text
    assert "by design" not in flat(render_realism(seeds, seeds, MANIFEST, catalog="c", out="o"))


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_default_profile_matches_abt_buy_train_within_tolerance():
    """The calibration reports/synth/realism.md publishes, re-derived -- on the train split only."""
    train, _ = split_by_entity(load_abt_buy(REAL_DATA), test_fraction=0.3, seed=0)
    catalog = generate(train, SynthConfig(target_records=20000, seed=0))
    gaps = calibration_gaps(
        pair_statistics(train, siblings=False), pair_statistics(catalog.records, siblings=False)
    )
    assert gaps
    assert max(abs(gap) for gap in gaps.values()) <= CALIBRATION_TOLERANCE
    # One split group per seed family, so no family can straddle a split.
    assert len({r.split_group for r in catalog.records}) == len({r.entity_id for r in train})
