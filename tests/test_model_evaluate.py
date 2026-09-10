"""Tests for the model report.

The unit tests here guard the reporting path, which is where `features/` was
last caught getting an invariant wrong one layer above the place it was being
checked. The integration tests at the bottom re-derive the published headline
against the real benchmark and skip when it has not been downloaded — so a green
run is not by itself evidence that any published number was reproduced.
`pytest -rs` reports the skips.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dedup.data.abt_buy import load_abt_buy
from dedup.eval.metrics import ThresholdPoint
from dedup.model.calibrate import OutOfFoldScores
from dedup.model.evaluate import (
    BASELINE_ORACLE_F1,
    BASELINE_TEST_F1,
    BASELINE_TEST_P_AT_K,
    BASELINE_TEST_PR_AUC,
    BASELINE_TEST_PRECISION,
    BASELINE_TEST_R_PRECISION,
    BASELINE_TEST_RECALL,
    BASELINE_THRESHOLD,
    SENSITIVITY_RATIOS,
    CalibrationStats,
    ModelReport,
    SplitStats,
    evaluate,
    render_markdown,
)
from dedup.model.threshold import BandSummary, CostModel

REAL_DATA = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"
BASELINE_REPORT = Path(__file__).parent.parent / "reports" / "baseline_tfidf.md"


def flat(markdown):
    """Collapse whitespace so a phrase the report wraps across lines still matches."""
    return " ".join(markdown.split())


def split_stats(found=50, total=50):
    return SplitStats(
        name="test",
        n_records=100,
        n_candidates=1000,
        n_positives_found=found,
        n_positives_total=total,
        pair_completeness=found / total,
        reduction_ratio=0.80,
    )


def summary(cost, n_review=5, n_missed=6):
    return BandSummary(
        cost=cost,
        n_pairs=1000,
        n_positives_total=50,
        n_auto_merge=40,
        n_review=n_review,
        n_auto_reject=960 - n_review,
        n_false_merge=1,
        n_missed=n_missed,
        n_review_positives=3,
        realized_cost=0.0,
    )


def synthetic_report(**changes):
    """A small hand-built report that exercises the rendering path with no data.

    The real-data report only ever reaches the branches Abt-Buy happens to take
    -- test PC 1.0000, a sensitivity table that barely moves, mostly confident
    misses -- so every other branch of "Reading this honestly" is invisible to
    it. Its defaults take the quiet branches; each test flips one condition.
    """
    point = ThresholdPoint(threshold=0.5, precision=0.9, recall=0.8, f1=0.8471)
    calibration = CalibrationStats(
        brier=0.01, ece=0.005, pr_auc=0.9, bins=[(0.0, 0.01, 0.01, 950), (0.9333, 0.97, 0.96, 50)]
    )
    base = ModelReport(
        dataset="synthetic",
        seed=0,
        test_fraction=0.3,
        n_folds=5,
        include_semantic=False,
        cost=CostModel(),
        n_records=300,
        n_entities=150,
        n_true_pairs=150,
        train=replace(split_stats(100, 100), name="train", n_records=200, n_candidates=3000),
        test=split_stats(),
        out_of_fold=OutOfFoldScores(
            scores=np.array([0.9, 0.1, 0.2]),
            labels=np.array([True, False, False]),
            n_folds=5,
            fold_candidates=(1, 1, 1),
            fold_positives=(1, 0, 0),
        ),
        threshold_point=point,
        test_point=point,
        test_oracle=point,
        test_precision_at_k={10: 1.0, 100: 0.5},
        test_r_precision=0.8,
        raw=calibration,
        calibrated=calibration,
        bands=summary(CostModel()),
        sensitivity=[
            summary(CostModel(false_merge=fm, false_split=fs)) for fm, fs in SENSITIVITY_RATIOS
        ],
        importance=[("code_token_jaccard", 100.0), ("desc_len_ratio", 50.0)],
        n_missed_confidently=5,
    )
    return replace(base, **changes)


# ---------------------------------------------------------------------------
# the sensitivity grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("false_merge", "false_split"), SENSITIVITY_RATIOS)
def test_every_sensitivity_ratio_leaves_a_review_band(false_merge, false_split):
    """A ratio that inverts the band would raise mid-report instead of rendering."""
    cost = CostModel(false_merge=false_merge, false_split=false_split)
    assert cost.auto_reject_threshold < cost.auto_merge_threshold


def test_the_sensitivity_grid_spans_an_order_of_magnitude():
    """The grid exists to show how little the ratio moves this dataset.

    A grid that only covered 20-25 would demonstrate nothing.
    """
    merges = [false_merge for false_merge, _ in SENSITIVITY_RATIOS]
    assert max(merges) / min(merges) >= 10


# ---------------------------------------------------------------------------
# against the real benchmark (skipped unless downloaded)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    if not REAL_DATA.is_dir():
        pytest.skip("Abt-Buy not downloaded; see CLAUDE.md > Data")
    return evaluate(load_abt_buy(REAL_DATA), dataset="abt-buy")


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_model_beats_the_published_baseline(report):
    """The claim the whole stage exists to make, on the baseline's own protocol."""
    assert report.test_point.f1 > BASELINE_TEST_F1
    # Guards against a leak as much as a regression: the probe that motivated
    # this stage landed near 0.89, and a number far above it would mean test
    # data reached the model rather than that the model improved.
    assert 0.80 < report.test_point.f1 < 0.95


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_threshold_is_chosen_off_in_sample_scores(report):
    """In-sample train scores would push the chosen threshold to the top.

    The threshold comes from out-of-fold predictions instead, so it sits at a
    plausible operating point rather than against the ceiling.
    """
    assert report.threshold_point.threshold < 0.99


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_calibration_does_not_change_the_ranking(report):
    """Platt is monotone, so PR-AUC must survive it untouched."""
    assert report.calibrated.pr_auc == pytest.approx(report.raw.pr_auc)


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_calibration_improves_the_calibration_error(report):
    """The reason the stage exists: the cost model consumes a probability."""
    assert report.calibrated.ece < report.raw.ece


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_calibrator_sees_the_measured_777_positives(report):
    """The measured reason for out-of-fold: 777 positives against a held-out split's 230.

    Pinned as the published number, not as a structural guarantee. Each fold is
    blocked as its own catalog, so the out-of-fold set holds the positives
    *per-fold* blocking found -- all 777 here, although blocking the whole train
    side at once finds 773. On a harder dataset the two can differ, and that
    would be a blocking result rather than a calibration bug.
    """
    oof = report.out_of_fold
    assert oof.n_positives == 777
    assert oof.n_positives == sum(oof.fold_positives)
    assert oof.n_positives <= report.train.n_positives_total
    assert oof.n_folds == 5


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_recall_is_divided_by_every_true_pair_in_the_split(report):
    """Not by the pairs blocking emitted, which pruning would otherwise flatter."""
    assert report.test.n_positives_total == 341
    assert report.bands.n_positives_total == report.test.n_positives_total
    # Whether blocking lost any test pairs is a separate question, pinned by
    # test_blocking_loss_is_measured_not_assumed; this one is the denominator.


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_bands_are_precision_heavy_by_construction(report):
    """The cost model prices a false merge at 20 reviews; it should show."""
    bands = report.bands
    assert bands.auto_merge_precision > bands.auto_merge_recall
    assert bands.n_false_merge < bands.n_missed
    assert bands.n_review < bands.n_auto_merge


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_auto_rejected_positives_are_confident_errors_not_near_misses(report):
    """What the report claims about where the lost recall went.

    Most of the auto-rejected true pairs score below 0.01 — the model is
    confidently wrong about them, so no threshold recovers them. This is the
    finding that points the next stage at `error-analyst` rather than at tuning.
    """
    assert report.n_missed_confidently > report.bands.n_missed / 2


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_vendor_code_columns_lead_the_gain_ranking(report):
    """The same claim `reports/features.md` makes univariately, on gain instead.

    If the code columns ever leave the top, the feature is broken rather than
    the claim being wrong.
    """
    top = [name for name, _ in report.importance[:5]]
    assert any("code" in name or "model_number" in name for name in top)


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_markdown_states_both_operating_points_and_the_caveats(report):
    markdown = flat(render_markdown(report))

    assert "auto-merge" in markdown and "auto-reject" in markdown and "review" in markdown
    assert str(BASELINE_TEST_F1) in markdown
    assert f"{report.test_point.f1:.4f}" in markdown
    # The honesty section must survive edits to the report.
    assert "not comparable to the baseline's; recall is" in markdown
    assert "not validated by this dataset" in markdown
    assert str(report.cost) in markdown


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_sensitivity_table_barely_moves_on_this_dataset(report):
    """The recorded reason the cost ratio cannot be fitted here.

    The calibrated distribution is bimodal, so an order of magnitude in `C_fm`
    changes the review queue by a few dozen pairs out of ~19k.
    """
    queues = [summary.n_review for summary in report.sensitivity]
    assert max(queues) - min(queues) < 0.01 * report.test.n_candidates


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_blocking_runs_inside_a_split(report):
    """Cross-split pairs are negatives by construction and must never be formed."""
    n_test = report.test.n_records
    assert report.test.n_candidates < n_test * (n_test - 1) // 2
    assert report.train.n_records + report.test.n_records == report.n_records


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_probabilities_stay_in_the_unit_interval(report):
    """Otherwise `assign_bands` would have rejected them, but say so here too."""
    for _, predicted, observed, _ in report.calibrated.bins:
        assert 0.0 <= predicted <= 1.0
        assert 0.0 <= observed <= 1.0
    assert np.isclose(sum(count for *_, count in report.calibrated.bins), report.test.n_candidates)


# ---------------------------------------------------------------------------
# Every interpretive sentence follows the numbers.
#
# On the real report: the branch Abt-Buy takes, and that it takes it for the
# measured reason. On the synthetic report: every other branch, which no
# dataset here reaches -- and an unreachable branch in a reporting path is
# exactly where an overclaim survives.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_blocking_loss_is_measured_not_assumed(report):
    assert report.test.n_blocking_missed == 0
    assert report.test.n_positives_found == report.test.n_positives_total
    text = flat(render_markdown(report))
    assert "Blocking emitted every one of them" in text
    assert "does not bind" in text


def test_the_synthetic_report_takes_the_quiet_branches_by_default():
    text = flat(render_markdown(synthetic_report()))
    assert "on the test split it does not bind" in text
    assert "Blocking emitted every one of them" in text
    assert "not validated by this dataset" in text
    assert "confidently wrong, not undecided" in text
    assert "`desc_len_ratio` ranks #2 on gain" in text


def test_a_split_where_blocking_loses_pairs_says_so_everywhere():
    """Both sentences that depend on blocking must flip together.

    `BandSummary.n_missed` counts auto-rejected *candidates* only, so a true
    pair no blocker emitted is in no band. The report has to name that loss,
    and the ceiling sentence must stop claiming the ceiling does not bind.
    """
    text = flat(render_markdown(synthetic_report(test=split_stats(found=41, total=50))))

    assert "does not bind" not in text
    assert "on the test split it binds" in text
    assert "capped at 0.8200" in text
    assert "Blocking emitted every one of them" not in text
    assert "9 true pairs never reached the model at all" in text
    assert "on top of 9 that blocking never emitted" in text


def test_a_ratio_that_moves_the_queue_is_called_load_bearing():
    report = synthetic_report()
    swung = replace(
        report,
        sensitivity=[
            replace(band, n_review=band.n_review + 100 * i)
            for i, band in enumerate(report.sensitivity)
        ],
    )
    text = flat(render_markdown(swung))

    assert "not validated by this dataset" not in text
    assert "load-bearing on this dataset" in text


def test_near_miss_rejections_are_not_called_confident_errors():
    """If most lost pairs sit just below `p_lo`, a threshold *is* the lever."""
    text = flat(render_markdown(synthetic_report(n_missed_confidently=1)))

    assert "confidently wrong" not in text
    assert "not a threshold problem" not in text
    assert "near misses" in text
    assert "higher `C_fs` would route them to review" in text


def test_no_auto_rejected_pair_is_stated_rather_than_explained():
    report = synthetic_report()
    clean = replace(report, bands=replace(report.bands, n_missed=0), n_missed_confidently=0)
    text = flat(render_markdown(clean))

    assert "No true pair that blocking emitted was auto-rejected" in text
    assert "real loss" not in text


def test_the_desc_len_ratio_note_appears_only_when_the_column_ranks():
    text = flat(render_markdown(synthetic_report(importance=[("code_token_jaccard", 100.0)])))
    assert "desc_len_ratio" not in text


# ---------------------------------------------------------------------------
# The baseline row is restated, not recomputed -- so it is checked against the
# committed report instead. Needs no dataset: reports/ is committed.
# ---------------------------------------------------------------------------


def test_the_restated_baseline_row_matches_the_committed_report():
    row = next(
        line
        for line in BASELINE_REPORT.read_text(encoding="utf-8").splitlines()
        if line.startswith("| title |")
    )
    cells = [cell.strip().strip("*") for cell in row.strip().strip("|").split("|")]
    _, f1, precision, recall, pr_auc, p10, p100, r_prec, threshold, oracle = cells

    assert float(f1) == BASELINE_TEST_F1
    assert float(precision) == BASELINE_TEST_PRECISION
    assert float(recall) == BASELINE_TEST_RECALL
    assert float(pr_auc) == BASELINE_TEST_PR_AUC
    assert float(p10) == BASELINE_TEST_P_AT_K[10]
    assert float(p100) == BASELINE_TEST_P_AT_K[100]
    assert float(r_prec) == BASELINE_TEST_R_PRECISION
    assert float(threshold) == BASELINE_THRESHOLD
    assert float(oracle) == BASELINE_ORACLE_F1


def test_the_baseline_row_has_no_blank_cells():
    """A dash in a comparison row reads as "not measured" when it was measured."""
    text = render_markdown(synthetic_report())
    row = next(line for line in text.splitlines() if line.startswith("| baseline (TF-IDF) |"))
    assert "—" not in row
