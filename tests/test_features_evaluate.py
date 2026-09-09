"""Tests for the per-feature diagnostic and its report.

Two of these guard metric traps CLAUDE.md names explicitly, and both would
produce a report that looks fine and is wrong:

  * `test_a_missed_positive_lowers_pr_auc` -- the recall denominator is every
    true pair in the split, not the ones blocking emitted. Divide by the
    survivors instead and pruning harder *improves* the reported number.
  * `test_a_distance_column_is_negated_before_ranking` -- a column declared as
    a distance must be flipped before it is ranked, or `price_abs_log_ratio`
    reports as useless and gets deleted for being strong.

The integration test at the bottom re-derives the published test-split pair
completeness against the real benchmark, and skips when it has not been
downloaded. `pytest -rs` reports the skip -- a green run is not by itself
evidence that the real files were checked.
"""

from pathlib import Path

import numpy as np
import pytest

from dedup.data.abt_buy import load_abt_buy
from dedup.features.base import FeatureSpec
from dedup.features.evaluate import (
    block,
    correlated_pairs,
    diagnose,
    evaluate,
    render_markdown,
)
from dedup.features.semantic import DEFAULT_MODEL, SemanticBlock, model_is_cached
from dedup.normalize import normalize

REAL_DATA = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"

SIMILARITY = FeatureSpec("sim", "doc")
DISTANCE = FeatureSpec("dist", "doc", higher_is_similar=False)


def covered(n):
    return np.ones(n, dtype=bool)


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


def test_a_perfectly_separating_column_scores_one():
    values = np.array([0.9, 0.8, 0.1, 0.0])
    labels = np.array([True, True, False, False])
    result = diagnose(values, labels, SIMILARITY, covered(4), n_positives_total=2)
    assert result.pr_auc == pytest.approx(1.0)


def test_a_distance_column_is_negated_before_ranking():
    """Low means similar, so the raw column ranks the positives last."""
    values = np.array([0.0, 0.1, 0.8, 0.9])
    labels = np.array([True, True, False, False])
    assert diagnose(values, labels, DISTANCE, covered(4), n_positives_total=2).pr_auc == (
        pytest.approx(1.0)
    )
    # The same numbers read as a similarity are the worst possible ranking.
    assert diagnose(values, labels, SIMILARITY, covered(4), n_positives_total=2).pr_auc < 0.6


def test_a_missed_positive_lowers_pr_auc():
    """The denominator is every true pair in the split, not the survivors.

    A candidate set holding 2 of 4 true pairs cannot score 1.0 however well
    it ranks the two it has.
    """
    values = np.array([0.9, 0.8, 0.1, 0.0])
    labels = np.array([True, True, False, False])
    complete = diagnose(values, labels, SIMILARITY, covered(4), n_positives_total=2)
    pruned = diagnose(values, labels, SIMILARITY, covered(4), n_positives_total=4)
    assert pruned.pr_auc < complete.pr_auc
    assert pruned.pr_auc == pytest.approx(0.5)


def test_class_means_ignore_the_fill():
    """Regression: an imputed pair must not be averaged into the class means.

    This is the missingness invariant one layer up from the vector, and the
    fixture reproduces `brand_equal`'s actual shape on Abt-Buy: the column
    points the right way wherever it is defined, but it is defined on a much
    smaller share of positives (1 of 4 here) than of negatives (4 of 4),
    because Abt carries no brand column and true pairs are almost all
    cross-source.

    Averaging the fill in makes the means 0.2500 against 0.5000 -- separation
    -0.25, so the column is flagged as pointing backwards. Over covered pairs
    they are 1.0000 against 0.5000, separation +0.5. The first reading is what
    put `brand_equal` in the report's "pointing the wrong way" section, and it
    was wrong.

    The `is_anti_predictive` assertion below fails against the pre-fix
    implementation, and that is deliberate: an earlier version of this test
    used a symmetric fixture on which the old code gave separation exactly
    0.0, so the same assertion passed either way and pinned nothing. A
    regression test that cannot fail is not one.
    """
    values = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    labels = np.array([True, True, True, True, False, False, False, False])
    covered = np.array([True, False, False, False, True, True, True, True])
    result = diagnose(values, labels, SIMILARITY, covered, n_positives_total=4)
    assert result.positive_mean == pytest.approx(1.0)
    assert result.negative_mean == pytest.approx(0.5)
    assert result.separation == pytest.approx(0.5)
    assert not result.is_anti_predictive
    assert result.n_covered_positives == 1
    assert result.n_covered == 5


def test_a_distance_columns_fill_is_not_ranked_as_perfect_agreement():
    """A fill of 0.0 negates to the top of a distance column's ranking.

    So an absent price would sort ahead of every pair with two real prices that
    actually agree. Restricting the ranking to covered pairs is what removes
    that; here the two imputed negatives must not displace the covered
    positive.
    """
    values = np.array([0.0, 0.0, 0.1, 5.0])
    labels = np.array([False, False, True, False])
    covered = np.array([False, False, True, True])
    result = diagnose(values, labels, DISTANCE, covered, n_positives_total=1)
    assert result.pr_auc == pytest.approx(1.0)


def test_coverage_comes_from_the_companion_indicator():
    values = np.array([0.5, 0.5, 0.0, 0.0])
    labels = np.array([True, False, True, False])
    result = diagnose(
        values, labels, SIMILARITY, np.array([True, True, False, False]), n_positives_total=2
    )
    assert result.coverage == pytest.approx(0.5)


def test_an_uncovered_column_reports_no_pr_auc():
    """A column with nothing to measure gets a dash, not a fabricated 0.0."""
    values = np.zeros(4)
    labels = np.array([True, False, True, False])
    result = diagnose(
        values, labels, SIMILARITY, np.zeros(4, dtype=bool), n_positives_total=2
    )
    assert result.coverage == 0.0
    assert result.pr_auc is None


def test_anti_predictive_detection_respects_the_declared_orientation():
    labels = np.array([True, True, False, False])
    backwards = np.array([0.1, 0.0, 0.9, 0.8])
    assert diagnose(
        backwards, labels, SIMILARITY, covered(4), n_positives_total=2
    ).is_anti_predictive
    # The identical numbers are correct for a column declared as a distance.
    assert not diagnose(
        backwards, labels, DISTANCE, covered(4), n_positives_total=2
    ).is_anti_predictive


def test_indicators_are_never_flagged_anti_predictive():
    """An indicator says whether a value exists; it makes no claim about
    which way duplication runs, so 'backwards' is meaningless for it."""
    indicator = FeatureSpec("ind", "doc", is_indicator=True)
    labels = np.array([True, True, False, False])
    values = np.array([0.0, 0.0, 1.0, 1.0])
    assert not diagnose(
        values, labels, indicator, covered(4), n_positives_total=2
    ).is_anti_predictive


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------


def test_a_duplicated_column_is_reported():
    values = np.array([[1.0, 1.0, 5.0], [2.0, 2.0, 1.0], [3.0, 3.0, 4.0], [4.0, 4.0, 2.0]])
    found = correlated_pairs(values, ["a", "b", "c"])
    assert ("a", "b", pytest.approx(1.0)) in [(x, y, r) for x, y, r in found]


def test_constant_columns_do_not_divide_by_zero():
    values = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert correlated_pairs(values, ["a", "constant"]) == []


# ---------------------------------------------------------------------------
# semantic block
# ---------------------------------------------------------------------------


def test_model_is_cached_answers_without_network():
    assert isinstance(model_is_cached(DEFAULT_MODEL), bool)


def test_semantic_block_declares_exactly_one_column():
    """Checkable with no weights on disk -- specs are class-level, and the
    import of sentence_transformers is deferred into the methods."""
    assert [spec.name for spec in SemanticBlock().specs] == ["title_embedding_cosine"]


@pytest.mark.skipif(
    not model_is_cached(DEFAULT_MODEL),
    reason=f"{DEFAULT_MODEL} not downloaded; --semantic is opt-in",
)
def test_semantic_cosine_is_high_for_a_paraphrase():
    records = [
        normalize_title("Sony Wireless Noise Cancelling Headphones"),
        normalize_title("Sony Cordless Noise Reducing Headset"),
    ]
    block_ = SemanticBlock()
    block_.fit(records)
    values = block_.transform(records, np.array([0]), np.array([1]))
    assert 0.0 <= values[0, 0] <= 1.0


def normalize_title(title):
    from dedup.schema import Record

    return normalize(Record(record_id="s:x", source="synthetic", entity_id="e1", title=title))


# ---------------------------------------------------------------------------
# Against the real benchmark (skipped unless downloaded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_report_reproduces_the_published_test_split_ceiling():
    """Re-derives `reports/features.md`. A drift here means the committed
    report no longer describes the code that produced it."""
    report = evaluate(load_abt_buy(REAL_DATA), dataset="abt-buy", test_fraction=0.3, seed=0)

    assert report.n_records == 2173
    assert report.n_true_pairs == 1118
    # The ceiling every column in the report inherits, matching reports/blocking.md.
    assert report.test.blocking.pair_completeness == pytest.approx(1.0)
    assert report.train.blocking.pair_completeness == pytest.approx(0.9949, abs=1e-4)
    assert report.test.n_positives_found == report.test.n_true_pairs


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_vendor_code_columns_dominate_the_ranking():
    """CLAUDE.md calls model-number extraction the single highest-value signal.

    If the code columns are not at the top of this table, the feature is
    broken -- not the claim.
    """
    report = evaluate(load_abt_buy(REAL_DATA), dataset="abt-buy", test_fraction=0.3, seed=0)
    ranked = sorted(
        report.diagnostics, key=lambda d: d.pr_auc if d.pr_auc is not None else -1.0, reverse=True
    )
    top_five = {d.spec.name for d in ranked[:5]}
    assert "code_token_jaccard" in top_five
    assert "model_number_exact" in top_five
    assert "title_tfidf_cosine" in top_five


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_markdown_names_every_column_and_both_splits():
    report = evaluate(load_abt_buy(REAL_DATA), dataset="abt-buy", test_fraction=0.3, seed=0)
    markdown = render_markdown(report)
    for diagnostic in report.diagnostics:
        assert f"`{diagnostic.spec.name}`" in markdown
    assert "| train |" in markdown and "| test |" in markdown


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_blocking_runs_inside_a_split_not_across_the_catalog():
    """Pair indices must be positions in the record list handed to `block`."""
    records = [normalize(record) for record in load_abt_buy(REAL_DATA)[:200]]
    run, scored = block(records)
    assert run.keys.max() < len(records) ** 2
    assert 0.0 <= scored.pair_completeness <= 1.0
