"""Tests for the cost model and the band assignment.

The one that carries the most weight is
`test_bands_are_the_argmin_of_expected_cost`: it checks the closed-form
thresholds against the definition they claim to implement, rather than against
themselves. Asserting `p_hi == 0.95` only proves the arithmetic was typed in
consistently; asserting the bands agree with `argmin(expected_costs)` across the
whole range proves the algebra is right.
"""

import numpy as np
import pytest

from dedup.model.threshold import (
    Band,
    CostModel,
    assign_bands,
    band_summary,
    expected_costs,
)

# ---------------------------------------------------------------------------
# the closed form
# ---------------------------------------------------------------------------


def test_default_cost_model_matches_the_recorded_decision():
    """CLAUDE.md records 20 : 2 : 1 -> p_hi 0.95, p_lo 0.50."""
    cost = CostModel()
    assert (cost.false_merge, cost.false_split, cost.review) == (20.0, 2.0, 1.0)
    assert cost.auto_merge_threshold == pytest.approx(0.95)
    assert cost.auto_reject_threshold == pytest.approx(0.50)


def test_bands_are_the_argmin_of_expected_cost():
    """The thresholds are a shortcut for the argmin; they must agree with it."""
    cost = CostModel()
    p = np.linspace(0.0, 1.0, 1001)
    bands = assign_bands(p, cost)
    coded = np.where(bands.auto_merge, 0, np.where(bands.review, 1, 2))
    assert np.array_equal(np.argmin(expected_costs(p, cost), axis=1), coded)


def test_only_the_ratios_to_review_cost_matter():
    """The absolute scale cancels, so the review action is the unit."""
    assert CostModel(20, 2, 1).auto_merge_threshold == CostModel(200, 20, 10).auto_merge_threshold
    assert CostModel(20, 2, 1).auto_reject_threshold == CostModel(200, 20, 10).auto_reject_threshold


def test_raising_the_false_merge_cost_raises_the_merge_bar():
    thresholds = [CostModel(false_merge=c).auto_merge_threshold for c in (10, 20, 50, 100)]
    assert thresholds == sorted(thresholds)
    assert all(t < 1.0 for t in thresholds)


def test_raising_the_false_split_cost_lowers_the_reject_bar():
    thresholds = [CostModel(false_split=c).auto_reject_threshold for c in (2, 5, 10)]
    assert thresholds == sorted(thresholds, reverse=True)


# ---------------------------------------------------------------------------
# rejected cost models
# ---------------------------------------------------------------------------


def test_an_inverted_band_is_rejected():
    """20 : 1 : 1 gives p_lo 1.0 against p_hi 0.95 -- plausible-looking, and empty."""
    with pytest.raises(ValueError, match="no review band"):
        CostModel(false_merge=20, false_split=1)


def test_a_review_cost_that_exceeds_its_worth_is_rejected():
    """Reviewing must be cheaper than the mistakes it prevents, or nothing is worth reviewing."""
    with pytest.raises(ValueError, match="no review band"):
        CostModel(false_merge=1.5, false_split=1.5, review=1.0)


@pytest.mark.parametrize("kwargs", [{"false_merge": 0}, {"false_split": -2}, {"review": 0}])
def test_non_positive_costs_are_rejected(kwargs):
    with pytest.raises(ValueError, match="finite and positive"):
        CostModel(**kwargs)


def test_a_raw_score_outside_the_unit_interval_is_rejected():
    """A ranking score fed in here would threshold cleanly and mean nothing."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assign_bands(np.array([-0.5, 0.2, 3.0]), CostModel())


# ---------------------------------------------------------------------------
# band assignment
# ---------------------------------------------------------------------------


def test_the_boundary_lands_in_the_more_decisive_band():
    """`>=` at both thresholds, matching eval.metrics.evaluate_at_threshold."""
    cost = CostModel()
    bands = assign_bands(np.array([0.95, 0.50]), cost)
    assert bands.auto_merge[0] and not bands.review[0]
    assert bands.review[1] and not bands.auto_reject[1]


def test_every_pair_lands_in_exactly_one_band():
    bands = assign_bands(np.random.default_rng(0).random(500), CostModel())
    stacked = np.stack([bands.auto_merge, bands.review, bands.auto_reject])
    assert np.array_equal(stacked.sum(axis=0), np.ones(500, dtype=int))


def test_labels_name_each_band():
    labels = assign_bands(np.array([0.99, 0.7, 0.1]), CostModel()).labels
    assert list(labels) == [Band.AUTO_MERGE, Band.REVIEW, Band.AUTO_REJECT]


# ---------------------------------------------------------------------------
# band_summary
# ---------------------------------------------------------------------------


def test_realized_cost_charges_each_mistake_its_own_price():
    """One false merge, one missed duplicate, one review: 20 + 2 + 1."""
    cost = CostModel()
    p = np.array([0.99, 0.99, 0.1, 0.1, 0.7])
    y = np.array([True, False, True, False, True])
    summary = band_summary(p, y, cost, n_positives_total=3)

    assert (summary.n_auto_merge, summary.n_review, summary.n_auto_reject) == (2, 1, 2)
    assert summary.n_false_merge == 1
    assert summary.n_missed == 1
    assert summary.realized_cost == pytest.approx(20.0 + 2.0 + 1.0)


def test_auto_merge_recall_divides_by_every_true_pair_in_the_split():
    """Not by the candidates -- otherwise discarding pairs improves the number.

    The same trap `eval/metrics.py` closes, restated one layer up: here the
    candidate set is what blocking emitted, and blocking must not be able to
    flatter the model by emitting less.
    """
    p = np.array([0.99, 0.99])
    y = np.array([True, True])
    blocked_in = band_summary(p, y, CostModel(), n_positives_total=2)
    pruned = band_summary(p, y, CostModel(), n_positives_total=10)

    assert blocked_in.auto_merge_recall == pytest.approx(1.0)
    assert pruned.auto_merge_recall == pytest.approx(0.2)


def test_summary_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="differ in shape"):
        band_summary(np.zeros(3), np.zeros(4, dtype=bool), CostModel(), n_positives_total=1)


def test_an_empty_candidate_set_does_not_divide_by_zero():
    summary = band_summary(
        np.empty(0), np.empty(0, dtype=bool), CostModel(), n_positives_total=0
    )
    assert summary.realized_cost == 0.0
    assert summary.auto_merge_precision == 0.0
    assert summary.review_fraction == 0.0
