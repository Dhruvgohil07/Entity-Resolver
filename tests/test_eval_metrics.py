"""Tests for the pairwise ranking metrics.

The theme is the one thing these differ from sklearn on: a pruned candidate
set must not flatter recall. Several tests below exist purely to make that
failure mode loud, because it is silent in every off-the-shelf metric.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from dedup.eval.metrics import (
    average_precision,
    best_f1,
    evaluate_at_threshold,
    precision_at_k,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Agreement with sklearn where the two are supposed to agree
# ---------------------------------------------------------------------------


def test_average_precision_matches_sklearn_on_a_complete_candidate_set():
    # When nothing is pruned, n_positives_total defaults to the positives
    # present and this must reduce exactly to average_precision_score. Pins
    # the step-wise convention -- a trapezoid would drift from sklearn here.
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = rng.random(500) < 0.1

    curve = precision_recall_curve(scores, labels)
    assert average_precision(curve) == pytest.approx(average_precision_score(labels, scores))


def test_average_precision_matches_sklearn_when_scores_tie_heavily():
    # Ties are the case where "one point per rank" and "one point per
    # reachable threshold" come apart, and char-3gram cosine produces plenty
    # of them (identical short titles score exactly 1.0).
    scores = np.array([0.9, 0.9, 0.9, 0.5, 0.5, 0.2])
    labels = np.array([1, 0, 1, 0, 1, 0], dtype=bool)

    curve = precision_recall_curve(scores, labels)
    assert average_precision(curve) == pytest.approx(average_precision_score(labels, scores))


# ---------------------------------------------------------------------------
# The pruned-candidate-set contract
# ---------------------------------------------------------------------------


def test_recall_charges_for_true_pairs_the_candidate_set_never_saw():
    # Three true pairs exist; blocking emitted only two of them. Recall tops
    # out at 2/3, not 1.0. Without n_positives_total this reads as perfect
    # recall and the blocker that dropped a match looks flawless.
    scores = np.array([0.9, 0.8, 0.1])
    labels = np.array([1, 1, 0], dtype=bool)

    curve = precision_recall_curve(scores, labels, n_positives_total=3)
    assert curve.recall.max() == pytest.approx(2 / 3)

    naive = precision_recall_curve(scores, labels)
    assert naive.recall.max() == pytest.approx(1.0)


def test_pruning_the_candidate_set_can_only_lower_the_score():
    # The regression that motivates the whole module: if dropping candidates
    # raised PR-AUC, every blocker would be "improved" by discarding pairs.
    rng = np.random.default_rng(7)
    scores = rng.random(2000)
    labels = rng.random(2000) < 0.05
    n_positives = int(labels.sum())

    full = average_precision(precision_recall_curve(scores, labels))

    kept = scores > 0.3  # a similarity floor, dropping some positives with it
    assert labels[kept].sum() < n_positives, "floor must actually drop a positive to be a test"
    pruned = average_precision(
        precision_recall_curve(scores[kept], labels[kept], n_positives_total=n_positives)
    )

    assert pruned <= full


def test_n_positives_total_below_the_observed_positives_is_rejected():
    # Means the caller's ground-truth count and its labels disagree -- worth
    # stopping on, since silently taking the larger would hide the bug.
    with pytest.raises(ValueError, match="smaller than"):
        precision_recall_curve(
            np.array([0.9, 0.8]), np.array([1, 1], dtype=bool), n_positives_total=1
        )


def test_evaluate_at_threshold_also_uses_the_full_denominator():
    scores = np.array([0.9, 0.4])
    labels = np.array([1, 0], dtype=bool)

    point = evaluate_at_threshold(scores, labels, 0.5, n_positives_total=4)
    assert point.precision == pytest.approx(1.0)
    assert point.recall == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Threshold semantics
# ---------------------------------------------------------------------------


def test_thresholds_are_inclusive_and_only_reachable_cuts_are_reported():
    # A threshold admits every pair sharing its score, so the three tied 0.9s
    # collapse to one operating point at P=2/3 -- there is no cut that keeps
    # the first 0.9 and drops the other two.
    scores = np.array([0.9, 0.9, 0.9, 0.4])
    labels = np.array([1, 1, 0, 1], dtype=bool)

    curve = precision_recall_curve(scores, labels)
    assert curve.thresholds.tolist() == [0.9, 0.4]
    assert curve.precision[0] == pytest.approx(2 / 3)

    # evaluate_at_threshold must agree with the curve at the same cut.
    point = evaluate_at_threshold(scores, labels, 0.9)
    assert point.precision == pytest.approx(curve.precision[0])
    assert point.recall == pytest.approx(curve.recall[0])


def test_best_f1_returns_an_operating_point_that_can_be_reproduced():
    rng = np.random.default_rng(3)
    scores = rng.random(400)
    labels = rng.random(400) < 0.2

    curve = precision_recall_curve(scores, labels)
    point = best_f1(curve)
    replayed = evaluate_at_threshold(scores, labels, point.threshold)

    assert replayed.precision == pytest.approx(point.precision)
    assert replayed.recall == pytest.approx(point.recall)
    assert replayed.f1 == pytest.approx(point.f1)


def test_threshold_above_every_score_scores_zero_rather_than_dividing_by_zero():
    point = evaluate_at_threshold(np.array([0.4, 0.2]), np.array([1, 0], dtype=bool), 0.9)
    assert (point.precision, point.recall, point.f1) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# precision@k
# ---------------------------------------------------------------------------


def test_precision_at_k_reads_the_top_of_the_ranking():
    scores = np.array([0.1, 0.9, 0.8, 0.7, 0.2])
    labels = np.array([0, 1, 1, 0, 1], dtype=bool)

    # Top 4 by score are 0.9, 0.8, 0.7, 0.2 -- three of which are positive.
    assert precision_at_k(scores, labels, 2) == pytest.approx(1.0)
    assert precision_at_k(scores, labels, 4) == pytest.approx(0.75)


def test_precision_at_k_divides_by_the_requested_k_not_the_candidates_available():
    # Two candidates, one of them a hit, but the caller asked about the first
    # 100 of a review queue: 98 of those seats are empty, and an empty seat
    # is not a hit. Dividing by the 2 that exist would report 0.5.
    scores = np.array([0.9, 0.8])
    labels = np.array([1, 0], dtype=bool)
    assert precision_at_k(scores, labels, 100) == pytest.approx(0.01)


def test_pruning_the_candidate_set_cannot_raise_precision_at_k():
    # The regression this function was rewritten for. Clamping k to the
    # surviving candidate count let a blocker that threw away 98% of its
    # pairs report P@100 = 0.611 for the very same 11 hits that score 0.11
    # unpruned -- rewarding exactly the behaviour blocking must be penalised
    # for. Same failure the recall denominator elsewhere in this module
    # exists to prevent, one metric over.
    scores = np.concatenate([np.linspace(0.99, 0.90, 18), np.linspace(0.5, 0.1, 982)])
    labels = np.zeros(1000, dtype=bool)
    labels[:11] = True

    unpruned = precision_at_k(scores, labels, 100)
    kept = scores > 0.89
    assert labels[kept].sum() == labels.sum(), "the floor must keep every hit for this to isolate k"

    assert precision_at_k(scores[kept], labels[kept], 100) == pytest.approx(unpruned)
    assert unpruned == pytest.approx(0.11)


def test_precision_at_k_resolves_a_boundary_tie_by_expectation_not_array_order():
    # Four pairs tie at 0.5 for the single seat below the top score, one of
    # them positive -- so the honest answer is 1 + 1/4 hits over 2 seats.
    # np.argpartition picked among the tied block arbitrarily, which made the
    # metric depend on the order the loader emitted rows in: the same
    # multiset returned 0.5 or 1.0 depending on the permutation.
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 0, 0, 0, 1], dtype=bool)

    results = [
        precision_at_k(scores[np.array(permutation)], labels[np.array(permutation)], 2)
        for permutation in ([0, 1, 2, 3, 4], [0, 4, 1, 2, 3], [0, 3, 4, 1, 2], [4, 3, 2, 1, 0])
    ]

    assert len(set(results)) == 1, f"order-dependent: {sorted(set(results))}"
    assert results[0] == pytest.approx(0.625)


def test_precision_at_k_rejects_a_non_positive_k():
    with pytest.raises(ValueError, match="must be positive"):
        precision_at_k(np.array([0.5]), np.array([1], dtype=bool), 0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mismatched_scores_and_labels_are_rejected():
    with pytest.raises(ValueError, match="equal shape"):
        precision_recall_curve(np.array([0.5, 0.4]), np.array([1], dtype=bool))


def test_a_nan_score_is_rejected_rather_than_silently_unreachable():
    # `nan >= threshold` is False at every cut, so a NaN-scored true pair can
    # never be admitted while still counting against recall -- the curve
    # reports a maximum no threshold can reach. schema.py guards price the
    # same way; a pair that cannot be scored belongs out of the candidate
    # set, not in it with a NaN.
    with pytest.raises(ValueError, match="NaN or infinite"):
        precision_recall_curve(np.array([0.9, np.nan, 0.1]), np.array([1, 1, 0], dtype=bool))

    with pytest.raises(ValueError, match="NaN or infinite"):
        precision_at_k(np.array([0.9, np.inf]), np.array([1, 0], dtype=bool), 2)


def test_a_curve_with_no_positives_is_rejected():
    # PR-AUC is undefined with an empty recall denominator; returning 0.0
    # would be indistinguishable from a genuinely useless ranking.
    with pytest.raises(ValueError, match="no positive pairs"):
        precision_recall_curve(np.array([0.5, 0.4]), np.array([0, 0], dtype=bool))
