"""Tests for out-of-fold calibration.

Two of these are leak tests, and they are the reason this file exists:

  * `test_the_featurizer_is_refit_for_every_fold` -- the fitted IDF weights are
    part of the feature definition, so a featurizer fit once over all of train
    would carry held-out text into the calibration set. That is a quiet leak: it
    changes no shape, raises nothing, and simply makes the calibration set
    slightly optimistic in a way no other assertion here would notice.
  * `test_calibration_cannot_change_the_ranking` -- Platt is monotone, so PR-AUC
    must be identical before and after. A calibrator that could reorder pairs
    would let a calibration step flatter a ranking metric it has no business
    improving.

The synthetic catalog is deliberately tiny and blockable: these tests are about
plumbing and leakage, not about model quality, and the published numbers are
re-derived in `test_model_evaluate.py` against the real benchmark instead.
"""

import numpy as np
import pytest

from dedup.eval.metrics import average_precision, precision_recall_curve
from dedup.model import calibrate as calibrate_module
from dedup.model.calibrate import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    fit_calibrated_scorer,
    out_of_fold_scores,
    reliability_bins,
)
from dedup.schema import Record

BRANDS = [
    "Panasonic", "Canon", "Sony", "Bose", "Cuisinart", "LG",
    "HP", "Nikon", "Epson", "Dell", "Asus", "Acer",
]

# A token whose char 3-grams appear nowhere else in the catalog, so its presence
# in a fitted vocabulary proves which records that vectorizer was fit on.
SENTINEL = "qzjvwx"
SENTINEL_ENTITY = "e0"
SENTINEL_GRAM = "zjv"


def catalog():
    """Twelve two-record entities; entity e0 carries the sentinel token."""
    records = []
    for index, brand in enumerate(BRANDS):
        code = f"XK{100 + index}A"
        extra = f" {SENTINEL}" if index == 0 else ""
        records += [
            Record(
                record_id=f"s:{index}a",
                source="synthetic",
                entity_id=f"e{index}",
                title=f"{brand} Digital Widget Model {code}{extra}",
                price=100.0 + index,
            ),
            Record(
                record_id=f"s:{index}b",
                source="synthetic",
                entity_id=f"e{index}",
                title=f"{brand} {code} Widget{extra}",
                description=f"{brand.lower()} widget",
            ),
        ]
    return records


def corpus_vocabulary(scorer):
    """The char n-gram vocabulary the scorer's shared text vectorizer was fit on."""
    for block in scorer.featurizer.blocks:
        vectorizer = getattr(block, "_vectorizer", None)
        if vectorizer is not None:
            return vectorizer.vocabulary_
    raise AssertionError("no fitted text vectorizer found on the featurizer")


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


def test_calibration_cannot_change_the_ranking():
    """Platt is monotone, so PR-AUC is identical before and after."""
    rng = np.random.default_rng(0)
    scores = rng.random(400)
    labels = rng.random(400) < scores  # correlated, so the curve is non-trivial

    calibrated = PlattCalibrator().fit(scores, labels).transform(scores)
    before = average_precision(
        precision_recall_curve(scores, labels, n_positives_total=int(labels.sum()))
    )
    after = average_precision(
        precision_recall_curve(calibrated, labels, n_positives_total=int(labels.sum()))
    )
    assert after == pytest.approx(before)
    assert np.array_equal(np.argsort(scores), np.argsort(calibrated))


def test_calibration_lowers_the_expected_calibration_error():
    """A deliberately overconfident score is pulled back toward its true rate."""
    rng = np.random.default_rng(1)
    truth = rng.random(2000)
    labels = rng.random(2000) < truth
    overconfident = np.clip(truth * 3.0 - 1.0, 0.0, 1.0)  # miscalibrated, still monotone

    calibrated = PlattCalibrator().fit(overconfident, labels).transform(overconfident)
    assert expected_calibration_error(calibrated, labels) < expected_calibration_error(
        overconfident, labels
    )


def test_calibrated_output_stays_a_probability():
    rng = np.random.default_rng(2)
    scores = rng.random(200)
    labels = rng.random(200) < scores
    calibrated = PlattCalibrator().fit(scores, labels).transform(np.array([0.0, 0.5, 1.0]))
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0


def test_a_calibrator_that_would_reverse_the_ranking_is_rejected():
    """Scores that rank positives *below* negatives give a negative Platt slope.

    Accepting it would silently reverse every ranking downstream: PR-AUC would
    change under calibration, and a threshold chosen on calibrated scores would
    select the opposite end of the list from the one the raw score ranks first.
    """
    rng = np.random.default_rng(4)
    scores = rng.random(400)
    labels = rng.random(400) > scores  # anti-correlated
    calibrator = PlattCalibrator()
    with pytest.raises(ValueError, match="slope"):
        calibrator.fit(scores, labels)
    with pytest.raises(RuntimeError, match="before fit"):
        calibrator.transform(scores)  # a rejected fit leaves nothing usable behind


def test_calibration_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        PlattCalibrator().fit(np.linspace(0, 1, 10), np.ones(10, dtype=bool))


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        PlattCalibrator().transform(np.array([0.5]))


def test_an_empty_score_array_calibrates_to_an_empty_array():
    scores = np.linspace(0, 1, 20)
    labels = np.arange(20) >= 10
    fitted = PlattCalibrator().fit(scores, labels)
    assert fitted.transform(np.empty(0)).shape == (0,)


# ---------------------------------------------------------------------------
# the fold loop
# ---------------------------------------------------------------------------


def test_the_featurizer_is_refit_for_every_fold(monkeypatch):
    """A featurizer fit once over all of train would leak held-out text.

    The sentinel token lives in exactly one entity. The model that scores the
    fold holding that entity must never have seen it; every other fold's model
    must have.
    """
    records = catalog()
    captured = []
    real_train_scorer = calibrate_module.train_scorer

    def spy(split, **kwargs):
        scorer = real_train_scorer(split, **kwargs)
        captured.append((split, scorer))
        return scorer

    monkeypatch.setattr(calibrate_module, "train_scorer", spy)
    out_of_fold_scores(records, n_folds=3, seed=0)

    assert len(captured) == 3
    holdouts = 0
    for split, scorer in captured:
        trained_on_sentinel = any(
            record.raw.entity_id == SENTINEL_ENTITY for record in split.records
        )
        has_gram = SENTINEL_GRAM in corpus_vocabulary(scorer)
        assert has_gram == trained_on_sentinel
        holdouts += not trained_on_sentinel
    assert holdouts == 1, "exactly one fold should hold the sentinel entity out"


def test_no_fold_scores_a_record_its_model_was_trained_on(monkeypatch):
    """The structural version of the same claim, on record ids rather than text."""
    records = catalog()
    seen = []
    real_train_scorer = calibrate_module.train_scorer

    monkeypatch.setattr(
        calibrate_module,
        "train_scorer",
        lambda split, **kwargs: (
            seen.append({r.raw.record_id for r in split.records}),
            real_train_scorer(split, **kwargs),
        )[1],
    )
    out_of_fold_scores(records, n_folds=3, seed=0)

    all_ids = {r.record_id for r in records}
    for trained_ids in seen:
        assert trained_ids < all_ids  # a proper subset -- something was held out
    assert set.union(*seen) == all_ids


def test_out_of_fold_covers_every_fold_once():
    oof = out_of_fold_scores(catalog(), n_folds=3, seed=0)
    assert oof.n_folds == 3
    assert len(oof.fold_candidates) == 3
    assert oof.n_pairs == sum(oof.fold_candidates)
    assert oof.n_positives == sum(oof.fold_positives)
    assert oof.scores.shape == oof.labels.shape


def test_the_calibrator_sees_more_positives_than_a_held_out_split_would():
    """The measured reason for out-of-fold over a third split, in miniature.

    Every true pair in train reaches the calibrator, rather than only those in
    one held-out slice of it.
    """
    records = catalog()
    oof = out_of_fold_scores(records, n_folds=3, seed=0)
    assert oof.n_positives == len(BRANDS)  # one per entity, all of them


# ---------------------------------------------------------------------------
# fit_calibrated_scorer
# ---------------------------------------------------------------------------


def test_the_final_booster_is_fit_on_every_train_record():
    """No training data is spent on calibration -- that is the point of folding."""
    records = catalog()
    fit = fit_calibrated_scorer(records, n_folds=3, seed=0)
    assert fit.scorer.calibrator is not None
    # The positive control for the fold test above: fit on everything, the
    # sentinel is present. Its absence there is therefore evidence, not an
    # artifact of how the vocabulary is read.
    assert SENTINEL_GRAM in corpus_vocabulary(fit.scorer)


def test_the_calibrated_scorer_returns_probabilities():
    from dedup.model.train import prepare

    records = catalog()
    fit = fit_calibrated_scorer(records, n_folds=3, seed=0)
    split = prepare(records)
    probabilities = fit.scorer.probabilities(split.records, split.pair_keys)
    assert probabilities.shape == (split.n_candidates,)
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0


# ---------------------------------------------------------------------------
# calibration metrics
# ---------------------------------------------------------------------------


def test_a_perfectly_calibrated_score_has_no_calibration_error():
    p = np.repeat([0.1, 0.9], 1000)
    rng = np.random.default_rng(3)
    labels = rng.random(2000) < p
    assert expected_calibration_error(p, labels) < 0.02


def test_brier_rewards_confident_correctness():
    labels = np.array([True, True, False, False])
    confident = np.array([0.99, 0.99, 0.01, 0.01])
    hedged = np.array([0.6, 0.6, 0.4, 0.4])
    assert brier_score(confident, labels) < brier_score(hedged, labels)


def test_reliability_bins_skip_empty_bins_and_cover_every_pair():
    p = np.array([0.02, 0.03, 0.97, 0.98])
    labels = np.array([False, False, True, True])
    bins = reliability_bins(p, labels, bins=10)
    assert len(bins) == 2
    assert sum(count for *_, count in bins) == 4


def test_empty_input_does_not_divide_by_zero():
    empty = np.empty(0)
    assert brier_score(empty, empty.astype(bool)) == 0.0
    assert expected_calibration_error(empty, empty.astype(bool)) == 0.0
