"""Contract tests for the pair scorer.

The one with real content is `test_probabilities_refuses_without_a_calibrator`.
LightGBM's `predict_proba` already returns numbers in [0, 1], so an uncalibrated
score would pass through `assign_bands` without complaint and produce bands that
are arithmetically valid and meaningless. Refusing is what turns CLAUDE.md's
calibration invariant into something the type system enforces rather than
something a reviewer has to notice.

`test_the_scorer_carries_the_featurizer_that_fit_it` is the train/serve one: a
booster whose 33 columns were produced by a different featurizer is scoring
different features under the same names.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dedup.features.vectorize import PairFeaturizer
from dedup.model.train import DEFAULT_PARAMS, prepare, train_scorer
from dedup.schema import Record

BRANDS = ["Panasonic", "Canon", "Sony", "Bose", "Cuisinart", "LG", "HP", "Nikon"]


def catalog():
    records = []
    for index, brand in enumerate(BRANDS):
        code = f"XK{100 + index}A"
        records += [
            Record(
                record_id=f"s:{index}a",
                source="synthetic",
                entity_id=f"e{index}",
                title=f"{brand} Digital Widget Model {code}",
                price=100.0 + index,
            ),
            Record(
                record_id=f"s:{index}b",
                source="synthetic",
                entity_id=f"e{index}",
                title=f"{brand} {code} Widget",
                description=f"{brand.lower()} widget",
            ),
        ]
    return records


@pytest.fixture(scope="module")
def split():
    return prepare(catalog())


@pytest.fixture(scope="module")
def scorer(split):
    return train_scorer(split)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_counts_positives_against_the_whole_split(split):
    """`n_positives_total` is every true pair, not the ones blocking emitted.

    They coincide here because this catalog blocks perfectly; the field exists
    so that when they diverge, recall is divided by the larger number.
    """
    assert split.n_positives_total == len(BRANDS)
    assert split.n_positives_found <= split.n_positives_total
    assert split.labels.shape == (split.n_candidates,)


def test_prepare_indices_are_positions_in_this_record_list(split):
    """Packed keys are split-local, which is why blocking runs inside a split."""
    n = len(split.records)
    assert split.pair_keys.max() < n * n


# ---------------------------------------------------------------------------
# PairScorer
# ---------------------------------------------------------------------------


def test_probabilities_refuses_without_a_calibrator(scorer, split):
    """An uncalibrated score would band cleanly and mean nothing."""
    assert scorer.calibrator is None
    with pytest.raises(RuntimeError, match="no calibrator"):
        scorer.probabilities(split.records, split.pair_keys)


def test_raw_scores_are_available_without_a_calibrator(scorer, split):
    """Ranking needs no calibration; only the cost model does."""
    scores = scorer.raw_scores(split.records, split.pair_keys)
    assert scores.shape == (split.n_candidates,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_the_scorer_carries_the_featurizer_that_fit_it(scorer):
    """A booster without its featurizer is not a servable artifact."""
    assert isinstance(scorer.featurizer, PairFeaturizer)
    assert scorer.feature_names == scorer.featurizer.names
    assert scorer.booster.n_features_in_ == len(scorer.feature_names)


def test_gain_importance_names_real_columns_in_descending_order(scorer):
    importance = scorer.gain_importance()
    names = [name for name, _ in importance]
    gains = [gain for _, gain in importance]

    assert set(names) == set(scorer.feature_names)
    assert gains == sorted(gains, reverse=True)


def test_an_empty_pair_set_scores_to_an_empty_array(scorer, split):
    assert scorer.raw_scores(split.records, np.empty(0, dtype=np.int64)).shape == (0,)


def test_with_calibrator_returns_a_new_scorer(scorer):
    """The scorer is frozen, so attaching a calibrator copies rather than mutates."""

    class Identity:
        def transform(self, scores):
            return scores

    attached = scorer.with_calibrator(Identity())
    assert attached is not scorer
    assert scorer.calibrator is None
    assert attached.booster is scorer.booster
    assert attached.featurizer is scorer.featurizer


def test_a_calibrated_scorer_returns_the_calibrators_output(scorer, split):
    class Constant:
        def transform(self, scores):
            return np.full(scores.shape, 0.25)

    probabilities = scorer.with_calibrator(Constant()).probabilities(
        split.records, split.pair_keys
    )
    assert np.all(probabilities == 0.25)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def test_training_is_deterministic(split):
    """Single-threaded and seeded, for the reason blocking/ann.py is.

    A committed report whose numbers drift between runs of identical input is
    not reproducible.
    """
    first = train_scorer(split, seed=0).raw_scores(split.records, split.pair_keys)
    second = train_scorer(split, seed=0).raw_scores(split.records, split.pair_keys)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("knob", ["scale_pos_weight", "is_unbalance", "class_weight"])
def test_the_default_params_do_not_reweight_the_classes(knob):
    """Reweighting is deliberately absent -- it would miscalibrate on purpose.

    A reweighted booster sits on a shifted prior, and the next stage here is a
    calibrator feeding a cost model that consumes probabilities. Turning one of
    these on would have to be undone by the thing it feeds.
    """
    assert knob not in DEFAULT_PARAMS


def test_params_can_be_overridden_without_losing_the_determinism_settings(split):
    scorer = train_scorer(split, params={"n_estimators": 5})
    assert scorer.booster.n_estimators == 5
    assert scorer.booster.get_params()["deterministic"] is True
    assert scorer.booster.get_params()["n_jobs"] == 1


def test_train_scorer_fits_its_own_featurizer(split):
    """There is no way to hand in a prefit one, which is what makes the
    out-of-fold loop leak-proof by construction rather than by discipline."""
    import inspect

    parameters = inspect.signature(train_scorer).parameters
    assert "featurizer" not in parameters


def test_a_scorer_is_frozen(scorer):
    """Swapping a featurizer under a fitted booster is the skew this prevents."""
    with pytest.raises(FrozenInstanceError):
        scorer.featurizer = PairFeaturizer()
