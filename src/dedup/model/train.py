"""LightGBM over the pair feature vector, bundled with the featurizer that fed it.

The stage boundary this module sits on is the one CLAUDE.md is most insistent
about: `normalize.py` runs identically in batch and at serve time, and anything
that exists only in the training path is a train/serve skew bug. A fitted
booster is not on its own a servable artifact, because the 33 columns it splits
on only mean what they meant during training if the *same fitted featurizer*
produced them -- the IDF weights inside `PairFeaturizer` are part of the feature
definition, not an optimization. So `PairScorer` carries booster, featurizer and
calibrator together, and nothing here ever returns a bare booster.

Two choices worth stating because both look like oversights:

**No class reweighting.** The reflex at this imbalance is `scale_pos_weight` or
`is_unbalance`, and it is wrong here for a specific reason: a reweighted booster
is *deliberately* miscalibrated -- its outputs sit on a shifted prior -- and the
next stage in this pipeline is a calibrator feeding a cost model that consumes
probabilities. Reweighting would have to be undone by the very thing it feeds.
It is also less tempting than it looks: post-blocking imbalance on Abt-Buy is
about 60 negatives per positive, not the thousands the pre-blocking case has.

**Hyperparameters are fixed, not tuned.** Tuning needs a third split, and tuning
against test is the leak the whole protocol exists to prevent. These values are
plain defaults that were never selected against any number in `reports/`.

Training runs single-threaded and deterministic for the same reason
`blocking/ann.py` builds its HNSW index on one thread: a committed report whose
numbers drift between runs of identical input is not reproducible, and the cost
here is seconds on a catalog this size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from lightgbm import LGBMClassifier

from dedup.blocking.defaults import block_split
from dedup.blocking.union import BlockerScore
from dedup.eval.splits import count_true_pairs
from dedup.features.vectorize import PairFeaturizer, pair_labels
from dedup.normalize import NormalizedRecord, normalize
from dedup.schema import Record

DEFAULT_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "verbose": -1,
    # Reproducibility, per the module docstring. `deterministic` requires one
    # of the force_*_wise modes to have any effect.
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
}


class Calibrator(Protocol):
    """Maps raw model scores to calibrated probabilities.

    A Protocol rather than an import so this module does not depend on
    `calibrate.py`, which depends on this one.
    """

    def transform(self, scores: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PreparedSplit:
    """One split, normalized and blocked, ready to featurize.

    Carries `n_positives_total` because every recall figure downstream must be
    divided by every true pair in the split rather than by the pairs blocking
    emitted -- otherwise pruning harder improves the reported number, the trap
    `eval/metrics.py` exists to close.
    """

    records: list[NormalizedRecord]
    pair_keys: np.ndarray
    labels: np.ndarray
    blocking: BlockerScore
    n_positives_total: int

    @property
    def n_candidates(self) -> int:
        return int(self.pair_keys.size)

    @property
    def n_positives_found(self) -> int:
        return int(self.labels.sum())


def prepare(records: Sequence[Record]) -> PreparedSplit:
    """Normalize, block and label one split.

    Blocking runs inside the split, so the packed pair indices are positions in
    *this* record list. That is what lets a fold be treated as a self-contained
    catalog during out-of-fold calibration.
    """
    normalized = [normalize(record) for record in records]
    run, blocking = block_split(normalized)
    return PreparedSplit(
        records=normalized,
        pair_keys=run.keys,
        labels=pair_labels(normalized, run.keys),
        blocking=blocking,
        n_positives_total=count_true_pairs(list(records)),
    )


@dataclass(frozen=True)
class PairScorer:
    """A servable pair model: fitted featurizer, booster, and optional calibrator."""

    featurizer: PairFeaturizer
    booster: LGBMClassifier
    calibrator: Calibrator | None = None

    @property
    def feature_names(self) -> list[str]:
        return self.featurizer.names

    def with_calibrator(self, calibrator: Calibrator) -> PairScorer:
        """A copy carrying `calibrator`; the scorer itself is frozen."""
        return replace(self, calibrator=calibrator)

    def raw_scores(
        self, records: Sequence[NormalizedRecord], pair_keys: np.ndarray
    ) -> np.ndarray:
        """Uncalibrated positive-class scores. Rank with these, never threshold them."""
        pair_keys = np.asarray(pair_keys, dtype=np.int64)
        if pair_keys.size == 0:
            return np.empty(0, dtype=np.float64)
        matrix = self.featurizer.transform(records, pair_keys)
        return np.asarray(self.booster.predict_proba(matrix.values)[:, 1], dtype=np.float64)

    def probabilities(
        self, records: Sequence[NormalizedRecord], pair_keys: np.ndarray
    ) -> np.ndarray:
        """Calibrated probabilities, which is what the cost model may consume.

        Refuses rather than falling back to the raw score. LightGBM's output is
        already in [0, 1] and would sail through `assign_bands` unnoticed, which
        is exactly how an uncalibrated score ends up behind a cost-derived
        threshold that then means nothing (CLAUDE.md, Invariants).
        """
        if self.calibrator is None:
            raise RuntimeError(
                "PairScorer has no calibrator, so it cannot return probabilities. The cost "
                "model consumes a probability, not a ranking score -- fit one with "
                "`model.calibrate.fit_calibrated_scorer`, or call `raw_scores` if you only "
                "need a ranking."
            )
        return np.asarray(self.calibrator.transform(self.raw_scores(records, pair_keys)))

    def gain_importance(self) -> list[tuple[str, float]]:
        """Feature names by total split gain, descending.

        Worth reporting next to `reports/features.md`'s univariate PR-AUCs: the
        two disagreeing is informative rather than alarming, since gain credits
        a column for interactions a univariate ranking cannot see.
        """
        gains = self.booster.booster_.feature_importance(importance_type="gain")
        return sorted(
            zip(self.feature_names, (float(g) for g in gains)),
            key=lambda row: -row[1],
        )


def train_scorer(
    split: PreparedSplit,
    *,
    params: dict[str, object] | None = None,
    include_semantic: bool = False,
    seed: int = 0,
) -> PairScorer:
    """Fit featurizer and booster on one prepared split.

    Pass the train split, never the catalog. The featurizer is fit here rather
    than accepted ready-made so that a caller cannot accidentally hand in one
    that has already seen the data it is about to score -- the leak that makes
    out-of-fold calibration worth doing at all.
    """
    featurizer = PairFeaturizer(include_semantic=include_semantic).fit(split.records)
    matrix = featurizer.transform(split.records, split.pair_keys)
    booster = LGBMClassifier(**{**DEFAULT_PARAMS, **(params or {}), "random_state": seed})
    booster.fit(matrix.values, split.labels)
    return PairScorer(featurizer=featurizer, booster=booster)
