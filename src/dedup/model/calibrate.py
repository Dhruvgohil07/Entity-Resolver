"""Raw booster scores -> calibrated probabilities, fit out of fold.

CLAUDE.md requires a calibrated probability because the cost model consumes one:
an uncalibrated score makes the threshold sweep meaningless. Two questions that
leaves open were settled by measurement and are recorded in CLAUDE.md's
decisions log; this module implements those answers rather than re-opening them.

**Fit out of fold, not on a held-out third split.** A calibrator fit on the rows
the trees were fit on is fit to memorized labels, so the third split is the
obvious alternative -- and on a catalog this size it is an expensive one. Five
entity-grouped folds give the calibrator 777 positives against a held-out
split's 230, and leave all 1521 train records for the final booster. Measured
on the raw booster score at 0.95, that lifts precision from 0.9669 -- a booster
trained on the remainder a held-out split would leave -- to 0.9845.

**Platt, not isotonic.** Isotonic is the usual first choice and it loses here
on resolution exactly where the cost model reads. Fit on a held-out split it
produces *one* distinct output level at or above 0.9, and that level is 1.0, so
a `p_hi` of 0.95 cannot separate anything inside a 253-pair atom -- moving the
threshold to 0.999 would change nothing. Out-of-fold isotonic manages five
distinct outputs above 0.9; Platt keeps 285. Platt was also the best calibrated
of the six variants measured (ECE 0.00085 against the raw booster's 0.00244).
Revisit at `synth/` scale, where isotonic's shape advantage gets the positives
it needs.

The load-bearing detail in the fold loop is that **the featurizer is refit per
fold, not just the booster**. The fitted IDF weights inside `PairFeaturizer` are
part of the feature definition, so a featurizer fit once over all of train would
carry held-out text into the calibration set -- a quiet leak that would undo the
only reason for calibrating out of fold. It is structural rather than
remembered: `train_scorer` always fits its own featurizer and offers no way to
pass one in.

Calibration metrics live here rather than in `eval/metrics.py` because they are
intrinsic to this stage and carry none of that module's denominator traps --
nothing here is a recall, so nothing here can be gamed by discarding pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from dedup.eval.splits import kfold_by_entity
from dedup.model.train import PairScorer, prepare, train_scorer
from dedup.schema import Record

DEFAULT_N_FOLDS = 5

# Effectively unregularized: Platt scaling is a two-parameter fit and the
# calibration set here holds hundreds of positives, so shrinkage would bias the
# map toward the identity for no variance saving worth having.
_PLATT_C = 1e10

RELIABILITY_BINS = 15


class PlattCalibrator:
    """Sigmoid map from raw score to probability: `sigmoid(a * s + b)`.

    `fit` rejects a slope `a <= 0`, so the map is always increasing. That matters
    beyond neatness: a calibrator that reordered pairs would change PR-AUC,
    letting calibration flatter a ranking metric it has no business improving,
    and `model/evaluate.py` chooses its threshold on calibrated scores on the
    strength of the map selecting the same rank the raw score would.
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels).astype(bool)
        if scores.shape != labels.shape:
            raise ValueError(f"scores and labels differ in shape: {scores.shape} vs {labels.shape}")
        if labels.all() or not labels.any():
            raise ValueError(
                "calibration needs both classes present; got "
                f"{int(labels.sum())} positives in {labels.size} pairs"
            )
        self._model = LogisticRegression(C=_PLATT_C).fit(scores.reshape(-1, 1), labels)
        slope = float(self._model.coef_[0, 0])
        if slope <= 0:
            self._model = None
            raise ValueError(
                f"Platt slope is {slope:.4g}, not positive: calibrated probabilities would rank "
                f"pairs in reverse or not at all. Calibration must never reorder pairs -- PR-AUC "
                f"would change under it -- so the raw scores carry no usable ordering here."
            )
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PlattCalibrator.transform called before fit")
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return np.empty(0, dtype=np.float64)
        return np.asarray(self._model.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=np.float64)


@dataclass(frozen=True)
class OutOfFoldScores:
    """Predictions on held-out folds, pooled -- the calibrator's training set."""

    scores: np.ndarray
    labels: np.ndarray
    n_folds: int
    fold_candidates: tuple[int, ...]
    fold_positives: tuple[int, ...]

    @property
    def n_pairs(self) -> int:
        return int(self.scores.size)

    @property
    def n_positives(self) -> int:
        return int(self.labels.sum())


def out_of_fold_scores(
    records: Sequence[Record],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
    params: dict[str, object] | None = None,
    include_semantic: bool = False,
) -> OutOfFoldScores:
    """Score every fold with a model that never saw any record of its entities.

    Each fold is blocked as its own catalog, which is the same thing
    `split_by_entity` guarantees one level up: cross-fold pairs are negatives by
    construction, so forming them would only add easy negatives the serving path
    never sees.
    """
    folds = kfold_by_entity(list(records), n_folds=n_folds, seed=seed)

    scores, labels, candidates, positives = [], [], [], []
    for index, held_out in enumerate(folds):
        rest = [record for other, fold in enumerate(folds) if other != index for record in fold]
        scorer = train_scorer(
            prepare(rest), params=params, include_semantic=include_semantic, seed=seed
        )
        held = prepare(held_out)
        scores.append(scorer.raw_scores(held.records, held.pair_keys))
        labels.append(held.labels)
        candidates.append(held.n_candidates)
        positives.append(held.n_positives_found)

    return OutOfFoldScores(
        scores=np.concatenate(scores),
        labels=np.concatenate(labels),
        n_folds=n_folds,
        fold_candidates=tuple(candidates),
        fold_positives=tuple(positives),
    )


@dataclass(frozen=True)
class CalibratedFit:
    """A calibrated scorer plus the out-of-fold evidence behind its calibrator."""

    scorer: PairScorer
    out_of_fold: OutOfFoldScores


def fit_calibrated_scorer(
    train_records: Sequence[Record],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
    params: dict[str, object] | None = None,
    include_semantic: bool = False,
) -> CalibratedFit:
    """Out-of-fold calibrator, final booster fit on all of train.

    The two are fit on different data on purpose: the calibrator needs
    predictions that were never fit to their own labels, the booster wants every
    record available. Pairing them is the standard cross-validated calibration
    arrangement, and it is why no training data has to be spent.
    """
    out_of_fold = out_of_fold_scores(
        train_records, n_folds=n_folds, seed=seed, params=params, include_semantic=include_semantic
    )
    calibrator = PlattCalibrator().fit(out_of_fold.scores, out_of_fold.labels)
    scorer = train_scorer(
        prepare(train_records), params=params, include_semantic=include_semantic, seed=seed
    )
    return CalibratedFit(scorer=scorer.with_calibrator(calibrator), out_of_fold=out_of_fold)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error against the 0/1 outcome.

    Proper but not decomposed: it mixes calibration with discrimination, so a
    model that ranks better can score better while being worse calibrated. Read
    it beside the ECE, never instead of it.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels).astype(np.float64)
    return float(np.mean((p - y) ** 2)) if p.size else 0.0


def reliability_bins(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = RELIABILITY_BINS
) -> list[tuple[float, float, float, int]]:
    """Equal-width bins as `(bin lower edge, predicted mean, observed rate, count)`.

    Equal-width rather than equal-count because the question is whether the
    number *means* what it says at each level of confidence, and equal-count
    bins on a bimodal score distribution would merge 0.02 with 0.97.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)

    out = []
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        out.append((float(edges[b]), float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    return out


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = RELIABILITY_BINS
) -> float:
    """Count-weighted mean gap between predicted probability and observed rate."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.size == 0:
        return 0.0
    total = 0.0
    for _, predicted, observed, count in reliability_bins(p, labels, bins=bins):
        total += (count / p.size) * abs(predicted - observed)
    return float(total)
