"""Pairwise ranking metrics for entity resolution.

Two things here are not what a default sklearn call gives you, and both are
CLAUDE.md invariants rather than preferences:

  * **PR-AUC, never ROC-AUC.** After blocking there are thousands of
    negatives per positive. ROC-AUC's false-positive-rate axis is divided by
    that huge negative count, so a model that ranks garbage above half the
    true pairs still scores ~0.99. Precision is divided by the *predicted*
    positives instead, so it actually moves.

  * **The candidate set is pruned, the recall denominator is not.** Every
    stage after blocking sees a subset of all N^2 pairs. A true pair that no
    blocker emitted, or that fell under a similarity floor, is a false
    negative that must still count against recall -- otherwise pruning
    harder *improves* the reported score, which is exactly backwards. So
    every function here takes `n_positives_total`, the number of true pairs
    over the whole catalog, and divides by that rather than by the positives
    that happen to be present in `labels`.

Passing `n_positives_total=None` falls back to `labels.sum()`, which is only
correct when the candidate set really is every pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrecisionRecallCurve:
    """One point per distinct score in the candidate set, ranked best first.

    `thresholds[i]` is a score to cut at inclusively (`score >= threshold`),
    which is the same convention `evaluate_at_threshold` applies.
    """

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    n_positives_total: int


@dataclass(frozen=True)
class ThresholdPoint:
    """A single operating point: one threshold and what it buys."""

    threshold: float
    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        return (
            f"threshold={self.threshold:.4f} P={self.precision:.4f} "
            f"R={self.recall:.4f} F1={self.f1:.4f}"
        )


def _f1(precision: float, recall: float) -> float:
    # Guarded because P and R are both 0 at any threshold above the top score.
    total = precision + recall
    return 0.0 if total == 0 else 2 * precision * recall / total


def _as_arrays(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must have equal shape")
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-d, got shape {scores.shape}")
    # NaN poisons silently rather than loudly: `nan >= threshold` is False at
    # every cut, so a NaN-scored true pair can never be admitted by any
    # threshold while still counting against the recall denominator -- the
    # curve just quietly reports a maximum that is not reachable. schema.py
    # rejects a non-finite price for the same reason; this is the same trap
    # one stage later.
    if not np.isfinite(scores).all():
        raise ValueError(
            f"{int((~np.isfinite(scores)).sum())} score(s) are NaN or infinite; "
            f"a pair that cannot be scored must be left out of the candidate set, "
            f"not scored NaN"
        )
    return scores, labels


def _resolve_total(labels: np.ndarray, n_positives_total: int | None) -> int:
    present = int(labels.sum())
    if n_positives_total is None:
        return present
    if n_positives_total < present:
        raise ValueError(
            f"n_positives_total={n_positives_total} is smaller than the "
            f"{present} positives present in labels"
        )
    return int(n_positives_total)


def precision_recall_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_positives_total: int | None = None,
) -> PrecisionRecallCurve:
    """Precision and recall at every distinct score, ranked best first.

    Not `sklearn.metrics.precision_recall_curve`: that one divides recall by
    the positives it can see. Here the divisor is `n_positives_total`, so
    true pairs that never reached the candidate set are charged to recall.
    With the default `None` the two agree.
    """
    scores, labels = _as_arrays(scores, labels)
    total_positives = _resolve_total(labels, n_positives_total)
    if total_positives == 0:
        raise ValueError("cannot build a PR curve with no positive pairs")
    if scores.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return PrecisionRecallCurve(empty, empty, empty, total_positives)

    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    ranked_labels = labels[order].astype(np.int64)

    true_positives = np.cumsum(ranked_labels)
    predicted = np.arange(1, ranked_scores.size + 1, dtype=np.int64)

    # A threshold admits *every* pair sharing that score, so only the last
    # index of each run of equal scores is a reachable operating point.
    # Reporting the intermediate ranks would claim a precision no threshold
    # can actually deliver -- with char-3gram cosine, exact ties are common.
    last_of_run = np.r_[np.nonzero(np.diff(ranked_scores))[0], ranked_scores.size - 1]

    return PrecisionRecallCurve(
        thresholds=ranked_scores[last_of_run],
        precision=true_positives[last_of_run] / predicted[last_of_run],
        recall=true_positives[last_of_run] / total_positives,
        n_positives_total=total_positives,
    )


def average_precision(curve: PrecisionRecallCurve) -> float:
    """PR-AUC as the step-wise sum of precision weighted by recall gained.

    sum (R_i - R_i-1) * P_i, matching `sklearn.average_precision_score`.
    Deliberately not a trapezoid/interpolated area: interpolation between two
    operating points assumes a classifier that can hit the midpoint, which
    biases the number upward -- badly so on a curve with the near-vertical
    drops that extreme class imbalance produces.
    """
    if curve.recall.size == 0:
        return 0.0
    recall = np.r_[0.0, curve.recall]
    return float(np.sum(np.diff(recall) * curve.precision))


def best_f1(curve: PrecisionRecallCurve) -> ThresholdPoint:
    """The operating point with the highest F1 on this curve.

    For the baseline only. Real thresholds come from expected cost, not
    argmax F1 (CLAUDE.md invariant) -- false merges and false splits are not
    equally expensive, and F1 assumes they are. The baseline has no cost
    model to answer to yet, and published Abt-Buy numbers are best-F1, so
    this is what makes it comparable.
    """
    if curve.thresholds.size == 0:
        return ThresholdPoint(threshold=float("inf"), precision=0.0, recall=0.0, f1=0.0)
    denominator = curve.precision + curve.recall
    scores = np.divide(
        2 * curve.precision * curve.recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    # np.argmax returns the *first* maximum and the curve runs from the
    # highest threshold down, so a tie resolves to the most conservative cut
    # -- the one admitting the fewest pairs. Stated rather than left to
    # argmax because it is load-bearing: false merges fuse two products and
    # corrupt the catalog, false splits merely leave a duplicate, so when F1
    # cannot tell two thresholds apart the higher one is the right default.
    best = int(np.argmax(scores))
    return ThresholdPoint(
        threshold=float(curve.thresholds[best]),
        precision=float(curve.precision[best]),
        recall=float(curve.recall[best]),
        f1=float(scores[best]),
    )


def evaluate_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    n_positives_total: int | None = None,
) -> ThresholdPoint:
    """Apply a fixed threshold (`score >= threshold`) to a scored candidate set.

    This is the honest way to report a baseline: pick the threshold on train,
    spend it here on test. `best_f1` on the test curve is an oracle that has
    already seen the answer.
    """
    scores, labels = _as_arrays(scores, labels)
    total_positives = _resolve_total(labels, n_positives_total)

    predicted_positive = scores >= threshold
    n_predicted = int(predicted_positive.sum())
    n_true_positive = int(np.count_nonzero(predicted_positive & labels))

    precision = n_true_positive / n_predicted if n_predicted else 0.0
    recall = n_true_positive / total_positives if total_positives else 0.0
    return ThresholdPoint(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
    )


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of the top-k ranked pairs that are true matches.

    The review-queue metric: a human works the queue top-down, so what
    matters is the purity of the first k pairs, not the shape of the whole
    curve.

    Two details are load-bearing, and an earlier version of this function got
    both wrong:

      * **The divisor is the requested k, never the number of candidates
        available.** Clamping k to `scores.size` turns "precision at 100"
        into "precision at however many pairs survived pruning", so a blocker
        that discards 97% of its candidates reports a *higher* precision@k
        for the same hits. That is the same backwards incentive the recall
        denominator above exists to remove, and `blocking/` would walk
        straight into it. Fewer than k candidates means the queue's remaining
        seats are empty, and an empty seat is not a hit.

      * **Ties straddling the k-th position are resolved by expectation, not
        by array order.** `np.argpartition` picks arbitrarily among equal
        scores, so the answer depended on the order the loader emitted rows
        in. Exact ties are common here (identical short titles score exactly
        1.0, and `synth/` will emit duplicate titles by construction). The
        tied block contributes its positive rate times the seats it fills --
        the same refusal to claim an unreachable operating point that
        `precision_recall_curve` makes by reporting only reachable cuts.
    """
    scores, labels = _as_arrays(scores, labels)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if scores.size == 0:
        return 0.0
    if k >= scores.size:
        # Every candidate is inside the queue; the shortfall stays as empty
        # seats in the divisor rather than shrinking it.
        return float(labels.sum()) / k

    boundary = float(np.partition(scores, -k)[-k])  # the k-th largest score
    above = scores > boundary
    hits = float(labels[above].sum())

    seats_left = k - int(above.sum())
    if seats_left > 0:
        tied = scores == boundary
        hits += float(labels[tied].sum()) * seats_left / int(tied.sum())
    return hits / k
