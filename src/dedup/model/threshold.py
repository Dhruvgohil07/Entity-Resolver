"""Two thresholds from three costs -- the band assignment, derived not swept.

CLAUDE.md's invariant: **thresholds come from expected cost, not argmax F1.**
Fusing two distinct products corrupts the catalog and is hard to undo; letting
a duplicate survive is the status quo the system was built to improve on. Those
are not the same mistake and a single best-F1 cut prices them as if they were.

The derivation is closed-form, so nothing here sweeps. For a calibrated
probability `p` that a pair is a true duplicate, and assuming a reviewer who
decides correctly:

    auto-merge   costs  (1 - p) * C_fm      paid when the pair was not a match
    auto-reject  costs       p  * C_fs      paid when it was
    review       costs            C_review  paid always

Taking the cheaper of merge-or-review, and of reject-or-review, gives

    p_hi = 1 - C_review / C_fm      above it, merging beats paying for review
    p_lo =     C_review / C_fs      below it, rejecting beats paying for review

Two things follow that are easy to miss. The absolute scale cancels -- only the
two ratios against the review cost matter, so **the review action is the unit**
and a cost model is fully specified by "how many reviews is one false merge
worth" and "how many is one missed duplicate worth". And the review band is
non-empty only when `C_review * (1/C_fm + 1/C_fs) < 1`, which is a real
constraint at plausible-looking numbers rather than a defensive check: at
20 : 1 : 1 it gives `p_lo` 1.0 against `p_hi` 0.95, an inverted band. That is
rejected loudly at construction instead of silently producing a system that
reviews everything or nothing.

The default 20 : 2 : 1 is a judgment call recorded in CLAUDE.md, not a fitted
value, and Abt-Buy cannot validate it -- only 64 of 18,819 test candidates fall
between 0.01 and 0.95, so any `C_fm` from 10 to 100 yields substantially the
same system. It stays a parameter for that reason. `C_fm` is set as high as 20
partly because the pairwise cost here *structurally* understates a false merge:
`cluster/` joins these decisions by connected components, so one bad edge fuses
two whole clusters rather than mispairing two records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Band(str, Enum):
    """Where a scored pair is routed.

    Declaration order is also the column order of `expected_costs` and the
    tie-breaking order of `assign_bands`: a tie at `p_hi` resolves to merge and
    a tie at `p_lo` to review, which is what makes the closed form agree with
    the `>=` comparisons below. `>=` is the convention
    `eval/metrics.evaluate_at_threshold` already uses, so an operating point
    means the same thing in both places.
    """

    AUTO_MERGE = "auto-merge"
    REVIEW = "review"
    AUTO_REJECT = "auto-reject"


@dataclass(frozen=True)
class CostModel:
    """Relative costs of the three outcomes, in units of one review action.

    Defaults are CLAUDE.md's recorded 20 : 2 : 1. They are deliberately ratios
    rather than currency: the algebra only ever divides by `review`, so quoting
    money here would imply a precision the numbers do not have.
    """

    false_merge: float = 20.0  # C_fm -- reviews' worth of harm in one bad merge
    false_split: float = 2.0  # C_fs -- reviews' worth of harm in one lost duplicate
    review: float = 1.0  # C_review -- the unit

    def __post_init__(self) -> None:
        for name in ("false_merge", "false_split", "review"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if self.auto_reject_threshold >= self.auto_merge_threshold:
            raise ValueError(
                f"cost model leaves no review band: p_lo={self.auto_reject_threshold:.4f} is not "
                f"below p_hi={self.auto_merge_threshold:.4f}. Reviewing is only worth its cost "
                f"when C_review * (1/C_fm + 1/C_fs) < 1; here that is "
                f"{self.review * (1 / self.false_merge + 1 / self.false_split):.4f}. Lower "
                f"C_review, or raise C_fm and C_fs."
            )

    @property
    def auto_merge_threshold(self) -> float:
        """`p_hi` -- at or above it, merging is cheaper than paying for a review."""
        return 1.0 - self.review / self.false_merge

    @property
    def auto_reject_threshold(self) -> float:
        """`p_lo` -- below it, rejecting is cheaper than paying for a review."""
        return self.review / self.false_split

    def __str__(self) -> str:
        return (
            f"C_fm={self.false_merge:g} C_fs={self.false_split:g} C_review={self.review:g} "
            f"-> p_hi={self.auto_merge_threshold:.4f} p_lo={self.auto_reject_threshold:.4f}"
        )


def expected_costs(probabilities: np.ndarray, cost: CostModel) -> np.ndarray:
    """Per-pair expected cost of each action, as an (n, 3) array in `Band` order.

    Needs no labels, which is the point: this is the quantity the *serving*
    path can compute, and `assign_bands` is just its argmin. Keeping it public
    means the closed-form thresholds can be tested against the definition they
    claim to implement, rather than against themselves.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    return np.stack(
        [
            (1.0 - p) * cost.false_merge,
            np.full(p.shape, cost.review, dtype=np.float64),
            p * cost.false_split,
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class BandAssignment:
    """Boolean masks over the scored pairs, one per band.

    Masks rather than a label array because every consumer wants a selection:
    `cluster/` takes the auto-merge edges, the review queue takes its own band,
    and the report counts all three.
    """

    auto_merge: np.ndarray
    review: np.ndarray
    auto_reject: np.ndarray
    cost: CostModel

    @property
    def labels(self) -> np.ndarray:
        """One `Band` value per pair, for export rather than for selection."""
        out = np.empty(self.auto_merge.shape, dtype=object)
        out[self.auto_merge] = Band.AUTO_MERGE
        out[self.review] = Band.REVIEW
        out[self.auto_reject] = Band.AUTO_REJECT
        return out


def assign_bands(probabilities: np.ndarray, cost: CostModel) -> BandAssignment:
    """Route each calibrated probability to the action that costs least.

    The probabilities must be **calibrated**. An uncalibrated ranking score fed
    in here produces thresholds that are arithmetically valid and mean nothing,
    which is why `model/train.PairScorer.probabilities` refuses to return
    anything until a calibrator is attached.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ValueError(
            f"probabilities must lie in [0, 1], got [{p.min():.4f}, {p.max():.4f}] -- "
            f"this looks like a raw score rather than a calibrated probability"
        )
    auto_merge = p >= cost.auto_merge_threshold
    auto_reject = p < cost.auto_reject_threshold
    return BandAssignment(
        auto_merge=auto_merge,
        review=~auto_merge & ~auto_reject,
        auto_reject=auto_reject,
        cost=cost,
    )


@dataclass(frozen=True)
class BandSummary:
    """What a cost model bought on a scored candidate set with known labels."""

    cost: CostModel
    n_pairs: int
    n_positives_total: int  # every true pair in the split, not just those blocked in
    n_auto_merge: int
    n_review: int
    n_auto_reject: int
    n_false_merge: int  # auto-merged and not a duplicate -- catalog corruption
    n_missed: int  # auto-rejected and a duplicate -- lost without review
    n_review_positives: int
    realized_cost: float

    @property
    def n_correct_merge(self) -> int:
        return self.n_auto_merge - self.n_false_merge

    @property
    def auto_merge_precision(self) -> float:
        return self.n_correct_merge / self.n_auto_merge if self.n_auto_merge else 0.0

    @property
    def auto_merge_recall(self) -> float:
        """Against every true pair in the split -- never against the candidates.

        Dividing by the surviving candidates would reward a blocker or a floor
        for discarding pairs, the trap `eval/metrics.py` exists to close.
        """
        return self.n_correct_merge / self.n_positives_total if self.n_positives_total else 0.0

    @property
    def review_fraction(self) -> float:
        return self.n_review / self.n_pairs if self.n_pairs else 0.0


def band_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    cost: CostModel,
    *,
    n_positives_total: int,
) -> BandSummary:
    """Apply a cost model to scored pairs and count what it cost.

    `realized_cost` is what ground truth says was actually paid, which is not
    the expected cost the thresholds minimized -- the thresholds only ever saw
    `p`. Reporting the realized number keeps the cost model falsifiable: if a
    ratio is wrong for the domain, this is where it shows up as a bill.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if p.shape != y.shape:
        raise ValueError(f"probabilities and labels differ in shape: {p.shape} vs {y.shape}")

    bands = assign_bands(p, cost)
    n_false_merge = int(np.count_nonzero(bands.auto_merge & ~y))
    n_missed = int(np.count_nonzero(bands.auto_reject & y))
    n_review = int(np.count_nonzero(bands.review))
    return BandSummary(
        cost=cost,
        n_pairs=int(p.size),
        n_positives_total=n_positives_total,
        n_auto_merge=int(np.count_nonzero(bands.auto_merge)),
        n_review=n_review,
        n_auto_reject=int(np.count_nonzero(bands.auto_reject)),
        n_false_merge=n_false_merge,
        n_missed=n_missed,
        n_review_positives=int(np.count_nonzero(bands.review & y)),
        realized_cost=(
            n_false_merge * cost.false_merge + n_missed * cost.false_split + n_review * cost.review
        ),
    )
