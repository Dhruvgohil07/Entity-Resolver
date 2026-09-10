"""The model report: the first number in this project comparable to the baseline.

`python -m dedup.model.evaluate --dataset abt-buy --out reports/model.md`

Two operating points are reported and both are needed, because they answer
different questions and only one of them is a fair comparison.

**Best-F1 threshold, chosen on train, spent on test.** This is exactly the
protocol `eval/baseline.py` follows, so it is the only like-for-like comparison
against the recorded baseline F1 of 0.5204. "Chosen on train" has a wrinkle a
learned model introduces and the baseline does not: in-sample train scores are
near-perfect, so a threshold picked on them is meaningless. The honest train-side
scores are the **out-of-fold** ones -- predictions on records the scoring model
never saw -- which the calibration step already produces. So the threshold comes
from there.

**The cost-derived bands.** This is what the system actually does, and it is not
an F1 at all: it is three populations and a bill. Reporting only the F1 would
hide the fact that the auto-merge band is chosen to be precision-heavy on
purpose, and reporting only the bands would leave the baseline uncontested.

What this report must not do is claim a clean win over the baseline. Blocking has
already discarded most of the pair space before the model sees any of it -- 91% of
the test triangle -- so the precision here is not measured on the same candidate
set as the baseline's, which scored every pair. The recall *is* comparable --
both divide by every true pair in the split -- and the "Reading this honestly"
section below states which is which.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dedup.data import DATASETS, load_dataset
from dedup.eval.metrics import (
    ThresholdPoint,
    average_precision,
    best_f1,
    evaluate_at_threshold,
    precision_at_k,
    precision_recall_curve,
)
from dedup.eval.splits import count_true_pairs, group_by_entity, split_by_entity
from dedup.model.calibrate import (
    DEFAULT_N_FOLDS,
    RELIABILITY_BINS,
    OutOfFoldScores,
    brier_score,
    expected_calibration_error,
    fit_calibrated_scorer,
    reliability_bins,
)
from dedup.model.threshold import BandSummary, CostModel, band_summary
from dedup.model.train import PreparedSplit, prepare
from dedup.schema import Record

# The title-only row of reports/baseline_tfidf.md, restated so this report is
# readable on its own. Not recomputed here: re-deriving it would mean rebuilding
# the baseline's full N^2 scoring on every model run. A test compares these
# against the committed report, so regenerating the baseline cannot leave them
# silently stale.
BASELINE_TEST_F1 = 0.5204
BASELINE_TEST_PRECISION = 0.4605
BASELINE_TEST_RECALL = 0.5982
BASELINE_TEST_PR_AUC = 0.4720
BASELINE_TEST_P_AT_K = {10: 0.600, 100: 0.620}
BASELINE_TEST_R_PRECISION = 0.504
BASELINE_THRESHOLD = 0.6243  # a cosine, not a probability
BASELINE_ORACLE_F1 = 0.5249

PRECISION_AT_K = (10, 100)
IMPORTANCE_ROWS = 10

# Ratios spanning the plausible range, to show how little the choice moves the
# system on this dataset. Every entry must leave a non-empty review band --
# `CostModel` rejects one that does not, which is the point of including 100:2.
SENSITIVITY_RATIOS = ((10.0, 2.0), (20.0, 2.0), (50.0, 2.0), (50.0, 5.0), (100.0, 2.0))

# Below this a probability is "confidently negative". Used only to describe
# where the auto-rejected true pairs sit, which is a diagnosis, not a metric.
CONFIDENT_NEGATIVE = 0.01

# If the review queue moves by less than this fraction of the candidates across
# the whole sensitivity grid, the cost ratio is not load-bearing on the dataset
# and the report says so; otherwise it says the opposite.
RATIO_INSENSITIVE_FRACTION = 0.01


@dataclass(frozen=True)
class SplitStats:
    """One side of the split after blocking."""

    name: str
    n_records: int
    n_candidates: int
    n_positives_found: int
    n_positives_total: int
    pair_completeness: float
    reduction_ratio: float

    @property
    def negatives_per_positive(self) -> float:
        found = max(self.n_positives_found, 1)
        return (self.n_candidates - self.n_positives_found) / found

    @property
    def n_blocking_missed(self) -> int:
        """True pairs no blocker emitted, so nothing downstream can recover them.

        Zero on this split, which is why the loss accounting below can be short.
        It is computed rather than assumed because a split where it is non-zero
        would otherwise under-report the loss: `BandSummary.n_missed` counts only
        auto-rejected *candidates*, and a pair blocking never produced is not a
        candidate.
        """
        return self.n_positives_total - self.n_positives_found


@dataclass(frozen=True)
class CalibrationStats:
    brier: float
    ece: float
    pr_auc: float
    bins: list[tuple[float, float, float, int]]


@dataclass(frozen=True)
class ModelReport:
    dataset: str
    seed: int
    test_fraction: float
    n_folds: int
    include_semantic: bool
    cost: CostModel
    n_records: int
    n_entities: int
    n_true_pairs: int
    train: SplitStats
    test: SplitStats
    out_of_fold: OutOfFoldScores
    threshold_point: ThresholdPoint  # chosen on the out-of-fold predictions
    test_point: ThresholdPoint  # that threshold, spent on test
    test_oracle: ThresholdPoint  # best F1 on test -- an upper bound, not a result
    test_precision_at_k: dict[int, float]
    test_r_precision: float
    raw: CalibrationStats
    calibrated: CalibrationStats
    bands: BandSummary
    sensitivity: list[BandSummary]
    importance: list[tuple[str, float]]
    n_missed_confidently: int  # auto-rejected true pairs scored below CONFIDENT_NEGATIVE


def _split_stats(name: str, split: PreparedSplit) -> SplitStats:
    n = len(split.records)
    all_pairs = n * (n - 1) // 2
    return SplitStats(
        name=name,
        n_records=n,
        n_candidates=split.n_candidates,
        n_positives_found=split.n_positives_found,
        n_positives_total=split.n_positives_total,
        pair_completeness=split.blocking.pair_completeness,
        reduction_ratio=1.0 - split.n_candidates / max(all_pairs, 1),
    )


def _calibration_stats(
    probabilities: np.ndarray, labels: np.ndarray, *, n_positives_total: int
) -> CalibrationStats:
    curve = precision_recall_curve(probabilities, labels, n_positives_total=n_positives_total)
    return CalibrationStats(
        brier=brier_score(probabilities, labels),
        ece=expected_calibration_error(probabilities, labels),
        pr_auc=average_precision(curve),
        bins=reliability_bins(probabilities, labels),
    )


def evaluate(
    records: list[Record],
    *,
    dataset: str,
    test_fraction: float = 0.3,
    seed: int = 0,
    n_folds: int = DEFAULT_N_FOLDS,
    cost: CostModel | None = None,
    include_semantic: bool = False,
) -> ModelReport:
    cost = cost or CostModel()
    train_records, test_records = split_by_entity(
        records, test_fraction=test_fraction, seed=seed
    )

    fit = fit_calibrated_scorer(
        train_records, n_folds=n_folds, seed=seed, include_semantic=include_semantic
    )
    train = prepare(train_records)
    test = prepare(test_records)

    raw_scores = fit.scorer.raw_scores(test.records, test.pair_keys)
    probabilities = fit.scorer.probabilities(test.records, test.pair_keys)

    # The threshold is chosen on out-of-fold predictions, which are the only
    # honest train-side scores a learned model has -- see the module docstring.
    # Choosing it on *calibrated* OOF probabilities looks circular, since the
    # calibrator was fit on these same scores, but introduces no optimism:
    # Platt is monotone, so `best_f1` selects a rank, and the raw and calibrated
    # OOF curves return the same operating point and the same test pairs.
    # `n_positives_total` is the whole train side rather than a per-fold sum,
    # which is correct only because entity-grouped folds keep every true pair
    # inside one fold -- a record-level fold split would break this denominator
    # silently.
    n_train_positives = count_true_pairs(train_records)
    out_of_fold = fit.out_of_fold
    oof_probabilities = fit.scorer.calibrator.transform(out_of_fold.scores)
    threshold_point = best_f1(
        precision_recall_curve(
            oof_probabilities, out_of_fold.labels, n_positives_total=n_train_positives
        )
    )

    test_curve = precision_recall_curve(
        probabilities, test.labels, n_positives_total=test.n_positives_total
    )
    summary = band_summary(
        probabilities, test.labels, cost, n_positives_total=test.n_positives_total
    )
    missed_confidently = int(
        np.count_nonzero(
            (probabilities < cost.auto_reject_threshold)
            & test.labels
            & (probabilities < CONFIDENT_NEGATIVE)
        )
    )

    return ModelReport(
        dataset=dataset,
        seed=seed,
        test_fraction=test_fraction,
        n_folds=n_folds,
        include_semantic=include_semantic,
        cost=cost,
        n_records=len(records),
        n_entities=len(group_by_entity(records)),
        n_true_pairs=count_true_pairs(records),
        train=_split_stats("train", train),
        test=_split_stats("test", test),
        out_of_fold=out_of_fold,
        threshold_point=threshold_point,
        test_point=evaluate_at_threshold(
            probabilities,
            test.labels,
            threshold_point.threshold,
            n_positives_total=test.n_positives_total,
        ),
        test_oracle=best_f1(test_curve),
        test_precision_at_k={
            k: precision_at_k(probabilities, test.labels, k) for k in PRECISION_AT_K
        },
        test_r_precision=precision_at_k(probabilities, test.labels, test.n_positives_total),
        raw=_calibration_stats(
            raw_scores, test.labels, n_positives_total=test.n_positives_total
        ),
        calibrated=_calibration_stats(
            probabilities, test.labels, n_positives_total=test.n_positives_total
        ),
        bands=summary,
        sensitivity=[
            band_summary(
                probabilities,
                test.labels,
                CostModel(false_merge=fm, false_split=fs, review=cost.review),
                n_positives_total=test.n_positives_total,
            )
            for fm, fs in SENSITIVITY_RATIOS
        ],
        importance=fit.scorer.gain_importance()[:IMPORTANCE_ROWS],
        n_missed_confidently=missed_confidently,
    )


def _band_row(summary: BandSummary) -> str:
    cost = summary.cost
    return (
        f"| {cost.false_merge:g} | {cost.false_split:g} "
        f"| {cost.auto_merge_threshold:.3f} | {cost.auto_reject_threshold:.3f} "
        f"| {summary.n_auto_merge:,} | {summary.n_false_merge} | {summary.n_review:,} "
        f"| {summary.n_missed} | {summary.auto_merge_precision:.4f} "
        f"| {summary.auto_merge_recall:.4f} | {summary.realized_cost:,.0f} |"
    )


def _reliability_row(low: float, predicted: float, observed: float, count: int) -> str:
    return (
        f"| {low:.2f}–{low + 1 / RELIABILITY_BINS:.2f} | {predicted:.4f} | {observed:.4f} "
        f"| {count:,} | {predicted - observed:+.4f} |"
    )


def _bullet(text: str) -> str:
    # Hyphens are never break points: `error-analyst` split across a line
    # renders as two words.
    return textwrap.fill(
        text,
        width=98,
        initial_indent="- ",
        subsequent_indent="  ",
        break_on_hyphens=False,
        break_long_words=False,
    )


def _reading_notes(report: ModelReport) -> list[str]:
    """The "Reading this honestly" bullets, each chosen from what was measured.

    No interpretive sentence here is printed unconditionally. The CLI takes any
    registered dataset, and a claim that is true on Abt-Buy and asserted
    regardless becomes a false claim on the first dataset where it is not --
    the same failure `features/evaluate.py` had with FILL_VALUE, one layer above
    the numbers it was describing. Branching on the *measurement* rather than
    on the dataset name also keeps CLAUDE.md's rule that nothing after `data/`
    may know which benchmark it is running on.
    """
    bands, cost, test, train = report.bands, report.cost, report.test, report.train
    notes = [
        (
            f"**Precision is not comparable to the baseline's; recall is.** Blocking discarded "
            f"{test.reduction_ratio:.2%} of the test triangle before the model saw anything, so "
            f"the negatives this model is scored against are the hard ones that survived. The "
            f"baseline scored the full N² triangle. Recall *is* like-for-like — both divide by "
            f"every true pair in the split, {test.n_positives_total} of them — so the honest "
            f"summary is that this beats the baseline as a **pipeline**, not that the classifier "
            f"beats TF-IDF on equal footing."
        )
    ]

    if test.n_blocking_missed == 0:
        notes.append(
            f"**The recall ceiling is inherited, and on the test split it does not bind.** "
            f"Blocking reaches PC {test.pair_completeness:.4f} on test "
            f"({train.pair_completeness:.4f} on train), so no test figure here is capped by the "
            f"blocker — a property of this split, not a general result."
        )
    else:
        notes.append(
            f"**The recall ceiling is inherited, and on the test split it binds.** Blocking "
            f"reaches PC {test.pair_completeness:.4f} on test, leaving {test.n_blocking_missed} "
            f"true pairs that no model can score, so every recall above is capped at "
            f"{test.pair_completeness:.4f}. When recall disappoints, check blocking first."
        )

    queues = [summary.n_review for summary in report.sensitivity]
    merges = [summary.cost.false_merge for summary in report.sensitivity]
    if max(queues) - min(queues) < RATIO_INSENSITIVE_FRACTION * max(test.n_candidates, 1):
        notes.append(
            f"**The cost ratio is not validated by this dataset.** Across the sensitivity grid "
            f"`C_fm` spans {min(merges):g}–{max(merges):g} and the review queue moves only "
            f"from {min(queues):,} to {max(queues):,} pairs out of {test.n_candidates:,}, because "
            f"few pairs score anywhere near either threshold. The ratio is a recorded judgment "
            f"call (CLAUDE.md), and it only becomes load-bearing once `synth/` populates the "
            f"middle of the distribution."
        )
    else:
        notes.append(
            f"**The cost ratio is load-bearing on this dataset.** Across the sensitivity grid the "
            f"review queue ranges from {min(queues):,} to {max(queues):,} pairs out of "
            f"{test.n_candidates:,}, so `C_fm` and `C_fs` should be set deliberately for this "
            f"catalog rather than left at the recorded default."
        )

    n_blocked_out = test.n_blocking_missed
    unreached = (
        f"{n_blocked_out} true pairs never reached the model at all, because no blocker "
        f"emitted them — a blocking problem, not a model one."
    )
    if bands.n_missed == 0:
        notes.append(
            "**No true pair that blocking emitted was auto-rejected.**"
            + (f" {unreached}" if n_blocked_out else "")
        )
    elif 2 * report.n_missed_confidently > bands.n_missed:
        blocking_part = (
            f"A further {unreached} " if n_blocked_out else "Blocking emitted every one of them. "
        )
        notes.append(
            f"**The auto-rejected true pairs are the real loss, and they are not a threshold "
            f"problem.** {bands.n_missed} true pairs fall below `p_lo`, and "
            f"{report.n_missed_confidently} of them score below {CONFIDENT_NEGATIVE} — the "
            f"model is confidently wrong, not undecided. {blocking_part}No threshold recovers a "
            f"pair the classifier buried; that is an `error-analyst` question for the next "
            f"stage, not a tuning knob."
        )
    else:
        notes.append(
            f"**Most auto-rejected true pairs are near misses.** {bands.n_missed} true pairs fall "
            f"below `p_lo`, but only {report.n_missed_confidently} score below "
            f"{CONFIDENT_NEGATIVE}; the rest sit between that and `p_lo` "
            f"({cost.auto_reject_threshold:.2f}), where a higher `C_fs` would route them to "
            f"review instead."
            + (f" A further {unreached}" if n_blocked_out else "")
        )

    ranks = {name: rank for rank, (name, _) in enumerate(report.importance, start=1)}
    if "desc_len_ratio" in ranks:
        notes.append(
            f"**`desc_len_ratio` ranks #{ranks['desc_len_ratio']} on gain, and that is "
            f"expected.** CLAUDE.md records it pointing backwards univariately on Abt-Buy — a "
            f"measured consequence of the deduplication framing — and a tree model uses a "
            f"backwards column correctly where a human reading the column name does not."
        )

    notes.append(
        "**No class reweighting, and hyperparameters are untuned.** Both are deliberate; see "
        "`model/train.py`. Tuning would need a third split, and tuning against test is the leak "
        "this protocol exists to prevent."
    )
    return [_bullet(note) for note in notes]


def render_markdown(report: ModelReport) -> str:
    """The reports/ artifact: the numbers, plus what makes them readable."""
    at_k = " | ".join(f"{report.test_precision_at_k[k]:.3f}" for k in PRECISION_AT_K)
    bands, cost = report.bands, report.cost
    sensitivity = "\n".join(_band_row(s) for s in report.sensitivity)
    reliability = "\n".join(_reliability_row(*row) for row in report.calibrated.bins)
    importance = "\n".join(
        f"| {rank} | `{name}` | {gain:,.0f} |"
        for rank, (name, gain) in enumerate(report.importance, start=1)
    )
    oof = report.out_of_fold
    lift = report.test_point.f1 - BASELINE_TEST_F1

    # Never assert that blocking lost nothing -- measure it. It is zero on this
    # split, but the sentence has to stay true on one where it is not.
    n_blocked_out = report.test.n_blocking_missed
    blocking_loss = (
        ""
        if not n_blocked_out
        else f", on top of {n_blocked_out} that blocking never emitted"
    )
    baseline_at_k = " | ".join(f"{BASELINE_TEST_P_AT_K[k]:.3f}" for k in PRECISION_AT_K)
    notes = "\n".join(_reading_notes(report))

    return f"""# Model: LightGBM over the pair vector, calibrated, cost-banded

The first number in this project comparable to the baseline's **test F1 \
{BASELINE_TEST_F1}** (`reports/baseline_tfidf.md`). Per-column PR-AUCs in
`reports/features.md` are not that number and never were.

Regenerate with:

```bash
python -m dedup.model.evaluate --dataset {report.dataset} --out reports/model.md
```

## Setup

- Dataset: `{report.dataset}` — {report.n_records:,} records, {report.n_entities:,} entities, \
{report.n_true_pairs:,} true pairs.
- Split: entity-grouped, `test_fraction={report.test_fraction}`, `seed={report.seed}` \
→ {report.train.n_records:,} train / {report.test.n_records:,} test records.
- Model: LightGBM over the {"34" if report.include_semantic else "33"}-column pair vector, \
featurizer fit on train only.
- Calibration: Platt, fit on out-of-fold predictions over {oof.n_folds} entity-grouped folds \
of train — {oof.n_pairs:,} pairs, {oof.n_positives:,} positives.
- Cost model: `{cost}`.

| split | records | candidates | true pairs found | in split | PC | RR | neg/pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | {report.train.n_records:,} | {report.train.n_candidates:,} \
| {report.train.n_positives_found:,} | {report.train.n_positives_total:,} \
| {report.train.pair_completeness:.4f} | {report.train.reduction_ratio:.4f} \
| {report.train.negatives_per_positive:.1f} |
| test | {report.test.n_records:,} | {report.test.n_candidates:,} \
| {report.test.n_positives_found:,} | {report.test.n_positives_total:,} \
| {report.test.pair_completeness:.4f} | {report.test.reduction_ratio:.4f} \
| {report.test.negatives_per_positive:.1f} |

## Results — threshold chosen on train, spent on test

The baseline's protocol, so this row is the like-for-like comparison. The
threshold is chosen on the **out-of-fold** train predictions, because in-sample
train scores from a fitted booster are near-perfect and a threshold picked on
them would mean nothing.

| | Test F1 | Test P | Test R | PR-AUC | P@10 | P@100 | R-prec | Threshold | Oracle F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model | **{report.test_point.f1:.4f}** | {report.test_point.precision:.4f} \
| {report.test_point.recall:.4f} | {report.calibrated.pr_auc:.4f} | {at_k} \
| {report.test_r_precision:.3f} | {report.threshold_point.threshold:.4f} \
| {report.test_oracle.f1:.4f} |
| baseline (TF-IDF) | {BASELINE_TEST_F1:.4f} | {BASELINE_TEST_PRECISION:.4f} \
| {BASELINE_TEST_RECALL:.4f} | {BASELINE_TEST_PR_AUC:.4f} | {baseline_at_k} \
| {BASELINE_TEST_R_PRECISION:.3f} | {BASELINE_THRESHOLD:.4f} | {BASELINE_ORACLE_F1:.4f} |

F1 {lift:+.4f} against the baseline. **Precision in these two rows is not measured
on the same candidate set** — see "Reading this honestly". Each row's threshold is
on its own scale: a cosine for the baseline, a calibrated probability for the model.

## Results — cost-derived bands

What the system actually does. Not an F1: three populations and a bill.

- **auto-merge** {bands.n_auto_merge:,} pairs, {bands.n_false_merge} of them wrong \
(precision {bands.auto_merge_precision:.4f}, recall {bands.auto_merge_recall:.4f} against \
all {report.test.n_positives_total} true pairs in the split)
- **review** {bands.n_review:,} pairs ({bands.review_fraction:.2%} of candidates), \
{bands.n_review_positives} of them true
- **auto-reject** {bands.n_auto_reject:,} pairs, losing {bands.n_missed} true pairs \
outright{blocking_loss}
- realized cost **{bands.realized_cost:,.0f}** review-equivalents

### Sensitivity to the cost ratio

| C_fm | C_fs | p_hi | p_lo | merge | bad | review | missed | auto-P | auto-R | cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{sensitivity}

## Calibration

| scores | Brier | ECE | PR-AUC |
| --- | ---: | ---: | ---: |
| raw booster | {report.raw.brier:.5f} | {report.raw.ece:.5f} | {report.raw.pr_auc:.4f} |
| Platt (out-of-fold) | {report.calibrated.brier:.5f} | {report.calibrated.ece:.5f} \
| {report.calibrated.pr_auc:.4f} |

PR-AUC is identical in both rows and must be: Platt is monotone, so it cannot
reorder pairs. Calibration moves what the number *means*, never the ranking —
which is also why it cannot flatter a ranking metric.

Reliability of the calibrated probabilities:

| bin | predicted | observed | pairs | gap |
| --- | ---: | ---: | ---: | ---: |
{reliability}

## Feature importance (LightGBM gain)

| # | feature | gain |
| ---: | --- | ---: |
{importance}

## Reading this honestly

{notes}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="abt-buy", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=None, help="override the dataset directory")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--folds", type=int, default=DEFAULT_N_FOLDS, help="entity-grouped calibration folds"
    )
    parser.add_argument("--cost-false-merge", type=float, default=CostModel().false_merge)
    parser.add_argument("--cost-false-split", type=float, default=CostModel().false_split)
    parser.add_argument("--cost-review", type=float, default=CostModel().review)
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="add the sentence-transformers column (downloads ~90 MB on first use)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the markdown report here")
    args = parser.parse_args(argv)

    records = load_dataset(args.dataset, args.root)
    report = evaluate(
        records,
        dataset=args.dataset,
        test_fraction=args.test_fraction,
        seed=args.seed,
        n_folds=args.folds,
        cost=CostModel(
            false_merge=args.cost_false_merge,
            false_split=args.cost_false_split,
            review=args.cost_review,
        ),
        include_semantic=args.semantic,
    )
    markdown = render_markdown(report)
    print(markdown)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
