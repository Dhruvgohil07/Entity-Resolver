"""Per-feature diagnostic -- what each column is worth before a model sees it.

`features/` cannot produce an F1. It has no classifier in it, and the number
that matters against the baseline's 0.5204 arrives with `model/`. What it can
produce, and what this CLI writes to `reports/features.md`, is the thing that
makes a later disappointment diagnosable: how much each column separates true
pairs from false ones on its own, and on how many pairs it holds a real value
at all.

That is worth committing because the failures it catches are silent ones. A
column that is constant, a column whose sign is backwards, and a column that
is imputed on 98% of pairs all look identical from inside `model/` -- they
show up as "the model did not improve much", and the search for why starts in
the wrong place.

Three rules the numbers here obey, each one a CLAUDE.md invariant rather than
a preference:

  * **PR-AUC, never ROC-AUC.** At roughly a thousand pairs per positive,
    ROC-AUC reads ~0.99 for a column with no practical value.
  * **The recall denominator is every true pair in the split**, not the ones
    blocking let through. So a column's PR-AUC is capped by the inherited
    blocking ceiling and cannot be inflated by pruning.
  * **The featurizer is fit on train and spent on test.** The table below is
    the test split, so it is measured the same way `reports/baseline_tfidf.md`
    measures the baseline.

A column ranked low here is not automatically dead. These are *univariate*
numbers, and LightGBM's whole advantage is interactions a single column cannot
express -- `code_best_ratio` near 1.0 means something quite different when
`model_number_exact` is 0 than when it is 1. Read this table to find broken
columns, not to prune the vector.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dedup.blocking.base import BlockerRun
from dedup.blocking.evaluate import default_blocker_set
from dedup.blocking.union import BlockerScore, ground_truth, score, union_run
from dedup.data import DATASETS, load_dataset
from dedup.eval.metrics import average_precision, precision_recall_curve
from dedup.eval.splits import count_true_pairs, group_by_entity, split_by_entity
from dedup.features.base import FeatureSpec
from dedup.features.vectorize import PairFeaturizer, pair_labels
from dedup.normalize import NormalizedRecord, normalize
from dedup.schema import Record

# Above this the two columns are near-duplicates. Not an error -- LightGBM is
# untroubled by collinearity -- but two columns costing two computations to
# say one thing is worth seeing in the report.
CORRELATION_ALERT = 0.95


@dataclass(frozen=True)
class FeatureDiagnostic:
    """One row of the feature table."""

    spec: FeatureSpec
    coverage: float  # fraction of pairs holding a real value, not the fill
    pr_auc: float | None  # None when nothing was covered
    # Class means over the *covered* pairs only. See `diagnose` for why.
    positive_mean: float
    negative_mean: float
    n_covered_positives: int
    n_covered: int

    @property
    def separation(self) -> float:
        return self.positive_mean - self.negative_mean

    @property
    def is_anti_predictive(self) -> bool:
        """True when the column points the opposite way to how it was declared."""
        if self.spec.is_indicator or self.coverage == 0.0:
            return False
        return self.separation < 0 if self.spec.higher_is_similar else self.separation > 0


@dataclass(frozen=True)
class SplitFeatures:
    """One side of the split, blocked and vectorized."""

    name: str
    n_records: int
    n_true_pairs: int
    blocking: BlockerScore
    labels: np.ndarray

    @property
    def n_candidates(self) -> int:
        return int(self.labels.size)

    @property
    def n_positives_found(self) -> int:
        return int(self.labels.sum())


@dataclass(frozen=True)
class FeatureReport:
    dataset: str
    seed: int
    test_fraction: float
    include_semantic: bool
    n_records: int
    n_entities: int
    n_true_pairs: int
    train: SplitFeatures
    test: SplitFeatures
    diagnostics: list[FeatureDiagnostic]
    correlated: list[tuple[str, str, float]]


def block(records: list[NormalizedRecord]) -> tuple[BlockerRun, BlockerScore]:
    """Run every blocker over one split and union the result.

    Blocking runs *inside* a split, never across the catalog. Cross-split
    pairs are all negatives by construction -- the entities were assigned to
    one side or the other -- so this loses no positives, and it keeps the
    packed pair indices meaningful as positions in this record list.
    """
    runs = [blocker.run(records) for blocker in default_blocker_set()]
    combined = union_run(runs)
    truth, n_true = ground_truth(records)
    return combined, score(combined, truth, len(records), n_true_pairs_total=n_true)


def diagnose(
    values: np.ndarray,
    labels: np.ndarray,
    spec: FeatureSpec,
    covered: np.ndarray,
    *,
    n_positives_total: int,
) -> FeatureDiagnostic:
    """One column's coverage, ranking power and class separation.

    **Everything here is measured on the covered pairs only**, and that is the
    whole correctness content of this function. Averaging the fill into the
    class means reads a null as a mismatch -- the exact error CLAUDE.md's
    missingness invariant exists to prevent, committed one layer up in the
    reporting path instead of in the vector.

    It is not a subtle effect. `brand_equal` is covered on 2.3% of positives
    against 24.3% of negatives, because Abt has no brand column and true pairs
    are overwhelmingly cross-source. Averaging 0.0 in at those rates reported
    the column as pointing *backwards* (0.0176 against 0.0890) when on the
    pairs where a brand actually exists it points forwards, and strongly
    (0.7500 against 0.3655).

    The same argument applies to the ranking. A fill of 0.0 ranks last for a
    similarity column, which is at least conservative; negated for a distance
    column it ranks *first*, so an absent price would sort as a perfect price
    match. Ranking only the covered pairs removes the asymmetry.

    `n_positives_total` stays the count of every true pair in the split rather
    than the covered ones. A column defined on a fifth of the candidates
    genuinely cannot rank the other four fifths, and charging it for that is
    the same rule the blocking denominator follows -- it is what stops a
    narrow column from looking strong by being asked less.
    """
    coverage = float(covered.mean()) if covered.size else 0.0
    oriented = values if spec.higher_is_similar else -values

    pr_auc: float | None = None
    if covered.any():
        curve = precision_recall_curve(
            oriented[covered], labels[covered], n_positives_total=n_positives_total
        )
        pr_auc = average_precision(curve)

    positives = values[covered & labels]
    negatives = values[covered & ~labels]
    return FeatureDiagnostic(
        spec=spec,
        coverage=coverage,
        pr_auc=pr_auc,
        positive_mean=float(positives.mean()) if positives.size else 0.0,
        negative_mean=float(negatives.mean()) if negatives.size else 0.0,
        n_covered_positives=int(positives.size),
        n_covered=int(covered.sum()),
    )


def correlated_pairs(
    values: np.ndarray, names: list[str], threshold: float = CORRELATION_ALERT
) -> list[tuple[str, str, float]]:
    """Column pairs whose absolute Pearson correlation exceeds `threshold`.

    Deliberately over the full matrix, fills included, unlike every other
    figure in this report. The question here is not "what does this column
    measure" but "do two columns cost two computations to say one thing", and
    the vector the model consumes is the one with the fills in it. Two columns
    that are near-identical *because* they share a missingness pattern are
    still near-identical to the model.

    Checked rather than assumed on Abt-Buy: restricting to co-covered pairs
    moves `digit_token_jaccard`/`digit_token_shared_count` from +0.959 to
    +0.951 and leaves `title_token_jaccard`/`title_idf_overlap` at +0.953, so
    neither alert here is a co-missingness artifact.
    """
    if values.shape[0] < 2:
        return []
    # Constant columns have zero variance and would divide by zero; they are
    # already visible in the table as zero separation.
    varying = values.std(axis=0) > 0
    if varying.sum() < 2:
        return []
    matrix = np.corrcoef(values[:, varying], rowvar=False)
    kept = [name for name, keep in zip(names, varying) if keep]
    out = []
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            if abs(matrix[i, j]) >= threshold:
                out.append((kept[i], kept[j], float(matrix[i, j])))
    return sorted(out, key=lambda row: -abs(row[2]))


def evaluate(
    records: list[Record],
    *,
    dataset: str,
    test_fraction: float = 0.3,
    seed: int = 0,
    include_semantic: bool = False,
) -> FeatureReport:
    train_records, test_records = split_by_entity(
        records, test_fraction=test_fraction, seed=seed
    )
    train = [normalize(record) for record in train_records]
    test = [normalize(record) for record in test_records]

    train_run, train_score = block(train)
    test_run, test_score = block(test)

    # Fit on train only. Fitting on the full catalog scores better and lets
    # test text into the IDF weights -- the same leak a pair-level split is,
    # just quieter (eval/baseline.py holds the identical line).
    featurizer = PairFeaturizer(include_semantic=include_semantic).fit(train)
    test_matrix = featurizer.transform(test, test_run.keys)
    test_labels = pair_labels(test, test_run.keys)
    train_labels = pair_labels(train, train_run.keys)

    n_positives_total = count_true_pairs(test_records)
    indicator_columns = {
        spec.name: test_matrix.column(spec.name) for spec in test_matrix.specs if spec.is_indicator
    }

    diagnostics = []
    for spec in test_matrix.specs:
        values = test_matrix.column(spec.name)
        if spec.companion_indicator is not None:
            covered = indicator_columns[spec.companion_indicator].astype(bool)
        else:
            covered = np.ones(values.shape[0], dtype=bool)
        diagnostics.append(
            diagnose(values, test_labels, spec, covered, n_positives_total=n_positives_total)
        )

    return FeatureReport(
        dataset=dataset,
        seed=seed,
        test_fraction=test_fraction,
        include_semantic=include_semantic,
        n_records=len(records),
        n_entities=len(group_by_entity(records)),
        n_true_pairs=count_true_pairs(records),
        train=SplitFeatures(
            name="train",
            n_records=len(train),
            n_true_pairs=count_true_pairs(train_records),
            blocking=train_score,
            labels=train_labels,
        ),
        test=SplitFeatures(
            name="test",
            n_records=len(test),
            n_true_pairs=n_positives_total,
            blocking=test_score,
            labels=test_labels,
        ),
        diagnostics=diagnostics,
        correlated=correlated_pairs(test_matrix.values, test_matrix.names),
    )


def _row(diagnostic: FeatureDiagnostic) -> str:
    spec = diagnostic.spec
    pr_auc = "—" if diagnostic.pr_auc is None else f"{diagnostic.pr_auc:.4f}"
    kind = "indicator" if spec.is_indicator else ("distance" if not spec.higher_is_similar else "")
    flag = " ⚠" if diagnostic.is_anti_predictive else ""
    return (
        f"| `{spec.name}`{flag} | {kind} | {diagnostic.coverage:.4f} | {pr_auc} "
        f"| {diagnostic.positive_mean:.4f} | {diagnostic.negative_mean:.4f} "
        f"| {diagnostic.separation:+.4f} | {diagnostic.n_covered_positives} |"
    )


def render_markdown(report: FeatureReport) -> str:
    """The reports/ artifact: numbers plus the caveats that make them readable."""
    ranked = sorted(
        report.diagnostics, key=lambda d: (d.pr_auc if d.pr_auc is not None else -1.0), reverse=True
    )
    rows = "\n".join(_row(diagnostic) for diagnostic in ranked)

    anti = [d for d in report.diagnostics if d.is_anti_predictive]
    anti_lines = (
        "\n".join(
            f"- `{d.spec.name}` — positives {d.positive_mean:.4f} against negatives "
            f"{d.negative_mean:.4f}, over {d.n_covered:,} covered pairs "
            f"({d.n_covered_positives} of them positive)."
            for d in anti
        )
        or "- None."
    )

    correlation_lines = (
        "\n".join(f"- `{a}` / `{b}` — r = {r:+.3f}" for a, b, r in report.correlated)
        or f"- No pair above |r| = {CORRELATION_ALERT}."
    )

    imbalance = report.test.n_candidates / max(report.test.n_true_pairs, 1)
    semantic = "on" if report.include_semantic else "off (`--semantic` enables it)"

    return f"""# Feature diagnostics

What each column of the pair vector is worth on its own, before any model sees
it. `features/` has no classifier in it, so there is no F1 here — the number
that answers `reports/baseline_tfidf.md`'s **test F1 0.5204** arrives with
`model/`. This table exists to catch the failures that are invisible from
inside a model: a constant column, a column whose sign is backwards, and a
column imputed on almost every pair all read as "the model did not improve".

Regenerate with:

```bash
python -m dedup.features.evaluate --dataset {report.dataset} --out reports/features.md
```

## Setup

- Dataset: `{report.dataset}` — {report.n_records} records, {report.n_entities} \
ground-truth entities, {report.n_true_pairs} true pairs.
- Split: entity-grouped, `test_fraction={report.test_fraction}`, `seed={report.seed}` \
→ {report.train.n_records} train / {report.test.n_records} test records.
- Featurizer fit on **train only**, measured on **test**. {len(report.diagnostics)} columns; \
semantic block {semantic}.
- Candidate pairs come from the full blocker union, run **inside each split**:

  | split | records | true pairs | candidates | PC | RR |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | train | {report.train.n_records} | {report.train.n_true_pairs} | \
{report.train.n_candidates:,} | {report.train.blocking.pair_completeness:.4f} | \
{report.train.blocking.reduction_ratio:.4f} |
  | test | {report.test.n_records} | {report.test.n_true_pairs} | \
{report.test.n_candidates:,} | {report.test.blocking.pair_completeness:.4f} | \
{report.test.blocking.reduction_ratio:.4f} |

- The test PC above is the **ceiling every number in this report inherits**. \
{report.test.n_positives_found} of {report.test.n_true_pairs} true pairs reached the \
candidate set; any that did not are counted as missed recall, never dropped from the denominator.
- Class balance on test: 1 positive per {imbalance:,.0f} candidate pairs.

## Per-feature diagnostics

Ranked by PR-AUC. **Coverage** is the fraction of candidate pairs where the column holds a
real value rather than the fill — the mean of the column's own `companion_indicator`, and
1.0 for columns that are always defined. **Sep** is positive mean minus negative mean, and
**n+** is how many covered pairs are true matches. ⚠ marks a column that separates the
classes in the opposite direction to how it was declared.

| feature | kind | coverage | PR-AUC | pos mean | neg mean | sep | n+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

Two things about how these are computed, both of which changed a published number here:

- **Every figure is over the covered pairs only.** Averaging the fill into a class mean
  reads a null as a mismatch — the error CLAUDE.md's missingness invariant exists to
  prevent, and it is just as wrong in a report as in a vector. `brand_equal` is covered on
  2.3% of positives against 24.3% of negatives, so averaging 0.0 in at those rates reported
  it as pointing *backwards*, when on pairs that actually have a brand it points forwards
  and strongly. The same restriction applies to the ranking, where a fill of 0.0 ranks last
  for a similarity column but — once negated — ranks *first* for a distance column, sorting
  an absent price as a perfect price match.
- **`distance` columns are negated before ranking**, so their PR-AUC is comparable with the
  rest. Without it `price_abs_log_ratio` would report as useless for being strong.

The recall denominator stays every true pair in the split, so a narrow column cannot look
strong by being asked less: `brand_equal` can address 8 of 341 true pairs, and its PR-AUC
is bounded near 0.02 accordingly. Read PR-AUC as *how much of the problem this column can
reach*, and the means as *whether it points the right way where it applies*. They answer
different questions and a column can score well on one and badly on the other.

## Columns pointing the wrong way

{anti_lines}

This is **not** a bug, and flipping the sign would be the wrong fix. It is the
deduplication framing showing through: every pair of the combined catalog is a candidate,
so same-side pairs (Buy against Buy) sit in the table alongside the cross-source pairs that
carry almost all the true matches. Abt descriptions average 249 characters against Buy's
34, so two descriptions of *similar* length are evidence of being same-side — which on this
dataset means evidence of being a non-match. A tree model uses that correctly. A human
reading the column name does not, which is why it is called out here.

Check `n+` before drawing a conclusion from any row in this section. A column covered on a
handful of true pairs can land here on noise, and the fix for that is more data, not a sign
flip. `brand_equal` was listed here in an earlier version of this report for a worse reason
than noise: the class means included the fill, so a missing brand was being counted as a
brand mismatch.

## Near-duplicate columns

{correlation_lines}

## Reading this honestly

- **These are univariate numbers and the model is not.** A low PR-AUC here means the
  column cannot separate the classes *alone*. `code_best_ratio` near 1.0 means one
  thing when `model_number_exact` is 1 and the opposite when it is 0, and no single-column
  metric can show that. Use this table to find broken columns, not to prune the vector.
- **No PR-AUC here is comparable to the baseline's 0.4720.** `title_tfidf_cosine` is
  very nearly the baseline's own similarity function, and it scores *higher* in this
  table — which measures the candidate set, not the column. The baseline ranks the full
  N² upper triangle; this ranks the {report.test.n_candidates:,} pairs blocking left,
  having already discarded {report.test.blocking.reduction_ratio:.2%} of them, almost all
  negatives. Precision rises because the negatives are gone, not because the column got
  better. The like-for-like comparison against 0.5204 is a trained, calibrated model
  scored end to end, and it does not exist until `model/`.
- **PR-AUC, not ROC-AUC.** At 1 positive per {imbalance:,.0f} pairs, ROC-AUC reads ~0.99
  for a column with no practical value (CLAUDE.md, Invariants).
- **Recall divides by every true pair in the split**, including the ones blocking never
  emitted. So no column's PR-AUC can exceed the test PC above, and pruning candidates
  cannot flatter these numbers.
- **Coverage below 1.0 is not a defect.** Every imputed column ships a companion indicator,
  so the model can tell a computed 0.0 from a filled one, and that pairing is enforced when
  the featurizer is constructed rather than by review. A low-coverage column still
  contributes more inside the vector than its row here suggests, because the model gets the
  indicator alongside it and this table scores the column alone.
- **The blocker parameters were swept on the full catalog**, so the PC figures inherited
  here are best-of-sweep rather than a clean estimate. Small bias, still unquantified —
  it is recorded as an open question in CLAUDE.md and not resolved by this report.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="abt-buy", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=None, help="override the dataset directory")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
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
