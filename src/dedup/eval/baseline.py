"""TF-IDF char-3gram cosine baseline -- the number every later stage owes.

CLAUDE.md, "Baseline to beat": one vectorizer, one cosine, one threshold. No
blocking, no learned model, no features. Every stage built after this has to
justify itself against the F1 recorded here, which only works if this number
is honest, so the protocol is deliberately unflattering to it:

  * The vectorizer is fit on the **train** split only, and the threshold is
    chosen on the train split only. Both are then spent on test. Fitting IDF
    on the full catalog is common and would score better, but it lets test
    text influence the weights -- the same class of leak as a pair-level
    split, just quieter.
  * Recall is divided by *all* true pairs in the split, not by the ones that
    survived into the candidate set. See `metrics.py`.
  * The test-set oracle (best F1 chosen on test) is reported alongside, so
    the gap between it and the honest number is visible rather than
    accidentally claimed.

Scoring is the full N^2 upper triangle, chunked. That is the correct shape
for a *pre-blocking* reference: it measures the similarity function with no
recall ceiling above it, which is the thing `blocking/` will later be
compared against. It is also why this does not scale past a few thousand
records -- which is the argument for `blocking/` existing at all, and is
demonstrated rather than asserted once `synth/` lands.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from dedup.data import DATASETS, NO_NOTES, DatasetNotes, load_dataset
from dedup.eval.metrics import (
    ThresholdPoint,
    average_precision,
    best_f1,
    evaluate_at_threshold,
    precision_at_k,
    precision_recall_curve,
)
from dedup.eval.splits import count_true_pairs, group_by_entity, split_by_entity
from dedup.normalize import normalize
from dedup.schema import Record

# char_wb, not char: char_wb pads each word with spaces before slicing, so a
# short model number ("SD1100") contributes n-grams anchored to its own
# boundaries instead of being blended into its neighbours. Cross-word grams
# mostly encode word order, which is exactly the signal that vendor titles
# reorder freely ("Sony PSLX350H Turntable" vs "Sony Turntable - PSLX350H").
DEFAULT_ANALYZER = "char_wb"
DEFAULT_NGRAM_RANGE = (3, 3)

# Pairs at or below this cosine are dropped from the candidate set instead of
# being carried as scored rows. 0.0 keeps everything strictly positive, so on
# a dataset this size nothing is lost and the measurement is exact. Raising it
# is the memory knob; `n_positives_total` means recall stays honest when it is
# raised, and `recall_ceiling` below reports what it cost.
DEFAULT_MIN_SIMILARITY = 0.0

DEFAULT_CHUNK_SIZE = 512
PRECISION_AT_K = (10, 100)
DEFAULT_OUT = "reports/baseline_tfidf.md"

# The passage here that makes a dataset-specific claim, and the generic text it
# prints when a dataset's notes leave it alone (see data/notes.py).
PASSAGE_SLOTS = frozenset({"baseline.framing"})

_FRAMING = """\
- **This is the deduplication framing.** Every pair of the {n_records} records is a
  candidate, and a pair is positive when the two records share an `entity_id`. That makes
  {n_true_pairs} pairs true here, counted by transitivity: an entity of k records holds
  C(k, 2) of them."""


def comparison_text(record: Record, *, include_description: bool) -> str:
    """The string this baseline compares, built through `normalize`.

    Goes through `normalize()` rather than reading `Record.title` directly so
    the baseline sits on the same canonical text as every later stage -- a
    baseline scored on differently-prepared text is not a baseline, it is a
    second experiment (CLAUDE.md, train/serve parity).
    """
    normalized = normalize(record)
    if not include_description or not normalized.normalized_description:
        return normalized.normalized_title
    return f"{normalized.normalized_title} {normalized.normalized_description}"


def fit_vectorizer(
    records: list[Record],
    *,
    include_description: bool,
    analyzer: str = DEFAULT_ANALYZER,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> TfidfVectorizer:
    """Fit IDF on these records only -- pass the train split, never the catalog."""
    # lowercase=False: normalize.py already casefolded this text, and it is
    # the module CLAUDE.md makes responsible for canonical form in both the
    # batch and serve paths. Leaving sklearn's default on would put a second,
    # weaker case rule (str.lower, not casefold) inside the fitted artifact
    # where normalize.py cannot see it. Verified a no-op on this corpus --
    # identical matrices either way -- so this buys ownership, not a number.
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=False)
    vectorizer.fit([comparison_text(r, include_description=include_description) for r in records])
    return vectorizer


@dataclass(frozen=True)
class ScoredPairs:
    """Every pair scoring above the floor, plus the counts pruning would hide."""

    left: np.ndarray  # index into the record list, always < right
    right: np.ndarray
    score: np.ndarray
    label: np.ndarray  # same entity_id
    n_records: int
    n_pairs_total: int  # N*(N-1)/2, including pairs pruned by the floor
    n_positives_total: int  # true pairs among all N*(N-1)/2, ditto

    @property
    def recall_ceiling(self) -> float:
        """Best recall any threshold could reach given what the floor kept."""
        if self.n_positives_total == 0:
            return 0.0
        return float(self.label.sum()) / self.n_positives_total


def score_pairs(
    records: list[Record],
    vectorizer: TfidfVectorizer,
    *,
    include_description: bool,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ScoredPairs:
    """Cosine over the full upper triangle, in row chunks.

    TfidfVectorizer L2-normalizes its rows, so the dot product *is* the
    cosine and no separate normalization step is needed.
    """
    n = len(records)
    if n < 2:
        raise ValueError(f"need at least 2 records to form a pair, got {n}")

    # Already CSR and L2-normalized -- transform() guarantees both.
    matrix = vectorizer.transform(
        [comparison_text(r, include_description=include_description) for r in records]
    )

    # Entity ids compared as integer codes: `entity_of[left] == entity_of[right]`
    # over a million pairs has to be array work, not Python string equality.
    entity_ids = [r.entity_id for r in records]
    if any(entity_id is None for entity_id in entity_ids):
        raise ValueError("every record must have an entity_id to be scored against ground truth")
    codes = {entity_id: index for index, entity_id in enumerate(sorted(set(entity_ids)))}
    entity_of = np.array([codes[entity_id] for entity_id in entity_ids], dtype=np.int32)

    columns = np.arange(n)
    lefts, rights, scores = [], [], []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = (matrix[start:stop] @ matrix.T).toarray()
        rows = np.arange(start, stop)
        # Upper triangle only: each unordered pair scored once, no self-pairs.
        keep = (columns[None, :] > rows[:, None]) & (block > min_similarity)
        row_index, column_index = np.nonzero(keep)
        lefts.append(rows[row_index])
        rights.append(column_index)
        scores.append(block[row_index, column_index])

    left = np.concatenate(lefts) if lefts else np.empty(0, dtype=np.int64)
    right = np.concatenate(rights) if rights else np.empty(0, dtype=np.int64)
    score = np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)

    return ScoredPairs(
        left=left,
        right=right,
        score=score,
        label=entity_of[left] == entity_of[right],
        n_records=n,
        n_pairs_total=n * (n - 1) // 2,
        n_positives_total=count_true_pairs(records),
    )


@dataclass(frozen=True)
class VariantResult:
    """One text configuration, measured end to end."""

    name: str
    include_description: bool
    n_features: int
    n_candidates: int
    recall_ceiling: float
    train_point: ThresholdPoint
    test_point: ThresholdPoint  # the train threshold, spent on test
    test_oracle: ThresholdPoint  # best F1 chosen on test -- an upper bound, not a result
    test_pr_auc: float
    test_precision_at_k: dict[int, float] = field(default_factory=dict)
    test_r_precision: float = 0.0


@dataclass(frozen=True)
class BaselineReport:
    dataset: str
    seed: int
    test_fraction: float
    analyzer: str
    ngram_range: tuple[int, int]
    min_similarity: float
    n_records: int
    n_entities: int
    n_true_pairs: int
    n_all_pairs: int
    n_train_records: int
    n_train_all_pairs: int
    n_test_records: int
    n_test_true_pairs: int
    n_test_all_pairs: int
    variants: list[VariantResult]


def run_variant(
    train: list[Record],
    test: list[Record],
    *,
    name: str,
    include_description: bool,
    analyzer: str = DEFAULT_ANALYZER,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> VariantResult:
    """Fit on train, choose the threshold on train, spend it on test."""
    vectorizer = fit_vectorizer(
        train,
        include_description=include_description,
        analyzer=analyzer,
        ngram_range=ngram_range,
    )
    scored = {
        "train": score_pairs(
            train,
            vectorizer,
            include_description=include_description,
            min_similarity=min_similarity,
        ),
        "test": score_pairs(
            test,
            vectorizer,
            include_description=include_description,
            min_similarity=min_similarity,
        ),
    }

    train_curve = precision_recall_curve(
        scored["train"].score,
        scored["train"].label,
        n_positives_total=scored["train"].n_positives_total,
    )
    train_point = best_f1(train_curve)

    test_pairs = scored["test"]
    test_curve = precision_recall_curve(
        test_pairs.score, test_pairs.label, n_positives_total=test_pairs.n_positives_total
    )

    return VariantResult(
        name=name,
        include_description=include_description,
        n_features=len(vectorizer.vocabulary_),
        n_candidates=int(test_pairs.score.size),
        recall_ceiling=test_pairs.recall_ceiling,
        train_point=train_point,
        test_point=evaluate_at_threshold(
            test_pairs.score,
            test_pairs.label,
            train_point.threshold,
            n_positives_total=test_pairs.n_positives_total,
        ),
        test_oracle=best_f1(test_curve),
        test_pr_auc=average_precision(test_curve),
        test_precision_at_k={
            k: precision_at_k(test_pairs.score, test_pairs.label, k) for k in PRECISION_AT_K
        },
        # R-precision: precision at k = the number of true pairs there are.
        # Scale-free, so it stays comparable across splits and datasets in a
        # way a fixed k does not.
        test_r_precision=precision_at_k(
            test_pairs.score, test_pairs.label, test_pairs.n_positives_total
        ),
    )


def run_baseline(
    records: list[Record],
    *,
    dataset: str,
    test_fraction: float = 0.3,
    seed: int = 0,
    analyzer: str = DEFAULT_ANALYZER,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> BaselineReport:
    train, test = split_by_entity(records, test_fraction=test_fraction, seed=seed)
    variants = [
        run_variant(
            train,
            test,
            name=name,
            include_description=include_description,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_similarity=min_similarity,
        )
        for name, include_description in (("title", False), ("title + description", True))
    ]
    n = len(records)
    n_test = len(test)
    return BaselineReport(
        dataset=dataset,
        seed=seed,
        test_fraction=test_fraction,
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_similarity=min_similarity,
        n_records=n,
        n_entities=len(group_by_entity(records)),
        n_true_pairs=count_true_pairs(records),
        n_all_pairs=n * (n - 1) // 2,
        n_train_records=len(train),
        n_train_all_pairs=len(train) * (len(train) - 1) // 2,
        n_test_records=n_test,
        n_test_true_pairs=count_true_pairs(test),
        n_test_all_pairs=n_test * (n_test - 1) // 2,
        variants=variants,
    )


@dataclass(frozen=True)
class BaselineRow:
    """One variant's results row of a committed baseline report, as a later report quotes it.

    Read from the committed file rather than recomputed -- recomputing means
    rescoring the full N^2 triangle on every model run -- and rather than copied
    into the quoting module as constants, so regenerating the baseline cannot
    leave a stale copy behind and the numbers always belong to the dataset the
    report names.
    """

    dataset: str
    path: str  # posix, as a report cites it
    f1: float
    precision: float
    recall: float
    pr_auc: float
    precision_at_k: dict[int, float]
    r_precision: float
    threshold: float  # a cosine, not a probability
    oracle_f1: float
    min_similarity: float  # pairs at or below this cosine were never scored; 0.0 is no floor
    recall_ceiling: float  # the most recall this variant could reach given that floor


_DATASET_LINE = re.compile(r"^- Dataset: `([^`]+)`", re.MULTILINE)
_FLOOR_LINE = re.compile(r"^- Similarity floor: (?:none|`([^`]+)`)", re.MULTILINE)


def read_baseline_row(path: Path, *, dataset: str, variant: str = "title") -> BaselineRow:
    """Parse `variant`'s results row out of a report `render_markdown` wrote.

    Refuses a report whose Setup names a different dataset. A model scored
    against another catalog's baseline yields two well-formed numbers and no
    comparison, and nothing after this line could tell.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    named = _DATASET_LINE.search(text)
    if named is None:
        raise ValueError(f"{path} names no dataset, so it is not a baseline report")
    if named.group(1) != dataset:
        raise ValueError(
            f"{path} is the baseline for {named.group(1)!r}, not {dataset!r}; a model scored "
            f"against another dataset's baseline is not a comparison"
        )
    prefix = f"| {variant} |"
    row = next((line for line in text.splitlines() if line.startswith(prefix)), None)
    if row is None:
        raise ValueError(f"{path} has no {variant!r} results row")

    cells = [cell.strip().strip("*") for cell in row.strip().strip("|").split("|")]
    n_at_k = len(PRECISION_AT_K)
    if len(cells) != 8 + n_at_k:
        raise ValueError(
            f"{path}: the {variant!r} row has {len(cells)} cells, expected {8 + n_at_k}"
        )
    _, f1, precision, recall, pr_auc = cells[:5]
    at_k = cells[5 : 5 + n_at_k]
    r_precision, threshold, oracle_f1 = cells[5 + n_at_k :]

    floor = _FLOOR_LINE.search(text)
    if floor is None:
        raise ValueError(f"{path} states no similarity floor, so its reach cannot be quoted")
    ceiling = re.search(
        rf"^- \*\*{re.escape(variant)}\*\* — .*recall ceiling ([0-9.]+)\)", text, re.MULTILINE
    )
    if ceiling is None:
        raise ValueError(f"{path} states no recall ceiling for {variant!r}")
    return BaselineRow(
        dataset=dataset,
        path=path.as_posix(),
        min_similarity=float(floor.group(1)) if floor.group(1) else 0.0,
        recall_ceiling=float(ceiling.group(1)),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        pr_auc=float(pr_auc),
        precision_at_k={k: float(value) for k, value in zip(PRECISION_AT_K, at_k)},
        r_precision=float(r_precision),
        threshold=float(threshold),
        oracle_f1=float(oracle_f1),
    )


def registered_baseline(notes: DatasetNotes, dataset: str) -> BaselineRow | None:
    """The baseline a dataset's notes register, or None when it has none yet.

    A registered report that does not exist warns instead of raising: a report
    on a catalog whose baseline has not been run is still a report, and it says
    that it compares against nothing.
    """
    if notes.baseline_report is None:
        return None
    if not notes.baseline_report.is_file():
        print(
            f"warning: {notes.baseline_report.as_posix()} not found; reporting without a "
            f"baseline",
            file=sys.stderr,
        )
        return None
    return read_baseline_row(notes.baseline_report, dataset=dataset)


def _results_row(variant: VariantResult) -> str:
    at_k = " | ".join(f"{variant.test_precision_at_k[k]:.3f}" for k in PRECISION_AT_K)
    return (
        f"| {variant.name} | **{variant.test_point.f1:.4f}** "
        f"| {variant.test_point.precision:.4f} | {variant.test_point.recall:.4f} "
        f"| {variant.test_pr_auc:.4f} | {at_k} | {variant.test_r_precision:.3f} "
        f"| {variant.train_point.threshold:.4f} | {variant.test_oracle.f1:.4f} |"
    )


def _variant_detail(variant: VariantResult, n_test_all_pairs: int) -> str:
    # Reduction ratio and recall ceiling are the pair a pruning stage is
    # always judged on (CLAUDE.md, blocking). The floor here is a pruner, so
    # it reports both -- otherwise a run that discarded 99% of pairs looks
    # identical in this section to one that discarded none.
    reduction = 1 - variant.n_candidates / max(n_test_all_pairs, 1)
    return (
        f"- **{variant.name}** — {variant.n_features:,} char n-gram features, "
        f"{variant.n_candidates:,} scored candidate pairs "
        f"(reduction ratio {reduction:.4f}, recall ceiling {variant.recall_ceiling:.4f}).\n"
        f"  - train: {variant.train_point}\n"
        f"  - test:  {variant.test_point}"
    )


def floor_caveat(row: BaselineRow) -> str | None:
    """What a raised similarity floor did to a quoted baseline row, or None without one.

    The floor discards pairs before scoring, so the row's reach stops at its recall
    ceiling. Figures at the train-chosen threshold are still exact when that
    threshold sits above the floor -- no discarded pair could have passed it -- but
    PR-AUC is integrated only up to the ceiling, so it is a lower bound.
    """
    if row.min_similarity <= DEFAULT_MIN_SIMILARITY:
        return None
    if row.threshold > row.min_similarity:
        exact = (
            f"Its F1, precision and recall are exact — the train-chosen threshold "
            f"{row.threshold:.4f} sits above the floor, so no discarded pair could have passed "
            f"it — but its PR-AUC ends at that ceiling and is a lower bound."
        )
    else:
        exact = (
            f"Its train-chosen threshold {row.threshold:.4f} sits at or below the floor, so every "
            f"figure in the row is a lower bound."
        )
    return (
        f"**The baseline row was scored above a similarity floor of {row.min_similarity:g}.** "
        f"Pairs at or below that cosine were never scored, so the baseline reaches recall "
        f"{row.recall_ceiling:.4f} at most rather than the full triangle's. {exact}"
    )


# Below this F1 gap between the test oracle and the train-chosen threshold, calling
# the threshold a bad transfer overstates what is closer to a rounding difference --
# an audit caught synth-20k's 0.0030 gap contradicting a flat "does not transfer"
# claim printed regardless of the number.
GAP_TRANSFERS_WELL = 0.01


def _threshold_transfer_bullet(report: BaselineReport, size_ratio: float) -> str:
    """Whether train F1 lands below test F1 is measured per catalog, never asserted.

    On Abt-Buy it lands below, for the reason the first branch gives. On synth-20k
    it lands above, and printing the Abt-Buy explanation there would explain away
    a gap that does not exist.

    Whether the train-chosen threshold itself transfers to test is a *separate*
    question from that size effect, and it is answered the same way in both
    branches: by the gap to the test oracle, not by asserting "does not transfer"
    regardless of how small that gap measures.
    """
    variant = report.variants[0]
    gap = variant.test_oracle.f1 - variant.test_point.f1
    if gap < GAP_TRANSFERS_WELL:
        transfer = (
            f"it transfers well: the test oracle reaches only {gap:.4f} F1 higher, so "
            f"re-tuning the cut with test labels in hand would have bought almost nothing "
            f"on this split"
        )
    else:
        transfer = (
            f"it does not transfer well: the test oracle reaches {gap:.4f} F1 higher, real "
            f"F1 a threshold search on test alone would have found"
        )
    if variant.train_point.f1 < variant.test_point.f1:
        return f"""- **Train F1 comes out *below* test F1, and that is not a bug.** The train side holds
  {report.n_train_all_pairs:,} pairs against test's {report.n_test_all_pairs:,} — {size_ratio:.1f}x as many —
  while true pairs grow only linearly with records. At a fixed cosine threshold the false
  positives scale with the pair count and the true positives do not, so precision, and
  with it F1, falls as the catalog grows — a size effect, separate from whether the
  train-chosen threshold carries over to test: {transfer}."""
    return f"""- **Train F1 comes out *above* test F1 on this catalog** ({variant.train_point.f1:.4f} against
  {variant.test_point.f1:.4f}), although the train side holds {report.n_train_all_pairs:,} pairs against
  test's {report.n_test_all_pairs:,} — {size_ratio:.1f}x as many. A fixed cosine threshold's false positives
  scale with the pair count and its true positives do not, which on its own pushes the larger
  side's precision down; here that size effect did not dominate. That size effect is separate
  from whether the train-chosen threshold carries over to test: {transfer}."""


def _scope_bullet(report: BaselineReport) -> str:
    """"No recall ceiling" is only true when no similarity floor was raised."""
    if report.min_similarity <= DEFAULT_MIN_SIMILARITY:
        return """- **No blocking.** Scoring is the full upper triangle, so this measures the similarity
  function with no recall ceiling above it. That is what makes it the right reference for
  `blocking/` — and why it will not scale to `synth/`."""
    ceiling = report.variants[0].recall_ceiling
    return f"""- **No blocking, but a similarity floor.** Scoring covers the full upper triangle, but pairs at
  or below cosine {report.min_similarity} were discarded to fit in memory, so the title variant is
  measured under a recall ceiling of {ceiling:.4f} rather than none. Recall still divides by every
  true pair, so the floor shows as lost recall — compare `blocking/` against this with that
  ceiling in mind."""


def render_markdown(
    report: BaselineReport, *, notes: DatasetNotes = NO_NOTES, out: str = DEFAULT_OUT
) -> str:
    """The reports/ artifact: numbers plus the caveats that make them readable.

    `notes` may replace the framing passage with the dataset's own, and `out` is
    the path the regenerate command names, so a report written anywhere
    reproduces itself instead of pointing at another dataset's file.
    """
    low, high = report.ngram_range
    imbalance = report.n_test_all_pairs / max(report.n_test_true_pairs, 1)
    size_ratio = report.n_train_all_pairs / max(report.n_test_all_pairs, 1)
    rows = "\n".join(_results_row(variant) for variant in report.variants)
    details = "\n".join(
        _variant_detail(variant, report.n_test_all_pairs) for variant in report.variants
    )
    floor = (
        "none — every pair with a non-zero cosine is scored"
        if report.min_similarity <= 0
        else f"`{report.min_similarity}` — pairs at or below it are discarded unscored"
    )
    floor_flag = (
        ""
        if report.min_similarity == DEFAULT_MIN_SIMILARITY
        else f" --min-similarity {report.min_similarity}"
    )
    transfer = _threshold_transfer_bullet(report, size_ratio)
    scope = _scope_bullet(report)
    framing = notes.passage(
        "baseline.framing",
        _FRAMING,
        n_records=report.n_records,
        n_true_pairs=report.n_true_pairs,
    )

    return f"""# Baseline: TF-IDF char-3gram cosine

The number every later stage is justified against (CLAUDE.md, "Baseline to
beat"). No blocking, no features, no learned model: one vectorizer, one
cosine, one threshold.

Regenerate with:

```bash
python -m dedup.eval.baseline --dataset {report.dataset}{floor_flag} --out {out}
```

## Setup

- Dataset: `{report.dataset}` — {report.n_records} records, \
{report.n_entities} ground-truth entities.
- {report.n_true_pairs} true pairs among {report.n_all_pairs:,} possible pairs.
- Vectorizer: `TfidfVectorizer(analyzer="{report.analyzer}", ngram_range=({low}, {high}))`, \
L2-normalized, so the dot product is the cosine.
- Split: entity-grouped, `test_fraction={report.test_fraction}`, `seed={report.seed}` \
→ {report.n_train_records} train / {report.n_test_records} test records.
- Test side: {report.n_test_true_pairs} true pairs in {report.n_test_all_pairs:,} pairs \
— 1 positive per {imbalance:,.0f} pairs.
- Similarity floor: {floor}. Recall is divided by all {report.n_test_true_pairs} test true pairs
  either way, so a raised floor shows up as lost recall rather than as a smaller denominator.

The vectorizer is fit on the train split and the threshold is chosen on the
train split; both are then spent on test. Recall is divided by every true pair
in the test split, including any the similarity floor never scored.

## Results

| Text | Test F1 | Test P | Test R | PR-AUC | P@10 | P@100 | R-prec | Threshold | Oracle F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

**Threshold** is the best-F1 cut chosen on train. **Oracle F1** is the best F1
reachable on test, chosen with the test labels in hand — not a result, just the
size of what threshold selection left on the table.

Per-variant detail:

{details}

## Reading this honestly

{framing}
- **PR-AUC, not ROC-AUC.** At 1 positive per {imbalance:,.0f} pairs, ROC-AUC reads ~0.99
  for a model with no practical value.
{transfer}
- **Best-F1 thresholds are a baseline convention, not the plan.** The real thresholds come
  from expected cost, because a false merge corrupts the catalog and a false split merely
  leaves a duplicate (CLAUDE.md, Invariants).
{scope}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="abt-buy", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=None, help="override the dataset directory")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--out", type=Path, default=None, help="write the markdown report here")
    args = parser.parse_args(argv)

    records = load_dataset(args.dataset, args.root)
    report = run_baseline(
        records,
        dataset=args.dataset,
        test_fraction=args.test_fraction,
        seed=args.seed,
        min_similarity=args.min_similarity,
    )
    markdown = render_markdown(
        report,
        notes=DATASETS[args.dataset].notes,
        out=DEFAULT_OUT if args.out is None else args.out.as_posix(),
    )
    print(markdown)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
