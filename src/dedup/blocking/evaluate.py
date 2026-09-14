"""The blocking table: blocker x pair completeness x reduction ratio.

`python -m dedup.blocking.evaluate --dataset abt-buy --out reports/blocking.md`

Two numbers, always together. Either alone is uninterpretable -- a blocker
that emits nothing has a perfect reduction ratio, and one that emits all N^2
has perfect completeness:

  * **Pair completeness** is the ceiling. If the union reaches 0.97, the
    finished system's recall cannot exceed 0.97 however good the classifier
    becomes, because the pairs it missed are never scored by anything.
  * **Reduction ratio** is the saving, reported beside the absolute candidate
    count because the ratio flatters what the count states plainly: at 200k
    records RR 0.999 still leaves 20M pairs for `features/` to compute.

There is no precision, F1 or accuracy column, and adding one would be a bug
rather than a low score.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dedup.blocking.base import Blocker, BlockerRun
from dedup.blocking.defaults import default_blocker_set
from dedup.blocking.pairs import total_pairs, unpack
from dedup.blocking.union import (
    BlockerScore,
    ground_truth,
    missed_pairs,
    score,
    union_run,
)
from dedup.data import DATASETS, NO_NOTES, DatasetNotes, load_dataset
from dedup.normalize import NormalizedRecord, normalize

MISSED_EXAMPLES = 10
DEFAULT_OUT = "reports/blocking.md"

# Passages that make a dataset-specific claim; see data/notes.py. None has
# generic text: a dataset nobody has written notes for gets no provenance or
# tuning claim at all, which is the only claim that is true of every catalog.
PASSAGE_SLOTS = frozenset({"blocking.intro", "blocking.provenance", "blocking.tuning"})

# Past this many possible pairs the exhaustive baseline cannot hold its scored
# pairs in 1 GiB -- it keeps two int64 indices and a float64 score per pair --
# which is where the reduction ratio stops flattering a small catalog and
# becomes the reason the pipeline can run at all.
BRUTE_FORCE_PAIRS = 2**30 // 24

_CEILING_BULLET = """\
- **Which ceiling applies depends on what is being compared.** The {ceiling:.4f} above is the
  batch-deduplication number over the whole catalog. When `features/` and `model/` report
  test-split recall against the baseline's test-split F1, the ceiling that binds them is
  the test-split one, not this."""


@dataclass(frozen=True)
class BlockingReport:
    dataset: str
    n_records: int
    n_true_pairs: int
    n_all_pairs: int
    rows: list[BlockerScore]
    union_row: BlockerScore
    missed_examples: list[tuple[str, str]]
    n_missed: int
    warnings: list[str]
    omitted: tuple[str, ...] = ()  # default blockers left out of this run, by name


def evaluate(
    records: Sequence[NormalizedRecord],
    blockers: Sequence[Blocker],
    *,
    dataset: str,
    omitted: Sequence[str] = (),
) -> BlockingReport:
    n = len(records)
    truth, n_true = ground_truth(records)

    runs: list[BlockerRun] = [blocker.run(records) for blocker in blockers]
    rows = [score(run, truth, n, n_true_pairs_total=n_true) for run in runs]

    combined = union_run(runs)
    union_score = score(combined, truth, n, n_true_pairs_total=n_true)

    missed = missed_pairs(combined.keys, truth)
    left, right = unpack(missed, n)
    examples = [
        (records[int(a)].raw.title, records[int(b)].raw.title)
        for a, b in zip(left[:MISSED_EXAMPLES], right[:MISSED_EXAMPLES])
    ]

    return BlockingReport(
        dataset=dataset,
        n_records=n,
        n_true_pairs=n_true,
        n_all_pairs=total_pairs(n),
        rows=rows,
        union_row=union_score,
        missed_examples=examples,
        n_missed=int(missed.size),
        warnings=[f"{run.name}: {w}" for run in runs for w in run.warnings],
        omitted=tuple(omitted),
    )


def _row(s: BlockerScore) -> str:
    build = "—" if s.build_seconds is None else f"{s.build_seconds:.1f}"
    query = "—" if s.query_seconds is None else f"{s.query_seconds:.1f}"
    return (
        f"| {s.name} | {s.params} | {s.n_candidates:,} | "
        f"{s.pair_completeness:.4f} | {s.reduction_ratio:.4f} | {build} | {query} |"
    )


def _omitted_bullet(omitted: Sequence[str]) -> str:
    """A union of fewer blockers is a lower ceiling, and must not pass for the default one."""
    if not omitted:
        return ""
    names = ", ".join(f"`{name}`" for name in omitted)
    return f"""- **Not every default blocker ran: {names} did not.** The union row is the ceiling of the
  blockers in the table, not of the default set every later stage inherits, so it is not
  comparable with a union that includes them. Run without the omission to measure that one."""


def _reduction_ratio_bullet(n_all_pairs: int) -> str:
    """Whether RR flatters a catalog is a property of its size, so it is measured."""
    if n_all_pairs <= BRUTE_FORCE_PAIRS:
        return f"""- **RR is flattered by a small N.** {n_all_pairs:,} possible pairs is a number a
  laptop can brute-force; the baseline does exactly that. The reduction ratio only
  becomes load-bearing when N² stops being computable."""
    return f"""- **At this N the reduction ratio is load-bearing.** {n_all_pairs:,} possible pairs is
  past the {BRUTE_FORCE_PAIRS:,} an exhaustive baseline can score in 1 GiB, so the candidate
  count above is the difference between a catalog `features/` can vectorize and one it
  cannot. Read RR beside that count, never instead of it."""


def render_markdown(
    report: BlockingReport,
    *,
    notes: DatasetNotes = NO_NOTES,
    out: str = DEFAULT_OUT,
    flags: str = "",
) -> str:
    """The reports/ artifact, with the dataset's own caveats where `notes` supply them.

    `out` and `flags` complete the regenerate command, so a report run with
    `--ann-components` or written outside `reports/` still reproduces itself.
    """
    body = "\n".join(_row(r) for r in report.rows)
    union_line = _row(report.union_row)
    ceiling = report.union_row.pair_completeness

    examples = "\n".join(
        f"{i}. `{first}`\n   `{second}`"
        for i, (first, second) in enumerate(report.missed_examples, start=1)
    )
    warnings = (
        "\n".join(f"- ⚠ {w}" for w in report.warnings)
        if report.warnings
        else "- No blocker dropped a block or otherwise lowered its own ceiling on this run."
    )
    intro = notes.passage("blocking.intro", "")
    intro_block = f"\n{intro}\n" if intro else ""
    if report.omitted:
        omitted_names = ", ".join(f"`{name}`" for name in report.omitted)
        ceiling_claim = f"""**The union row is the only one the rest of the pipeline inherits \
on this run** — but this run left {omitted_names} out. Its pair completeness of \
**{ceiling:.4f}** is therefore a **lower bound** on the default blocker set's ceiling, not the
ceiling itself: the omitted blocker could only add candidates, never remove them, so the
default set's true completeness is at least this high. The
{report.n_missed} true pairs no blocker *here* emitted are never scored by `features/`,
never seen by `model/`, and never reach `cluster/` — on this run. Whether the omitted
blocker recovers any of them is not measured until it runs."""
    else:
        ceiling_claim = f"""**The union row is the only one the rest of the pipeline inherits.** Its pair
completeness of **{ceiling:.4f}** is a hard ceiling on system recall: the
{report.n_missed} true pairs no blocker emitted are never scored by `features/`,
never seen by `model/`, and never reach `cluster/`. No amount of model work
recovers them."""
    honest = "\n".join(
        bullet
        for bullet in (
            notes.passage("blocking.provenance", ""),
            _omitted_bullet(report.omitted),
            _reduction_ratio_bullet(report.n_all_pairs),
            notes.passage("blocking.tuning", ""),
            _CEILING_BULLET.format(ceiling=ceiling),
        )
        if bullet
    )

    return f"""# Blocking: candidate generation and the recall ceiling

`{report.dataset}` — N = {report.n_records:,} records, \
{report.n_true_pairs:,} ground-truth pairs, {report.n_all_pairs:,} possible pairs.
Blockers are deterministic given the records, so no seed applies. Candidate counts and
completeness figures reproduce exactly between runs — the ANN index is built
single-threaded for that reason. Only the wall-clock columns vary.
{intro_block}
Regenerate with:

```bash
python -m dedup.blocking.evaluate --dataset {report.dataset}{flags} --out {out}
```

## The table

| blocker | params | candidates | PC | RR | build s | query s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{body}
{union_line}

**PC** is pair completeness — true pairs surviving, divided by all \
{report.n_true_pairs:,} ground-truth pairs (never by the survivors).
**RR** is reduction ratio — the fraction of the {report.n_all_pairs:,} possible pairs discarded.

There is deliberately no precision, F1 or accuracy column. A blocker's output is
overwhelmingly non-duplicates by construction; discarding non-duplicates is the job,
not an error.

### Ceiling caveats

{warnings}

## What this means

{ceiling_claim}

Individual blockers are *expected* to be mediocre. They earn their place by failing
differently — a blocker with low standalone PC is worth keeping if it lifts the union,
and a blocker that raises the candidate count without lifting the union is pure cost.

## The pairs blocking missed

{report.n_missed} of {report.n_true_pairs:,} true pairs never became candidates. This
set, not the count, is where the next blocker's design comes from:

{examples}

## Reading this honestly

{honest}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="abt-buy", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=None, help="override the dataset directory")
    parser.add_argument(
        "--ann-components",
        type=int,
        default=None,
        help="project ann's vectors with SVD before indexing -- needed once the dense index "
        "outgrows its memory budget, and it changes ann's candidates",
    )
    parser.add_argument(
        "--without",
        action="append",
        default=[],
        metavar="NAME",
        help="leave out every default blocker whose name starts with NAME (repeatable); the "
        "report names what was left out, since its union is then not the default ceiling",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the markdown report here")
    args = parser.parse_args(argv)

    records = [normalize(record) for record in load_dataset(args.dataset, args.root)]
    blockers = default_blocker_set(ann_components=args.ann_components)
    for prefix in args.without:
        if not any(blocker.name.startswith(prefix) for blocker in blockers):
            parser.error(f"--without {prefix!r} matches no default blocker")
    omitted = [b.name for b in blockers if any(b.name.startswith(p) for p in args.without)]
    report = evaluate(
        records,
        [blocker for blocker in blockers if blocker.name not in omitted],
        dataset=args.dataset,
        omitted=omitted,
    )
    flags = "" if args.ann_components is None else f" --ann-components {args.ann_components}"
    flags += "".join(f" --without {prefix}" for prefix in args.without)
    markdown = render_markdown(
        report,
        notes=DATASETS[args.dataset].notes,
        out=DEFAULT_OUT if args.out is None else args.out.as_posix(),
        flags=flags,
    )
    print(markdown)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
