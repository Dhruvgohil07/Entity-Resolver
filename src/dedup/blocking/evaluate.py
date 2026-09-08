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

from dedup.blocking.ann import AnnBlocker
from dedup.blocking.base import Blocker, BlockerRun
from dedup.blocking.lsh import MinHashLSHBlocker
from dedup.blocking.pairs import total_pairs, unpack
from dedup.blocking.sorted_neighborhood import SortedNeighborhoodBlocker
from dedup.blocking.standard import default_blockers
from dedup.blocking.union import (
    BlockerScore,
    ground_truth,
    missed_pairs,
    score,
    union_run,
)
from dedup.data import DATASETS, load_dataset
from dedup.normalize import NormalizedRecord, normalize

MISSED_EXAMPLES = 10


def default_blocker_set() -> list[Blocker]:
    """Every blocker, including the ones measurement says do not earn their place.

    `lsh` contributes zero marginal completeness on this benchmark and
    `sorted_neighborhood` buys 0.0044 for 30k candidates. They stay in the
    table because a negative result someone can re-run is evidence, and the
    same result asserted from a deleted experiment is not.
    """
    return [
        *default_blockers(),
        SortedNeighborhoodBlocker(window=20),
        MinHashLSHBlocker(threshold=0.4, shingles="token"),
        AnnBlocker(neighbours=10),
    ]


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


def evaluate(
    records: Sequence[NormalizedRecord],
    blockers: Sequence[Blocker],
    *,
    dataset: str,
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
    )


def _row(s: BlockerScore) -> str:
    build = "—" if s.build_seconds is None else f"{s.build_seconds:.1f}"
    query = "—" if s.query_seconds is None else f"{s.query_seconds:.1f}"
    return (
        f"| {s.name} | {s.params} | {s.n_candidates:,} | "
        f"{s.pair_completeness:.4f} | {s.reduction_ratio:.4f} | {build} | {query} |"
    )


def render_markdown(report: BlockingReport) -> str:
    body = "\n".join(_row(r) for r in report.rows)
    union_line = _row(report.union_row)
    ceiling = report.union_row.pair_completeness

    examples = "\n".join(
        f"{i}. `{abt}`\n   `{buy}`" for i, (abt, buy) in enumerate(report.missed_examples, start=1)
    )
    warnings = (
        "\n".join(f"- ⚠ {w}" for w in report.warnings)
        if report.warnings
        else "- No blocker dropped a block or otherwise lowered its own ceiling on this run."
    )

    return f"""# Blocking: candidate generation and the recall ceiling

`{report.dataset}` — N = {report.n_records:,} records, \
{report.n_true_pairs:,} ground-truth pairs, {report.n_all_pairs:,} possible pairs.
Blockers are deterministic given the records, so no seed applies. Candidate counts and
completeness figures reproduce exactly between runs — the ANN index is built
single-threaded for that reason. Only the wall-clock columns vary.

**These parameters were chosen against these same numbers.** The window, neighbour
count and document-frequency cutoff were selected by sweeping them over this full
catalog, not over a held-out split, so the figures below are fit to this dataset and
are optimistic as an estimate of what these settings would do on an unseen one. See
*Reading this honestly* for the held-out ceilings.

Regenerate with:

```bash
python -m dedup.blocking.evaluate --dataset {report.dataset} --out reports/blocking.md
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

**The union row is the only one the rest of the pipeline inherits.** Its pair
completeness of **{ceiling:.4f}** is a hard ceiling on system recall: the
{report.n_missed} true pairs no blocker emitted are never scored by `features/`,
never seen by `model/`, and never reach `cluster/`. No amount of model work
recovers them.

Individual blockers are *expected* to be mediocre. They earn their place by failing
differently — a blocker with low standalone PC is worth keeping if it lifts the union,
and a blocker that raises the candidate count without lifting the union is pure cost.

## The pairs blocking missed

{report.n_missed} of {report.n_true_pairs:,} true pairs never became candidates. This
set, not the count, is where the next blocker's design comes from:

{examples}

## Reading this honestly

- **Abt-Buy is pre-blocked.** It ships as two curated catalogs of ~1,000 records each,
  already scoped to overlapping product ranges. A high union PC here says the benchmark
  is small and clean, not that blocking is solved. `synth/` — 200k to 1M records with
  known ground truth — is where this stage earns its keep, and where a dense ANN index
  stops fitting in memory.
- **RR is flattered by a small N.** {report.n_all_pairs:,} possible pairs is a number a
  laptop can brute-force; the baseline does exactly that. The reduction ratio only
  becomes load-bearing when N² stops being computable.
- **The parameters were tuned on the catalog they are scored on.** No held-out split was
  used to pick the window, neighbour count or df cutoff, so treat the union figure as a
  best-of-sweep number rather than a clean estimate. Measured for comparison on the
  entity-grouped split (`seed=0`), where the same settings give union pair completeness
  **0.9949 on train and 1.0000 on test** — so the selection bias here is small, but it is
  present and unmeasured until parameters are chosen on train alone.
- **Which ceiling applies depends on what is being compared.** The {ceiling:.4f} above is the
  batch-deduplication number over the whole catalog. When `features/` and `model/` report
  test-split recall against the baseline's test-split F1, the ceiling that binds them is
  the test-split one, not this.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="abt-buy", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=None, help="override the dataset directory")
    parser.add_argument("--out", type=Path, default=None, help="write the markdown report here")
    args = parser.parse_args(argv)

    records = [normalize(record) for record in load_dataset(args.dataset, args.root)]
    report = evaluate(records, default_blocker_set(), dataset=args.dataset)
    markdown = render_markdown(report)
    print(markdown)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
