"""Every committed report regenerates to the file that is committed.

CLAUDE.md's rule is that each report ends with a "Regenerate with" block
recording the exact command that produced it, and that changing `normalize.py`
moves every downstream number. Nothing enforced the pair: a change to
extraction or the model left the committed reports quietly stale, and the only
guard was remembering to re-run them. This file is that guard -- it re-runs each
report's *own* recorded command and compares the result with the committed file.

What makes this possible rather than merely desirable was measured before it was
written (`docs/plans/2026-10-06-phase-1-stale-numbers.md`): all twelve committed
reports reproduce, and the blocking reports' two wall-clock columns are the only
cells anywhere in the set that move between runs. `reports/blocking.md` already
states that contract in its own header -- "Candidate counts and completeness
figures reproduce exactly between runs -- the ANN index is built single-threaded
for that reason. Only the wall-clock columns vary." So the comparison is
byte-exact with those two columns masked, and `test_the_mask_touches_only_the
_wall_clock_columns` pins the mask, because the mask is the one place this test
could go blind.

Three things about the environment are load-bearing, each found by running it:

  * **The renderers read sibling reports off disk.** `registered_baseline` opens
    the path in `DatasetNotes.baseline_report` and *warns to stderr* rather than
    raising when it is missing, so a report regenerated beside an empty
    `reports/` silently says "no baseline report is registered for this dataset"
    and differs in prose. The sandbox therefore holds a copy of the committed
    tree.
  * **The `--out` path leaks into derived prose and re-wraps it.**
    `cluster/evaluate.py` derives its cross-references to `reports/model.md`
    from its own `--out` directory, so regenerating to a temporary path changes
    four lines and re-wraps a paragraph -- which substituting the path back
    afterwards cannot undo. Hence a sandbox that mirrors the repo layout and the
    committed *relative* `--out`, with only `--root` made absolute.
  * **`reports/synth/realism.md` names a data path.** Its `--out` is the catalog
    it reads, not a report, so whatever the command references under `data/` is
    staged into the sandbox at the same relative path. That is a general rule
    here rather than a special case for one report.

Gating: the five `abt-buy` reports cost 105 s together and run whenever the
benchmark is present; the seven that touch a synthetic catalog cost 1,820 s and
run only under `DEDUP_REPRODUCE_SLOW`. Both tiers skip with a reason rather than
being deselected by a marker, because CLAUDE.md's testing contract is that
`pytest -rs` tells you what did not run and a deselected test is invisible.
"""

from __future__ import annotations

import importlib
import os
import shlex
import shutil
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path

import pytest

from dedup.data import DATASETS

REPO_ROOT = Path(__file__).parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

# Eleven reports say "Regenerate with:"; realism.md says "Rendered against the
# catalog already on disk. Re-render with:" because it re-renders against a
# fixed artifact rather than regenerating one. Both are matched on the suffix.
PROVENANCE_MARKERS = ("Regenerate with:", "Re-render with:")

# The env var that opts into the expensive tier. Named for what it buys rather
# than for a marker, since it gates data availability and cost together.
SLOW_ENV = "DEDUP_REPRODUCE_SLOW"

# The two columns `blocking/evaluate.py` renders from a stopwatch. Anchored on
# the header text the renderer itself writes, so a renamed or reordered column
# fails loudly here instead of silently widening what gets masked.
WALL_CLOCK_HEADERS = ("build s", "query s")
MASK_CELL = " ~wall clock~ "


# ---------------------------------------------------------------------------
# Reading the recorded command out of a committed report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedReport:
    """A committed report and the command it records as having produced it."""

    path: str  # repo-relative, posix -- also the parametrize id
    module: str  # e.g. "dedup.blocking.evaluate"
    argv: list[str]  # everything after `python -m <module>`
    out_flag: str  # "--out", or "--report" for realism.md
    dataset: str  # the catalog named by --dataset / --seed-dataset
    command: str  # the recorded line verbatim, for failure messages

    @property
    def data_paths(self) -> list[str]:
        """Relative `data/` paths the command names, to stage into the sandbox.

        Only `realism.md` has any: its `--out` is the catalog it reads. The
        other eleven rely on the dataset's default root, which is why `--root`
        is injected instead.
        """
        return [value for value in self.argv if value.startswith("data/")]

    @property
    def is_default_tier(self) -> bool:
        """True for the reports that run without an opt-in.

        The cheap tier is "runs on `abt-buy`'s raw catalog and nothing else".
        `realism.md` falls outside it despite taking a second, because it reads
        a synthetic catalog -- tiering by the data touched rather than by a
        measured runtime keeps the rule one sentence long.
        """
        return self.dataset == "abt-buy" and not any(
            path.startswith("data/synth") for path in self.data_paths
        )


def _recorded_command(text: str, report: Path) -> str:
    """The single command line inside the first fenced block after the marker."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not any(line.rstrip().endswith(marker) for marker in PROVENANCE_MARKERS):
            continue
        fence = next(
            (i for i in range(index + 1, min(index + 5, len(lines))) if lines[i].startswith("```")),
            None,
        )
        if fence is None:
            break
        body = []
        for following in lines[fence + 1 :]:
            if following.startswith("```"):
                break
            body.append(following)
        if len(body) != 1:
            raise AssertionError(
                f"{report}: the provenance block holds {len(body)} lines; it must hold exactly "
                f"the one command that produced the report"
            )
        return body[0].strip()
    raise AssertionError(
        f"{report}: no provenance block found. Every report in reports/ must record the exact "
        f"command that produced it under one of {PROVENANCE_MARKERS} -- CLAUDE.md, Commands."
    )


def _parse(report: Path) -> RecordedReport:
    relative = report.relative_to(REPO_ROOT).as_posix()
    command = _recorded_command(report.read_text(encoding="utf-8"), report)
    tokens = shlex.split(command)
    if tokens[:2] != ["python", "-m"]:
        raise AssertionError(
            f"{relative}: recorded command is {command!r}; it must be a `python -m <module>` "
            f"invocation so this test can run it in-process"
        )
    module, argv = tokens[2], tokens[3:]

    # --report for realism.md, whose --out names the catalog it reads.
    out_flag = "--report" if "--report" in argv else "--out"
    dataset = "abt-buy"
    for flag in ("--dataset", "--seed-dataset"):
        if flag in argv:
            dataset = argv[argv.index(flag) + 1]
    return RecordedReport(
        path=relative,
        module=module,
        argv=argv,
        out_flag=out_flag,
        dataset=dataset,
        command=command,
    )


def recorded_reports() -> list[RecordedReport]:
    return [_parse(path) for path in sorted(REPORTS_DIR.rglob("*.md"))]


REPORTS = recorded_reports()
CASES = [pytest.param(report, id=report.path) for report in REPORTS]


# ---------------------------------------------------------------------------
# Masking the one thing that legitimately moves
# ---------------------------------------------------------------------------


def _flag_value(argv: list[str], flag: str) -> str | None:
    """The value following `flag`, or None if it is absent.

    Not a zip of alternating tokens: `--leave-one-out` is a bare store_true
    flag, and two of the committed commands carry one, which shifts every pair
    after it.
    """
    if flag not in argv:
        return None
    index = argv.index(flag) + 1
    return argv[index] if index < len(argv) else None


def _cells(row: str) -> list[str]:
    """The cells of a markdown table row, without the leading/trailing pipes."""
    return row.split("|")[1:-1]


def _is_separator(row: str) -> bool:
    return all(set(cell.strip()) <= {"-", ":"} and cell.strip() for cell in _cells(row))


def mask_wall_clock(text: str) -> str:
    """Blank the blocker table's `build s` / `query s` cells.

    Anchored on the header row the renderer writes, and applied only to the data
    rows beneath it, so a renamed column or a reordered table stops being masked
    rather than quietly masking something else. The union row already renders
    both cells as an em dash, and is masked like any other row.
    """
    out: list[str] = []
    in_table = False
    for line in text.splitlines():
        if not line.startswith("|"):
            in_table = False
            out.append(line)
            continue
        cells = _cells(line)
        if tuple(cell.strip() for cell in cells[-2:]) == WALL_CLOCK_HEADERS:
            in_table = True
            out.append(line)  # the header itself is compared, not masked
            continue
        if in_table and not _is_separator(line):
            cells[-2:] = [MASK_CELL, MASK_CELL]
            out.append("|" + "|".join(cells) + "|")
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Running a recorded command in a sandbox that mirrors the repo
# ---------------------------------------------------------------------------


def _regenerate(report: RecordedReport, sandbox: Path, monkeypatch) -> Path:
    """Run the recorded command inside `sandbox`, returning what it wrote."""
    shutil.copytree(REPORTS_DIR, sandbox / "reports")
    for relative in report.data_paths:
        source = REPO_ROOT / relative
        destination = sandbox / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)

    # The dataset is the one thing that cannot be relative: the sandbox is not
    # the repo, so the registry's `default_root` would not resolve. Everything
    # else -- `--out` above all -- stays exactly as committed, because the
    # renderers derive prose from it.
    argv = [*report.argv, "--root", str(REPO_ROOT / DATASETS[report.dataset].default_root)]
    monkeypatch.chdir(sandbox)
    assert importlib.import_module(report.module).main(argv) == 0, report.command
    return sandbox / (_flag_value(report.argv, report.out_flag) or report.path)


def _require(report: RecordedReport) -> None:
    """Skip, visibly, when the data is absent or the tier is not opted into."""
    root = REPO_ROOT / DATASETS[report.dataset].default_root
    if not root.is_dir():
        pytest.skip(f"{report.dataset} not downloaded at {root}; see CLAUDE.md > Setup")
    for relative in report.data_paths:
        if not (REPO_ROOT / relative).is_dir():
            pytest.skip(f"{relative} absent; see CLAUDE.md > Setup")
    if not report.is_default_tier and not os.environ.get(SLOW_ENV):
        # Phrased for the tier rather than this report: the tier is "touches a
        # synthetic catalog", and `realism.md` is in it at one second while
        # `synth/model.md` is 11 minutes. Claiming minutes for all seven would
        # be the kind of unconditional sentence this project keeps out of its
        # reports, and a skip reason is read the same way.
        pytest.skip(
            f"{report.path} reads a synthetic catalog, so it is outside the default tier "
            f"(the seven together cost ~30 min); set {SLOW_ENV}=1 to reproduce it"
        )


# ---------------------------------------------------------------------------
# The provenance blocks themselves -- checkable with no data at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("report", CASES)
def test_the_recorded_command_writes_the_report_it_is_recorded_in(report):
    """A block naming some other path is not a regeneration command for this
    file, and would send a reader to the wrong report. Cheap to get wrong by
    copy-paste when a new report is added, and invisible until someone runs it."""
    written = _flag_value(report.argv, report.out_flag)
    assert written == report.path, (
        f"{report.path} records `{report.command}`, which writes to {written!r}"
    )


def test_every_committed_report_is_covered():
    """The suite discovers reports by globbing rather than from a list, so a new
    report is guarded the moment it is committed. This asserts the glob found
    the tree rather than an empty directory -- a passing-because-empty
    parametrize is the failure mode that would make every test above vacuous."""
    assert len(REPORTS) >= 12, f"found only {len(REPORTS)} reports under {REPORTS_DIR}"
    assert {report.module for report in REPORTS} >= {
        "dedup.eval.baseline",
        "dedup.blocking.evaluate",
        "dedup.features.evaluate",
        "dedup.model.evaluate",
        "dedup.cluster.evaluate",
    }, "a stage's report is missing from reports/"


def test_the_mask_touches_only_the_wall_clock_columns():
    """The mask is the one place this test can go blind, so it is pinned against
    the committed files: it must change every blocker row's last two cells and
    nothing else. Widening it to a candidate count or a PC would let a real
    regression through, which is exactly what this asserts cannot happen."""
    blocking = [r for r in REPORTS if r.module == "dedup.blocking.evaluate"]
    assert blocking, "no blocking report found to pin the mask against"
    for report in blocking:
        text = (REPO_ROOT / report.path).read_text(encoding="utf-8")
        masked = mask_wall_clock(text)
        assert masked != text, f"{report.path}: the mask matched nothing"

        differing = [
            (before, after)
            for before, after in zip(text.splitlines(), masked.splitlines())
            if before != after
        ]
        for before, after in differing:
            kept_before, kept_after = _cells(before)[:-2], _cells(after)[:-2]
            assert kept_before == kept_after, (
                f"{report.path}: the mask altered a column other than the wall-clock pair:\n"
                f"  {before}\n  {after}"
            )
            assert _cells(after)[-2:] == [MASK_CELL, MASK_CELL]
        # Every data row of the one blocker table, and nothing outside it.
        assert len(differing) == 7, (
            f"{report.path}: masked {len(differing)} rows; the blocker table has six blockers "
            f"and a union row"
        )


def test_the_mask_leaves_a_report_without_wall_clock_columns_alone():
    """The complement of the test above: a report whose tables have no such
    header must come through untouched, or the mask is matching on shape rather
    than on the renderer's own column names."""
    for report in REPORTS:
        if report.module == "dedup.blocking.evaluate":
            continue
        text = (REPO_ROOT / report.path).read_text(encoding="utf-8")
        assert mask_wall_clock(text) == text, f"{report.path} was masked and should not have been"


# ---------------------------------------------------------------------------
# The reproduction itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("report", CASES)
def test_a_committed_report_regenerates_to_itself(report, tmp_path, monkeypatch, capsys):
    """The guard this file exists for. A change to extraction, the featurizer,
    the model or a renderer that moves a published figure fails here, naming the
    report and the line, instead of leaving the committed file stale."""
    _require(report)
    produced = _regenerate(report, tmp_path, monkeypatch)
    # Every CLI prints the report to stdout as well as writing `--out`, so
    # without this the diff below would be followed by the whole report again as
    # captured output, burying the one line that matters.
    capsys.readouterr()
    assert produced.is_file(), f"{report.command} wrote nothing to {produced}"

    expected = mask_wall_clock((REPO_ROOT / report.path).read_text(encoding="utf-8"))
    actual = mask_wall_clock(produced.read_text(encoding="utf-8"))
    if expected == actual:
        return
    diff = "\n".join(
        unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"committed {report.path}",
            tofile="regenerated",
            lineterm="",
        )
    )
    pytest.fail(
        f"{report.path} no longer matches what its own command produces.\n\n{diff}\n\n"
        f"If the change is intended, regenerate the report from the repo root with:\n"
        f"    {report.command}\n"
        f"and say in the commit body what moved and why (CLAUDE.md, Conventions)."
    )
