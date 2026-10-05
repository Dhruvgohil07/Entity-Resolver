"""Smoke tests for the five report CLIs' `main()` -- the one surface the rest of
this suite never touches.

CLAUDE.md records these five as "implemented but untested": every other test
calls the functions beneath them (`evaluate`, `render_markdown`) directly, so
coverage shows every line of argument parsing, dataset resolution, `--out`
writing and every flag branch unexecuted. Breaking any of them would not fail a
test run. They were instead run by hand at each sync -- "checks made once, not
guards" -- and this file makes them guards.

What these assert is deliberately *plumbing, not numbers*: the exit code, that
`--out` wrote a file, and that a flag actually reached the rendered report. The
published figures are pinned by the per-stage report tests, which call
`evaluate()` directly and so cannot notice a broken CLI; these notice a broken
CLI and cannot notice a wrong figure. The two halves are complementary and
neither replaces the other.

`baseline`, `blocking` and `features` run on the committed 7-record fixtures, so
they guard a fresh clone with no benchmark downloaded. `model` and `cluster`
cannot, and the reason is worth recording rather than rediscovering: an
entity-grouped 0.3 split of the fixtures' 4 entities leaves 2 in train, so
`--folds 5` fails the fold check outright and `--folds 2` gets past it only to
hand LightGBM an empty blocked candidate set ("Found array with 0 sample(s)").
Both are therefore `@needs_data`, joining the tests here that skip when
`data/raw/` is absent, and both run at `--folds 2` (cluster also `--restarts 1`)
because they check wiring rather than reproduce a report -- the committed
settings are what the report tests cover.
"""

from pathlib import Path

import pytest

from dedup.blocking import evaluate as blocking_cli
from dedup.cluster import evaluate as cluster_cli
from dedup.eval import baseline as baseline_cli
from dedup.features import evaluate as features_cli
from dedup.model import evaluate as model_cli
from dedup.model.train import PairScorer, read_scorer_manifest

FIXTURES = Path(__file__).parent / "fixtures" / "abt-buy"
REAL_DATA = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"
needs_data = pytest.mark.skipif(
    not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data"
)

# Every CLI takes the same two, which is what lets a fixture drive them at all.
ON_FIXTURES = ["--dataset", "abt-buy", "--root", str(FIXTURES)]

FIXTURE_CLIS = [
    pytest.param(baseline_cli, id="baseline"),
    pytest.param(blocking_cli, id="blocking"),
    pytest.param(features_cli, id="features"),
]


@pytest.mark.parametrize("module", FIXTURE_CLIS)
def test_a_report_cli_exits_zero_and_writes_the_file_out_names(module, tmp_path):
    out = tmp_path / "report.md"
    assert module.main([*ON_FIXTURES, "--out", str(out)]) == 0
    assert out.is_file(), "--out was accepted but nothing was written"
    text = out.read_text(encoding="utf-8")
    # A markdown report, not an empty file or a traceback written to disk.
    assert text.startswith("# "), text[:80]
    assert len(text.splitlines()) > 10


@pytest.mark.parametrize("module", FIXTURE_CLIS)
def test_a_report_cli_prints_the_report_when_out_is_omitted(module, capsys):
    """`--out` is optional and the default path is stdout, which is how every
    command in CLAUDE.md's Commands block is shown before `--out` is added."""
    assert module.main(ON_FIXTURES) == 0
    printed = capsys.readouterr().out
    assert printed.lstrip().startswith("# ")


@pytest.mark.parametrize("module", FIXTURE_CLIS)
def test_a_report_cli_refuses_an_unknown_dataset_before_doing_any_work(module):
    """argparse `choices` over the live `DATASETS` registry, so a name no loader
    knows is refused at parse time rather than failing somewhere in `data/`."""
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--dataset", "not-a-real-dataset"])
    assert excinfo.value.code == 2


def test_the_baselines_similarity_floor_reaches_the_report(tmp_path):
    """`--min-similarity` is the flag that discards pairs unscored, so a report
    that does not state the floor it used is not reproducible from itself."""
    out = tmp_path / "baseline.md"
    assert baseline_cli.main([*ON_FIXTURES, "--min-similarity", "0.3", "--out", str(out)]) == 0
    assert "Similarity floor: `0.3`" in out.read_text(encoding="utf-8")


def test_blockings_without_flag_omits_the_blocker_and_calls_the_union_a_lower_bound(tmp_path):
    """The `--without` path, including the ceiling-vs-lower-bound wording a
    previous audit found contradicting itself. The union of a reduced set bounds
    the default set's ceiling from below; it is not that ceiling."""
    out = tmp_path / "blocking.md"
    assert blocking_cli.main([*ON_FIXTURES, "--without", "lsh", "--out", str(out)]) == 0
    text = " ".join(out.read_text(encoding="utf-8").split())
    assert "lower bound" in text
    assert "lsh (minhash)" in text, "the omitted blocker is not named"


def test_blockings_leave_one_out_flag_renders_the_marginal_table(tmp_path):
    """Opt-in because it costs one extra union per blocker; the committed
    Abt-Buy and synth-20k reports both pass it, `blocking-200k.md` does not."""
    out = tmp_path / "blocking.md"
    assert blocking_cli.main([*ON_FIXTURES, "--leave-one-out", "--out", str(out)]) == 0
    assert "| blocker | marginal candidates | marginal PC |" in out.read_text(encoding="utf-8")


def test_blockings_scale_flags_are_accepted_together(tmp_path):
    """The three flags `synth-200k` needs. Values here are tiny because the
    fixture is; what this guards is that they parse and reach the blockers,
    which is the half no other test covers."""
    out = tmp_path / "blocking.md"
    assert (
        blocking_cli.main(
            [
                *ON_FIXTURES,
                "--ann-components", "2",
                "--ann-neighbours", "3",
                "--lsh-max-neighbours", "50",
                "--out", str(out),
            ]
        )
        == 0
    )
    assert out.is_file()


@needs_data
def test_the_model_cli_writes_a_report_and_save_scorer_persists_a_loadable_one(tmp_path):
    """`--save-scorer` is the only place in the project that persists a scorer,
    and `service/batch.py` only ever loads one -- so a break here is a break in
    the handoff between the evaluation half and the serving half."""
    out = tmp_path / "model.md"
    root = tmp_path / "scorer"
    assert (
        model_cli.main(
            ["--folds", "2", "--cost-false-merge", "10", "--out", str(out), "--save-scorer", str(root)]
        )
        == 0
    )
    assert out.read_text(encoding="utf-8").startswith("# ")

    # The artifact is loadable by the class `service/batch.py` loads it with,
    # and it arrives *calibrated* -- an uncalibrated scorer loads fine and then
    # refuses `probabilities`, which would strand the serving half at runtime
    # rather than here. `load` already enforces manifest-against-pickle
    # agreement on the feature names, so what is checked here is that the CLI
    # saved a servable scorer at all.
    scorer = PairScorer.load(root)
    manifest = read_scorer_manifest(root)
    assert scorer.calibrator is not None, "--save-scorer wrote an uncalibrated scorer"
    assert manifest["feature_names"] == scorer.feature_names
    assert manifest["n_features"] == len(scorer.feature_names)


@needs_data
def test_a_cost_model_with_no_review_band_is_refused_through_the_cli():
    """C_fm 10 with C_review 2 gives p_lo 1.0 against p_hi 0.8 -- an inverted
    band. `threshold.py` rejects it, and the CLI must surface that rather than
    silently reporting bands nothing can fall into."""
    with pytest.raises(ValueError, match="no review band"):
        model_cli.main(["--folds", "2", "--cost-false-merge", "10", "--cost-review", "2"])


@needs_data
def test_the_cluster_cli_writes_a_report(tmp_path):
    out = tmp_path / "cluster.md"
    assert cluster_cli.main(["--folds", "2", "--restarts", "1", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# ")
    # Every clusterer the report compares must appear, since the comparison is
    # the report's whole point -- correlation clustering is kept precisely to
    # show the objective's own winner losing on realized cost.
    assert "average linkage" in text
    assert "correlation clustering" in text
