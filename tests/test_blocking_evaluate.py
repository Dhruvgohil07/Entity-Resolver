"""Tests for the blocking measurement contract.

The defect these exist to prevent is specific and has reached publication:
computing pair completeness over the pairs that *survived* blocking makes it
identically 1.0, so the worse the blocker, the better it scores. It is the
same shape as the `precision_at_k` defect an audit caught in eval/metrics.py
-- any denominator taken from the survivors rewards discarding candidates.
"""

from pathlib import Path

import numpy as np
import pytest

from dedup.blocking.ann import AnnBlocker
from dedup.blocking.base import BlockerRun
from dedup.blocking.evaluate import default_blocker_set, evaluate, render_markdown
from dedup.blocking.pairs import pack, total_pairs, union
from dedup.blocking.standard import StandardBlocker, code_token_keys, model_number_keys
from dedup.blocking.union import ground_truth, missed_pairs, score, union_run
from dedup.data.abt_buy import load_abt_buy
from dedup.normalize import normalize
from dedup.schema import Record

REAL_DATA = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"

# The union pair completeness measured when reports/blocking.md was generated.
# A band, not an exact value: faiss HNSW is approximate, and the point is that
# the ceiling stays where the report claims.
PUBLISHED_UNION_PC = (0.97, 1.0)


def rec(rid: str, title: str, entity: str):
    return normalize(Record(record_id=rid, source="synthetic", entity_id=entity, title=title))


@pytest.fixture
def catalog():
    # Three real duplicate pairs (e1, e2, e3) plus two singletons.
    return [
        rec("s:1", "Panasonic 2-Line Telephone System - KXTS208W", "e1"),
        rec("s:2", "Panasonic KX-TS208W Corded Phone", "e1"),
        rec("s:3", "Canon PowerShot SD1100 Digital Camera", "e2"),
        rec("s:4", "Canon SD1100 IS PowerShot Camera Silver", "e2"),
        rec("s:5", "Cuisinart Food Processor DLC2009CHB", "e3"),
        rec("s:6", "Cuisinart DLC2009CHB Processor", "e3"),
        rec("s:7", "Weber Genesis Gas Grill 3841001", "e4"),
        rec("s:8", "Netgear Wireless Router WNR2000", "e5"),
    ]


# ---------------------------------------------------------------------------
# The denominator
# ---------------------------------------------------------------------------


def test_ground_truth_counts_every_true_pair(catalog):
    truth, n_true = ground_truth(catalog)
    assert n_true == 3  # three size-2 entities; singletons contribute nothing
    assert truth.size == 3


def test_pair_completeness_divides_by_all_true_pairs_not_the_survivors(catalog):
    # A blocker that emits exactly one of the three true pairs scores 1/3.
    # Scored against its own output it would score 1.0 -- the bug.
    truth, n_true = ground_truth(catalog)
    n = len(catalog)
    one_true_pair = pack([0], [1], n)
    run = BlockerRun("toy", "-", one_true_pair, 0.0, 0.0)

    result = score(run, truth, n, n_true_pairs_total=n_true)
    assert result.pair_completeness == pytest.approx(1 / 3)
    assert result.n_true_pairs_found == 1
    assert result.n_true_pairs_total == 3


def test_scoring_against_a_survivor_derived_denominator_is_rejected(catalog):
    # The regression. Passing the count of true pairs *present in the
    # candidate set* instead of the full ground-truth count must fail loudly
    # rather than quietly report a flattering 1.0.
    truth, _ = ground_truth(catalog)
    n = len(catalog)
    run = BlockerRun("toy", "-", pack([0], [1], n), 0.0, 0.0)

    survivors_only = 1
    with pytest.raises(ValueError, match="never by the survivors"):
        score(run, truth, n, n_true_pairs_total=survivors_only)


def test_ground_truth_cross_checks_its_two_computations(catalog):
    # `count_true_pairs` sums C(size,2); `true_pair_keys` enumerates. If they
    # disagreed, every completeness number downstream would be wrong.
    truth, n_true = ground_truth(catalog)
    assert truth.size == n_true


# ---------------------------------------------------------------------------
# Reduction ratio
# ---------------------------------------------------------------------------


def test_reduction_ratio_is_against_all_possible_pairs(catalog):
    truth, n_true = ground_truth(catalog)
    n = len(catalog)
    run = BlockerRun("toy", "-", pack([0, 2], [1, 3], n), 0.0, 0.0)

    result = score(run, truth, n, n_true_pairs_total=n_true)
    assert result.reduction_ratio == pytest.approx(1 - 2 / total_pairs(n))


def test_a_blocker_emitting_everything_has_zero_reduction_and_full_completeness(catalog):
    truth, n_true = ground_truth(catalog)
    n = len(catalog)
    left, right = np.triu_indices(n, k=1)
    run = BlockerRun("all pairs", "-", pack(left, right, n), 0.0, 0.0)

    result = score(run, truth, n, n_true_pairs_total=n_true)
    assert result.pair_completeness == pytest.approx(1.0)
    assert result.reduction_ratio == pytest.approx(0.0)


def test_a_blocker_emitting_nothing_has_perfect_reduction_and_no_completeness(catalog):
    # Why the two numbers are never reported apart: this blocker looks
    # excellent on one of them.
    truth, n_true = ground_truth(catalog)
    run = BlockerRun("nothing", "-", np.empty(0, dtype=np.int64), 0.0, 0.0)

    result = score(run, truth, len(catalog), n_true_pairs_total=n_true)
    assert result.reduction_ratio == pytest.approx(1.0)
    assert result.pair_completeness == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The union
# ---------------------------------------------------------------------------


def test_union_completeness_is_at_least_every_member(catalog):
    truth, n_true = ground_truth(catalog)
    n = len(catalog)
    runs = [
        StandardBlocker("model", model_number_keys).run(catalog),
        StandardBlocker("codes", code_token_keys).run(catalog),
        AnnBlocker(neighbours=2).run(catalog),
    ]
    combined = score(union_run(runs), truth, n, n_true_pairs_total=n_true)

    for run in runs:
        assert combined.pair_completeness >= score(
            run, truth, n, n_true_pairs_total=n_true
        ).pair_completeness


def test_the_union_row_omits_timings(catalog):
    # Build and query seconds are not additive in a way a reader could act on.
    runs = [StandardBlocker("codes", code_token_keys).run(catalog)]
    combined = union_run(runs)
    assert combined.build_seconds is None and combined.query_seconds is None


def test_missed_pairs_are_the_true_pairs_no_blocker_emitted(catalog):
    truth, _ = ground_truth(catalog)
    n = len(catalog)
    caught = pack([0], [1], n)

    missed = missed_pairs(caught, truth)
    assert missed.size == 2
    assert not np.isin(missed, caught).any()


def test_union_of_one_blocker_is_that_blocker(catalog):
    run = StandardBlocker("codes", code_token_keys).run(catalog)
    assert np.array_equal(union(run.keys), run.keys)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_carries_the_header_a_reproducible_table_needs(catalog):
    report = evaluate(catalog, [AnnBlocker(neighbours=2)], dataset="synthetic")
    markdown = render_markdown(report)

    assert "N = 8 records" in markdown
    assert "3 ground-truth pairs" in markdown
    assert "union (all)" in markdown


def test_report_has_no_precision_or_f1_column(catalog):
    # A precision column on a blocker table is a wrong column, not a low
    # score: discarding non-duplicates is the job.
    report = evaluate(catalog, [AnnBlocker(neighbours=2)], dataset="synthetic")
    header = render_markdown(report).split("\n| --- |")[0].lower()

    for forbidden in ("precision", "f1", "accuracy"):
        assert forbidden not in header


def test_report_states_the_benchmark_is_pre_blocked(catalog):
    # Without this the union PC reads as "blocking is solved".
    markdown = render_markdown(evaluate(catalog, [AnnBlocker(neighbours=2)], dataset="synthetic"))
    assert "pre-blocked" in markdown


# ---------------------------------------------------------------------------
# Against the real benchmark (skipped unless downloaded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_published_union_ceiling_still_reproduces():
    # Pins reports/blocking.md. This union PC is the recall ceiling every
    # later stage inherits, so silent drift is a problem in itself.
    records = [normalize(r) for r in load_abt_buy(REAL_DATA)]
    report = evaluate(records, default_blocker_set(), dataset="abt-buy")

    assert report.n_records == 2173
    assert report.n_true_pairs == 1118
    low, high = PUBLISHED_UNION_PC
    assert low < report.union_row.pair_completeness < high
    # ann is the strongest single blocker -- the finding that justified
    # faiss being a dependency at all.
    best = max(report.rows, key=lambda r: r.pair_completeness)
    assert best.name.startswith("ann")
