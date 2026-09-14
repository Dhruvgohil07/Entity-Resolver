"""Tests for the cluster report.

As in `tests/test_model_evaluate.py`: the unit tests guard the reporting path, and a
synthetic report reaches the branches of "Reading this honestly" that Abt-Buy never
takes. The integration tests at the bottom re-derive the measurement taken while
planning this stage and skip when the benchmark has not been downloaded -- so a
green run is not by itself evidence that any published number was reproduced.
`pytest -rs` reports the skips.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from dedup.cluster.bcubed import BCubedScore, ClusterErrors, ClusterPairCounts
from dedup.cluster.evaluate import (
    AVERAGE_LINKAGE,
    COMPONENTS_BEST_F1,
    COMPONENTS_HI,
    COMPONENTS_LO,
    CORRELATION,
    MAX_FUSED_LISTED,
    ORACLE,
    SEPARATED,
    SEPARATED_WITH_SPLIT,
    SINGLETONS,
    STILL_FUSED,
    ClusterReport,
    FusedCluster,
    FusedMember,
    MethodRow,
    evaluate,
    render_markdown,
)
from dedup.data.abt_buy import load_abt_buy
from dedup.model.threshold import CostModel

REAL_DATA = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"
needs_data = pytest.mark.skipif(
    not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data"
)


def flat(markdown):
    """Collapse whitespace so a phrase the report wraps across lines still matches."""
    return " ".join(markdown.split())


def method_row(
    key,
    *,
    precision=0.99,
    recall=0.90,
    fused=0,
    split=10,
    implied=0,
    review=5,
    largest=2,
    true_merged=45,
    expected=100.0,
    realized=100.0,
    method="connected components",
):
    return MethodRow(
        key=key,
        method=method,
        criterion=f"criterion of {key}",
        errors=ClusterErrors(n_clusters=60, n_fused_clusters=fused, n_split_entities=split),
        largest=largest,
        n_implied_pairs=implied,
        n_review=review,
        bcubed=BCubedScore(precision=precision, recall=recall, n_records=100),
        pairs=ClusterPairCounts(
            n_merged_pairs=50, n_true_merged_pairs=true_merged, n_true_pairs=50
        ),
        expected_cost=expected,
        realized_cost=realized,
    )


def fused_cluster(linkage=SEPARATED, correlation=SEPARATED, title="Widget 100 - W100"):
    return FusedCluster(
        members=(
            FusedMember("e1", title),
            FusedMember("e1", "Widget W100"),
            FusedMember("e2", "Widget 100 Red - W100R"),
        ),
        n_edges=2,
        n_implied_pairs=1,
        cross_entity_edges=(0.97,),
        outcomes={AVERAGE_LINKAGE: linkage, CORRELATION: correlation},
    )


def synthetic_report(**changes):
    """A hand-built report whose defaults take the branches Abt-Buy takes.

    Each test below flips one condition, so every other branch of "Reading this
    honestly" is exercised without a dataset.
    """
    rows = (
        method_row(SINGLETONS, precision=1.0, recall=0.5, split=50, largest=1, true_merged=0,
                   review=40, expected=150.0, realized=100.0, method="all singletons"),
        method_row(COMPONENTS_HI, fused=2, implied=3, largest=4, expected=90.0, realized=90.0),
        method_row(COMPONENTS_LO, precision=0.98, recall=0.92, fused=3, implied=6, largest=5,
                   review=0, expected=110.0, realized=95.0),
        method_row(COMPONENTS_BEST_F1, precision=0.90, recall=0.96, fused=10, implied=50,
                   largest=11, review=0, expected=300.0, realized=250.0),
        method_row(AVERAGE_LINKAGE, precision=0.995, review=7, expected=80.0, realized=80.0,
                   method="average linkage"),
        method_row(CORRELATION, precision=0.996, review=7, expected=70.0, realized=70.0,
                   method="correlation clustering"),
        method_row(ORACLE, precision=1.0, recall=1.0, split=0, true_merged=50, expected=1200.0,
                   realized=0.0),
    )
    base = ClusterReport(
        dataset="synthetic",
        seed=0,
        test_fraction=0.3,
        n_folds=5,
        include_semantic=False,
        n_restarts=8,
        cost=CostModel(),
        best_f1_threshold=0.0099,
        n_records=100,
        n_entities=50,
        n_true_pairs=50,
        n_candidates=1000,
        n_true_candidates=50,
        band_realized_cost=265.0,
        rows=rows,
        fused=(fused_cluster(), fused_cluster(STILL_FUSED, STILL_FUSED)),
    )
    return replace(base, **changes)


def with_row(report, key, **fields):
    """The report with one row rebuilt from `fields`."""
    return replace(
        report,
        rows=tuple(method_row(key, **fields) if row.key == key else row for row in report.rows),
    )


# ---------------------------------------------------------------------------
# Every interpretive sentence follows the numbers.
# ---------------------------------------------------------------------------


def test_the_synthetic_report_takes_the_abt_buy_branches_by_default():
    text = flat(render_markdown(synthetic_report()))
    assert "B-cubed flatters doing nothing on this split" in text
    assert "makes worse clusters, and chaining is why" in text
    assert "B³ F1 falls with it" in text
    assert "Chaining happens at `p_hi` as well: 2 of its clusters" in text
    assert "Correlation clustering leaves 1 of the 2 fused" in text
    assert "The ceiling does not bind on this split" in text
    assert "ground truth agree on the winner" in text
    assert "the truth is not the cheapest partition" in text
    assert "Review is a third outcome here too" in text
    assert "Clustering lowers the bill, not only the entity count" in text


def test_a_partition_no_cheaper_than_the_bands_is_not_credited_with_a_lower_bill():
    text = flat(render_markdown(synthetic_report(band_realized_cost=50.0)))
    assert "lowers the bill" not in text
    assert "Clustering buys entities here, not a lower bill" in text


def test_every_row_is_rendered_with_its_criterion_and_review_count():
    report = synthetic_report()
    text = render_markdown(report)
    for row in report.rows:
        assert f"criterion of {row.key}" in text
    assert "leaves 5 pairs queued, average linkage 7 pairs" in flat(text)


def test_a_low_floor_is_not_called_flattering():
    report = with_row(
        synthetic_report(), SINGLETONS, precision=1.0, recall=0.2, method="all singletons"
    )
    text = flat(render_markdown(report))
    assert "flatters" not in text
    assert "singleton floor is low on this split" in text


def test_a_threshold_that_does_not_chain_is_not_blamed_for_chaining():
    report = synthetic_report()
    hi = report.row(COMPONENTS_HI)
    unchained = replace(
        report,
        rows=tuple(
            replace(hi, key=COMPONENTS_BEST_F1) if row.key == COMPONENTS_BEST_F1 else row
            for row in report.rows
        ),
    )
    text = flat(render_markdown(unchained))
    assert "chaining is why" not in text
    assert "did not cost precision at the best-pairwise-F1 threshold" in text


def test_a_chaining_threshold_that_still_raises_f1_is_called_a_trade():
    """Precision lost to fused products is not redeemed by a higher F1."""
    report = with_row(
        synthetic_report(), COMPONENTS_BEST_F1, precision=0.97, recall=0.99, fused=10,
        implied=50, largest=11, review=0, expected=300.0, realized=250.0,
    )
    text = flat(render_markdown(report))
    assert "trades cluster precision for recall, and chaining is why" in text
    assert "B³ F1 still rises" in text
    assert "makes worse clusters" not in text


def test_no_fusion_at_p_hi_is_stated_in_both_places():
    text = flat(render_markdown(synthetic_report(fused=())))
    assert "nothing to name here" in text
    assert "fused no two entities on this split" in text
    assert "Chaining happens at `p_hi`" not in text


def test_fusions_correlation_undoes_are_not_blamed_on_the_model():
    text = flat(render_markdown(synthetic_report(fused=(fused_cluster(),))))
    assert "None survives correlation clustering" in text
    assert "error-analyst" not in text


def test_a_separation_that_splits_an_entity_is_counted_apart():
    report = synthetic_report(fused=(fused_cluster(correlation=SEPARATED_WITH_SPLIT),))
    text = flat(render_markdown(report))
    assert "and correlation clustering 0; separating them but splitting an entity: " in text
    assert "splitting an entity: correlation clustering 1." in text


def test_a_fusion_only_average_linkage_undoes_is_put_down_to_merge_order():
    """Greedy luck is not credited to the objective when the objective prefers the fusion."""
    text = flat(render_markdown(synthetic_report(fused=(fused_cluster(SEPARATED, STILL_FUSED),))))
    assert "Average linkage separates 1 of those anyway. That is its merge order" in text


def test_a_binding_ceiling_reports_what_closure_recovered():
    report = with_row(
        synthetic_report(n_true_candidates=45), ORACLE, precision=1.0, recall=0.96, split=2,
        true_merged=47, expected=1200.0, realized=0.0,
    )
    text = flat(render_markdown(report))
    assert "does not bind" not in text
    assert "The ceiling binds" in text
    assert "recovers 2 more" in text
    assert "B³ recall 0.9600" in text


def test_disagreement_between_the_objective_and_ground_truth_is_reported():
    report = with_row(
        synthetic_report(), AVERAGE_LINKAGE, precision=0.995, review=7, expected=80.0,
        realized=60.0, method="average linkage",
    )
    text = flat(render_markdown(report))
    assert "ground truth disagree on the winner" in text
    assert "ground truth agree" not in text


def test_a_cheapest_truth_points_at_the_search_rather_than_the_scorer():
    report = with_row(
        synthetic_report(), ORACLE, precision=1.0, recall=1.0, split=0, true_merged=50,
        expected=60.0, realized=0.0,
    )
    text = flat(render_markdown(report))
    assert "the truth is the cheapest partition measured" in text
    assert "the truth is not the cheapest partition" not in text


# ---------------------------------------------------------------------------
# the chaining listing
# ---------------------------------------------------------------------------


def test_counts_of_one_are_singular():
    text = flat(render_markdown(synthetic_report()))
    assert "2 edges at `p_hi`, 1 implied pair;" in text


def test_the_fused_listing_names_each_methods_outcome():
    text = flat(render_markdown(synthetic_report()))
    assert "Average linkage: **separated**; correlation clustering: **separated**" in text
    assert "Average linkage: **still fused**; correlation clustering: **still fused**" in text


def test_the_fused_listing_is_capped():
    many = tuple(fused_cluster() for _ in range(MAX_FUSED_LISTED + 2))
    text = flat(render_markdown(synthetic_report(fused=many)))
    assert f"the first {MAX_FUSED_LISTED} of the {MAX_FUSED_LISTED + 2}" in text
    assert f"**{MAX_FUSED_LISTED + 1}." not in text


def test_a_title_with_a_pipe_does_not_break_its_table():
    text = render_markdown(synthetic_report(fused=(fused_cluster(title="Widget | 100"),)))
    assert "Widget \\| 100" in text


# ---------------------------------------------------------------------------
# against the real benchmark (skipped unless downloaded)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    if not REAL_DATA.is_dir():
        pytest.skip("Abt-Buy not downloaded; see CLAUDE.md > Data")
    return evaluate(load_abt_buy(REAL_DATA), dataset="abt-buy")


@needs_data
def test_components_at_p_hi_reproduce_the_planning_measurement(report):
    """Measured before this package existed, with scipy and a B-cubed written inline.

    Two independent implementations agreeing is what makes the committed numbers
    worth quoting.
    """
    hi = report.row(COMPONENTS_HI)
    assert hi.bcubed.precision == pytest.approx(0.9880, abs=5e-5)
    assert hi.bcubed.recall == pytest.approx(0.9054, abs=5e-5)
    assert hi.errors.n_fused_clusters == 4
    assert hi.n_implied_pairs == 11
    assert len(report.fused) == 4


@needs_data
def test_every_row_scores_every_record_and_every_true_pair(report):
    assert report.n_records == 652
    assert report.n_true_pairs == 341
    for row in report.rows:
        assert row.bcubed.n_records == report.n_records
        assert row.pairs.n_true_pairs == report.n_true_pairs


@needs_data
def test_merging_nothing_bills_exactly_what_the_bands_route(report):
    """The cross-stage check that the two reports bill on one set of terms.

    With no merges, every pair `reports/model.md` sends to auto-merge (280) or review
    (13) is queued, and its 56 auto-rejected true pairs are the only misses.
    """
    floor = report.row(SINGLETONS)
    assert floor.n_review == 280 + 13
    assert floor.realized_cost == 293 * 1 + 56 * 2


@needs_data
def test_merging_the_whole_review_band_leaves_nothing_to_review(report):
    assert report.row(COMPONENTS_LO).n_review == 0


@needs_data
def test_the_floor_and_the_ceiling(report):
    floor, oracle = report.row(SINGLETONS), report.row(ORACLE)
    assert floor.bcubed.precision == 1.0
    assert floor.bcubed.recall == pytest.approx(0.4923, abs=5e-5)
    assert oracle.bcubed.f1 == 1.0
    assert report.n_recovered_by_closure == 0


@needs_data
def test_the_best_pairwise_f1_threshold_chains(report):
    hi, best = report.row(COMPONENTS_HI), report.row(COMPONENTS_BEST_F1)
    assert best.bcubed.precision < hi.bcubed.precision
    assert best.n_implied_pairs > hi.n_implied_pairs
    assert best.largest > hi.largest
    assert "chaining is why" in flat(render_markdown(report))


@needs_data
def test_lower_thresholds_only_ever_add_merges(report):
    """Components at a lower threshold coarsens components at a higher one."""
    keys = (SINGLETONS, COMPONENTS_HI, COMPONENTS_LO, COMPONENTS_BEST_F1)
    merged = [report.row(key).pairs.n_merged_pairs for key in keys]
    assert merged == sorted(merged)


@needs_data
def test_correlation_is_never_worse_on_its_own_objective(report):
    found = report.row(CORRELATION).expected_cost
    assert found <= report.row(AVERAGE_LINKAGE).expected_cost + 1e-6
    assert found <= report.row(COMPONENTS_HI).expected_cost + 1e-6


@needs_data
def test_the_chaining_section_names_every_fused_cluster(report):
    text = render_markdown(report)
    for cluster in report.fused:
        for member in cluster.members:
            assert member.title.replace("|", "\\|") in text


@needs_data
def test_the_bands_alone_bill_what_reports_model_md_bills(report):
    """7 false merges at 20, 13 reviews, 56 misses at 2 -- nothing lost to blocking on test."""
    assert report.band_realized_cost == 7 * 20 + 13 * 1 + 56 * 2


@needs_data
def test_a_lower_expected_cost_does_not_buy_the_better_partition_here(report):
    """Measured, not designed: the objective's winner loses on ground truth.

    Correlation clustering finds the lower expected cost and average linkage the
    lower realized cost, and the report must say so rather than crown the objective.
    Under the model's probabilities the true partition is not even cheap, which is
    why no search closes the gap.
    """
    linkage, correlation = report.row(AVERAGE_LINKAGE), report.row(CORRELATION)
    assert correlation.expected_cost < linkage.expected_cost
    assert linkage.realized_cost < correlation.realized_cost
    assert linkage.realized_cost < report.band_realized_cost
    assert linkage.errors.n_fused_clusters == 1
    assert report.row(ORACLE).expected_cost > correlation.expected_cost
    text = flat(render_markdown(report))
    assert "ground truth disagree on the winner" in text
    assert "the truth is not the cheapest partition" in text
    assert "Clustering lowers the bill" in text
