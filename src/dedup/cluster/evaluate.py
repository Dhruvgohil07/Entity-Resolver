"""The cluster report: entities rather than pairs, scored with B-cubed.

`python -m dedup.cluster.evaluate --dataset abt-buy --out reports/cluster.md`

The pipeline runs in-process up to calibrated test probabilities, exactly as
`model/evaluate.py` runs it -- `model/` persists no artifact yet -- and then every
clusterer reads the same scored graph of the test split. Each row answers to three
measurements, and they are deliberately different kinds of number:

  * **B-cubed precision and recall** over every record in the split: the cluster
    quality metric CLAUDE.md requires, and the only one here that scores records
    rather than pairs. It scores the automatic partition, before any queued review
    is resolved.
  * **Closure-pair precision and recall**: every pair the partition merges,
    emitted by blocking or not, against every true pair in the split. Pairwise,
    but over the partition rather than the candidate edges, so unlike an
    edge-level F1 it sees what chaining merged.
  * **Expected and realized cost**, on the bands' three outcomes: `C_fm` per merged
    pair, `C_review` per pair left apart at `p >= p_lo`, `C_fs` per true pair left
    apart below it. The objective `base.py` defines under the model's
    probabilities, next to what ground truth says it cost -- billed on the same
    terms as `reports/model.md`.

Two rows frame the rest and neither is a method. **All singletons** is the floor,
and it is higher than it looks when entities are small. **Components over the true
candidate edges** is the ceiling: what any clusterer could reach from the pairs
blocking emitted.

One row is there to fail. CLAUDE.md requires connected-components chaining to be
demonstrated, not designed around silently, so components also runs at the
threshold that maximizes *pairwise* F1 -- chosen on out-of-fold train predictions,
as `model/evaluate.py` chooses it -- and "Chaining, shown" names every cluster the
auto-merge band fuses once its edges are closed under transitivity.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dedup.cluster.agglomerative import average_linkage
from dedup.cluster.base import (
    ScoredGraph,
    canonical_labels,
    expected_partition_cost,
    review_mask,
)
from dedup.cluster.bcubed import (
    BCubedScore,
    ClusterErrors,
    ClusterPairCounts,
    bcubed,
    closure_pair_counts,
    cluster_errors,
)
from dedup.cluster.components import cluster_shapes, connected_components, implied_pairs
from dedup.cluster.correlation import DEFAULT_RESTARTS, correlation_clusters
from dedup.data import DATASETS, load_dataset
from dedup.eval.metrics import best_f1, precision_recall_curve
from dedup.eval.splits import count_true_pairs, split_by_entity
from dedup.model.calibrate import DEFAULT_N_FOLDS, fit_calibrated_scorer
from dedup.model.threshold import CostModel, band_summary
from dedup.model.train import prepare
from dedup.normalize import NormalizedRecord
from dedup.schema import Record

# Row keys, in report order.
SINGLETONS = "singletons"
COMPONENTS_HI = "components-p_hi"
COMPONENTS_LO = "components-p_lo"
COMPONENTS_BEST_F1 = "components-best-f1"
AVERAGE_LINKAGE = "average-linkage"
CORRELATION = "correlation"
ORACLE = "oracle"

# The cost-based methods, whose treatment of each fused cluster is reported.
FIXES = (AVERAGE_LINKAGE, CORRELATION)
FIX_NAMES = {AVERAGE_LINKAGE: "average linkage", CORRELATION: "correlation clustering"}

SEPARATED = "separated"
SEPARATED_WITH_SPLIT = "separated, but split an entity"
STILL_FUSED = "still fused"

MAX_FUSED_LISTED = 10

# A singleton floor at or above this B-cubed F1 gets the note that B-cubed
# flatters doing nothing here; below it, the note says the opposite.
FLOOR_IS_HIGH = 0.5


@dataclass(frozen=True)
class MethodRow:
    """One partition of the test split and everything measured about it."""

    key: str
    method: str
    criterion: str
    errors: ClusterErrors
    largest: int
    n_implied_pairs: int
    n_review: int  # pairs left apart at p >= p_lo, queued for a reviewer
    bcubed: BCubedScore
    pairs: ClusterPairCounts
    expected_cost: float
    realized_cost: float


@dataclass(frozen=True)
class FusedMember:
    entity_id: str
    title: str


@dataclass(frozen=True)
class FusedCluster:
    """A cluster connected components at `p_hi` built from more than one entity."""

    members: tuple[FusedMember, ...]
    n_edges: int
    n_implied_pairs: int
    cross_entity_edges: tuple[float, ...]  # probabilities, highest first
    outcomes: dict[str, str]  # row key of each cost-based method -> what it did here

    @property
    def n_entities(self) -> int:
        return len({member.entity_id for member in self.members})


@dataclass(frozen=True)
class ClusterReport:
    dataset: str
    seed: int
    test_fraction: float
    n_folds: int
    include_semantic: bool
    n_restarts: int
    cost: CostModel
    best_f1_threshold: float  # chosen on out-of-fold train predictions
    n_records: int  # test split
    n_entities: int
    n_true_pairs: int
    n_candidates: int
    n_true_candidates: int  # true pairs blocking emitted
    band_realized_cost: float  # the pairwise bands alone, billed on the rows' terms
    rows: tuple[MethodRow, ...]
    fused: tuple[FusedCluster, ...]

    def row(self, key: str) -> MethodRow:
        for row in self.rows:
            if row.key == key:
                return row
        raise KeyError(key)

    @property
    def n_recovered_by_closure(self) -> int:
        """True pairs the ceiling merges that blocking never emitted.

        Transitivity over true edges only ever joins records of one entity, so
        every pair it merges beyond the emitted ones is a true pair recovered --
        which is how cluster recall can exceed blocking's pair completeness.
        """
        return self.row(ORACLE).pairs.n_true_merged_pairs - self.n_true_candidates


def _measure(
    key: str,
    method: str,
    criterion: str,
    labels: np.ndarray,
    *,
    truth: np.ndarray,
    graph: ScoredGraph,
    edge_labels: np.ndarray,
    cost: CostModel,
    edge_graph: ScoredGraph,
    edge_threshold: float,
) -> MethodRow:
    labels = canonical_labels(labels)
    pairs = closure_pair_counts(labels, truth)
    queued = review_mask(labels, graph, cost)
    n_review = int(np.count_nonzero(queued))
    return MethodRow(
        key=key,
        method=method,
        criterion=criterion,
        errors=cluster_errors(labels, truth),
        largest=int(np.bincount(labels).max()),
        n_implied_pairs=implied_pairs(labels, edge_graph, edge_threshold),
        n_review=n_review,
        bcubed=bcubed(labels, truth),
        pairs=pairs,
        expected_cost=expected_partition_cost(labels, graph, cost),
        realized_cost=pairs.realized_cost(
            cost,
            n_review=n_review,
            n_review_true=int(np.count_nonzero(queued & edge_labels)),
        ),
    )


def _outcome(labels: np.ndarray, entities: list[str]) -> str:
    """What a partition did with one fused cluster's records, judged inside that cluster."""
    entities_in_group: dict[int, set[str]] = defaultdict(set)
    groups_of_entity: dict[str, set[int]] = defaultdict(set)
    for label, entity in zip(labels.tolist(), entities):
        entities_in_group[label].add(entity)
        groups_of_entity[entity].add(label)
    if any(len(found) > 1 for found in entities_in_group.values()):
        return STILL_FUSED
    if any(len(found) > 1 for found in groups_of_entity.values()):
        return SEPARATED_WITH_SPLIT
    return SEPARATED


def _fused_clusters(
    labels: np.ndarray,
    *,
    truth: np.ndarray,
    graph: ScoredGraph,
    threshold: float,
    records: Sequence[NormalizedRecord],
    fixes: dict[str, np.ndarray],
) -> tuple[FusedCluster, ...]:
    labels = canonical_labels(labels)
    members_of: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        members_of[label].append(index)

    left, right, p = graph.edges(min_p=threshold)
    crossing = (labels[left] == labels[right]) & (truth[left] != truth[right])
    cross_edges: dict[int, list[float]] = defaultdict(list)
    for record, probability in zip(left[crossing].tolist(), p[crossing].tolist()):
        cross_edges[int(labels[record])].append(probability)

    fused = []
    for shape in cluster_shapes(labels, graph, threshold):
        members = members_of[shape.cluster]
        entities = [str(truth[index]) for index in members]
        if len(set(entities)) < 2:
            continue
        ordered = sorted(members, key=lambda index: (str(truth[index]), index))
        fused.append(
            FusedCluster(
                members=tuple(
                    FusedMember(entity_id=str(truth[index]), title=records[index].raw.title)
                    for index in ordered
                ),
                n_edges=shape.n_edges,
                n_implied_pairs=shape.n_implied_pairs,
                cross_entity_edges=tuple(sorted(cross_edges[shape.cluster], reverse=True)),
                outcomes={
                    key: _outcome(partition[members], entities) for key, partition in fixes.items()
                },
            )
        )
    return tuple(fused)


def evaluate(
    records: list[Record],
    *,
    dataset: str,
    test_fraction: float = 0.3,
    seed: int = 0,
    n_folds: int = DEFAULT_N_FOLDS,
    cost: CostModel | None = None,
    include_semantic: bool = False,
    n_restarts: int = DEFAULT_RESTARTS,
) -> ClusterReport:
    cost = cost or CostModel()
    train_records, test_records = split_by_entity(
        records, test_fraction=test_fraction, seed=seed
    )
    fit = fit_calibrated_scorer(
        train_records, n_folds=n_folds, seed=seed, include_semantic=include_semantic
    )
    test = prepare(test_records)
    graph = ScoredGraph(
        n_records=len(test.records),
        pair_keys=test.pair_keys,
        probabilities=fit.scorer.probabilities(test.records, test.pair_keys),
    )
    truth = np.array([record.raw.entity_id for record in test.records], dtype=object)

    # The pairwise best-F1 threshold, chosen on out-of-fold train predictions
    # exactly as model/evaluate.py chooses it, and never on test. Its row exists
    # to show what that threshold does once its edges are closed under
    # transitivity.
    out_of_fold = fit.out_of_fold
    best_threshold = best_f1(
        precision_recall_curve(
            fit.scorer.calibrator.transform(out_of_fold.scores),
            out_of_fold.labels,
            n_positives_total=count_true_pairs(train_records),
        )
    ).threshold

    p_hi, p_lo = cost.auto_merge_threshold, cost.auto_reject_threshold
    # The ceiling reads labels, never probabilities: an edge exactly where
    # blocking emitted a true pair. Its costs are still read under the model's
    # probabilities, like every other row's.
    oracle = ScoredGraph(graph.n_records, graph.pair_keys, test.labels.astype(np.float64))

    # The bands on their own -- every pair decided alone, no entities -- billed on
    # the rows' terms. BandSummary bills candidates only, so true pairs blocking
    # never emitted are added as misses, as the rows' closure counts bill them.
    bands = band_summary(
        graph.probabilities, test.labels, cost, n_positives_total=test.n_positives_total
    )
    band_realized_cost = (
        bands.realized_cost
        + (test.n_positives_total - test.n_positives_found) * cost.false_split
    )

    at_p_hi = connected_components(graph, p_hi)
    linkage = average_linkage(graph, cost)
    correlation = correlation_clusters(graph, cost, n_restarts=n_restarts, seed=seed)

    components = "connected components"
    partitions = [
        (SINGLETONS, "all singletons", "no merges — the floor", np.arange(graph.n_records),
         graph, p_hi),
        (COMPONENTS_HI, components, f"p ≥ {p_hi:.4f}, `p_hi`", at_p_hi, graph, p_hi),
        (COMPONENTS_LO, components, f"p ≥ {p_lo:.4f}, `p_lo` — review band merged unreviewed",
         connected_components(graph, p_lo), graph, p_lo),
        (COMPONENTS_BEST_F1, components, f"p ≥ {best_threshold:.4f}, best pairwise F1",
         connected_components(graph, best_threshold), graph, best_threshold),
        (AVERAGE_LINKAGE, FIX_NAMES[AVERAGE_LINKAGE],
         "mean merge gain > 0 — a lone pair at p > `p_hi`", linkage, graph, p_hi),
        (CORRELATION, FIX_NAMES[CORRELATION], "lowest expected cost found", correlation,
         graph, p_hi),
        (ORACLE, components, "true candidate edges — the ceiling",
         connected_components(oracle, 1.0), oracle, 1.0),
    ]
    rows = tuple(
        _measure(
            key,
            method,
            criterion,
            labels,
            truth=truth,
            graph=graph,
            edge_labels=test.labels,
            cost=cost,
            edge_graph=edge_graph,
            edge_threshold=threshold,
        )
        for key, method, criterion, labels, edge_graph, threshold in partitions
    )

    return ClusterReport(
        dataset=dataset,
        seed=seed,
        test_fraction=test_fraction,
        n_folds=n_folds,
        include_semantic=include_semantic,
        n_restarts=n_restarts,
        cost=cost,
        best_f1_threshold=best_threshold,
        n_records=graph.n_records,
        n_entities=len(set(truth.tolist())),
        n_true_pairs=test.n_positives_total,
        n_candidates=test.n_candidates,
        n_true_candidates=test.n_positives_found,
        band_realized_cost=band_realized_cost,
        rows=rows,
        fused=_fused_clusters(
            at_p_hi,
            truth=truth,
            graph=graph,
            threshold=p_hi,
            records=test.records,
            fixes={AVERAGE_LINKAGE: linkage, CORRELATION: correlation},
        ),
    )


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


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


def _paragraph(text: str) -> str:
    return textwrap.fill(text, width=98, break_on_hyphens=False, break_long_words=False)


def _method_row(row: MethodRow) -> str:
    return (
        f"| {row.method} | {row.criterion} | {row.errors.n_clusters:,} | {row.largest} "
        f"| {row.errors.n_fused_clusters} | {row.errors.n_split_entities} "
        f"| {row.n_implied_pairs:,} | {row.n_review:,} | {row.bcubed.precision:.4f} "
        f"| {row.bcubed.recall:.4f} | **{row.bcubed.f1:.4f}** | {row.pairs.precision:.4f} "
        f"| {row.pairs.recall:.4f} | {row.expected_cost:,.1f} | {row.realized_cost:,.0f} |"
    )


def _plural(count: int, noun: str) -> str:
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def _row_name(row: MethodRow, rows: Sequence[MethodRow], *, capital: bool = False) -> str:
    """A row's method, qualified by its criterion only when another row shares the method."""
    shared = sum(1 for other in rows if other.method == row.method) > 1
    name = f"{row.method} at {row.criterion}" if shared else row.method
    return name[:1].upper() + name[1:] if capital else name


def _fused_block(index: int, cluster: FusedCluster) -> str:
    edges = ", ".join(f"{p:.4f}" for p in cluster.cross_entity_edges)
    outcomes = "; ".join(
        f"{FIX_NAMES[key]}: **{outcome}**" for key, outcome in cluster.outcomes.items()
    )
    members = "\n".join(
        f"| `{_escape(member.entity_id)}` | {_escape(member.title)} |"
        for member in cluster.members
    )
    summary = _paragraph(
        f"**{index}. {len(cluster.members)} records, {cluster.n_entities} entities** — "
        f"{_plural(cluster.n_edges, 'edge')} at `p_hi`, "
        f"{_plural(cluster.n_implied_pairs, 'implied pair')}; "
        f"cross-entity edges scored {edges}. {outcomes[:1].upper()}{outcomes[1:]}."
    )
    return f"{summary}\n\n| entity | title |\n| --- | --- |\n{members}"


def _fused_section(report: ClusterReport) -> str:
    if not report.fused:
        return _paragraph(
            "Connected components at `p_hi` fused no two entities on this split, so there is "
            "nothing to name here. The failure itself is pinned by "
            "`tests/test_cluster_components.py`, and the best-pairwise-F1 row above measures "
            "how much a lower threshold chains."
        )
    shown = report.fused[:MAX_FUSED_LISTED]
    scope = (
        f"the first {len(shown)} of the {len(report.fused)}"
        if len(report.fused) > len(shown)
        else f"all {len(report.fused)}"
    )
    intro = _paragraph(
        f"Connected components at `p_hi` is the auto-merge band with its edges closed under "
        f"transitivity. These are {scope} of its clusters that hold more than one entity, and "
        f"what each cost-based method did with the same records: **{SEPARATED}**, "
        f"**{SEPARATED_WITH_SPLIT}**, or **{STILL_FUSED}**."
    )
    blocks = [_fused_block(index, cluster) for index, cluster in enumerate(shown, start=1)]
    return "\n\n".join([intro, *blocks])


def _reading_notes(report: ClusterReport) -> list[str]:
    """The "Reading this honestly" bullets, each chosen from what was measured.

    As in `model/evaluate.py`: no interpretive sentence is printed unconditionally,
    because the CLI takes any registered dataset and a claim true on Abt-Buy and
    asserted regardless is false on the first dataset where it is not. Branching on
    the measurement rather than the dataset name also keeps the rule that nothing
    after `data/` knows which benchmark it is running on.
    """
    floor = report.row(SINGLETONS)
    hi = report.row(COMPONENTS_HI)
    best = report.row(COMPONENTS_BEST_F1)
    linkage = report.row(AVERAGE_LINKAGE)
    oracle = report.row(ORACLE)
    notes = []

    mean_size = report.n_records / max(report.n_entities, 1)
    if floor.bcubed.f1 >= FLOOR_IS_HIGH:
        notes.append(
            f"**B-cubed flatters doing nothing on this split.** Leaving every record a "
            f"singleton scores B³ F1 {floor.bcubed.f1:.4f}: precision is 1 by construction, and "
            f"recall is {floor.bcubed.recall:.4f} because entities average {mean_size:.2f} "
            f"records. Read every row against that floor rather than against zero — connected "
            f"components at `p_hi` is {hi.bcubed.f1 - floor.bcubed.f1:+.4f} above it."
        )
    else:
        notes.append(
            f"**The singleton floor is low on this split** — B³ F1 {floor.bcubed.f1:.4f}, with "
            f"entities averaging {mean_size:.2f} records — so the B-cubed figures above are not "
            f"inflated by small entities."
        )

    if best.bcubed.precision < hi.bcubed.precision and best.n_implied_pairs > hi.n_implied_pairs:
        if best.bcubed.f1 < hi.bcubed.f1:
            heading = "makes worse clusters, and chaining is why"
            f1_part = f"B³ F1 falls with it, {hi.bcubed.f1:.4f} to {best.bcubed.f1:.4f}."
        else:
            heading = "trades cluster precision for recall, and chaining is why"
            f1_part = (
                f"B³ F1 still rises, {hi.bcubed.f1:.4f} to {best.bcubed.f1:.4f}, because recall "
                f"gained more than precision lost — which is why no F1, pairwise or B-cubed, "
                f"should pick the threshold: the precision lost here is fused products."
            )
        notes.append(
            f"**The threshold that maximizes pairwise F1 {heading}.** At p ≥ "
            f"{report.best_f1_threshold:.4f}, chosen on out-of-fold train predictions, connected "
            f"components merges {best.n_implied_pairs:,} pairs no edge supports, against "
            f"{hi.n_implied_pairs:,} at `p_hi`. The largest cluster grows from {hi.largest} "
            f"records to {best.largest}, fused clusters from {hi.errors.n_fused_clusters} to "
            f"{best.errors.n_fused_clusters}, and B³ precision falls from "
            f"{hi.bcubed.precision:.4f} to {best.bcubed.precision:.4f}. {f1_part}"
        )
    else:
        notes.append(
            f"**Chaining did not cost precision at the best-pairwise-F1 threshold on this "
            f"split.** At p ≥ {report.best_f1_threshold:.4f} connected components reaches B³ "
            f"precision {best.bcubed.precision:.4f} against {hi.bcubed.precision:.4f} at `p_hi`, "
            f"with {best.n_implied_pairs:,} implied pairs against {hi.n_implied_pairs:,}. The "
            f"failure is still pinned by `tests/test_cluster_components.py`; this data does not "
            f"show it."
        )

    fused = report.fused
    if not fused:
        notes.append(
            "**Connected components at `p_hi` fused no two entities on this split**, so the "
            "cost-based methods have no fusion to undo here and are compared on cost and "
            "B-cubed alone."
        )
    else:

        def count(key: str, outcome: str) -> int:
            return sum(1 for cluster in fused if cluster.outcomes[key] == outcome)

        text = (
            f"**Chaining happens at `p_hi` as well: {len(fused)} of its clusters fuse more than "
            f"one product.** Average linkage cleanly separates {count(AVERAGE_LINKAGE, SEPARATED)} "
            f"of them and correlation clustering {count(CORRELATION, SEPARATED)}"
        )
        partial = [
            f"{FIX_NAMES[key]} {count(key, SEPARATED_WITH_SPLIT)}"
            for key in FIXES
            if count(key, SEPARATED_WITH_SPLIT)
        ]
        if partial:
            text += f"; separating them but splitting an entity: {', '.join(partial)}"
        text += "."
        survivors = count(CORRELATION, STILL_FUSED)
        if survivors:
            text += (
                f" Correlation clustering leaves {survivors} of the {len(fused)} fused, and it "
                f"keeps a record in a cluster only while no single move lowers expected cost "
                f"under the model's own probabilities — so it is those probabilities holding "
                f"them together, and the fix belongs to an `error-analyst` pass over `model/` "
                f"and `features/`, not to this stage."
            )
            greedy_only = sum(
                1
                for cluster in fused
                if cluster.outcomes[CORRELATION] == STILL_FUSED
                and cluster.outcomes[AVERAGE_LINKAGE] != STILL_FUSED
            )
            if greedy_only:
                text += f" Average linkage separates {greedy_only} of those anyway."
                if report.row(CORRELATION).expected_cost < linkage.expected_cost:
                    text += (
                        " That is its merge order, not the objective: correlation clustering "
                        "reached a lower expected cost overall with them fused."
                    )
        else:
            text += " None survives correlation clustering."
        notes.append(text)

    if report.n_true_candidates == report.n_true_pairs:
        notes.append(
            f"**The ceiling does not bind on this split.** Blocking emitted all "
            f"{report.n_true_pairs:,} true pairs, so components over the true candidate edges "
            f"rebuilds every entity (B³ F1 {oracle.bcubed.f1:.4f}) and nothing above is capped "
            f"by blocking — a property of this split, not a general result."
        )
    else:
        notes.append(
            f"**The ceiling binds.** Blocking emitted {report.n_true_candidates:,} of "
            f"{report.n_true_pairs:,} true pairs, and transitivity over the true ones recovers "
            f"{report.n_recovered_by_closure:,} more, so the best any clusterer here can reach "
            f"is B³ recall {oracle.bcubed.recall:.4f}. Cluster recall can exceed blocking's pair "
            f"completeness — an entity of three needs only two of its pairs emitted — but not "
            f"by more than that."
        )

    methods = [row for row in report.rows if row.key not in (SINGLETONS, ORACLE)]
    by_expected = min(methods, key=lambda row: row.expected_cost)
    by_realized = min(methods, key=lambda row: row.realized_cost)
    if by_expected.key == by_realized.key:
        notes.append(
            f"**The objective and ground truth agree on the winner.** "
            f"{_row_name(by_expected, report.rows, capital=True)} has the lowest expected cost "
            f"of the methods above ({by_expected.expected_cost:,.1f}) and the lowest realized "
            f"cost ({by_realized.realized_cost:,.0f}), so the model's probabilities are good "
            f"enough to choose a partition with on this split."
        )
    else:
        notes.append(
            f"**The objective and ground truth disagree on the winner.** "
            f"{_row_name(by_expected, report.rows, capital=True)} has the lowest expected cost "
            f"({by_expected.expected_cost:,.1f}), but {_row_name(by_realized, report.rows)} the "
            f"lowest realized cost ({by_realized.realized_cost:,.0f}). The objective reads the "
            f"model's probabilities, so where the two disagree it is those probabilities that "
            f"are wrong about some pairs."
        )

    if oracle.expected_cost > by_expected.expected_cost:
        notes.append(
            f"**Under the model's probabilities the truth is not the cheapest partition.** The "
            f"ceiling row costs {oracle.expected_cost:,.1f} in expectation against "
            f"{by_expected.expected_cost:,.1f} for the best partition found, because merging a "
            f"true pair the model scores low is priced as a false merge. No search over this "
            f"objective reaches the ceiling, however thorough; that gap is the scorer's to close."
        )
    else:
        notes.append(
            f"**Under the model's probabilities the truth is the cheapest partition measured** "
            f"({oracle.expected_cost:,.1f} in expectation, against "
            f"{by_expected.expected_cost:,.1f} for the best found), so a better search over this "
            f"objective could still close some of the gap to the ceiling."
        )

    notes.append(
        f"**Review is a third outcome here too, and B-cubed does not see it.** Every pair a "
        f"partition leaves apart falls back to its band — queued for review at p ≥ `p_lo`, "
        f"rejected below — so a lone pair merges only where the bands would auto-merge it, and "
        f"both cost columns bill on the same terms as `reports/model.md`. The B-cubed figures "
        f"score the partition before any review is resolved: connected components at `p_hi` "
        f"leaves {_plural(hi.n_review, 'pair')} queued, average linkage "
        f"{_plural(linkage.n_review, 'pair')}."
    )

    if by_realized.realized_cost < report.band_realized_cost:
        notes.append(
            f"**Clustering lowers the bill, not only the entity count.** On those same terms the "
            f"pairwise bands alone — every pair decided on its own, producing no entities at "
            f"all — bill {report.band_realized_cost:,.0f}; "
            f"{_row_name(by_realized, report.rows)} bills {by_realized.realized_cost:,.0f}."
        )
    else:
        notes.append(
            f"**Clustering buys entities here, not a lower bill.** On those same terms the "
            f"pairwise bands alone bill {report.band_realized_cost:,.0f}, no more than the "
            f"cheapest partition ({by_realized.realized_cost:,.0f})."
        )
    return [_bullet(note) for note in notes]


def render_markdown(report: ClusterReport) -> str:
    """The reports/ artifact: the numbers, plus what makes them readable."""
    cost = report.cost
    table = "\n".join(_method_row(row) for row in report.rows)
    notes = "\n".join(_reading_notes(report))
    columns = "34" if report.include_semantic else "33"

    return f"""# Clusters: entities from the scored pair graph

Pairs become entities here, and the unit of quality changes with them: B-cubed over
records, reported separately from anything pairwise (CLAUDE.md, Invariants). The
pairwise figures for the same scorer are in `reports/model.md`.

Regenerate with:

```bash
python -m dedup.cluster.evaluate --dataset {report.dataset} --out reports/cluster.md
```

## Setup

- Dataset: `{report.dataset}`, test split — {report.n_records:,} records, \
{report.n_entities:,} entities, {report.n_true_pairs:,} true pairs. Blocking emitted \
{report.n_candidates:,} candidate pairs holding {report.n_true_candidates:,} of them.
- Split: entity-grouped, `test_fraction={report.test_fraction}`, `seed={report.seed}`.
- Scorer: the `reports/model.md` pipeline rerun in-process — LightGBM over the \
{columns}-column pair vector, Platt fit out of fold over {report.n_folds} entity-grouped folds.
- Cost model: `{cost}`. A pair a partition leaves apart falls back to its band — review at \
p ≥ `p_lo`, rejection below — so the cost-based methods merge only where that beats both, \
which for a lone pair is exactly `p_hi` (`cluster/base.py`).
- Correlation clustering: local search from components at `p_hi`, from average linkage and \
from {report.n_restarts} seeded pivot orders; the lowest expected cost wins.

## Results

| method | criterion | clusters | largest | fused | split entities | implied pairs | review \
| B³ P | B³ R | B³ F1 | pair P | pair R | E[cost] | cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: \
| ---: | ---: |
{table}

- **fused**: clusters holding records of more than one entity. **split entities**: entities \
spread over more than one cluster.
- **implied pairs**: merged pairs no edge at the row's criterion supports — what transitivity \
added. The cost-based rows are measured against edges at p ≥ `p_hi`, the pairs the bands \
auto-merge.
- **review**: pairs left in different clusters at p ≥ `p_lo`, queued for a reviewer.
- **pair P / R**: every pair the partition merges, candidate or not, against all \
{report.n_true_pairs:,} true pairs in the split.
- **E[cost]**: the objective under the model's probabilities, with unemitted pairs at p = 0. \
**cost**: what ground truth bills with the reviewer assumed correct — {cost.false_merge:g} per \
false merge, {cost.review:g} per queued pair, {cost.false_split:g} per true pair left apart \
outside the queue, the terms `reports/model.md` bills its bands on.

## Chaining, shown

{_fused_section(report)}

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
    parser.add_argument(
        "--restarts",
        type=int,
        default=DEFAULT_RESTARTS,
        help="seeded pivot orders for correlation clustering",
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
        n_restarts=args.restarts,
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
