"""Batch dedup: blocking -> scoring -> clustering over a catalog with no ground truth.

`python -m dedup.service.batch --dataset abt-buy --scorer artifacts/scorer --db data/service.duckdb`

Deliberately not `model.train.prepare()`: that function's blocking call
(`blocking.defaults.block_split`) goes through `eval.splits.group_by_entity`,
which raises on a record with no `entity_id` -- correct for evaluation, where
the point is to check candidates against a known answer, and wrong here, where
`entity_id` is `None` on every record because nobody knows the answer yet
(`schema.py`'s own docstring: that is what a genuinely new record looks like).
`blocking.defaults.block_unlabeled` is the labels-free twin this module runs
instead.

`run_batch` is pure -- no I/O, no DB. `main()` is the CLI: it loads a dataset,
loads a scorer `model.evaluate --save-scorer` already produced, runs the
pipeline, and writes the result via `service.store.write_run`. This CLI's job
is to populate the store, not to render a `reports/` markdown artifact -- there
is no ground truth here to score a report against.

Clustering is fixed to average linkage, the committed headline clusterer per
`reports/cluster.md` (B3 P 0.9985 / R 0.8978 on Abt-Buy, bills 155
review-equivalents against the pairwise bands' 265) -- not a CLI flag in v1.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dedup.blocking.defaults import block_unlabeled
from dedup.cluster.agglomerative import average_linkage
from dedup.cluster.base import ScoredGraph, review_mask
from dedup.data import DATASETS, load_dataset
from dedup.model.threshold import BandAssignment, CostModel, assign_bands
from dedup.model.train import PairScorer, read_scorer_manifest
from dedup.normalize import NormalizedRecord, normalize
from dedup.schema import Record
from dedup.service import store

CLUSTERER = "average_linkage"


@dataclass(frozen=True)
class BatchRun:
    """One catalog, blocked, scored and clustered -- no ground truth anywhere."""

    records: list[NormalizedRecord]
    pair_keys: np.ndarray
    probabilities: np.ndarray  # calibrated, aligned to pair_keys
    cluster_labels: np.ndarray  # canonical, one per record position
    review_pairs: np.ndarray  # boolean mask over pair_keys: review_mask()'s output
    bands: BandAssignment  # candidate-level, informational -- no BandSummary without labels
    cost: CostModel


def run_batch(records: Sequence[Record], *, scorer: PairScorer, cost: CostModel) -> BatchRun:
    """Blocking -> scoring -> clustering over a catalog with no ground truth."""
    normalized = [normalize(record) for record in records]
    run = block_unlabeled(normalized)
    probabilities = scorer.probabilities(normalized, run.keys)
    graph = ScoredGraph(len(normalized), run.keys, probabilities)
    labels = average_linkage(graph, cost)
    return BatchRun(
        records=normalized,
        pair_keys=run.keys,
        probabilities=probabilities,
        cluster_labels=labels,
        review_pairs=review_mask(labels, graph, cost),
        bands=assign_bands(probabilities, cost),
        cost=cost,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="abt-buy", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=None, help="override the dataset directory")
    parser.add_argument(
        "--scorer", type=Path, required=True, help="a PairScorer.save() root to load"
    )
    parser.add_argument("--db", type=Path, required=True, help="the DuckDB file to write into")
    parser.add_argument("--cost-false-merge", type=float, default=CostModel().false_merge)
    parser.add_argument("--cost-false-split", type=float, default=CostModel().false_split)
    parser.add_argument("--cost-review", type=float, default=CostModel().review)
    args = parser.parse_args(argv)

    records = load_dataset(args.dataset, args.root)
    scorer = PairScorer.load(args.scorer)
    scorer_sha256 = read_scorer_manifest(args.scorer)["artifact_sha256"]
    cost = CostModel(
        false_merge=args.cost_false_merge,
        false_split=args.cost_false_split,
        review=args.cost_review,
    )

    batch = run_batch(records, scorer=scorer, cost=cost)
    conn = store.connect(args.db)
    run_id = store.write_run(
        conn,
        dataset=args.dataset,
        scorer_root=str(args.scorer),
        scorer_sha256=scorer_sha256,
        cost=cost,
        clusterer=CLUSTERER,
        records=batch.records,
        pair_keys=batch.pair_keys,
        probabilities=batch.probabilities,
        cluster_labels=batch.cluster_labels,
        review_pairs=batch.review_pairs,
        bands=batch.bands,
    )
    conn.close()

    n_clusters = int(batch.cluster_labels.max()) + 1 if batch.records else 0
    print(
        f"run {run_id}: {len(batch.records):,} records, {batch.pair_keys.size:,} candidates, "
        f"{n_clusters:,} clusters, {int(batch.review_pairs.sum()):,} queued for review",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
