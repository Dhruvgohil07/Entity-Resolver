"""End-to-end tests for the batch CLI: a real scorer, a real (small) catalog,
a real DuckDB file.

The one CLI-calling test in `service/`, matching the project's convention of
one such test per stage (`tests/test_synth_generate.py`'s is the model).
"""

from pathlib import Path

import pytest

from dedup.data.abt_buy import load_abt_buy
from dedup.model.calibrate import PlattCalibrator
from dedup.model.threshold import CostModel
from dedup.model.train import PairScorer, prepare, train_scorer
from dedup.schema import Record
from dedup.service import store
from dedup.service.batch import main, run_batch

FIXTURES = Path(__file__).parent / "fixtures" / "abt-buy"

# Abt-Buy's committed fixtures are only 7 records / 4 entities -- too few
# candidate pairs for LightGBM to produce a discriminative raw score (every
# score comes out identical, which Platt then rightly refuses to calibrate).
# The scorer is trained on a separate, larger synthetic catalog -- the same
# shape test_model_train.py's own fixtures use -- and scores the fixtures at
# serve time, exactly as service/batch.py's real use ("a scorer trained
# elsewhere, applied to this catalog") already assumes.
_BRANDS = ["Panasonic", "Canon", "Sony", "Bose", "Cuisinart", "LG", "HP", "Nikon"]


def _training_catalog():
    records = []
    for index, brand in enumerate(_BRANDS):
        code = f"XK{100 + index}A"
        records += [
            Record(
                record_id=f"s:{index}a",
                source="synthetic",
                entity_id=f"e{index}",
                title=f"{brand} Digital Widget Model {code}",
                price=100.0 + index,
            ),
            Record(
                record_id=f"s:{index}b",
                source="synthetic",
                entity_id=f"e{index}",
                title=f"{brand} {code} Widget",
                description=f"{brand.lower()} widget",
            ),
        ]
    return records


@pytest.fixture(scope="module")
def scorer_root(tmp_path_factory):
    """A real, calibrated scorer -- trained and saved once, reused by every test here."""
    split = prepare(_training_catalog())
    scorer = train_scorer(split)
    raw = scorer.raw_scores(split.records, split.pair_keys)
    calibrated = scorer.with_calibrator(PlattCalibrator().fit(raw, split.labels))
    root = tmp_path_factory.mktemp("scorer")
    calibrated.save(root, metadata={"dataset": "synthetic-fixture"})
    return root


def test_run_batch_needs_no_ground_truth(scorer_root):
    """The regression `block_unlabeled` exists to prevent: a genuine batch
    catalog has entity_id=None on every record, and run_batch must not touch
    ground truth anywhere in the pipeline it drives."""
    scorer = PairScorer.load(scorer_root)
    unlabeled = [r.model_copy(update={"entity_id": None}) for r in load_abt_buy(FIXTURES)]
    assert all(r.entity_id is None for r in unlabeled)

    batch = run_batch(unlabeled, scorer=scorer, cost=CostModel())

    assert len(batch.records) == len(unlabeled)
    assert batch.cluster_labels.shape == (len(unlabeled),)
    assert batch.review_pairs.shape == batch.pair_keys.shape


def test_the_cli_writes_a_run_with_clusters_and_a_review_queue(scorer_root, tmp_path):
    db = tmp_path / "service.duckdb"
    exit_code = main(
        [
            "--dataset", "abt-buy",
            "--root", str(FIXTURES),
            "--scorer", str(scorer_root),
            "--db", str(db),
        ]
    )
    assert exit_code == 0

    conn = store.connect(db)
    runs, total = store.list_runs(conn)
    assert total == 1
    run = runs[0]
    assert run.dataset == "abt-buy"
    assert run.n_records == 7
    assert run.clusterer == "average_linkage"

    records, n_records = store.list_records(conn, run.run_id, limit=100)
    assert n_records == run.n_records == len(records)

    clusters, n_clusters = store.list_clusters(conn, run.run_id, limit=100)
    assert n_clusters == run.n_clusters
    assert sum(c.size for c in clusters) == run.n_records

    # Every queued pair references clusters this same run actually produced --
    # the correctness point store.py's docstring calls out (review_mask, not
    # assign_bands, is what keeps this true).
    queue, _ = store.list_review_queue(conn, run.run_id, limit=100)
    cluster_ids = {c.cluster_id for c in clusters}
    for item in queue:
        assert item.left_cluster_id in cluster_ids
        assert item.right_cluster_id in cluster_ids
        assert item.left_cluster_id != item.right_cluster_id  # apart, or it would not be queued
        assert item.is_match is None


def test_a_second_run_against_the_same_db_adds_a_second_run(scorer_root, tmp_path):
    db = tmp_path / "service.duckdb"
    args = ["--dataset", "abt-buy", "--root", str(FIXTURES), "--scorer", str(scorer_root), "--db", str(db)]
    assert main(args) == 0
    assert main(args) == 0

    conn = store.connect(db)
    _, total = store.list_runs(conn)
    assert total == 2
