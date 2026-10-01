"""Tests for the `build_index` CLI: enables online lookup for a real written run."""

from pathlib import Path

import numpy as np
import pytest

from dedup.blocking.ann import AnnIndex
from dedup.blocking.pairs import pack
from dedup.blocking.standard_index import InvertedIndex
from dedup.model.threshold import BandAssignment, CostModel
from dedup.normalize import normalize
from dedup.schema import Record
from dedup.service import store
from dedup.service.build_index import main


def rec(record_id: str, title: str):
    return normalize(Record(record_id=record_id, source="synthetic", title=title))


def write_small_run(conn) -> str:
    records = [
        rec("r1", "widget deluxe xy123z"),
        rec("r2", "widget deluxe xy123z v2"),
        rec("r3", "unrelated gadget"),
    ]
    n = len(records)
    pair_keys = pack(np.array([0, 0, 1]), np.array([1, 2, 2]), n)
    probabilities = np.array([0.99, 0.6, 0.01])
    cost = CostModel()
    bands = BandAssignment(
        auto_merge=np.array([True, False, False]),
        review=np.array([False, True, False]),
        auto_reject=np.array([False, False, True]),
        cost=cost,
    )
    return store.write_run(
        conn,
        dataset="synthetic",
        scorer_root="artifacts/scorer",
        scorer_sha256="deadbeef",
        cost=cost,
        clusterer="average_linkage",
        records=records,
        pair_keys=pair_keys,
        probabilities=probabilities,
        cluster_labels=np.array([0, 0, 1]),
        review_pairs=np.array([False, True, False]),
        bands=bands,
    )


def test_the_cli_populates_run_indexes(tmp_path):
    db = tmp_path / "x.duckdb"
    conn = store.connect(db)
    run_id = write_small_run(conn)
    conn.close()

    index_root = tmp_path / "index"
    exit_code = main(
        ["--db", str(db), "--run-id", run_id, "--index-root", str(index_root)]
    )
    assert exit_code == 0

    conn = store.connect(db)
    row = store.get_run_indexes(conn, run_id)
    assert row is not None
    assert Path(row.ann_root).is_dir()
    assert Path(row.standard_root).is_dir()

    # The saved indexes are genuinely loadable and queryable.
    ann = AnnIndex.load(Path(row.ann_root))
    found = ann.query_one(rec("r4", "widget deluxe xy123z v3"))
    assert any(record_id == "r1" for record_id, _ in found)

    inverted = InvertedIndex.load(Path(row.standard_root))
    # r1/r2 share no code-shaped token (plain words only), so this at least
    # confirms load() succeeds and query_one runs without error.
    inverted.query_one(rec("r4", "widget deluxe xy123z v3"))


def test_the_cli_refuses_an_unknown_run(tmp_path):
    db = tmp_path / "x.duckdb"
    store.connect(db).close()
    with pytest.raises(SystemExit):
        main(["--db", str(db), "--run-id", "not-a-real-run", "--index-root", str(tmp_path / "idx")])


def test_rerunning_rebuilds_and_overwrites_the_pointers(tmp_path):
    db = tmp_path / "x.duckdb"
    conn = store.connect(db)
    run_id = write_small_run(conn)
    conn.close()

    root_a = tmp_path / "index-a"
    root_b = tmp_path / "index-b"
    assert main(["--db", str(db), "--run-id", run_id, "--index-root", str(root_a)]) == 0
    assert main(["--db", str(db), "--run-id", run_id, "--index-root", str(root_b)]) == 0

    conn = store.connect(db)
    row = store.get_run_indexes(conn, run_id)
    assert row.ann_root == str(root_b / "ann")
