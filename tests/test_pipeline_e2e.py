"""End-to-end walks through the whole pipeline, asserting the *handoffs*.

Every stage in this project is well covered on its own, and that is exactly the
gap: a per-stage test verifies what a stage computes from input it constructs
itself, so a stage pair whose contract has drifted apart can leave both tests
green. What follows walks the stages in order and checks the boundaries -- the
things CLAUDE.md lists as invariants, at the seam where one stage hands to the
next, rather than inside either.

Two walks, because the pipeline has two halves that meet at one artifact:

1. `test_the_stage_contracts_hold_from_the_raw_csv_through_features` runs on the
   committed fixtures, so it guards a fresh clone with no benchmark downloaded.
   It stops at `features/` because the fixtures cannot drive `model/`: an
   entity-grouped 0.3 split of 4 entities leaves 2 in train, and blocking
   emits no candidate pairs from them at all (see `test_cli_smoke.py`).

2. `test_a_scorer_the_model_cli_saved_drives_batch_index_and_online_lookup`
   is the cross-half chain, and it is the one thing nothing else covered.
   `model/evaluate.py --save-scorer` is the only place in the project that
   persists a scorer and `service/batch.py` only ever loads one, so that file
   on disk *is* the interface between the evaluation half and the serving half
   -- and until now no test carried an artifact across it. This one goes
   model CLI -> batch CLI -> build_index CLI -> `POST /lookup` -> a review
   decision, each step through the real entry point.

Deliberately not asserted here: any published figure. These prove the pieces
compose; `reports/` and the per-stage report tests prove the numbers.
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dedup.blocking.defaults import block_split, block_unlabeled
from dedup.blocking.pairs import total_pairs, unpack
from dedup.data.abt_buy import load_abt_buy
from dedup.eval.splits import count_true_pairs, split_by_entity
from dedup.features.vectorize import PairFeaturizer
from dedup.model import evaluate as model_cli
from dedup.normalize import normalize
from dedup.service import store
from dedup.service.app import create_app
from dedup.service.batch import main as batch_main
from dedup.service.build_index import main as build_index_main

FIXTURES = Path(__file__).parent / "fixtures" / "abt-buy"
REAL_DATA = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"
needs_data = pytest.mark.skipif(
    not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data"
)


def test_the_stage_contracts_hold_from_the_raw_csv_through_features():
    # --- data/ -> Record ------------------------------------------------
    records = load_abt_buy(FIXTURES)
    assert records, "the committed fixtures loaded empty"
    # Every record carries a label, which is what makes an entity-grouped
    # split possible at all. Loaders assign singletons to unmatched rows, so
    # None here would be a loader bug, not an absent ground truth.
    assert all(r.entity_id for r in records)
    # cp1252, not UTF-8: a mojibaked title would still be a str, so the check
    # that matters is that the replacement character never appears.
    assert not any("�" in r.title for r in records)

    # --- eval/splits: the leak invariant -------------------------------
    train, test = split_by_entity(records)
    train_entities = {r.entity_id for r in train}
    test_entities = {r.entity_id for r in test}
    assert not (train_entities & test_entities), "an entity spans both splits -- pair-level leak"
    assert len(train) + len(test) == len(records)

    # --- normalize/ -> NormalizedRecord --------------------------------
    normalized = [normalize(r) for r in train]
    assert len(normalized) == len(train)
    # Composition, not replacement: the raw Record survives for the review
    # queue, which needs the code as the source printed it.
    assert [n.raw.record_id for n in normalized] == [r.record_id for r in train]

    # --- blocking/ -> packed candidate pairs ---------------------------
    run, score = block_split(normalized)
    n = len(normalized)
    assert run.keys.dtype == np.int64, "the packed-int64 representation is load-bearing at scale"
    assert np.all(np.diff(run.keys) > 0), "keys must be sorted and deduplicated"
    left, right = unpack(run.keys, n)
    assert np.all(left < right), "a pair must be upper-triangular"
    assert np.all(right < n), "an index escaped this split's record list"
    # Pair completeness and reduction ratio are only meaningful together --
    # either alone is gamed by moving a threshold.
    assert 0.0 <= score.pair_completeness <= 1.0
    assert 0.0 <= score.reduction_ratio <= 1.0
    # The recall denominator comes from the records, never the candidate set --
    # a denominator that shrank with the candidate set would reward pruning.
    assert score.n_true_pairs_total == count_true_pairs(train)
    assert score.n_true_pairs_found <= score.n_true_pairs_total
    assert len(run.keys) <= total_pairs(n)

    # --- features/ -> FeatureMatrix ------------------------------------
    featurizer = PairFeaturizer(include_semantic=False).fit(normalized)
    matrix = featurizer.transform(normalized, run.keys)
    assert matrix.values.shape == (len(run.keys), len(featurizer.names))
    assert matrix.values.dtype == np.float64
    # A NaN reaching LightGBM is treated as a value with meaning; missingness
    # is carried by explicit indicators instead (invariant I5).
    assert np.isfinite(matrix.values).all(), "a non-finite value reached the feature matrix"

    names = set(featurizer.names)
    for spec in matrix.specs:
        if spec.imputed:
            assert spec.companion_indicator is not None, spec.name
            assert spec.companion_indicator in names, (
                f"{spec.name} imputes but its indicator {spec.companion_indicator} is not a column"
            )

    # --- the labels-free twin agrees ------------------------------------
    # `service/` cannot call `block_split` (its records have no entity_id), so
    # the two must not drift; a second blocker list is how train/serve skew
    # starts.
    unlabeled = [normalize(r.model_copy(update={"entity_id": None})) for r in train]
    assert np.array_equal(block_unlabeled(unlabeled).keys, run.keys)


@needs_data
def test_a_scorer_the_model_cli_saved_drives_batch_index_and_online_lookup(tmp_path):
    scorer_root = tmp_path / "scorer"
    db = tmp_path / "service.duckdb"
    index_root = tmp_path / "index"

    # --- 1. the evaluation half persists the artifact -------------------
    # Trained on the real catalog, because the fixtures cannot train a scorer.
    # --folds 2 keeps it quick; this checks the handoff, not the numbers.
    assert (
        model_cli.main(
            ["--folds", "2", "--out", str(tmp_path / "model.md"), "--save-scorer", str(scorer_root)]
        )
        == 0
    )
    assert (scorer_root / "manifest.json").is_file()

    # --- 2. the serving half loads it and dedupes a catalog -------------
    # A scorer trained elsewhere applied to this catalog, which is the posture
    # `service/batch.py` is built for. The fixtures stand in for "some catalog
    # nobody knows the answers for" and keep this fast.
    assert (
        batch_main(
            [
                "--dataset", "abt-buy",
                "--root", str(FIXTURES),
                "--scorer", str(scorer_root),
                "--db", str(db),
            ]
        )
        == 0
    )

    conn = store.connect(db)
    runs, total = store.list_runs(conn)
    assert total == 1
    run_id = runs[0].run_id
    # The batch CLI is the only writer of every table, so all three must be
    # populated by that one invocation.
    assert store.iter_records(conn, run_id), "records table came back empty"
    assert store.list_clusters(conn, run_id)[1] > 0, "clusters table came back empty"
    n_queued = store.list_review_queue(conn, run_id)[1]
    conn.close()

    # --- 3. online lookup is opt-in per run -----------------------------
    # A run has no queryable index until this CLI builds one and flips the
    # `run_indexes` row on; refitting `ann`'s vectorizer is the same unbounded
    # operation that kept batch execution out of the request path.
    assert build_index_main(
        ["--db", str(db), "--run-id", run_id, "--index-root", str(index_root)]
    ) == 0

    # --- 4. the API answers a lookup against that run -------------------
    # TestClient as a context manager, so the lifespan that opens the DB
    # connection actually runs.
    with TestClient(create_app(db_path=db)) as client:
        assert client.get("/health").json() == {"status": "ok"}

        response = client.post(
            f"/runs/{run_id}/lookup",
            json={"source": "abt_buy", "title": "Panasonic KX-TS208W Corded Phone"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        # Whatever it decides, it must decide *something* and name the record it
        # wrote and the cluster that record now belongs to. Merging into an
        # existing cluster and opening a new singleton are both valid outcomes
        # for this input; returning neither is not.
        assert body["decision"] in {"auto_merge", "review", "new_cluster"}
        assert body["record"]["record_id"]
        assert body["cluster"]["cluster_id"]
        # The new record is readable back through the run it joined, which is
        # what `apply_lookup`'s single transaction exists to guarantee.
        written = client.get(f"/runs/{run_id}/records/{body['record']['record_id']}")
        assert written.status_code == 200, written.text
        assert written.json()["cluster_id"] == body["cluster"]["cluster_id"]

        # --- 5. the review queue a run produced is decidable ------------
        queue = client.get(f"/runs/{run_id}/review-queue").json()
        assert queue["total"] == n_queued, "the API and the store disagree on the queue"
        if queue["total"]:
            review_id = queue["items"][0]["review_id"]
            decided = client.post(
                f"/runs/{run_id}/review-queue/{review_id}/decision",
                json={"is_match": True, "reviewer_id": "e2e"},
            )
            assert decided.status_code == 200, decided.text
            assert decided.json()["is_match"] is True
            # A decided row leaves the pending queue, which is what makes the
            # queue a work list rather than a log.
            assert client.get(f"/runs/{run_id}/review-queue").json()["total"] == queue["total"] - 1
