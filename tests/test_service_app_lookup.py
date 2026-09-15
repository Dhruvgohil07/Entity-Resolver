"""`TestClient` tests for `POST /runs/{run_id}/lookup`: a real trained scorer,
a real batch run, real persisted ann/standard indexes -- the same "real
scorer, real catalog" posture `test_service_batch.py` uses, because the
route's decision logic (`service/lookup.py`) is already exhaustively unit
tested against a stub scorer in `test_service_lookup.py`; what this file
proves is the wiring: store writes, index updates, response shape, the
per-run_id lock, and the 404/409 error paths.

The training catalog gives a genuinely bimodal scorer (matching the
bimodal-probability property CLAUDE.md records for the real pipeline), which
makes an organic "review, no merge" case hard to elicit from real scores --
so that one case alone swaps in a stub scorer via `app.state.scorer_cache`,
the same substitution mechanism the route itself uses for a cache hit.
"""


import numpy as np
import pytest
from fastapi.testclient import TestClient

from dedup.model.calibrate import PlattCalibrator
from dedup.model.threshold import CostModel
from dedup.model.train import PairScorer, prepare, read_scorer_manifest, train_scorer
from dedup.schema import Record
from dedup.service import store
from dedup.service.app import create_app
from dedup.service.batch import run_batch
from dedup.service.build_index import build_index

_BRANDS = ["Panasonic", "Canon", "Sony", "Bose", "Cuisinart", "LG", "HP", "Nikon"]


def _training_catalog():
    """Same shape as `test_service_batch.py`'s: two listings per entity, one
    leading with the vendor code and one trailing it, so `model_number_exact`
    and `code_token_jaccard` carry a real, learnable signal."""
    records = []
    for index, brand in enumerate(_BRANDS):
        code = f"XK{100 + index}A"
        records += [
            Record(
                record_id=f"s:{index}a", source="synthetic", entity_id=f"e{index}",
                title=f"{brand} Digital Widget Model {code}", price=100.0 + index,
            ),
            Record(
                record_id=f"s:{index}b", source="synthetic", entity_id=f"e{index}",
                title=f"{brand} {code} Widget", description=f"{brand.lower()} widget",
            ),
        ]
    return records


def _batch_catalog():
    """`_training_catalog()` plus two singleton entities the scorer never
    trained on -- `service/lookup.py` now prices a merge against a
    candidate cluster's *total* size (`cluster/base.merge_credit`'s real
    objective, not one best edge), and every `_training_catalog()` entity
    ends up in a 2-member cluster once `run_batch` clusters it, where a
    single strong edge can never alone clear `C_fm * cluster_size`
    (max one-edge credit is `1 + 1*C_fm` = 21, under `2 * C_fm` = 40 for the
    default cost). These two singletons give a genuine size-1 cluster to
    merge into, so auto-merge stays testable without re-deriving what a
    16-record booster happens to generalize to across two listings at once."""
    return _training_catalog() + [
        Record(record_id="s:8a", source="synthetic", entity_id="e8",
               title="Zenith Digital Widget Model XK108A"),
        Record(record_id="s:9a", source="synthetic", entity_id="e9",
               title="Ferrari Digital Widget Model XK109A"),
    ]


@pytest.fixture(scope="module")
def scorer_root(tmp_path_factory):
    split = prepare(_training_catalog())
    scorer = train_scorer(split)
    raw = scorer.raw_scores(split.records, split.pair_keys)
    calibrated = scorer.with_calibrator(PlattCalibrator().fit(raw, split.labels))
    root = tmp_path_factory.mktemp("scorer")
    calibrated.save(root, metadata={"dataset": "lookup-fixture"})
    return root


@pytest.fixture
def lookup_app(scorer_root, tmp_path):
    """A real batch run over `_training_catalog()`, indexed for lookup, and
    an app wired against it -- one fresh DB/index dir per test, since a
    lookup mutates both."""
    cost = CostModel()
    scorer = PairScorer.load(scorer_root)
    unlabeled = [r.model_copy(update={"entity_id": None}) for r in _batch_catalog()]
    batch = run_batch(unlabeled, scorer=scorer, cost=cost)

    db = tmp_path / "x.duckdb"
    conn = store.connect(db)
    run_id = store.write_run(
        conn,
        dataset="synthetic",
        scorer_root=str(scorer_root),
        scorer_sha256=read_scorer_manifest(scorer_root)["artifact_sha256"],
        cost=cost,
        clusterer="average_linkage",
        records=batch.records,
        pair_keys=batch.pair_keys,
        probabilities=batch.probabilities,
        cluster_labels=batch.cluster_labels,
        review_pairs=batch.review_pairs,
        bands=batch.bands,
    )
    build_index(conn, run_id, tmp_path / "index")
    conn.close()

    app = create_app(db_path=db)
    with TestClient(app) as client:
        yield client, app, run_id


def _lookup(client, run_id, title, **kw):
    body = {"source": "synthetic", "title": title, **kw}
    return client.post(f"/runs/{run_id}/lookup", json=body)


def test_auto_merge_joins_the_matching_cluster(lookup_app):
    """Targets `s:8a` (Zenith) deliberately: it is the one singleton
    cluster in this fixture. Every `_training_catalog()` entity clusters to
    size 2, where a single strong edge can never alone clear
    `C_fm * cluster_size` (see `_batch_catalog()`'s docstring) -- so a
    reliable, deterministic auto-merge case needs a real size-1 cluster,
    not a two-member one this tiny booster may or may not generalize to
    across both listings at once."""
    client, _, run_id = lookup_app
    existing = client.get(f"/runs/{run_id}/records/s:8a").json()

    resp = _lookup(client, run_id, "Zenith Digital Widget Model XK108A Version2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "auto_merge"
    assert body["record"]["cluster_id"] == existing["cluster_id"]
    assert body["cluster"]["cluster_id"] == existing["cluster_id"]
    assert body["cluster"]["size"] == 2  # s:8a plus this one
    assert body["secondary_reviews"] == []

    # Persisted, not just returned: the record is really in the store...
    stored = client.get(f"/runs/{run_id}/records/{body['record']['record_id']}").json()
    assert stored["cluster_id"] == existing["cluster_id"]
    # ...and findable by a second lookup through the re-saved index -- and
    # the now-2-member cluster still clears the aggregate merge threshold,
    # since this record matches every one of its members strongly.
    again = _lookup(client, run_id, "Zenith Digital Widget Model XK108A Version3")
    assert again.json()["decision"] == "auto_merge"
    assert again.json()["record"]["cluster_id"] == existing["cluster_id"]
    assert again.json()["cluster"]["size"] == 3


def test_new_cluster_for_an_unrelated_record(lookup_app):
    client, _, run_id = lookup_app
    resp = _lookup(client, run_id, "Zephyr Gadget Model ZZ999Z")
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "new_cluster"
    assert body["cluster"]["size"] == 1
    assert body["secondary_reviews"] == []


def test_a_record_scoring_high_against_two_clusters_merges_exactly_one(lookup_app):
    """The design's central multi-candidate rule: merge the single
    highest-scoring cluster, queue every other qualifying one for review --
    never both merged, never both only queued. Uses the two singleton
    clusters (`s:8a`/`s:9a`) for the same reason `test_auto_merge_joins_
    the_matching_cluster` does: a reliable merge needs a real size-1
    cluster under the aggregate `merge_credit` rule."""
    client, _, run_id = lookup_app
    cluster8 = client.get(f"/runs/{run_id}/records/s:8a").json()["cluster_id"]
    cluster9 = client.get(f"/runs/{run_id}/records/s:9a").json()["cluster_id"]

    resp = _lookup(client, run_id, "Digital Widget Model XK108A XK109A")
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "auto_merge"
    # Both clusters score identically here; the tie breaks on the smaller
    # cluster_id (agglomerative.py's own convention), which is cluster8's.
    assert body["record"]["cluster_id"] == cluster8
    assert len(body["secondary_reviews"]) == 1
    review = body["secondary_reviews"][0]
    assert cluster9 in (review["left_record"]["cluster_id"], review["right_record"]["cluster_id"])

    # The review row is really queued, findable through the normal route.
    queue = client.get(f"/runs/{run_id}/review-queue").json()
    assert any(item["review_id"] == review["review_id"] for item in queue["items"])


def test_review_only_when_no_cluster_clears_auto_merge(lookup_app):
    """No real score naturally lands in the review band on this small,
    bimodal-scoring catalog (the same bimodal shape CLAUDE.md records for the
    real pipeline), so this one case swaps in a stub scorer -- exercising the
    route's own cache-hit path, not a different code path."""
    client, app, run_id = lookup_app

    class _AlwaysMid:
        def probabilities(self, records, pair_keys):
            return np.full(len(pair_keys), 0.6)  # inside the default [0.5, 0.95) band

    app.state.scorer_cache[run_id] = _AlwaysMid()

    resp = _lookup(client, run_id, "Panasonic Digital Widget Model XK100A Version2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "review"
    assert body["cluster"]["size"] == 1  # nothing cleared p_hi, so it lands as a new singleton
    assert len(body["secondary_reviews"]) >= 1
    assert all(r["probability"] == pytest.approx(0.6) for r in body["secondary_reviews"])


def test_a_duplicate_record_id_is_refused_with_409(lookup_app):
    client, _, run_id = lookup_app
    resp = _lookup(client, run_id, "Anything", record_id="s:0a")
    assert resp.status_code == 409


def test_lookup_against_a_run_with_no_index_is_refused_with_409(scorer_root, tmp_path):
    cost = CostModel()
    scorer = PairScorer.load(scorer_root)
    unlabeled = [r.model_copy(update={"entity_id": None}) for r in _training_catalog()]
    batch = run_batch(unlabeled, scorer=scorer, cost=cost)

    db = tmp_path / "y.duckdb"
    conn = store.connect(db)
    run_id = store.write_run(
        conn,
        dataset="synthetic",
        scorer_root=str(scorer_root),
        scorer_sha256=read_scorer_manifest(scorer_root)["artifact_sha256"],
        cost=cost,
        clusterer="average_linkage",
        records=batch.records,
        pair_keys=batch.pair_keys,
        probabilities=batch.probabilities,
        cluster_labels=batch.cluster_labels,
        review_pairs=batch.review_pairs,
        bands=batch.bands,
    )
    conn.close()  # deliberately no build_index call

    app = create_app(db_path=db)
    with TestClient(app) as client:
        resp = _lookup(client, run_id, "Anything")
        assert resp.status_code == 409


def test_lookup_against_an_unknown_run_is_404(lookup_app):
    client, _, _ = lookup_app
    resp = _lookup(client, "not-a-real-run", "Anything")
    assert resp.status_code == 404
