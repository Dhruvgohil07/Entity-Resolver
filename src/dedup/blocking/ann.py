"""Approximate-nearest-neighbour blocking over character TF-IDF vectors.

The strongest single blocker measured on Abt-Buy: pair completeness 0.9562 at
reduction ratio 0.9938, beating every exact-key blocker combined. It succeeds
where keys fail because it needs no shared token at all -- "Bose 161WH" and
"Boss 161 Speaker" have a source typo in the brand and still land near each
other in character-3gram space.

Built on `faiss.IndexHNSWFlat`, which is why `faiss-cpu` is a dependency:
`hnswlib` ships no PyPI wheel for any platform and needs an MSVC toolchain
(CLAUDE.md records the direct `pip download` probe). Same HNSW algorithm.

The vectorizer comes from `dedup.eval.baseline.fit_vectorizer` rather than a
local copy of the char-3gram configuration, so this blocker searches the same
text space the TF-IDF baseline is scored in. Two definitions of "the vector
for a record" that drift apart is the train/serve skew CLAUDE.md warns about.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from dedup.blocking.base import BlockerRun, timed
from dedup.blocking.pairs import pack
from dedup.eval.baseline import (
    DEFAULT_ANALYZER,
    DEFAULT_NGRAM_RANGE,
    comparison_text,
    fit_vectorizer,
)
from dedup.normalize import NormalizedRecord

# Whether a record's description ever joins its title in the ann vector space.
# One binding, shared by every fit and every transform below -- changing it in
# one place and not the other is a train/serve skew the same shape as fitting
# on one text and searching another.
_INCLUDE_DESCRIPTION = False

DEFAULT_NEIGHBOURS = 10
DEFAULT_M = 32
DEFAULT_EF_CONSTRUCTION = 80
DEFAULT_EF_SEARCH = 100

# HNSW needs dense vectors, and dense char-3gram TF-IDF is N x |vocab| float32
# -- 59 MB at Abt-Buy's 2,173 x 6,804, but ~24 GB at a 200k-record synth
# catalog. Past this budget, pass `n_components` to project the matrix down
# with SVD first; the guard exists so that limit surfaces as an explanation
# rather than as an out-of-memory kill.
DENSE_BUDGET_BYTES = 2 * 1024**3


def _fit_vectors(
    records: Sequence[NormalizedRecord], n_components: int | None
) -> tuple[TfidfVectorizer, TruncatedSVD | None, np.ndarray]:
    """Fit the vectorizer (+ optional SVD) on `records`; return both, plus
    their vectors. The one definition of "the vector for a record" this
    module has -- `AnnBlocker.run` (batch, self-querying) and `AnnIndex.build`
    (serve-time, incremental) both call this rather than each fitting its
    own, which is exactly the train/serve skew CLAUDE.md's `code_key`
    anecdote warns two implementations of the same rule eventually become.
    """
    raw = [record.raw for record in records]
    vectorizer = fit_vectorizer(
        raw,
        include_description=_INCLUDE_DESCRIPTION,
        analyzer=DEFAULT_ANALYZER,
        ngram_range=DEFAULT_NGRAM_RANGE,
    )
    matrix = vectorizer.transform(
        [comparison_text(record, include_description=_INCLUDE_DESCRIPTION) for record in raw]
    )

    if n_components is not None:
        svd = TruncatedSVD(n_components=n_components, random_state=0).fit(matrix)
        vectors = np.ascontiguousarray(svd.transform(matrix).astype(np.float32))
        # SVD output is not unit-length, and inner product only equals
        # cosine on unit vectors. Normalizing here keeps the similarity
        # the index reports comparable with the un-reduced path.
        faiss.normalize_L2(vectors)
        return vectorizer, svd, vectors

    needed = matrix.shape[0] * matrix.shape[1] * 4
    if needed > DENSE_BUDGET_BYTES:
        raise MemoryError(
            f"dense TF-IDF would need {needed / 1024**3:.1f} GB for "
            f"{matrix.shape[0]:,} x {matrix.shape[1]:,}. Pass n_components to project "
            f"with SVD first, or block a smaller catalog."
        )
    # TfidfVectorizer L2-normalizes its rows, so inner product is cosine.
    vectors = np.ascontiguousarray(matrix.toarray().astype(np.float32))
    return vectorizer, None, vectors


def _transform_one(
    vectorizer: TfidfVectorizer, svd: TruncatedSVD | None, record: NormalizedRecord
) -> np.ndarray:
    """One new record through an already-fitted vectorizer (+ SVD) -- the
    serve-time half of `_fit_vectors`'s definition, never refit. A vocabulary
    term this record introduces that the fit never saw is silently dropped by
    `vectorizer.transform`, the same way any out-of-vocabulary term is; that
    is the accepted, unmeasured cost of staying incremental (CLAUDE.md, Open
    questions).
    """
    matrix = vectorizer.transform(
        [comparison_text(record.raw, include_description=_INCLUDE_DESCRIPTION)]
    )
    if svd is not None:
        vector = np.ascontiguousarray(svd.transform(matrix).astype(np.float32))
        faiss.normalize_L2(vector)
        return vector
    return np.ascontiguousarray(matrix.toarray().astype(np.float32))


class AnnBlocker:
    def __init__(
        self,
        name: str = "ann (faiss HNSW)",
        *,
        neighbours: int = DEFAULT_NEIGHBOURS,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        n_components: int | None = None,
        deterministic: bool = True,
        params: str = "",
    ) -> None:
        if neighbours < 1:
            raise ValueError(f"neighbours must be at least 1, got {neighbours}")
        self.name = name
        detail = f"M={m}, ef={ef_search}, k={neighbours}"
        self.params = params or (detail if n_components is None else f"{detail}, svd={n_components}")
        self._neighbours = neighbours
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._n_components = n_components
        self._deterministic = deterministic

    def _vectors(self, records: Sequence[NormalizedRecord]) -> np.ndarray:
        _, _, vectors = _fit_vectors(records, self._n_components)
        return vectors

    def run(self, records: Sequence[NormalizedRecord]) -> BlockerRun:
        n = len(records)
        # HNSW graph construction is parallelized, and the order threads link
        # nodes changes the graph -- measured on Abt-Buy, repeated runs of the
        # identical input gave 14,608 / 14,603 / 14,602 candidates. Pair
        # completeness was unmoved, but a committed report whose candidate
        # count drifts between runs is not reproducible, so the build is
        # pinned to one thread. Set deterministic=False to trade that back for
        # build speed on a large catalog.
        previous_threads = faiss.omp_get_max_threads()
        if self._deterministic:
            faiss.omp_set_num_threads(1)
        try:
            with timed() as build:
                vectors = self._vectors(records)
                index = faiss.IndexHNSWFlat(vectors.shape[1], self._m, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = self._ef_construction
                index.add(vectors)

            with timed() as query:
                index.hnsw.efSearch = self._ef_search
                # k+1 because a record is always its own nearest neighbour.
                _, neighbours = index.search(vectors, min(self._neighbours + 1, n))
                rows = np.repeat(np.arange(n, dtype=np.int64), neighbours.shape[1])
                cols = neighbours.astype(np.int64).ravel()
                # HNSW returns -1 when it finds fewer than k neighbours.
                keep = (cols >= 0) & (cols != rows)
                keys = pack(rows[keep], cols[keep], n)
        finally:
            faiss.omp_set_num_threads(previous_threads)

        return BlockerRun(
            name=self.name,
            params=self.params,
            keys=keys,
            build_seconds=build.seconds,
            query_seconds=query.seconds,
        )


INDEX_FILE = "index.faiss"
VECTORIZER_FILE = "vectorizer.pkl"
SVD_FILE = "svd.pkl"
RECORD_IDS_FILE = "record_ids.json"
MANIFEST_FILE = "manifest.json"


class AnnIndex:
    """`service/`'s online-lookup half of this module: `AnnBlocker.run` stays a
    pure, one-shot, self-querying batch call -- this splits the same vector
    definition (`_fit_vectors`/`_transform_one`, above) into build (fit once,
    from a batch), `query_one` (one new record, no refit, against the index
    as it stood before this record was ever added), and `add` (grow the index
    so a later lookup sees it too).

    Persistence mirrors `model.train.PairScorer.save`/`.load` exactly: a
    directory, a `manifest.json` carrying an `artifact_sha256`, refuse on
    hash mismatch or a missing manifest. `faiss.write_index`/`read_index`
    round-trip the HNSW graph (including `efConstruction`/`efSearch`,
    confirmed); the fitted vectorizer and SVD, both plain scikit-learn
    objects, pickle the same way `PairScorer`'s fields already do.
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        svd: TruncatedSVD | None,
        index: faiss.Index,
        record_ids: list[str],
        *,
        neighbours: int,
        ef_search: int,
    ) -> None:
        self._vectorizer = vectorizer
        self._svd = svd
        self._index = index
        self._record_ids = record_ids
        self._neighbours = neighbours
        self._index.hnsw.efSearch = ef_search

    @classmethod
    def build(
        cls,
        records: Sequence[NormalizedRecord],
        record_ids: Sequence[str],
        *,
        neighbours: int = DEFAULT_NEIGHBOURS,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        n_components: int | None = None,
        deterministic: bool = True,
    ) -> AnnIndex:
        if len(records) != len(record_ids):
            raise ValueError(
                f"records ({len(records)}) and record_ids ({len(record_ids)}) must match"
            )
        vectorizer, svd, vectors = _fit_vectors(records, n_components)

        # Parallel construction races when multiple new nodes link
        # concurrently -- the same reason AnnBlocker.run pins this. A single
        # add() (below) has nothing else to race against, so it does not.
        previous_threads = faiss.omp_get_max_threads()
        if deterministic:
            faiss.omp_set_num_threads(1)
        try:
            index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.add(vectors)
        finally:
            faiss.omp_set_num_threads(previous_threads)

        return cls(
            vectorizer,
            svd,
            index,
            list(record_ids),
            neighbours=neighbours,
            ef_search=ef_search,
        )

    def query_one(self, record: NormalizedRecord) -> list[tuple[str, float]]:
        """`(record_id, inner-product score)`, sorted descending.

        Against the index as it stood before this record was ever added --
        callers must query before add(), never after, for the same record.
        """
        if not self._record_ids:
            return []
        vector = _transform_one(self._vectorizer, self._svd, record)
        k = min(self._neighbours, len(self._record_ids))
        scores, neighbours = self._index.search(vector, k)
        out = [
            (self._record_ids[int(position)], float(score))
            for score, position in zip(scores[0], neighbours[0])
            if position >= 0
        ]
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out

    def add(self, record: NormalizedRecord, record_id: str) -> None:
        """Grow the index by one record, so a later `query_one` finds it."""
        vector = _transform_one(self._vectorizer, self._svd, record)
        self._index.add(vector)
        self._record_ids.append(record_id)

    def save(self, root: Path) -> str:
        """Write this index to `root`, return the hash `load` will verify."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(root / INDEX_FILE))
        digest = hashlib.sha256((root / INDEX_FILE).read_bytes()).hexdigest()
        (root / VECTORIZER_FILE).write_bytes(pickle.dumps(self._vectorizer))
        if self._svd is not None:
            (root / SVD_FILE).write_bytes(pickle.dumps(self._svd))
        (root / RECORD_IDS_FILE).write_text(
            json.dumps(self._record_ids), encoding="utf-8"
        )
        manifest = {
            "artifact_sha256": digest,
            "n_records": len(self._record_ids),
            "has_svd": self._svd is not None,
            "neighbours": self._neighbours,
            "ef_search": int(self._index.hnsw.efSearch),
            "created_at": datetime.now(UTC).isoformat(),
        }
        (root / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return digest

    @classmethod
    def load(cls, root: Path) -> AnnIndex:
        """The inverse of `save`, refused if the artifact no longer matches
        its manifest (mirrors `PairScorer.load`'s refusal triad)."""
        root = Path(root)
        manifest_path = root / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{root} has no {MANIFEST_FILE}, so it is not a saved AnnIndex. Save one with "
                f"AnnIndex.save or `python -m dedup.service.build_index` (CLAUDE.md > Commands)."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = (root / INDEX_FILE).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest["artifact_sha256"]:
            raise ValueError(
                f"{root / INDEX_FILE} does not match the hash in its manifest -- it was "
                f"edited, truncated, or regenerated without its manifest"
            )
        record_ids = json.loads((root / RECORD_IDS_FILE).read_text(encoding="utf-8"))
        if len(record_ids) != manifest["n_records"]:
            raise ValueError(
                f"{root} holds {len(record_ids)} record id(s); its manifest says "
                f"{manifest['n_records']}"
            )
        index = faiss.read_index(str(root / INDEX_FILE))
        vectorizer = pickle.loads((root / VECTORIZER_FILE).read_bytes())
        svd = None
        if manifest["has_svd"]:
            svd = pickle.loads((root / SVD_FILE).read_bytes())
        return cls(
            vectorizer,
            svd,
            index,
            record_ids,
            neighbours=manifest["neighbours"],
            ef_search=manifest["ef_search"],
        )
