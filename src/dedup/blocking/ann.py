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

from collections.abc import Sequence

import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD

from dedup.blocking.base import BlockerRun, timed
from dedup.blocking.pairs import pack
from dedup.eval.baseline import (
    DEFAULT_ANALYZER,
    DEFAULT_NGRAM_RANGE,
    comparison_text,
    fit_vectorizer,
)
from dedup.normalize import NormalizedRecord

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
        raw = [record.raw for record in records]
        # One binding, used by both the fit and the transform below. Written
        # as two separate literals, changing one would silently fit the
        # vectorizer on different text than it searches.
        include_description = False
        vectorizer = fit_vectorizer(
            raw,
            include_description=include_description,
            analyzer=DEFAULT_ANALYZER,
            ngram_range=DEFAULT_NGRAM_RANGE,
        )
        # Transform through `comparison_text` rather than reading
        # `record.normalized_title` directly. The two are equal today, but
        # `fit_vectorizer` fits on comparison_text -- routing both halves
        # through the same function is what stops a later change to it from
        # silently fitting on one text and searching another.
        matrix = vectorizer.transform(
            [comparison_text(record, include_description=include_description) for record in raw]
        )

        if self._n_components is not None:
            reduced = TruncatedSVD(n_components=self._n_components, random_state=0).fit_transform(
                matrix
            )
            vectors = np.ascontiguousarray(reduced.astype(np.float32))
            # SVD output is not unit-length, and inner product only equals
            # cosine on unit vectors. Normalizing here keeps the similarity
            # the index reports comparable with the un-reduced path.
            faiss.normalize_L2(vectors)
            return vectors

        needed = matrix.shape[0] * matrix.shape[1] * 4
        if needed > DENSE_BUDGET_BYTES:
            raise MemoryError(
                f"dense TF-IDF would need {needed / 1024**3:.1f} GB for "
                f"{matrix.shape[0]:,} x {matrix.shape[1]:,}. Pass n_components to project "
                f"with SVD first, or block a smaller catalog."
            )
        # TfidfVectorizer L2-normalizes its rows, so inner product is cosine.
        return np.ascontiguousarray(matrix.toarray().astype(np.float32))

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
