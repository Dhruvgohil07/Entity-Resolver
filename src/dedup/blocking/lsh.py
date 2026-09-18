"""MinHash-LSH blocking: approximate Jaccard neighbours, sublinearly.

Sorted-neighborhood needs a good sort key and exact keys need an exact match;
LSH needs neither. It hashes each record's shingle set so that records with
similar sets land in the same bucket with high probability, without ever
computing the N^2 Jaccard matrix.

**Measured caveat, recorded because it is surprising.** With token shingles at
threshold 0.4 on Abt-Buy, this blocker reaches pair completeness 0.5894 on its
own and contributes *exactly zero* marginal completeness to the union -- 20,539
extra candidate pairs for no additional true pair. Product titles are short
(often six to ten tokens), so a token-shingle set is tiny and Jaccard over it
is coarse: dropping one token moves the similarity a long way.

That is a result about this configuration on this dataset, not about LSH.
`shingles="char"` gives a much denser set, and `synth/`'s token-drop corruption
is precisely the failure mode LSH is supposed to absorb. Both knobs stay
exposed so the claim can be re-tested rather than inherited.

**`max_neighbours` bounds memory the way `standard.py`'s `max_block_size` does.**
At 200k records a sibling family's shared vocabulary can put a whole family in
one record's query result, and nothing capped how many candidates one record's
`index.query` call could contribute -- accumulated across every record, that
grew the raw pair count past what a single array allocation could hold. Unlike
a dropped standard-blocker block (symmetric: every member loses every pair),
this cap is per-record and asymmetric: dropping record A's contribution only
loses pair (A, B) if B's own query also fails to surface A. `index.query` was
never guaranteed symmetric anyway, so this weakens no promise the blocker
already made; determinism holds because the enumeration order over records is
fixed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from datasketch import MinHash, MinHashLSH

from dedup.blocking.base import BlockerRun, timed
from dedup.blocking.pairs import pack
from dedup.normalize import NormalizedRecord

DEFAULT_THRESHOLD = 0.4
DEFAULT_NUM_PERM = 128
DEFAULT_CHAR_SHINGLE = 4


def _shingles(text: str, mode: str, char_k: int) -> set[bytes]:
    if mode == "token":
        return {t.encode("utf-8") for t in text.split()}
    if mode == "char":
        if len(text) < char_k:
            return {text.encode("utf-8")} if text else set()
        return {text[i : i + char_k].encode("utf-8") for i in range(len(text) - char_k + 1)}
    raise ValueError(f"shingles must be 'token' or 'char', got {mode!r}")


class MinHashLSHBlocker:
    def __init__(
        self,
        name: str = "lsh (minhash)",
        *,
        threshold: float = DEFAULT_THRESHOLD,
        num_perm: int = DEFAULT_NUM_PERM,
        shingles: str = "token",
        char_shingle: int = DEFAULT_CHAR_SHINGLE,
        max_neighbours: int | None = None,
        params: str = "",
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        if shingles not in ("token", "char"):
            raise ValueError(f"shingles must be 'token' or 'char', got {shingles!r}")
        if max_neighbours is not None and max_neighbours < 1:
            raise ValueError(f"max_neighbours must be at least 1, got {max_neighbours}")
        self.name = name
        detail = shingles if shingles == "token" else f"char{char_shingle}"
        cap = "" if max_neighbours is None else f", cap={max_neighbours}"
        self.params = params or f"{num_perm}p, t={threshold}, {detail}{cap}"
        self._threshold = threshold
        self._num_perm = num_perm
        self._shingles = shingles
        self._char_shingle = char_shingle
        self._max_neighbours = max_neighbours
        self.dropped_records = 0

    def run(self, records: Sequence[NormalizedRecord]) -> BlockerRun:
        n = len(records)

        with timed() as build:
            # MinHash's permutations are seeded (datasketch defaults to seed=1),
            # so the same records always produce the same signatures -- the
            # candidate set must not drift between runs of the same report.
            index = MinHashLSH(threshold=self._threshold, num_perm=self._num_perm)
            signatures = []
            for position, record in enumerate(records):
                sketch = MinHash(num_perm=self._num_perm)
                for shingle in _shingles(
                    record.normalized_title, self._shingles, self._char_shingle
                ):
                    sketch.update(shingle)
                signatures.append(sketch)
                index.insert(str(position), sketch)

        with timed() as query:
            self.dropped_records = 0
            lefts, rights = [], []
            for position, sketch in enumerate(signatures):
                neighbours = [int(other) for other in index.query(sketch) if int(other) != position]
                if self._max_neighbours is not None and len(neighbours) > self._max_neighbours:
                    self.dropped_records += 1
                    continue
                for neighbour in neighbours:
                    lefts.append(position)
                    rights.append(neighbour)
            keys = (
                pack(np.array(lefts, dtype=np.int64), np.array(rights, dtype=np.int64), n)
                if lefts
                else np.empty(0, dtype=np.int64)
            )

        warnings: tuple[str, ...] = ()
        if self.dropped_records:
            message = (
                f"{self.dropped_records} record(s) had more than max_neighbours="
                f"{self._max_neighbours} LSH neighbours and were dropped, so any true pair "
                f"depending on one of them is not in this candidate set"
            )
            warnings = (message,)

        return BlockerRun(
            name=self.name,
            params=self.params,
            keys=keys,
            build_seconds=build.seconds,
            query_seconds=query.seconds,
            warnings=warnings,
        )
