"""Sentence-embedding cosine over titles -- the one column string metrics cannot give.

CLAUDE.md's argument for carrying embeddings at all: they are complementary
to string distance, not a replacement. They catch paraphrase, where two
titles describe the same product with no shared vocabulary, and they miss
fine distinctions -- `WH-1000XM4` against `WH-1000XM5` is near-identical to an
encoder. The string and code columns are exactly the opposite on both counts,
which is the reason to have both in one vector rather than to pick a winner.

**Opt-in, and off by default.** Two reasons, both practical rather than
principled:

  * The model is a ~90 MB download the rest of the pipeline does not need, so
    a fresh clone can build every other column with no network at all.
  * It is by far the slowest block here, and until `model/` exists there is no
    measurement that says the column earns its cost.

`sentence_transformers` is imported inside the methods, never at module
scope, so importing `dedup.features` stays cheap and offline. `model_is_cached`
lets tests and the CLI check for the weights without triggering a download --
a test that silently pulls 90 MB on a fresh machine is a test that fails in CI
for reasons unrelated to the code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dedup.features.base import FeatureSpec, empty_columns
from dedup.normalize import NormalizedRecord

# Small, fast, and the standard default for this task. Pinned by name so a
# report can be tied to the encoder that produced it.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64


def model_is_cached(model_name: str = DEFAULT_MODEL) -> bool:
    """True when the weights are already on disk, checked without network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - huggingface_hub ships with s-t
        return False
    return isinstance(try_to_load_from_cache(model_name, "config.json"), str)


class SemanticBlock:
    """Cosine between encoded titles.

    Encoding is a pure function of one record, so nothing about this column is
    corpus-dependent and `fit` exists only to load the weights -- which makes
    a missing model fail at fit time, where the error is legible, rather than
    part-way through a transform.

    Titles are encoded once per record and indexed by pair, not encoded per
    pair. On Abt-Buy that is 2,173 encodes against 84,117 -- the same
    arithmetic that makes blocking worth having.
    """

    specs = (
        FeatureSpec(
            "title_embedding_cosine",
            "Cosine between sentence-embedded titles -- catches paraphrase with no shared words.",
        ),
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None

    def fit(self, records: Sequence[NormalizedRecord]) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SemanticBlock.transform called before fit")

        out = empty_columns(len(left), self.specs)
        if len(left) == 0:
            return out

        # normalize_embeddings=True makes the dot product the cosine, the same
        # trick TfidfVectorizer's L2 rows allow in string.py.
        embeddings = self._model.encode(
            [record.normalized_title for record in records],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float64)
        out[:, 0] = np.einsum(
            "ij,ij->i", embeddings[np.asarray(left)], embeddings[np.asarray(right)]
        )
        return out
