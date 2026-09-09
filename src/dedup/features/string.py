"""Text similarity between two records: titles, vendor codes, descriptions, brand.

The bulk of the vector. Four pairwise-local blocks live here -- nothing
fitted, so each is computable for a single pair at serve time with no
artifact -- plus `CorpusTextBlock`, which carries the IDF weights and the
TF-IDF space and therefore has a real `fit`.

Two decisions worth restating, both taken from measurements already recorded
in CLAUDE.md rather than re-derived here:

  * **Description is never concatenated onto the title.** Gluing the two
    together drops the baseline's test F1 from 0.5204 to 0.4349 and its P@10
    from 0.600 to 0.000, because Abt descriptions average 249 characters and
    Buy's average 34 with 40% empty -- so a concatenated Abt vector that is
    mostly description faces a Buy vector that is mostly title. Description
    appears here as its own columns behind its own indicator instead.

  * **The cross title-vs-description columns target a specific failure.**
    Blocking cannot recover `LG Over-The-Range White Microwave Oven -
    LMV1680WH` against `LG 1.6 cu.ft. Over the Range`: the second title is
    truncated and carries no code. What that record does have is a
    description, and matching one side's title against the other side's
    description is the only column here that can see it.

`rapidfuzz.process.cpdist` scores the aligned pair lists element-wise in C++
rather than looping in Python -- the reason CLAUDE.md picks rapidfuzz over
fuzzywuzzy at all. It is deterministic regardless of `workers`, having no
shared state across pairs, so unlike `ann.py`'s index build it needs no
pinning to a single thread.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

from dedup.blocking.standard import code_token_keys
from dedup.eval.baseline import DEFAULT_ANALYZER, DEFAULT_NGRAM_RANGE
from dedup.features.base import FeatureSpec, empty_columns
from dedup.normalize import NormalizedRecord

# Element-wise scoring is independent per pair, so threads change the speed
# and not the answer.
WORKERS = -1


def _cpdist(left_texts: list[str], right_texts: list[str], scorer: Any) -> np.ndarray:
    """One rapidfuzz scorer over aligned pair lists, rescaled to [0, 1]."""
    if not left_texts:
        return np.empty(0, dtype=np.float64)
    scores = process.cpdist(left_texts, right_texts, scorer=scorer, workers=WORKERS)
    return np.asarray(scores, dtype=np.float64) / 100.0


def _jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _containment(a: set[str], b: set[str]) -> float:
    """Overlap measured against the *smaller* side.

    Jaccard punishes a short title for being short: a five-token marketplace
    stub that is a perfect subset of a twenty-token Abt title scores 0.25.
    Containment scores it 1.0, which is what makes this the column that
    speaks to truncated titles.
    """
    smaller = min(len(a), len(b))
    return len(a & b) / smaller if smaller else 0.0


def _ratio(first: float, second: float) -> float:
    """min/max, so the column is symmetric in the pair and lives in [0, 1]."""
    shorter, longer = min(first, second), max(first, second)
    return shorter / longer if longer else 0.0


def _common_prefix(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


class TitleBlock:
    """Pairwise-local title similarity.

    Defined for every pair with nothing to impute: `schema.py` refuses a blank
    title, so both sides always have text. That is why no column here carries
    a companion indicator.
    """

    specs = (
        FeatureSpec("title_ratio", "Levenshtein-based similarity over the whole title."),
        FeatureSpec("title_token_sort_ratio", "Ratio after sorting tokens -- absorbs reordering."),
        FeatureSpec("title_token_set_ratio", "Ratio over token sets -- absorbs extra words."),
        FeatureSpec("title_partial_ratio", "Best ratio of the shorter title against a window."),
        FeatureSpec("title_token_jaccard", "Shared title tokens over their union."),
        FeatureSpec("title_token_containment", "Shared title tokens over the smaller set."),
        FeatureSpec("title_len_ratio", "Shorter title over longer, in characters."),
        FeatureSpec("title_common_prefix_ratio", "Common prefix over the shorter title."),
    )

    def fit(self, records: Sequence[NormalizedRecord]) -> None:
        """No-op: every column here reads only the two records of the pair."""

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        titles = [record.normalized_title for record in records]
        tokens = [set(title.split()) for title in titles]

        left_titles = [titles[i] for i in left]
        right_titles = [titles[j] for j in right]

        out = empty_columns(len(left), self.specs)
        out[:, 0] = _cpdist(left_titles, right_titles, fuzz.ratio)
        out[:, 1] = _cpdist(left_titles, right_titles, fuzz.token_sort_ratio)
        out[:, 2] = _cpdist(left_titles, right_titles, fuzz.token_set_ratio)
        out[:, 3] = _cpdist(left_titles, right_titles, fuzz.partial_ratio)
        for row, (i, j) in enumerate(zip(left, right)):
            out[row, 4] = _jaccard(tokens[i], tokens[j])
            out[row, 5] = _containment(tokens[i], tokens[j])
            out[row, 6] = _ratio(len(titles[i]), len(titles[j]))
            shorter = min(len(titles[i]), len(titles[j]))
            out[row, 7] = _common_prefix(titles[i], titles[j]) / shorter if shorter else 0.0
        return out


class CodeBlock:
    """Vendor codes -- the highest-value signal in the pipeline.

    The model-number comparison runs on `model_number_key`, the
    separator-stripped form, because Abt writes `KXTS208W` where Buy writes
    `KX-TS208W` for the same Panasonic phone. Code-shaped title tokens come
    from `blocking.standard.code_token_keys` rather than a local
    reimplementation: that function is already built on the public
    `normalize.code_key`, and a second copy of a rule that must agree with
    blocking's is exactly how a serve-time index stops reproducing the batch
    keys.

    `code_best_ratio` deliberately scores near-miss codes high -- `WH-1000XM4`
    against `WH-1000XM5` is 0.9. That is not a bug to correct here. Paired
    with `model_number_exact` (which reads 0 for that pair) it is what lets
    the model separate "same code" from "adjacent code", which is a
    distinction a single column cannot express.
    """

    specs = (
        FeatureSpec(
            "model_number_exact",
            "Extracted vendor codes agree, compared in separator-stripped form.",
            imputed=True,
            companion_indicator="model_number_both_present",
        ),
        FeatureSpec(
            "model_number_prefix_ratio",
            "Common prefix of the two codes over the shorter one.",
            imputed=True,
            companion_indicator="model_number_both_present",
        ),
        FeatureSpec(
            "code_token_jaccard",
            "Shared code-shaped title tokens over their union.",
            imputed=True,
            companion_indicator="code_tokens_both_present",
        ),
        FeatureSpec(
            "code_token_shared_count",
            "How many code-shaped tokens both titles carry. Zero is an answer, not a gap.",
        ),
        FeatureSpec(
            "code_best_ratio",
            "Best similarity between any code token of one side and any of the other.",
            imputed=True,
            companion_indicator="code_tokens_both_present",
        ),
    )

    def fit(self, records: Sequence[NormalizedRecord]) -> None:
        """No-op: `code_key` is a pure function, so there is nothing to fit."""

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        model_keys = [record.model_number_key for record in records]
        code_tokens = [set(keys) for keys in code_token_keys(records)]

        out = empty_columns(len(left), self.specs)
        for row, (i, j) in enumerate(zip(left, right)):
            a_code, b_code = model_keys[i], model_keys[j]
            if a_code and b_code:
                out[row, 0] = 1.0 if a_code == b_code else 0.0
                shorter = min(len(a_code), len(b_code))
                out[row, 1] = _common_prefix(a_code, b_code) / shorter if shorter else 0.0

            a_tokens, b_tokens = code_tokens[i], code_tokens[j]
            out[row, 3] = float(len(a_tokens & b_tokens))
            if a_tokens and b_tokens:
                out[row, 2] = _jaccard(a_tokens, b_tokens)
                # Across the pair only. The best match *within* one title
                # would be a property of that record, not of the pair.
                out[row, 4] = max(fuzz.ratio(a, b) for a in a_tokens for b in b_tokens) / 100.0
        return out


class DescriptionBlock:
    """Pairwise-local description similarity, imputed when either side lacks one."""

    specs = (
        FeatureSpec(
            "desc_token_jaccard",
            "Shared description tokens over their union.",
            imputed=True,
            companion_indicator="desc_both_present",
        ),
        FeatureSpec(
            "desc_len_ratio",
            "Shorter description over longer -- the measured Abt/Buy length asymmetry.",
            imputed=True,
            companion_indicator="desc_both_present",
        ),
    )

    def fit(self, records: Sequence[NormalizedRecord]) -> None:
        """No-op."""

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        descriptions = [record.normalized_description or "" for record in records]
        tokens = [set(text.split()) for text in descriptions]

        out = empty_columns(len(left), self.specs)
        for row, (i, j) in enumerate(zip(left, right)):
            if descriptions[i] and descriptions[j]:
                out[row, 0] = _jaccard(tokens[i], tokens[j])
                out[row, 1] = _ratio(len(descriptions[i]), len(descriptions[j]))
        return out


class BrandBlock:
    """Brand equality -- dead on Abt-Buy, and shipped anyway.

    Abt has no brand column at all, so `normalized_brand` is None on every Abt
    record and this column is imputed on every Abt-Buy pair: coverage 0.0000
    in the report. That is the honest way to carry a feature this dataset
    cannot support. The indicator tells the model the column is empty here,
    and the column starts working unchanged on a dataset that populates brand
    -- which is the whole point of `data/` being the only stage that knows a
    dataset's name.
    """

    specs = (
        FeatureSpec(
            "brand_equal",
            "Normalized brands agree.",
            imputed=True,
            companion_indicator="brand_both_present",
        ),
    )

    def fit(self, records: Sequence[NormalizedRecord]) -> None:
        """No-op."""

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        brands = [record.normalized_brand for record in records]
        out = empty_columns(len(left), self.specs)
        for row, (i, j) in enumerate(zip(left, right)):
            if brands[i] and brands[j]:
                out[row, 0] = 1.0 if brands[i] == brands[j] else 0.0
        return out


class CorpusTextBlock:
    """The fitted columns: IDF-weighted token overlap and TF-IDF cosines.

    `fit` must see the **train split only**. Fitting IDF on the full catalog
    scores better and lets test text influence the weights -- the same class
    of leak as a pair-level split, just quieter. `eval/baseline.py` states the
    same protocol for the same reason, and a comparison between the two is
    only like-for-like if both obey it.

    Titles and descriptions share one vectorizer rather than getting one each.
    They have to: the cross columns compare one record's title against the
    other's description, and a cosine between vectors from two separately
    fitted spaces is not a similarity, it is a coincidence.
    """

    specs = (
        FeatureSpec(
            "title_idf_overlap",
            "Shared title tokens weighted by IDF, over the union's weight.",
        ),
        FeatureSpec(
            "title_tfidf_cosine",
            "Cosine of char n-gram TF-IDF titles -- the baseline's similarity, as one column.",
        ),
        FeatureSpec(
            "desc_tfidf_cosine",
            "Cosine of char n-gram TF-IDF descriptions.",
            imputed=True,
            companion_indicator="desc_both_present",
        ),
        FeatureSpec(
            "cross_title_desc_cosine_max",
            "Better of (A title vs B description) and (B title vs A description).",
            imputed=True,
            companion_indicator="desc_any_present",
        ),
        FeatureSpec(
            "cross_title_desc_cosine_min",
            "Worse of the two cross directions; equals the max when only one side has text.",
            imputed=True,
            companion_indicator="desc_any_present",
        ),
    )

    def __init__(
        self,
        analyzer: str = DEFAULT_ANALYZER,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    ) -> None:
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self._vectorizer: TfidfVectorizer | None = None
        self._idf: dict[str, float] = {}
        self._unseen_idf = 0.0

    def fit(self, records: Sequence[NormalizedRecord]) -> None:
        titles = [record.normalized_title for record in records]
        descriptions = [r.normalized_description for r in records if r.normalized_description]

        # lowercase=False for the reason eval/baseline.py gives: normalize.py
        # already casefolded this text and owns that rule in both the batch
        # and serve paths. sklearn's str.lower would bake a second, weaker
        # case rule into the fitted artifact where normalize.py cannot see it.
        self._vectorizer = TfidfVectorizer(
            analyzer=self.analyzer, ngram_range=self.ngram_range, lowercase=False
        )
        self._vectorizer.fit(titles + descriptions)

        # Token-level IDF, kept separate from the char n-gram space above:
        # this column is about *which words* two titles share, which char
        # n-grams deliberately blur together.
        document_frequency: dict[str, int] = {}
        for title in titles:
            for token in set(title.split()):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        n_documents = max(len(titles), 1)
        self._idf = {
            token: math.log(n_documents / count) + 1.0
            for token, count in document_frequency.items()
        }
        # A token unseen in training is maximally rare, not weightless: an
        # unrecognized vendor code is the most informative thing a serve-time
        # title can contain, and weighting it 0 would discard exactly that.
        self._unseen_idf = math.log(n_documents) + 1.0

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("CorpusTextBlock.transform called before fit")

        titles = [record.normalized_title for record in records]
        descriptions = [record.normalized_description or "" for record in records]
        title_tokens = [set(title.split()) for title in titles]

        out = empty_columns(len(left), self.specs)
        if len(left) == 0:
            return out

        # TfidfVectorizer L2-normalizes its rows, so a row-wise product summed
        # across features *is* the cosine -- no separate normalization step.
        title_matrix = self._vectorizer.transform(titles)
        desc_matrix = self._vectorizer.transform(descriptions)

        left_index = np.asarray(left)
        right_index = np.asarray(right)

        def cosine(first: Any, first_rows: np.ndarray, second: Any, second_rows: np.ndarray):
            return np.asarray(
                first[first_rows].multiply(second[second_rows]).sum(axis=1)
            ).ravel()

        out[:, 1] = cosine(title_matrix, left_index, title_matrix, right_index)
        desc_cosine = cosine(desc_matrix, left_index, desc_matrix, right_index)
        cross_a = cosine(title_matrix, left_index, desc_matrix, right_index)
        cross_b = cosine(title_matrix, right_index, desc_matrix, left_index)

        has_description = np.array([bool(text) for text in descriptions])
        left_has = has_description[left_index]
        right_has = has_description[right_index]

        both_have = left_has & right_has
        out[both_have, 2] = desc_cosine[both_have]

        # A side with no description contributes no cross value at all. Filling
        # it with 0.0 here would drag the min down and read as disagreement,
        # when the truth is that there was nothing to compare.
        available = np.stack([right_has, left_has])
        stacked = np.where(available, np.stack([cross_a, cross_b]), np.nan)
        # Sliced to the rows that have at least one direction before reducing:
        # nanmax over an all-NaN column is both a warning and a NaN, and a NaN
        # anywhere in a feature vector propagates through the whole model.
        any_cross = available.any(axis=0)
        if any_cross.any():
            reducible = stacked[:, any_cross]
            out[any_cross, 3] = np.nanmax(reducible, axis=0)
            out[any_cross, 4] = np.nanmin(reducible, axis=0)

        for row, (i, j) in enumerate(zip(left, right)):
            shared = title_tokens[i] & title_tokens[j]
            union = title_tokens[i] | title_tokens[j]
            union_weight = sum(self._idf.get(t, self._unseen_idf) for t in union)
            if union_weight:
                shared_weight = sum(self._idf.get(t, self._unseen_idf) for t in shared)
                out[row, 0] = shared_weight / union_weight
        return out
