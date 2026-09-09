"""`PairFeaturizer` -- assembles every block into one matrix, in a fixed order.

Shaped like an sklearn transformer on purpose, because the fit/transform
boundary is what makes CLAUDE.md's train/serve-parity invariant enforceable
rather than aspirational:

  * **`fit` sees the train split and nothing else.** Fitting the IDF weights
    or the TF-IDF space on the full catalog scores better and lets test text
    influence the weights -- the same class of leak as a pair-level split,
    just quieter. `eval/baseline.py` follows the identical protocol, which is
    what makes the two numbers comparable.
  * **`fit` never sees labels.** Nothing here is supervised; that is `model/`.
  * **The fitted featurizer is the serve-time artifact.** One record arriving
    at the service is transformed by this same object against the same
    weights, so a column cannot mean one thing in training and another in
    production.

Column order is the block order below, and it is part of the contract: a model
trained when column 7 meant `title_common_prefix_ratio` silently mispredicts
if column 7 later means something else. Blocks are appended, never inserted,
and `FeatureMatrix` carries the specs alongside the values so the mapping
travels with the data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dedup.blocking.pairs import unpack
from dedup.blocking.union import true_pair_keys
from dedup.eval.baseline import DEFAULT_ANALYZER, DEFAULT_NGRAM_RANGE
from dedup.features.base import FeatureBlock, FeatureMatrix, FeatureSpec
from dedup.features.missingness import MissingnessBlock
from dedup.features.numeric import DigitTokenBlock, PriceBlock
from dedup.features.string import (
    BrandBlock,
    CodeBlock,
    CorpusTextBlock,
    DescriptionBlock,
    TitleBlock,
)
from dedup.normalize import NormalizedRecord


def validate_registry(specs: Sequence[FeatureSpec]) -> None:
    """Reject a feature set that breaks the missingness invariant (I5).

    Called from `__init__`, so a block whose columns are mis-declared fails
    when the featurizer is constructed rather than several stages later, when
    the symptom is a model that learned from fill values and no obvious cause.
    """
    names = [spec.name for spec in specs]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"duplicate feature names: {', '.join(sorted(duplicates))}")

    indicators = {spec.name for spec in specs if spec.is_indicator}
    for spec in specs:
        if spec.is_indicator:
            if spec.imputed:
                raise ValueError(
                    f"{spec.name} is an indicator and cannot itself be imputed -- an indicator "
                    f"that can be missing answers nothing"
                )
            continue
        if spec.imputed and spec.companion_indicator is None:
            raise ValueError(
                f"{spec.name} is imputed but names no companion_indicator; a filled value the "
                f"model cannot distinguish from a real one is what invariant I5 forbids"
            )
        if spec.companion_indicator is not None and spec.companion_indicator not in indicators:
            raise ValueError(
                f"{spec.name} names companion_indicator {spec.companion_indicator!r}, which is "
                f"not a registered indicator column"
            )


class PairFeaturizer:
    """Candidate pairs in, feature matrix out."""

    def __init__(
        self,
        *,
        include_semantic: bool = False,
        analyzer: str = DEFAULT_ANALYZER,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    ) -> None:
        blocks: list[FeatureBlock] = [
            TitleBlock(),
            CodeBlock(),
            DescriptionBlock(),
            BrandBlock(),
            CorpusTextBlock(analyzer=analyzer, ngram_range=ngram_range),
            PriceBlock(),
            DigitTokenBlock(),
            MissingnessBlock(),
        ]
        if include_semantic:
            # Appended last so that turning it on does not renumber any
            # existing column.
            from dedup.features.semantic import SemanticBlock

            blocks.append(SemanticBlock())

        self.blocks = blocks
        self.include_semantic = include_semantic
        self._fitted = False
        validate_registry(self.specs)

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        return tuple(spec for block in self.blocks for spec in block.specs)

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def fit(self, records: Sequence[NormalizedRecord]) -> PairFeaturizer:
        """Fit every block on these records. Pass the train split, never the catalog."""
        for block in self.blocks:
            block.fit(records)
        self._fitted = True
        return self

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        pair_keys: np.ndarray,
    ) -> FeatureMatrix:
        """Vectorize packed candidate pairs against `records`.

        `pair_keys` are `blocking/pairs.py`'s packed int64 form, so the
        representation blocking settled on carries through unchanged and the
        indices are positions in *this* record sequence -- which is why
        blocking has to run inside a split rather than across the catalog.
        """
        if not self._fitted:
            raise RuntimeError(
                "PairFeaturizer.transform called before fit; the fitted IDF weights are part "
                "of the feature definition, not an optimization"
            )
        left, right = unpack(np.asarray(pair_keys, dtype=np.int64), len(records))
        columns = [block.transform(records, left, right) for block in self.blocks]
        values = (
            np.concatenate(columns, axis=1)
            if columns
            else np.empty((len(left), 0), dtype=np.float64)
        )
        return FeatureMatrix(values=values, specs=self.specs)

    def fit_transform(
        self,
        records: Sequence[NormalizedRecord],
        pair_keys: np.ndarray,
    ) -> FeatureMatrix:
        """Only correct on the train split. On test, `fit` elsewhere and `transform` here."""
        return self.fit(records).transform(records, pair_keys)


def pair_labels(records: Sequence[NormalizedRecord], pair_keys: np.ndarray) -> np.ndarray:
    """1 where the two records of a candidate pair share an `entity_id`.

    Truth comes from `blocking.union.true_pair_keys`, which expands each
    entity group into all C(size, 2) pairs -- so a size-3 cluster contributes
    three positives, not the two a pairwise mapping file lists.
    """
    truth = true_pair_keys(records)
    return np.isin(np.asarray(pair_keys, dtype=np.int64), truth)
