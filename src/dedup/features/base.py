"""The contract every feature block satisfies.

A feature block turns a candidate pair into some fixed number of columns. It
is deliberately smaller than "a feature builder": each block owns one family
(title strings, vendor codes, price, missingness), declares its columns up
front as `FeatureSpec`s, and `vectorize.py` concatenates them in a fixed
order. That is what makes the column layout a property of the registry rather
than of the order somebody happened to write the code in -- a model trained
on column 7 meaning `title_token_jaccard` breaks silently if column 7 later
means something else.

Two things here exist to make CLAUDE.md invariants mechanical rather than
remembered:

  * **`companion_indicator` (invariant I5).** A feature that falls back to a
    fill value on some pairs must name the indicator column that says so.
    A null price is not a price mismatch; without the indicator the model
    learns from the fill value as though it were data. `imputed=True` with no
    companion is rejected by a test that walks the whole registry, so a
    feature added later inherits the check by being registered.

  * **`higher_is_similar`.** Most features rise with similarity; a few
    (`price_abs_log_ratio`) fall. The diagnostic in `evaluate.py` negates the
    falling ones before ranking, because otherwise a strong-but-inverted
    feature reports as a useless one and gets "fixed" by deletion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dedup.normalize import NormalizedRecord

# What an undefined feature is filled with. 0.0 rather than NaN because
# LightGBM's own NaN handling would make the choice invisible, and rather than
# -1 because a similarity of -1 is a value the model could otherwise reach.
# Safe *only* because every imputed feature ships a companion indicator: the
# model can tell "similarity 0.0" from "no similarity to compute".
FILL_VALUE = 0.0


@dataclass(frozen=True)
class FeatureSpec:
    """One column: what it is called, which way it points, and its indicator."""

    name: str
    doc: str

    # False for a distance-shaped column, where a smaller number means a more
    # likely duplicate. Read by evaluate.py, never by the model -- LightGBM
    # does not care about orientation, human readers of the report do.
    higher_is_similar: bool = True

    # True for the 0/1 columns in missingness.py. They are excluded from the
    # companion-indicator requirement, being the thing it points at.
    is_indicator: bool = False

    # True when this feature falls back to FILL_VALUE on some pairs.
    imputed: bool = False

    # The indicator column that distinguishes a real value from the fill.
    # Required when `imputed` is True (invariant I5), None otherwise.
    companion_indicator: str | None = None


@dataclass(frozen=True)
class FeatureMatrix:
    """The vectors, plus the specs that say what each column means.

    Kept together deliberately: an ndarray of feature values with the names
    living somewhere else is how a column-order bug survives review.
    """

    values: np.ndarray  # (n_pairs, n_features), float64
    specs: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-d (n_pairs, n_features), got {self.values.shape}")
        if self.values.shape[1] != len(self.specs):
            raise ValueError(
                f"values has {self.values.shape[1]} columns but {len(self.specs)} specs "
                f"were declared -- a block returned a width it did not declare"
            )

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    @property
    def n_pairs(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def column(self, name: str) -> np.ndarray:
        """One feature's values across every pair, by name rather than index."""
        try:
            index = self.names.index(name)
        except ValueError:
            raise KeyError(f"no feature named {name!r}; have {', '.join(self.names)}") from None
        return self.values[:, index]


class FeatureBlock(Protocol):
    """One family of columns.

    `fit` sees records only -- never labels, and never the test split. Blocks
    whose columns are pairwise-local (a rapidfuzz ratio over two titles)
    implement it as a no-op; blocks carrying a fitted artifact (IDF weights, a
    vectorizer) do the work there so that the artifact is reusable at serve
    time. A block that computed corpus statistics inside `transform` would
    give a different answer for one pair than for the same pair inside a
    batch, which is train/serve skew (CLAUDE.md, I8).
    """

    @property
    def specs(self) -> tuple[FeatureSpec, ...]: ...

    def fit(self, records: Sequence[NormalizedRecord]) -> None: ...

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray: ...


def empty_columns(n_pairs: int, specs: Sequence[FeatureSpec]) -> np.ndarray:
    """A correctly-shaped block of fill values for a transform to write into."""
    return np.full((n_pairs, len(specs)), FILL_VALUE, dtype=np.float64)


# ---------------------------------------------------------------------------
# What "present" means
# ---------------------------------------------------------------------------
#
# These live here, in the contract module, rather than in the blocks that use
# them. An indicator is a claim about *another* column -- `desc_both_present`
# asserts that `desc_token_jaccard` holds a real value -- so the predicate
# that gates the feature and the predicate that sets the indicator have to be
# the same code. Two copies that drift produce an indicator that lies, which
# is worse than having no indicator at all: the model would learn to trust a
# fill value.
#
# `Record` documents the convention these rest on: None means the source did
# not have the column, "" means the column was there and empty. Both are
# absent for feature purposes -- there is no similarity to compute against an
# empty string -- but only the loader is allowed to collapse the distinction,
# and it does not.


def has_description(record: NormalizedRecord) -> bool:
    return bool(record.normalized_description)


def has_brand(record: NormalizedRecord) -> bool:
    return bool(record.normalized_brand)


def has_model_number(record: NormalizedRecord) -> bool:
    return bool(record.model_number_key)


def has_price(record: NormalizedRecord) -> bool:
    """Price is read off the raw record: `normalize` does not touch numbers.

    None is the only absent value. `schema.py` rejects NaN and infinity at the
    boundary precisely so this check does not have to be `math.isfinite`, and
    0.0 is a real price, not a missing one.
    """
    return record.raw.price is not None
