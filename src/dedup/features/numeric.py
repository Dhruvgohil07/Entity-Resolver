"""Numeric comparisons: price, and the bare numbers inside a title.

Price is the obvious one and the weaker one. On Abt-Buy it is absent on 61% of
Abt rows and 46% of Buy rows, so `price_both_present` is 0 on most pairs and
the model has to be told that rather than shown a fabricated difference --
which is the whole argument for CLAUDE.md's missingness invariant. Price also
disagrees legitimately between vendors for the *same* product, so a large gap
is weak evidence of a non-match while a tiny gap is decent evidence of a
match. Both columns point the same way and both are inverted.

Digit tokens are the less obvious one and cover a real gap. `code_token_keys`
requires a token to hold both a letter and a digit, so a pure number is
invisible to it -- yet `161` (Bose 161), `1600` (PIXMA iP1600) and `32`
(a screen size) all discriminate. These columns pick those up.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from dedup.features.base import FeatureSpec, empty_columns, has_price
from dedup.normalize import NormalizedRecord


def digit_token_sets(records: Sequence[NormalizedRecord]) -> list[set[str]]:
    """Pure-digit tokens of each normalized title.

    Public because `missingness.py` sets `digit_tokens_both_present` from the
    same sets these columns are computed from -- an indicator derived from a
    second, similar-looking rule is an indicator that eventually lies.

    Single digits are dropped: a lone "2" is a pack count or a line number and
    matches almost everything, so it costs more in false agreement than it
    earns.
    """
    return [
        {token for token in record.normalized_title.split() if token.isdigit() and len(token) > 1}
        for record in records
    ]


class PriceBlock:
    """Price agreement, imputed on any pair where either side has no price."""

    specs = (
        FeatureSpec(
            "price_abs_log_ratio",
            "|log((1+a)/(1+b))| -- a $10 gap counts differently at $20 than at $2000.",
            higher_is_similar=False,
            imputed=True,
            companion_indicator="price_both_present",
        ),
        FeatureSpec(
            "price_rel_diff",
            "|a-b| / max(a, b).",
            higher_is_similar=False,
            imputed=True,
            companion_indicator="price_both_present",
        ),
    )

    def fit(self, records: Sequence[NormalizedRecord]) -> None:
        """No-op: both columns read only the two prices of the pair."""

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        prices = [record.raw.price for record in records]
        present = [has_price(record) for record in records]

        out = empty_columns(len(left), self.specs)
        for row, (i, j) in enumerate(zip(left, right)):
            if not (present[i] and present[j]):
                continue
            a, b = float(prices[i]), float(prices[j])
            # log1p, not log: schema.py permits a price of 0.0 (non-negative,
            # finite) and log(0) is -inf, which would propagate through the
            # whole vector with nothing pointing back at the cause.
            #
            # The +1 shift costs exact scale-freeness at the bottom of the
            # range: a 2x gap reads 0.647 at $10-$20 against 0.693 at
            # $1000-$2000, converging to log(2) as prices grow. That is a real
            # distortion and it is accepted rather than worked around, because
            # it is confined to single-digit prices -- Abt-Buy's lowest is
            # $1.75, with two rows under $5 -- and the column is consumed by a
            # tree model that splits on thresholds rather than reading the
            # value as a ratio.
            out[row, 0] = abs(math.log1p(a) - math.log1p(b))
            larger = max(a, b)
            out[row, 1] = abs(a - b) / larger if larger else 0.0
        return out


class DigitTokenBlock:
    """Agreement on the bare numbers in the two titles."""

    specs = (
        FeatureSpec(
            "digit_token_jaccard",
            "Shared multi-digit title tokens over their union.",
            imputed=True,
            companion_indicator="digit_tokens_both_present",
        ),
        FeatureSpec(
            "digit_token_shared_count",
            "How many multi-digit tokens both titles carry. Zero is an answer, not a gap.",
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
        digits = digit_token_sets(records)

        out = empty_columns(len(left), self.specs)
        for row, (i, j) in enumerate(zip(left, right)):
            a, b = digits[i], digits[j]
            out[row, 1] = float(len(a & b))
            if a and b:
                out[row, 0] = len(a & b) / len(a | b)
        return out
