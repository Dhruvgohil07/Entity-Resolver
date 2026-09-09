"""Explicit indicator columns -- CLAUDE.md's missingness invariant, as code.

A null price is not a price mismatch. Every column in `string.py` and
`numeric.py` that falls back to `FILL_VALUE` names one of the indicators here
as its `companion_indicator`, and `vectorize.py` refuses to assemble a matrix
where that name does not resolve. So the invariant is checked by construction
rather than remembered: a feature added later either declares its indicator or
fails to build.

Two of these are pairs of columns describing three states rather than two, and
that is deliberate:

    price_both_present  price_neither_present   meaning
    ------------------  ---------------------   -------
                     1                      0   comparable
                     0                      0   exactly one side priced
                     0                      1   nothing to compare

A single `price_is_missing` column collapses the middle row into one of the
outer ones, and the middle row is the interesting one -- a record with no
price at all behaves differently from a pair where one vendor published a
price and the other did not. `desc_both_present` / `desc_any_present` splits
the same three ways, and it is the reason the plan's original
`desc_either_missing` was dropped: that column is exactly `1 -
desc_both_present`, carrying no information and adding a perfectly collinear
feature for the model to split on.

The predicates themselves live in `base.py`, shared with the blocks whose
columns they gate. An indicator computed from its own private copy of "does
this record have a description" is an indicator that eventually disagrees with
the feature it describes, which is worse than no indicator at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dedup.blocking.standard import code_token_keys
from dedup.features.base import (
    FeatureSpec,
    empty_columns,
    has_brand,
    has_description,
    has_model_number,
    has_price,
)
from dedup.features.numeric import digit_token_sets
from dedup.normalize import NormalizedRecord


class MissingnessBlock:
    """The 0/1 columns every imputed feature points at."""

    specs = (
        FeatureSpec(
            "price_both_present",
            "Both records carry a price, so the price columns hold real values.",
            is_indicator=True,
        ),
        FeatureSpec(
            "price_neither_present",
            "Neither record carries a price -- distinct from exactly one missing.",
            is_indicator=True,
        ),
        FeatureSpec(
            "desc_both_present",
            "Both records carry a non-empty description.",
            is_indicator=True,
        ),
        FeatureSpec(
            "desc_any_present",
            "At least one record carries a description, which is what the cross columns need.",
            is_indicator=True,
        ),
        FeatureSpec(
            "brand_both_present",
            "Both records carry a brand. Always 0 on Abt-Buy: Abt has no brand column.",
            is_indicator=True,
        ),
        FeatureSpec(
            "model_number_both_present",
            "Both records yielded a vendor code from `normalize`.",
            is_indicator=True,
        ),
        FeatureSpec(
            "code_tokens_both_present",
            "Both titles contain at least one code-shaped token.",
            is_indicator=True,
        ),
        FeatureSpec(
            "digit_tokens_both_present",
            "Both titles contain at least one multi-digit token.",
            is_indicator=True,
        ),
    )

    def fit(self, records: Sequence[NormalizedRecord]) -> None:
        """No-op: presence is a property of a record, not of the corpus."""

    def transform(
        self,
        records: Sequence[NormalizedRecord],
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        price = np.array([has_price(record) for record in records])
        description = np.array([has_description(record) for record in records])
        brand = np.array([has_brand(record) for record in records])
        model_number = np.array([has_model_number(record) for record in records])
        code_tokens = np.array([bool(keys) for keys in code_token_keys(records)])
        digit_tokens = np.array([bool(tokens) for tokens in digit_token_sets(records)])

        out = empty_columns(len(left), self.specs)
        if len(left) == 0:
            return out

        left_index = np.asarray(left)
        right_index = np.asarray(right)

        out[:, 0] = (price[left_index] & price[right_index]).astype(np.float64)
        out[:, 1] = (~price[left_index] & ~price[right_index]).astype(np.float64)
        out[:, 2] = (description[left_index] & description[right_index]).astype(np.float64)
        out[:, 3] = (description[left_index] | description[right_index]).astype(np.float64)
        out[:, 4] = (brand[left_index] & brand[right_index]).astype(np.float64)
        out[:, 5] = (model_number[left_index] & model_number[right_index]).astype(np.float64)
        out[:, 6] = (code_tokens[left_index] & code_tokens[right_index]).astype(np.float64)
        out[:, 7] = (digit_tokens[left_index] & digit_tokens[right_index]).astype(np.float64)
        return out
