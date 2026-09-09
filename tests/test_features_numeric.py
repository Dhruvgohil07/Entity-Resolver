"""Behaviour tests for the price and digit-token columns.

Price is the column CLAUDE.md's missingness invariant was written for: it is
absent on 61% of Abt rows and 46% of Buy rows, so most pairs have no price
comparison to make and the model has to be told that rather than shown a
fabricated difference.

The zero-price case has its own test because it is the one that would fail
loudly and late: `schema.py` permits a price of 0.0 (finite, non-negative),
`log(0)` is -inf, and an infinity in one column propagates through the whole
vector with nothing pointing back at the cause.
"""

import numpy as np
import pytest

from dedup.features.base import FILL_VALUE
from dedup.features.numeric import DigitTokenBlock, PriceBlock, digit_token_sets
from dedup.normalize import normalize
from dedup.schema import Record

PAIR = (np.array([0]), np.array([1]))


def rec(title, **kwargs):
    return normalize(
        Record(record_id="s:x", source="synthetic", entity_id="e1", title=title, **kwargs)
    )


def one(block, records, column):
    values = block.transform(records, *PAIR)
    return values[0, [spec.name for spec in block.specs].index(column)]


# ---------------------------------------------------------------------------
# PriceBlock
# ---------------------------------------------------------------------------


def test_equal_prices_give_zero_distance():
    records = [rec("Canon Camera", price=199.0), rec("Canon Camera", price=199.0)]
    assert one(PriceBlock(), records, "price_abs_log_ratio") == pytest.approx(0.0)
    assert one(PriceBlock(), records, "price_rel_diff") == pytest.approx(0.0)


def test_price_columns_are_imputed_when_either_side_has_none():
    records = [rec("Canon Camera", price=199.0), rec("Canon Camera")]
    assert one(PriceBlock(), records, "price_abs_log_ratio") == FILL_VALUE
    assert one(PriceBlock(), records, "price_rel_diff") == FILL_VALUE


def test_a_zero_price_stays_finite():
    """log(0) is -inf; log1p(0) is 0. A price of 0.0 is legal per schema.py."""
    records = [rec("Free Sample", price=0.0), rec("Canon Camera", price=199.0)]
    values = PriceBlock().transform(records, *PAIR)
    assert np.isfinite(values).all()


def test_both_prices_zero_is_agreement_not_a_division_by_zero():
    records = [rec("Free Sample", price=0.0), rec("Free Sample", price=0.0)]
    assert one(PriceBlock(), records, "price_rel_diff") == 0.0


def test_relative_difference_is_symmetric_and_bounded():
    records = [rec("Canon Camera", price=100.0), rec("Canon Camera", price=150.0)]
    reversed_records = list(reversed(records))
    forward = one(PriceBlock(), records, "price_rel_diff")
    backward = one(PriceBlock(), reversed_records, "price_rel_diff")
    assert forward == pytest.approx(backward)
    assert 0.0 <= forward <= 1.0


def test_price_columns_are_declared_as_distances():
    """A smaller gap means a more likely duplicate, so `evaluate.py` has to
    negate these before ranking or they report as useless."""
    assert all(not spec.higher_is_similar for spec in PriceBlock().specs)


def test_the_log_column_is_scale_free_away_from_the_shift():
    """A 2x gap reads the same at $100 and at $10,000 -- the reason both price
    columns are carried rather than just the relative one."""
    block = PriceBlock()
    cheap = [rec("Widget", price=100.0), rec("Widget", price=200.0)]
    dear = [rec("Widget", price=10_000.0), rec("Widget", price=20_000.0)]
    assert one(block, cheap, "price_abs_log_ratio") == pytest.approx(
        one(block, dear, "price_abs_log_ratio"), rel=0.01
    )


def test_the_log1p_shift_costs_scale_freeness_at_single_digit_prices():
    """The accepted cost of being defined at zero, pinned so it is a known
    property rather than a surprise.

    A 2x gap reads ~0.647 at $10-$20 against log(2) = 0.693 at $1000-$2000.
    Abt-Buy's lowest price is $1.75 and only two rows fall under $5, so the
    distortion is confined to a corner of the range that barely exists -- and
    a tree model splits on thresholds rather than reading the value as a
    ratio.
    """
    block = PriceBlock()
    cheap = [rec("Widget", price=10.0), rec("Widget", price=20.0)]
    dear = [rec("Widget", price=1000.0), rec("Widget", price=2000.0)]
    assert one(block, cheap, "price_abs_log_ratio") == pytest.approx(0.647, abs=0.001)
    assert one(block, dear, "price_abs_log_ratio") == pytest.approx(0.693, abs=0.001)


# ---------------------------------------------------------------------------
# DigitTokenBlock
# ---------------------------------------------------------------------------


def test_single_digits_are_dropped_but_longer_numbers_kept():
    """A lone "2" is a pack count and matches almost everything, so it costs
    more in false agreement than it earns."""
    (tokens,) = digit_token_sets([rec("Panasonic 2 Line Phone 1600 Series")])
    assert tokens == {"1600"}


def test_digit_tokens_cover_what_code_tokens_cannot():
    """`code_token_keys` requires both a letter and a digit in one token, so a
    bare model year or capacity is invisible to it."""
    records = [rec("Canon Ink For PIXMA 1600 Printer"), rec("Canon 1600 Ink Cartridge")]
    assert one(DigitTokenBlock(), records, "digit_token_shared_count") == 1.0
    assert one(DigitTokenBlock(), records, "digit_token_jaccard") > 0


def test_digit_jaccard_is_imputed_when_either_title_has_no_number():
    records = [rec("Canon Ink For PIXMA 1600 Printer"), rec("Canon Ink Cartridge")]
    assert one(DigitTokenBlock(), records, "digit_token_jaccard") == FILL_VALUE
    # The count is not imputed: zero shared numbers is a real answer.
    assert one(DigitTokenBlock(), records, "digit_token_shared_count") == 0.0
