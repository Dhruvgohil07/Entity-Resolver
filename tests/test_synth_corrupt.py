"""Tests for the corruption operators that turn one product into several listings.

Two properties matter beyond "it produces variety". Every operator is a pure
function of its generator, so a catalog regenerates byte for byte; and the
operators corrupt the way real listings differ without anticipating
`normalize.py` -- so which unit spellings normalization absorbs is pinned here as a
known property rather than left to be discovered in a report.
"""

from dataclasses import fields, replace

import numpy as np
import pytest
from rapidfuzz.distance import Levenshtein

from dedup.normalize import code_key, normalize
from dedup.schema import Record
from dedup.synth.corrupt import (
    REGION_SUFFIXES,
    CorruptionProfile,
    corrupt,
    hide_code,
    punctuate_code,
    respell_units,
    typo,
)
from dedup.synth.families import Product

PRODUCT = Product(
    title='Samsung 32" LCD TV Black - LN32A330',
    description="Samsung 32 inch LCD television with HDMI inputs and remote",
    brand="Samsung",
    price=499.99,
    code="LN32A330",
    alt_code="0617B002",
)

NOTHING = CorruptionProfile(**{spec.name: 0.0 for spec in fields(CorruptionProfile)})


def rng(seed=0):
    return np.random.default_rng(seed)


def normalized(title):
    return normalize(Record(record_id="p", source="synthetic", title=title)).normalized_title


# ---------------------------------------------------------------------------
# The whole listing
# ---------------------------------------------------------------------------


def test_a_profile_that_fires_nothing_returns_the_product_unchanged():
    listing = corrupt(PRODUCT, rng(), NOTHING)
    assert listing.title == PRODUCT.title
    assert (listing.description, listing.brand, listing.price) == (
        PRODUCT.description,
        PRODUCT.brand,
        PRODUCT.price,
    )
    assert listing.operators == ()


def test_corruption_is_a_pure_function_of_the_generator():
    profile = CorruptionProfile()
    for seed in range(20):
        assert corrupt(PRODUCT, rng(seed), profile) == corrupt(PRODUCT, rng(seed), profile)


def test_no_profile_produces_a_blank_title():
    """Record rejects a blank title, so a corruption that makes one would crash generation.

    A title with a word in it keeps a word however hard it is corrupted. A title
    that never had one -- valid as a Record, if useless -- comes back exactly as it
    was, since there is nothing to corrupt it into.
    """
    brutal = replace(NOTHING, code_drop=1.0, token_drop=1.0, typo=1.0, abbreviation=1.0)
    for title, code in (("TV - AB12", "AB12"), ("AB12", "AB12")):
        product = replace(PRODUCT, title=title, code=code)
        for seed in range(50):
            assert any(char.isalnum() for char in corrupt(product, rng(seed), brutal).title)

    wordless = replace(PRODUCT, title="- -", code=None)
    for seed in range(50):
        assert corrupt(wordless, rng(seed), brutal).title == "- -"


# ---------------------------------------------------------------------------
# Vendor codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["KXTS208W", "KX-TS208W", "DVPFX820/R", "LN32A330"])
def test_code_punctuation_never_changes_the_code_key(code):
    for seed in range(20):
        assert code_key(punctuate_code(code, rng(seed))) == code_key(code)


def test_a_region_suffix_makes_the_printed_code_a_different_code():
    profile = replace(NOTHING, code_region_suffix=1.0)
    for seed in range(20):
        last = corrupt(PRODUCT, rng(seed), profile).title.split()[-1]
        assert last in {PRODUCT.code + suffix for suffix in REGION_SUFFIXES}
        assert code_key(last) != code_key(PRODUCT.code)


def test_an_alternate_code_replaces_the_vendor_code():
    """The CL41CL-against-0617B002 case: one product, two numbering schemes."""
    listing = corrupt(PRODUCT, rng(), replace(NOTHING, code_alternate=1.0))
    assert PRODUCT.alt_code in listing.title
    assert PRODUCT.code not in listing.title


def test_a_hidden_code_is_gone_from_every_listing_of_the_entity():
    hidden = hide_code(PRODUCT)
    assert (hidden.code, hidden.alt_code) == (None, None)
    assert hidden.title == 'Samsung 32" LCD TV Black'
    for seed in range(20):
        title = corrupt(hidden, rng(seed), replace(NOTHING, code_alternate=1.0)).title
        assert PRODUCT.code not in title and PRODUCT.alt_code not in title


def test_a_dropped_code_leaves_no_dangling_separator():
    listing = corrupt(PRODUCT, rng(), replace(NOTHING, code_drop=1.0))
    assert listing.title == 'Samsung 32" LCD TV Black'


# ---------------------------------------------------------------------------
# Words and units
# ---------------------------------------------------------------------------


def test_unit_spellings_normalize_absorbs_normalize_equal():
    absorbed = {normalized(t) for t in ('TV 32" x', "TV 32' x", "TV 32 in. x", "TV 32 inch x")}
    assert absorbed == {"tv 32 in x"}


def test_the_hyphenated_inch_is_a_spelling_normalize_does_not_absorb():
    """Pinned as a known property: a synthetic pair differing only here is not a normalized match."""
    assert normalized("TV 32-inch x") != normalized('TV 32" x')


def test_respelled_units_land_on_one_of_the_two_normalized_forms():
    forms = set()
    for seed in range(40):
        title, _ = respell_units('Samsung 32" LCD TV', rng(seed), 1.0)
        forms.add(normalized(title))
    assert forms == {"samsung 32 in lcd tv", "samsung 32-inch lcd tv"}


def test_typo_edits_at_most_one_character_and_skips_short_tokens():
    changed = 0
    for seed in range(100):
        new = typo("Turntable", rng(seed))
        assert Levenshtein.distance("Turntable", new) <= 2  # a transposition counts as 2
        changed += new != "Turntable"
    assert changed > 80
    assert typo("TV", rng()) == "TV"


# ---------------------------------------------------------------------------
# Brand, price, description
# ---------------------------------------------------------------------------


def test_brand_can_be_withheld_blank_or_written_another_way():
    assert corrupt(PRODUCT, rng(), replace(NOTHING, brand_none=1.0)).brand is None
    assert corrupt(PRODUCT, rng(), replace(NOTHING, brand_empty=1.0)).brand == ""
    alias = corrupt(PRODUCT, rng(), replace(NOTHING, brand_alias=1.0)).brand
    assert alias.startswith("Samsung ") and alias != "Samsung"


def test_price_is_positive_when_shown_and_sometimes_withheld():
    profile = replace(NOTHING, price_none=0.5, price_jitter_sigma=1.0)
    prices = [corrupt(PRODUCT, rng(seed), profile).price for seed in range(200)]
    assert any(price is None for price in prices)
    assert all(price > 0 for price in prices if price is not None)


def test_invalid_rates_are_rejected():
    with pytest.raises(ValueError, match="probability"):
        CorruptionProfile(typo=1.5)
    with pytest.raises(ValueError, match="more than 1"):
        CorruptionProfile(code_drop=0.6, code_alternate=0.6)
