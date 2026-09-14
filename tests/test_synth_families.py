"""Tests for sibling derivation -- the distinct products `synth/` exists to generate.

The property carrying the weight is distinctness. Two entities sharing a vendor
code or a normalized title would be ground truth contradicting itself, and every
metric computed on it would be wrong in a way no metric could show.
"""

import numpy as np
from rapidfuzz.distance import Levenshtein

from dedup.normalize import code_key, normalize
from dedup.schema import Record
from dedup.synth.families import (
    COLOUR_OF_SUFFIX,
    Product,
    Reservations,
    alternate_code,
    base_product,
    derive_family,
    far_sibling,
    near_sibling,
    normalized_title,
    recolour,
    resize,
)

PHONE = Product(
    title="Panasonic 2-Line White Phone - KX-TS208W",
    description="Corded KX-TS208W phone with caller ID",
    brand="Panasonic",
    price=49.0,
    code="KX-TS208W",
)


def rng(seed=0):
    return np.random.default_rng(seed)


def seed(record_id, title, **fields):
    return Record(record_id=record_id, source="synthetic", entity_id="seed:e1", title=title, **fields)


# ---------------------------------------------------------------------------
# The seed product
# ---------------------------------------------------------------------------


def test_base_product_prefers_a_listing_that_carries_a_code():
    product = base_product(
        [
            seed("seed:a", "Panasonic Corded Phone"),
            seed("seed:b", "Panasonic 2-Line Phone - KXTS208W"),
        ]
    )
    assert product.code == "KXTS208W"
    assert product.seed_record_id == "seed:b"


def test_base_product_fills_fields_a_benchmark_splits_across_listings():
    """Abt-Buy prints the brand on its Buy side only; the product should still have one."""
    product = base_product(
        [
            seed("seed:a", "Sony Turntable - PSLX350H", description="Belt drive turntable"),
            seed("seed:b", "Sony PSLX350H Turntable", brand="Sony", price=99.0),
        ]
    )
    assert (product.brand, product.price, product.description) == (
        "Sony",
        99.0,
        "Belt drive turntable",
    )


# ---------------------------------------------------------------------------
# Siblings
# ---------------------------------------------------------------------------


def test_a_near_sibling_is_one_or_two_characters_of_code_away():
    siblings = [s for s in (near_sibling(PHONE, rng(i)) for i in range(200)) if s is not None]
    assert siblings
    for sibling in siblings:
        assert 1 <= Levenshtein.distance(code_key(PHONE.code), code_key(sibling.code)) <= 2
        assert sibling.code in sibling.title
        assert PHONE.code not in sibling.title
        assert PHONE.code not in sibling.description
        assert sibling.kind == "near"


def test_a_far_sibling_keeps_the_series_prefix_and_moves_the_code_further():
    siblings = [s for s in (far_sibling(PHONE, rng(i)) for i in range(100)) if s is not None]
    assert siblings
    for sibling in siblings:
        assert sibling.code.startswith("KX")
        assert Levenshtein.distance(code_key(PHONE.code), code_key(sibling.code)) >= 3
        assert sibling.kind == "far"


def test_a_colour_suffix_swap_changes_the_colour_word_with_it():
    """A phone whose code now ends in B is listed as black, not still as white."""
    recoloured = set()
    for i in range(300):
        sibling = near_sibling(PHONE, rng(i), mutations=("suffix",), resize_rate=0.0)
        if sibling is None:
            continue
        colour = COLOUR_OF_SUFFIX.get(sibling.code[-1])
        if colour is not None:
            assert colour.capitalize() in sibling.title.split()
            recoloured.add(sibling.code[-1])
    assert recoloured


def test_recolour_leaves_a_title_without_a_colour_word_alone():
    assert recolour("Panasonic 2-Line Phone", "BK") == "Panasonic 2-Line Phone"
    assert recolour("Panasonic White Phone", "ZZ") == "Panasonic White Phone"


def test_resize_changes_a_size_and_never_a_digit_inside_a_code():
    title = 'Samsung 32" LCD TV - LN32A330'
    resized = [t for t in (resize(title, rng(i)) for i in range(30)) if t is not None]
    assert resized
    for new in resized:
        assert new != title
        assert new.endswith('" LCD TV - LN32A330')


def test_a_product_with_no_code_can_only_differ_by_size():
    product = Product(title='Sharp 32" Widescreen TV', description=None, brand="Sharp", price=None, code=None)
    assert far_sibling(product, rng()) is None
    sibling = next(s for s in (near_sibling(product, rng(i)) for i in range(20)) if s is not None)
    assert sibling.code is None
    assert sibling.title != product.title


def test_an_alternate_code_is_a_different_numbering_scheme_normalize_still_reads():
    alt = alternate_code(rng())
    assert len(alt) == 8 and alt[:4].isdigit() and alt[4].isalpha() and alt[5:].isdigit()
    view = normalize(Record(record_id="p", source="synthetic", title=f"Canon Ink Tank - {alt}"))
    assert view.model_number_key == code_key(alt)


# ---------------------------------------------------------------------------
# Distinctness across the catalog
# ---------------------------------------------------------------------------


def test_no_two_products_in_a_family_share_a_code_key_or_a_title():
    reservations = Reservations()
    reservations.reserve(PHONE)
    products, dropped = derive_family(
        PHONE, n_near=15, n_far=15, rng=rng(), reservations=reservations
    )
    keys = [code_key(product.code) for product in products]
    titles = [normalized_title(product).normalized_title for product in products]
    assert len(set(keys)) == len(keys)
    assert len(set(titles)) == len(titles)
    assert len(products) - 1 + dropped == 30


def test_a_request_with_no_near_variant_left_is_drawn_far_before_it_is_dropped():
    """KX-TS208W has a few dozen near variants; asking for 60 must not lose the rest."""
    reservations = Reservations()
    reservations.reserve(PHONE)
    products, dropped = derive_family(
        PHONE, n_near=60, n_far=0, rng=rng(), reservations=reservations
    )
    kinds = [product.kind for product in products[1:]]
    assert "far" in kinds
    assert len(kinds) + dropped == 60
    assert dropped < 5


def test_a_code_taken_elsewhere_in_the_catalog_is_never_derived():
    reservations = Reservations()
    reservations.reserve(PHONE)
    neighbour = Product(title="Panasonic Phone - KX-TS209W", description=None, brand=None, price=None, code="KX-TS209W")
    reservations.reserve(neighbour)

    products, _ = derive_family(PHONE, n_near=40, n_far=0, rng=rng(3), reservations=reservations)
    assert code_key(neighbour.code) not in {code_key(product.code) for product in products[1:]}


def test_a_root_with_no_code_turns_far_requests_into_size_siblings():
    product = Product(title='Sharp 32" Widescreen TV', description=None, brand="Sharp", price=None, code=None)
    reservations = Reservations()
    reservations.reserve(product)
    products, dropped = derive_family(product, n_near=0, n_far=3, rng=rng(), reservations=reservations)
    assert all(p.kind in {"root", "near"} for p in products)
    assert len(products) - 1 + dropped == 3
