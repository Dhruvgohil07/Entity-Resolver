"""Tests for the shared normalize step (NormalizedRecord / normalize()).

Model-number and unit examples are pulled from real Abt-Buy / Amazon-Google
rows researched while designing this module, not synthetic ones -- see the
approved plan for the exact sources.
"""

from dedup.normalize import normalize
from dedup.schema import Record


def test_nfkd_normalizes_equivalent_unicode_compositions_to_same_form():
    # "\xe9" is precomposed U+00E9 (LATIN SMALL LETTER E WITH ACUTE);
    # "é" is the canonically-equivalent decomposed form (plain "e" +
    # COMBINING ACUTE ACCENT). Same rendered text, different code points --
    # NFKD's actual guarantee is that both fold to the same normalized form
    # (it does not strip accents, so this isn't an accent-insensitivity
    # claim).
    precomposed = Record(record_id="abt_buy:1", source="abt_buy", title="Caf\xe9 Blender")
    decomposed = Record(record_id="abt_buy:2", source="abt_buy", title="Café Blender")
    assert precomposed.title != decomposed.title
    assert normalize(precomposed).normalized_title == normalize(decomposed).normalized_title


def test_casefold_ignores_case_differences():
    upper = Record(record_id="abt_buy:1", source="abt_buy", title="SONY TURNTABLE")
    lower = Record(record_id="abt_buy:2", source="abt_buy", title="sony turntable")
    assert normalize(upper).normalized_title == normalize(lower).normalized_title


def test_whitespace_collapsed_and_stripped():
    record = Record(record_id="abt_buy:1", source="abt_buy", title="Sony   Turntable\t-  PSLX350H")
    normalized_title = normalize(record).normalized_title
    assert "  " not in normalized_title
    assert normalized_title == normalized_title.strip()


def test_unit_canonicalization_cu_ft():
    record = Record(
        record_id="abt_buy:1",
        source="abt_buy",
        title="Widget",
        description="LG TROMM WM2688HNM 4.2 Cu. Ft. Navy Blue Front Load Washer",
    )
    normalized_description = normalize(record).normalized_description
    assert "cu ft" in normalized_description
    assert "cu." not in normalized_description


def test_brand_alias_resolves_known_variant():
    record = Record(record_id="abt_buy:1", source="abt_buy", title="Widget", brand="Hewlett-Packard")
    assert normalize(record).normalized_brand == "hp"


def test_brand_alias_passthrough_for_unlisted_brand():
    record = Record(record_id="abt_buy:1", source="abt_buy", title="Widget", brand="Sony")
    assert normalize(record).normalized_brand == "sony"


def test_model_number_trailing_convention():
    record = Record(record_id="abt_buy:1", source="abt_buy", title="Sony Turntable - PSLX350H")
    assert normalize(record).model_number == "PSLX350H"


def test_model_number_trailing_convention_ignores_mid_title_digits():
    record = Record(
        record_id="abt_buy:5",
        source="abt_buy",
        title="Bose 27028 161 Bookshelf Pair Speakers In White - 161WH",
    )
    assert normalize(record).model_number == "161WH"


def test_model_number_undelimited_mid_title_fallback():
    record = Record(
        record_id="abt_buy:9", source="abt_buy", title="Netgear ProSafe FS105 Ethernet Switch"
    )
    assert normalize(record).model_number == "FS105"


def test_model_number_leading_convention():
    record = Record(
        record_id="amazon_google:1",
        source="amazon_google",
        title="cd384-aisk9= cisco ios advanced ip services - complete package - cd",
    )
    assert normalize(record).model_number == "CD384-AISK9"


def test_model_number_absent_returns_none():
    record = Record(record_id="amazon_google:2", source="amazon_google", title="learning quickbooks 2007")
    assert normalize(record).model_number is None


def test_normalize_is_pure_and_deterministic():
    record = Record(
        record_id="abt_buy:1",
        source="abt_buy",
        title="Sony Turntable - PSLX350H",
        description="A turntable.",
        brand="Sony",
    )
    assert normalize(record) == normalize(record)


def test_raw_field_round_trips_original_record_unchanged():
    record = Record(record_id="abt_buy:1", source="abt_buy", title="Sony Turntable - PSLX350H")
    assert normalize(record).raw == record


def test_description_none_stays_none():
    record = Record(record_id="abt_buy:1", source="abt_buy", title="Widget", description=None)
    assert normalize(record).normalized_description is None
