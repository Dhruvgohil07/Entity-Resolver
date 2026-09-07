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


# ---------------------------------------------------------------------------
# Regressions: unit-rule boundary bugs
#
# The four tests below pin defects found in code review. Each one passed
# silently before the fix, which is the point -- normalize() has no output
# that looks obviously wrong, so only an assertion catches it.
# ---------------------------------------------------------------------------


def test_cu_ft_rule_consumes_its_trailing_period():
    # `cu\.?\s*ft\.?\b` backtracks off the final "." to satisfy \b, leaving
    # "cu ft." -- which then compares unequal to a source that wrote "cu ft".
    record = Record(
        record_id="abt_buy:1",
        source="abt_buy",
        title="Frigidaire 4.2 Cu. Ft. Compact Refrigerator",
    )
    assert "cu ft compact" in normalize(record).normalized_title


def test_abbreviations_with_trailing_period_are_canonicalized():
    # `\bin\.\b` / `\boz\.\b` / `\blbs?\.\b` can never fire: \b after a
    # literal "." demands a word character, but these abbreviations are
    # followed by a space or end-of-string.
    record = Record(
        record_id="abt_buy:2",
        source="abt_buy",
        title="Sony Speaker 12.5 in. deep 8 lbs. 6 oz.",
    )
    normalized_title = normalize(record).normalized_title
    assert normalized_title == "sony speaker 12.5 in deep 8 lb 6 oz"


def test_foot_mark_and_inch_mark_map_to_different_units():
    feet = Record(record_id="abt_buy:3", source="abt_buy", title="Monster 6' HDMI Cable")
    inches = Record(record_id="abt_buy:4", source="abt_buy", title='Sony 6" LCD Monitor')
    assert "6 ft" in normalize(feet).normalized_title
    assert "6 in" in normalize(inches).normalized_title


def test_possessive_apostrophe_after_digit_is_left_alone():
    record = Record(record_id="abt_buy:5", source="abt_buy", title="Best of the 1980's Collection")
    assert normalize(record).normalized_title == "best of the 1980's collection"


def test_spelled_out_unit_needs_a_preceding_number():
    # "feet"/"foot" are only units after a digit; "Foot Massager" is a
    # product name and must survive normalization intact.
    record = Record(
        record_id="abt_buy:6", source="abt_buy", title="Sunbeam Foot Massager 6 feet cord"
    )
    assert normalize(record).normalized_title == "sunbeam foot massager 6 ft cord"


# ---------------------------------------------------------------------------
# Regressions: specs misread as model numbers
#
# model_number is the highest-weight blocking key, so a spec accepted here
# does not just add noise -- it fuses every product sharing that spec into
# one block.
# ---------------------------------------------------------------------------


def test_trailing_convention_requires_the_code_to_follow_the_delimiter():
    # " - " appears, and the last token is model-shaped, but the last token
    # is not what follows the delimiter. The looser "is ' - ' anywhere in
    # the title" test returns the spec "12MP" here.
    record = Record(
        record_id="abt_buy:7",
        source="abt_buy",
        title="Canon PowerShot Digital Camera - 12MP Black",
    )
    assert normalize(record).model_number is None


def test_spec_token_is_not_taken_as_a_model_number():
    for title in (
        "Netgear Router 1200W - Refurbished",
        "Seagate External Drive 500GB USB",
        "Sony Camcorder 1080P Handheld",
    ):
        record = Record(record_id="abt_buy:8", source="abt_buy", title=title)
        assert normalize(record).model_number is None, title


def test_vendor_codes_colliding_with_unit_suffixes_still_qualify():
    # "wh" and "a" are deliberately absent from the spec-suffix list: 161WH
    # and 8500A are real vendor codes, not watt-hours and amps.
    bose = Record(record_id="abt_buy:9", source="abt_buy", title="Bose Speakers White - 161WH")
    hp = Record(record_id="abt_buy:10", source="abt_buy", title="HP Officejet Pro 8500A Printer")
    assert normalize(bose).model_number == "161WH"
    assert normalize(hp).model_number == "8500A"
