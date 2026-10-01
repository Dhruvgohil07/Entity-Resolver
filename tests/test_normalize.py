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


def test_both_quote_marks_fold_to_inches():
    # Measured against the real CSVs, not assumed from typography: all 249
    # digit+' occurrences in Abt.csv/Buy.csv are inch measurements ("3.0'
    # LCD Display", "32' to 50' LCD", "1-1/8' Dome Tweeter") and none are
    # feet. A source that writes 32' must match one that writes 32".
    apostrophe = Record(record_id="abt_buy:3", source="abt_buy", title="Sony 32' LCD TV")
    quote = Record(record_id="abt_buy:4", source="abt_buy", title='Sony 32" LCD TV')
    assert normalize(apostrophe).normalized_title == normalize(quote).normalized_title
    assert "32 in" in normalize(apostrophe).normalized_title


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


def test_a_code_with_a_separator_the_shape_gate_rejected_is_still_extracted():
    # The regression this pins: the shape gate used to test the raw token
    # against ^[A-Za-z0-9-]+$, which rejects "/". On this real Abt-Buy title
    # the trailing, correct code failed that gate, extraction fell through to
    # the *leading* convention, and the record was keyed under a different
    # listing's code -- producing the only cluster fusion in the pipeline
    # that survived both cost-based clusterers, plus a false negative on the
    # same record's true partner. Gating on code_key(token) is the fix: the
    # gate and the key it feeds now agree about what a code is.
    record = Record(
        record_id="abt_buy:30",
        source="abt_buy",
        title="Samsung YP-S2ZW 1GB Flash MP3 Player - YP-S2ZG/XAA",
    )
    normalized = normalize(record)
    assert normalized.model_number == "YP-S2ZG/XAA"
    assert normalized.model_number_key == "yps2zgxaa"


def test_a_code_wearing_any_separator_code_key_strips_is_still_extracted():
    # Not just "/": code_key discards everything outside [a-z0-9], so any
    # separator a vendor prints inside a code must still reach the gate.
    #
    # Named for the direction it actually pins. An earlier name claimed the
    # gate and code_key "agree on every separator", which is deliberately no
    # longer true in general -- the gate reads the printed token as well as
    # the key, because the key cannot tell "9N00.101" from "802.11n". See
    # test_a_decimal_measurement_is_not_a_code_but_a_dotted_code_is.
    for title, expected in (
        ("Apple iPod Shuffle 1GB MP3 Player - MB226LL/A", "MB226LL/A"),
        ("TOMTOM GPS Receiver Accessory Kit - 9N00.101", "9N00.101"),
        ("Hoover Wide Path Bag Up Vac - #U5140900", "#U5140900"),
        ("Canon Deluxe Soft Case - 2595B002(AA)", "2595B002(AA)"),
    ):
        record = Record(record_id="abt_buy:31", source="abt_buy", title=title)
        assert normalize(record).model_number == expected, title


def test_a_spec_wearing_a_separator_is_still_rejected():
    # The other half of gating on the key: a separator must not smuggle a
    # spec *past* the check either. "18-200MM" and "1.5TB" both reduce to
    # the digits-then-unit form the spec list is written against, which the
    # raw-token gate never saw because of the "-" and the ".".
    for title in (
        "Nikon Zoom Lens - 18-200MM",
        "Western Digital External Drive - 1.5TB",
        "SONY PS2 8 MB Memory Card - 2-pk",
    ):
        record = Record(record_id="abt_buy:32", source="abt_buy", title=title)
        assert normalize(record).model_number is None, title


def test_an_aperture_is_not_a_model_number_and_does_not_fuse_two_camera_kits():
    # The regression this pins, found by review on the real catalog. Gating
    # the shape on code_key alone let "f/3.5-5.6G" -- a lens aperture -- pass,
    # because stripping the punctuation leaves "f3556g", which carries a
    # letter and a digit and is not <digits><unit>. Both of these are real
    # Abt-Buy titles for *different* products (entities e00465 and e00883),
    # and both extracted the same code, giving their pair the single shared
    # key "f3556g" and setting model_number_exact and
    # model_number_prefix_ratio to 1. Both columns are monotone-constrained,
    # so a false key there can only push the pair's score up.
    d60 = Record(
        record_id="abt_buy:abt:34",
        source="abt_buy",
        title="Nikon D60 Digital SLR Camera with 18-55mm f/3.5-5.6G VR Lens - 25438",
    )
    d90 = Record(
        record_id="abt_buy:abt:35",
        source="abt_buy",
        title="Nikon D90 SLR Camera with 18-105mm f/3.5-5.6G ED VR Lens - 25448",
    )
    assert normalize(d60).model_number is None
    assert normalize(d90).model_number is None
    # The point is not that each is None but that they no longer collide.
    assert normalize(d60).model_number_key == normalize(d90).model_number_key is None


def test_a_one_character_prefix_before_a_slash_is_an_abbreviation_not_a_code():
    # "w/" is "with", "f/" an f-stop, "b/w" black-and-white. Every real
    # slash-bearing code measured on Abt-Buy carries at least two characters
    # before the slash, "CV/FO-10" being the shortest of the 83 the gate is
    # there to admit.
    for title in (
        "Logitech Gaming Keyboard w/PS2 Adapter",
        "Canon Zoom Lens f/4-5.6 Telephoto",
    ):
        record = Record(record_id="abt_buy:36", source="abt_buy", title=title)
        assert normalize(record).model_number is None, title

    kept = Record(record_id="abt_buy:37", source="abt_buy", title="Casio Adapter - CV/FO-10")
    assert normalize(kept).model_number == "CV/FO-10"


def test_a_spec_segment_condemns_the_whole_slashed_token():
    # "MOUNT/12-VOLT" is not a spec read as a whole -- it fails the
    # <digits><unit> shape, so a check over the full key passes it -- and is
    # obviously one read a segment at a time. Same for "1080p/60Hz".
    for title in (
        "Peerless TV Mount/12-Volt Accessory Kit",
        "Sharp Projector 1080p/60Hz Input Module",
    ):
        record = Record(record_id="abt_buy:38", source="abt_buy", title=title)
        assert normalize(record).model_number is None, title


def test_a_decimal_measurement_is_not_a_code_but_a_dotted_code_is():
    # Both survive code_key identically ("80211n", "9n00101"), so the printed
    # form is the only place the difference exists: a measurement starts with
    # the digits, a dotted vendor code carries a letter before its first ".".
    # Every dotted code on the real catalog has one -- 9N00.101, 9S00.006,
    # 1EG0.052.00, 9UEA.017.00 -- and every spec wearing a dot does not.
    for title in (
        "Netgear Wireless Router 802.11n Dual Band",
        "Onkyo 7.1-Channel Home Theater Receiver",
        "Kodak 10.1-Megapixel Digital Camera",
    ):
        record = Record(record_id="abt_buy:39", source="abt_buy", title=title)
        assert normalize(record).model_number is None, title

    for title, expected in (
        ("TOMTOM GPS Receiver Accessory Kit - 9N00.101", "9N00.101"),
        ("Targus Laptop Case - 1EG0.052.00", "1EG0.052.00"),
    ):
        record = Record(record_id="abt_buy:40", source="abt_buy", title=title)
        assert normalize(record).model_number == expected, title


def test_a_short_code_is_measured_as_the_source_wrote_it():
    # The 4-character floor stays on the token as printed, not on its key:
    # "XM-6" and "GR-4" are real vendor codes whose keys ("xm6", "gr4") are
    # three characters. Moving the floor onto the key silently dropped them.
    for title, expected in (
        ("Audiovox XM6 Outdoor Home Antenna - XM-6", "XM-6"),
        ("Cuisinart GR-4 Griddler Four Multifunctional Cooking Options", "GR-4"),
    ):
        record = Record(record_id="abt_buy:33", source="abt_buy", title=title)
        assert normalize(record).model_number == expected, title


def test_vendor_codes_colliding_with_unit_suffixes_still_qualify():
    # "wh" and "a" are deliberately absent from the spec-suffix list: 161WH
    # and 8500A are real vendor codes, not watt-hours and amps.
    bose = Record(record_id="abt_buy:9", source="abt_buy", title="Bose Speakers White - 161WH")
    hp = Record(record_id="abt_buy:10", source="abt_buy", title="HP Officejet Pro 8500A Printer")
    assert normalize(bose).model_number == "161WH"
    assert normalize(hp).model_number == "8500A"


# ---------------------------------------------------------------------------
# model_number_key -- the blocking form of a vendor code
# ---------------------------------------------------------------------------


def test_the_same_code_in_two_house_styles_shares_one_key():
    # The finding that motivated this field. Abt writes KXTS208W, Buy writes
    # KX-TS208W, and it is the same Panasonic phone. As printed vendor codes
    # both are correct, so model_number keeps them apart; as blocking keys
    # they must agree, and an exact model_number blocker reaches pair
    # completeness 0.3336 against 0.5832 for this form.
    abt = Record(record_id="abt_buy:abt:1", source="abt_buy", title="Panasonic Phone - KXTS208W")
    buy = Record(record_id="abt_buy:buy:1", source="abt_buy", title="Panasonic KX-TS208W Corded")

    assert normalize(abt).model_number != normalize(buy).model_number
    assert normalize(abt).model_number_key == normalize(buy).model_number_key == "kxts208w"


def test_the_printed_vendor_code_is_preserved_alongside_the_key():
    # Two fields rather than one: a human working the review queue needs the
    # code as the source printed it, not the comparison form.
    record = Record(record_id="abt_buy:2", source="abt_buy", title="Panasonic Fax - KX-FA83")
    normalized = normalize(record)

    assert normalized.model_number == "KX-FA83"
    assert normalized.model_number_key == "kxfa83"


def test_code_key_is_the_one_rule_blocking_and_normalize_share():
    # blocking/standard.py once carried its own copy of this, built on
    # str.lower rather than casefold. Two implementations that must agree is
    # how train/serve skew starts: a serve-time inverted index built from the
    # second copy cannot reproduce the batch blocking key.
    from dedup.blocking.standard import code_token_keys
    from dedup.normalize import code_key

    assert code_key("KX-TS208W") == "kxts208w"
    assert code_key("PS/LX350H") == "pslx350h"

    record = Record(record_id="abt_buy:4", source="abt_buy", title="Panasonic KX-TS208W Corded")
    assert code_key(normalize(record).model_number) in set(code_token_keys([normalize(record)])[0])


def test_code_key_folds_accents_which_plain_folding_deliberately_does_not():
    # Documented difference, not an accident of the regex: dropping
    # everything outside [a-z0-9] also drops the combining marks NFKD leaves
    # behind. Wanted for a blocking key, where a missed match costs a true
    # pair and a spurious one costs a cheap comparison -- but `_fold` states
    # it does not strip accents, so the divergence is asserted here.
    from dedup.normalize import _fold, code_key

    assert _fold("S8ÜX390") != "s8ux390"
    assert code_key("S8ÜX390") == "s8ux390"


def test_code_key_does_not_depend_on_how_folded_its_input_already_was():
    # The regression. `code_key` has two callers feeding it differently
    # prepared strings: the model-number path passes a code taken from the
    # raw title, blocking/ passes tokens from the already-NFKD'd one. Without
    # decomposing internally, a precomposed "Ü" is dropped whole ("s8x390")
    # while a decomposed one keeps its base letter ("s8ux390"), so the same
    # vendor code keys two ways and the two blockers never meet.
    import unicodedata

    from dedup.normalize import code_key

    precomposed = "S8ÜX390"
    decomposed = unicodedata.normalize("NFKD", precomposed)
    assert precomposed != decomposed, "the two spellings must actually differ"

    assert code_key(precomposed) == code_key(decomposed) == "s8ux390"


def test_no_model_number_means_no_key():
    record = Record(record_id="abt_buy:3", source="abt_buy", title="Cotton Kitchen Towel Set")
    normalized = normalize(record)

    assert normalized.model_number is None
    assert normalized.model_number_key is None
