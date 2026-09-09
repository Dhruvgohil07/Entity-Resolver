"""Behaviour tests for the text blocks.

Chosen to pin the *reason* each column exists rather than its exact value, the
same way `test_blocking_blockers.py` pins why each blocker family is there.
Every case below is a failure mode recorded in CLAUDE.md against real Abt-Buy
rows, not an invented string:

  * `KXTS208W` / `KX-TS208W` -- one Panasonic phone, two vendor spellings.
    Worth pair completeness 0.3354 -> 0.5349 in blocking; the same separator
    problem would silently halve this column.
  * `WH-1000XM4` / `WH-1000XM5` -- one character apart and genuinely different
    products. The near-miss column must score them high while the exact column
    scores them 0; that pairing is the signal.
  * A truncated marketplace title -- the failure blocking cannot close, and
    the reason `title_token_containment` and the cross description columns are
    in the vector at all.
"""

import numpy as np
import pytest

from dedup.features.base import FILL_VALUE
from dedup.features.string import (
    BrandBlock,
    CodeBlock,
    CorpusTextBlock,
    DescriptionBlock,
    TitleBlock,
)
from dedup.normalize import normalize
from dedup.schema import Record

PAIR = (np.array([0]), np.array([1]))


def rec(title, **kwargs):
    return normalize(
        Record(record_id="s:x", source="synthetic", entity_id="e1", title=title, **kwargs)
    )


def one(block, records, column):
    """The named column's value for the single pair (0, 1)."""
    values = block.transform(records, *PAIR)
    return values[0, [spec.name for spec in block.specs].index(column)]


# ---------------------------------------------------------------------------
# CodeBlock
# ---------------------------------------------------------------------------


def test_model_number_matches_across_a_separator_difference():
    """Abt writes KXTS208W, Buy writes KX-TS208W. Both are correct as printed."""
    records = [
        rec("Panasonic 2-Line Integrated Telephone System - KXTS208W"),
        rec("Panasonic KX-TS208W Corded Phone"),
    ]
    assert one(CodeBlock(), records, "model_number_exact") == 1.0


def test_near_miss_codes_are_separated_by_the_exact_column_not_the_ratio():
    """`WH-1000XM4` and `WH-1000XM5` are different products.

    The ratio column is *supposed* to score them high -- it measures string
    similarity, and they are similar. What distinguishes them is that the
    exact column reads 0. A single column cannot express that; the pair can.
    """
    records = [rec("Sony WH-1000XM4 Wireless Headphones"), rec("Sony WH-1000XM5 Headphones")]
    assert one(CodeBlock(), records, "model_number_exact") == 0.0
    assert one(CodeBlock(), records, "code_best_ratio") > 0.8


def test_code_columns_are_imputed_when_one_side_has_no_code():
    records = [rec("Canon PowerShot SD1100 Digital Camera"), rec("Digital Camera Silver")]
    assert one(CodeBlock(), records, "model_number_exact") == FILL_VALUE
    assert one(CodeBlock(), records, "code_token_jaccard") == FILL_VALUE
    # The count is not imputed: zero shared code tokens is a real answer.
    assert one(CodeBlock(), records, "code_token_shared_count") == 0.0


def test_shared_count_is_zero_rather_than_missing_when_codes_disagree():
    records = [rec("Canon PowerShot SD1100 Camera"), rec("Nikon Coolpix S550 Camera")]
    assert one(CodeBlock(), records, "code_token_shared_count") == 0.0


# ---------------------------------------------------------------------------
# TitleBlock
# ---------------------------------------------------------------------------


def test_containment_beats_jaccard_on_a_truncated_title():
    """The measured failure mode: a short marketplace stub inside a long title.

    Jaccard punishes the stub for being short. Containment is the column that
    can see it, which is why both are in the vector.
    """
    records = [
        rec("LG Over-The-Range White Microwave Oven - LMV1680WH"),
        rec("LG Over the Range Microwave"),
    ]
    block = TitleBlock()
    assert one(block, records, "title_token_containment") > one(
        block, records, "title_token_jaccard"
    )


def test_title_columns_are_never_imputed():
    """schema.py refuses a blank title, so both sides always have text."""
    assert all(not spec.imputed for spec in TitleBlock().specs)


def test_reordered_titles_score_higher_on_token_sort_than_on_raw_ratio():
    records = [rec("Sony Turntable - PSLX350H"), rec("PSLX350H Turntable Sony")]
    block = TitleBlock()
    assert one(block, records, "title_token_sort_ratio") > one(block, records, "title_ratio")


# ---------------------------------------------------------------------------
# DescriptionBlock and BrandBlock
# ---------------------------------------------------------------------------


def test_description_columns_are_imputed_when_either_side_lacks_one():
    records = [rec("Canon Camera", description="a digital camera"), rec("Canon Camera")]
    assert one(DescriptionBlock(), records, "desc_token_jaccard") == FILL_VALUE


def test_empty_string_description_counts_as_absent():
    """Record's convention keeps None and "" distinct, but neither gives text
    to compare -- only the loader may collapse them, and it does not."""
    records = [rec("Canon Camera", description="a digital camera"), rec("Canon Camera",
                                                                        description="")]
    assert one(DescriptionBlock(), records, "desc_token_jaccard") == FILL_VALUE


def test_brand_is_imputed_when_one_side_has_no_brand_column():
    """Abt-Buy's shape: Abt has no brand column at all, so every cross-source
    pair lands here."""
    records = [rec("Canon Camera", brand="Canon"), rec("Canon Camera")]
    assert one(BrandBlock(), records, "brand_equal") == FILL_VALUE


def test_brand_equality_is_computed_when_both_sides_have_one():
    records = [rec("Canon Camera", brand="Canon"), rec("Canon Printer", brand="canon")]
    assert one(BrandBlock(), records, "brand_equal") == 1.0


# ---------------------------------------------------------------------------
# CorpusTextBlock
# ---------------------------------------------------------------------------


CORPUS = [
    rec("Canon PowerShot SD1100 Digital Camera", description="a compact digital camera"),
    rec("Nikon Coolpix S550 Digital Camera"),
    rec("Sony Cybershot W120 Digital Camera"),
    rec("Panasonic Lumix TZ5 Digital Camera"),
]


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        CorpusTextBlock().transform(CORPUS, *PAIR)


def test_a_rare_shared_token_outweighs_a_common_one():
    """The point of IDF weighting: sharing "camera" here says nothing, because
    every record has it. Sharing a code says almost everything."""
    block = CorpusTextBlock()
    block.fit(CORPUS)
    common = [rec("Digital Camera Black"), rec("Digital Camera Silver")]
    rare = [rec("Digital Camera XQ7734B"), rec("Digital Camera XQ7734B")]
    assert one(block, rare, "title_idf_overlap") > one(block, common, "title_idf_overlap")


def test_an_unseen_token_is_treated_as_maximally_rare():
    """Serve-time behaviour: an unrecognized vendor code is the most
    informative thing a new title can carry, so weighting it zero would
    discard exactly the signal that matters most."""
    block = CorpusTextBlock()
    block.fit(CORPUS)
    unseen = [rec("Digital Camera ZZQQ9981"), rec("Digital Camera ZZQQ9981")]
    seen_common = [rec("Digital Camera"), rec("Digital Camera")]
    # Both pairs are identical strings, so both score 1.0 -- what is being
    # checked is that the unseen token does not divide by a zero weight.
    assert one(block, unseen, "title_idf_overlap") == pytest.approx(1.0)
    assert one(block, seen_common, "title_idf_overlap") == pytest.approx(1.0)


def test_cross_description_columns_agree_when_only_one_side_has_text():
    """With one direction available, max and min are the same single value --
    not a spread against a fabricated zero."""
    block = CorpusTextBlock()
    block.fit(CORPUS)
    records = [
        rec("Canon PowerShot SD1100"),
        rec("Canon Camera", description="canon powershot sd1100 compact camera"),
    ]
    assert one(block, records, "cross_title_desc_cosine_max") == pytest.approx(
        one(block, records, "cross_title_desc_cosine_min")
    )
    assert one(block, records, "cross_title_desc_cosine_max") > 0


def test_cross_description_columns_are_imputed_when_neither_side_has_text():
    block = CorpusTextBlock()
    block.fit(CORPUS)
    records = [rec("Canon PowerShot SD1100"), rec("Canon Camera")]
    assert one(block, records, "cross_title_desc_cosine_max") == FILL_VALUE
    assert one(block, records, "cross_title_desc_cosine_min") == FILL_VALUE


def test_cross_description_reduction_produces_no_nan():
    """Regression: reducing an all-absent row with nanmax gave both a warning
    and a NaN, and one NaN poisons the whole feature vector."""
    block = CorpusTextBlock()
    block.fit(CORPUS)
    records = [
        rec("Canon PowerShot SD1100"),
        rec("Canon Camera"),
        rec("Nikon Coolpix", description="a nikon compact camera"),
    ]
    values = block.transform(records, np.array([0, 0, 1]), np.array([1, 2, 2]))
    assert np.isfinite(values).all()


def test_titles_and_descriptions_share_one_fitted_space():
    """The cross columns compare a title against a description, so a cosine
    between two separately fitted spaces would be a coincidence, not a
    similarity. Identical text must therefore score 1.0 across the two."""
    block = CorpusTextBlock()
    block.fit(CORPUS)
    text = "canon powershot sd1100 digital camera"
    records = [rec(text), rec("Unrelated Title", description=text)]
    assert one(block, records, "cross_title_desc_cosine_max") == pytest.approx(1.0, abs=1e-6)
