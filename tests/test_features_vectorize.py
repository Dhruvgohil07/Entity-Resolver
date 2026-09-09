"""Contract tests for the feature registry and the assembled matrix.

The important ones here are about CLAUDE.md's missingness invariant (I5), and
they are written to walk the *live* registry rather than a hand-listed set of
columns. A feature added to any block next month inherits these checks by
being registered, which is the same trick `test_blocking_blockers.py` uses to
make every blocker inherit the pair-shape checks by parametrization.

`test_imputed_columns_hold_exactly_the_fill_value` is the one that would
actually catch the bug the invariant exists for: a column that quietly writes
a computed-looking number where it has no data, so the model learns from a
value it cannot distinguish from a real one.
"""

import numpy as np
import pytest

from dedup.blocking.pairs import pack
from dedup.features import FILL_VALUE, FeatureMatrix, FeatureSpec, PairFeaturizer, pair_labels
from dedup.features.vectorize import validate_registry
from dedup.normalize import normalize
from dedup.schema import Record


def rec(record_id, title, entity_id="e1", **kwargs):
    return normalize(
        Record(record_id=record_id, source="synthetic", entity_id=entity_id, title=title, **kwargs)
    )


CATALOG = [
    rec("s:1", "Panasonic 2-Line Telephone System - KXTS208W", "e1", price=99.99),
    rec("s:2", "Panasonic KX-TS208W Corded Phone", "e1", description="two line corded phone"),
    rec("s:3", "Canon PowerShot SD1100 Digital Camera", "e2", price=199.0, brand="Canon"),
    rec("s:4", "Canon SD1100 IS PowerShot Camera Silver", "e2", brand="Canon"),
    rec("s:5", "Cuisinart Food Processor DLC2009CHB", "e3"),
]

ALL_PAIRS = pack(
    np.array([0, 0, 1, 2, 3], dtype=np.int64),
    np.array([1, 2, 2, 3, 4], dtype=np.int64),
    len(CATALOG),
)


def fitted():
    return PairFeaturizer().fit(CATALOG)


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------


def test_every_imputed_column_names_a_real_indicator():
    """I5, checked against the shipped registry rather than a copy of it."""
    featurizer = PairFeaturizer()
    indicators = {spec.name for spec in featurizer.specs if spec.is_indicator}
    for spec in featurizer.specs:
        if spec.imputed:
            assert spec.companion_indicator in indicators, (
                f"{spec.name} imputes but its companion {spec.companion_indicator!r} "
                f"is not a registered indicator"
            )


def test_registry_has_no_duplicate_names():
    names = PairFeaturizer().names
    assert len(names) == len(set(names))


def test_column_count_is_in_the_documented_range():
    """CLAUDE.md promises a ~25-35 dim vector; a silent drift out of that band
    means a block was dropped or duplicated."""
    assert 25 <= len(PairFeaturizer().names) <= 35


def test_semantic_block_appends_without_renumbering():
    """Turning the optional column on must not move any existing column.

    Column order is part of the trained-model contract: a model that learned
    column 7 as `title_common_prefix_ratio` mispredicts silently if column 7
    later means something else.
    """
    plain = PairFeaturizer().names
    with_semantic = PairFeaturizer(include_semantic=True).names
    assert with_semantic[: len(plain)] == plain
    assert with_semantic[len(plain) :] == ["title_embedding_cosine"]


def test_validate_registry_rejects_imputed_without_companion():
    with pytest.raises(ValueError, match="names no companion_indicator"):
        validate_registry([FeatureSpec("a", "doc", imputed=True)])


def test_validate_registry_rejects_companion_that_is_not_an_indicator():
    with pytest.raises(ValueError, match="not a registered indicator"):
        validate_registry(
            [
                FeatureSpec("a", "doc", imputed=True, companion_indicator="b"),
                FeatureSpec("b", "doc"),  # a plain column, not is_indicator
            ]
        )


def test_validate_registry_rejects_an_imputed_indicator():
    with pytest.raises(ValueError, match="cannot itself be imputed"):
        validate_registry([FeatureSpec("a", "doc", is_indicator=True, imputed=True)])


def test_validate_registry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate feature names"):
        validate_registry([FeatureSpec("a", "doc"), FeatureSpec("a", "doc")])


# ---------------------------------------------------------------------------
# fit / transform
# ---------------------------------------------------------------------------


def test_transform_before_fit_raises():
    """The fitted IDF weights are part of the feature definition, so an
    unfitted transform is wrong rather than merely unoptimized."""
    with pytest.raises(RuntimeError, match="before fit"):
        PairFeaturizer().transform(CATALOG, ALL_PAIRS)


def test_matrix_shape_and_names_agree_with_the_registry():
    matrix = fitted().transform(CATALOG, ALL_PAIRS)
    assert matrix.values.shape == (ALL_PAIRS.size, len(matrix.specs))
    assert matrix.names == PairFeaturizer().names


def test_matrix_rejects_a_width_that_was_not_declared():
    with pytest.raises(ValueError, match="a block returned a width it did not declare"):
        FeatureMatrix(values=np.zeros((2, 3)), specs=(FeatureSpec("a", "doc"),))


def test_column_lookup_is_by_name():
    matrix = fitted().transform(CATALOG, ALL_PAIRS)
    assert matrix.column("title_ratio").shape == (ALL_PAIRS.size,)
    with pytest.raises(KeyError, match="no feature named"):
        matrix.column("not_a_feature")


def test_every_value_is_finite():
    """A NaN anywhere propagates through the entire vector with no error to
    trace it back to -- which is why schema.py rejects NaN prices upstream and
    why the cross-description columns reduce only over rows that have a value."""
    matrix = fitted().transform(CATALOG, ALL_PAIRS)
    assert np.isfinite(matrix.values).all()


def test_imputed_columns_hold_exactly_the_fill_value():
    """The invariant that matters: where the indicator says 0, the column must
    be the fill and nothing else.

    If a column ever writes a computed-looking number on a pair it has no data
    for, the indicator becomes a lie and the model learns from the fill.
    """
    matrix = fitted().transform(CATALOG, ALL_PAIRS)
    for spec in matrix.specs:
        if not spec.imputed:
            continue
        uncovered = matrix.column(spec.companion_indicator) == 0.0
        assert np.all(matrix.column(spec.name)[uncovered] == FILL_VALUE), (
            f"{spec.name} holds a non-fill value on a pair where "
            f"{spec.companion_indicator} is 0"
        )


def test_empty_pair_set_gives_an_empty_matrix():
    matrix = fitted().transform(CATALOG, np.empty(0, dtype=np.int64))
    assert matrix.values.shape == (0, len(matrix.specs))


def test_transform_is_reproducible():
    featurizer = fitted()
    first = featurizer.transform(CATALOG, ALL_PAIRS)
    second = featurizer.transform(CATALOG, ALL_PAIRS)
    np.testing.assert_array_equal(first.values, second.values)


def test_fitting_on_one_set_and_transforming_another_works():
    """The serve-time shape: weights fitted on train, spent on records the
    featurizer has never seen."""
    featurizer = PairFeaturizer().fit(CATALOG[:3])
    unseen = [rec("s:9", "Sony WH-1000XM4 Headphones", "e9"), rec("s:10", "Sony WH1000XM4", "e9")]
    matrix = featurizer.transform(unseen, pack(np.array([0]), np.array([1]), 2))
    assert matrix.column("model_number_exact")[0] == 1.0


def test_pair_labels_follow_entity_id():
    labels = pair_labels(CATALOG, ALL_PAIRS)
    # ALL_PAIRS is (0,1) (0,2) (1,2) (2,3) (3,4): same entity only for
    # (0,1) -> e1 and (2,3) -> e2.
    np.testing.assert_array_equal(labels, [True, False, False, True, False])
