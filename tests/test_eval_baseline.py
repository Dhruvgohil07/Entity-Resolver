"""Tests for the TF-IDF char-3gram baseline.

Most of these run on synthetic records rather than the committed Abt-Buy
fixtures: the fixtures hold 7 records in 4 entities, which is too few to
split into two sides that both contain a true pair. The fixtures are still
used where real text shape matters -- ranking a genuine duplicate above a
genuine non-duplicate.

The one test that reads data/raw/ is marked and skips when the benchmark has
not been downloaded, matching tests/test_data_abt_buy.py.
"""

import random
import string
from pathlib import Path

import numpy as np
import pytest

from dedup.data.abt_buy import load_abt_buy
from dedup.eval.baseline import (
    comparison_text,
    fit_vectorizer,
    render_markdown,
    run_baseline,
    run_variant,
    score_pairs,
)
from dedup.eval.splits import split_by_entity
from dedup.schema import Record

FIXTURES = Path(__file__).parent / "fixtures" / "abt-buy"
REAL_DATA = Path(__file__).parent.parent / "data" / "raw" / "abt-buy"

# The F1 this baseline scored on the real benchmark when reports/baseline_tfidf.md
# was generated. Held as a band, not an exact value -- sklearn's tokenizer and
# IDF smoothing can shift the last digits across versions, and the point of the
# assertion is that the number stays in the neighbourhood it was published in.
PUBLISHED_TITLE_F1 = (0.45, 0.60)


def make_pairs(n_entities: int, *, seed: int = 0) -> list[Record]:
    """Two listings per product, reworded -- the real failure the baseline faces.

    The vocabulary is generated and sliced disjointly per entity rather than
    hand-written. A first attempt gave every title the same boilerplate
    ("Acme Widget ... Stainless") and only varied a model code, which put a
    floor of ~0.9 cosine under *unrelated* pairs: the baseline scored 0.0 F1
    on data described as easy. Shared boilerplate is a real catalog problem,
    but it belongs in a test that is about boilerplate, not in the end-to-end
    smoke test.
    """
    rng = random.Random(seed)
    words = ["".join(rng.choices(string.ascii_lowercase, k=7)) for _ in range(n_entities * 6)]

    records = []
    for index in range(n_entities):
        product = words[index * 6 : index * 6 + 5]
        # Second listing: same product, words rotated and one dropped.
        rotated = (product[2:] + product[:2])[:4]
        for side, title in enumerate((" ".join(product), " ".join(rotated))):
            records.append(
                Record(
                    record_id=f"synthetic:{side}:{index:04d}",
                    source="synthetic",
                    entity_id=f"synthetic:e{index:04d}",
                    title=title,
                )
            )
    return records


@pytest.fixture
def fixture_records() -> list[Record]:
    return load_abt_buy(FIXTURES)


# ---------------------------------------------------------------------------
# The text being compared
# ---------------------------------------------------------------------------


def test_comparison_text_is_the_normalized_text_not_the_raw_title():
    # Train/serve parity: the baseline must score the same canonical string
    # every later stage sees, or it is measuring a different system.
    record = Record(
        record_id="synthetic:1",
        source="synthetic",
        title="Sharp 32' LCD TV",
        description="A Great TV",
    )
    assert comparison_text(record, include_description=False) == "sharp 32 in lcd tv"


def test_description_is_appended_only_when_asked_for():
    record = Record(
        record_id="synthetic:1", source="synthetic", title="Sharp TV", description="Widescreen"
    )
    assert comparison_text(record, include_description=True) == "sharp tv widescreen"
    assert comparison_text(record, include_description=False) == "sharp tv"


def test_an_empty_description_does_not_leave_trailing_whitespace():
    # Abt-Buy has blank descriptions ("" -- column present, value empty), and
    # a trailing space would change the char_wb n-grams for those records only.
    record = Record(record_id="synthetic:1", source="synthetic", title="Sharp TV", description="")
    assert comparison_text(record, include_description=True) == "sharp tv"


# ---------------------------------------------------------------------------
# Pair enumeration
# ---------------------------------------------------------------------------


def test_every_unordered_pair_is_scored_exactly_once(fixture_records):
    vectorizer = fit_vectorizer(fixture_records, include_description=False)
    # -1.0 keeps even the zero-similarity pairs, so this counts the full
    # upper triangle rather than whatever survived the floor.
    scored = score_pairs(
        fixture_records, vectorizer, include_description=False, min_similarity=-1.0
    )

    n = len(fixture_records)
    assert scored.score.size == n * (n - 1) // 2 == scored.n_pairs_total
    assert bool((scored.left < scored.right).all()), "pairs must be upper-triangle only"
    assert len({(int(a), int(b)) for a, b in zip(scored.left, scored.right)}) == scored.score.size


def test_chunking_does_not_change_the_result(fixture_records):
    vectorizer = fit_vectorizer(fixture_records, include_description=False)
    whole = score_pairs(fixture_records, vectorizer, include_description=False, chunk_size=1000)
    chunked = score_pairs(fixture_records, vectorizer, include_description=False, chunk_size=2)

    assert np.array_equal(whole.left, chunked.left)
    assert np.array_equal(whole.right, chunked.right)
    assert whole.score == pytest.approx(chunked.score)


def test_labels_follow_entity_id_including_the_transitive_same_side_pair(fixture_records):
    # The fixture's size-3 entity is one Abt record and two Buy records, so
    # one of its three true pairs is Buy-vs-Buy. A cross-source-only framing
    # would miss it; this baseline must not.
    vectorizer = fit_vectorizer(fixture_records, include_description=False)
    scored = score_pairs(
        fixture_records, vectorizer, include_description=False, min_similarity=-1.0
    )

    ids = [record.record_id for record in fixture_records]
    matched = {
        frozenset((ids[left], ids[right]))
        for left, right, label in zip(scored.left, scored.right, scored.label)
        if label
    }
    assert frozenset(("abt_buy:buy:98", "abt_buy:buy:99")) in matched
    assert scored.n_positives_total == len(matched) == 4


def test_a_record_without_an_entity_id_cannot_be_scored():
    records = make_pairs(2)
    records.append(Record(record_id="synthetic:new", source="synthetic", title="unlabeled thing"))
    vectorizer = fit_vectorizer(records[:4], include_description=False)
    with pytest.raises(ValueError, match="entity_id"):
        score_pairs(records, vectorizer, include_description=False)


# ---------------------------------------------------------------------------
# Ranking behaviour
# ---------------------------------------------------------------------------


def test_a_reworded_duplicate_outranks_an_unrelated_product(fixture_records):
    vectorizer = fit_vectorizer(fixture_records, include_description=False)
    scored = score_pairs(
        fixture_records, vectorizer, include_description=False, min_similarity=-1.0
    )
    ids = [record.record_id for record in fixture_records]
    by_pair = {
        frozenset((ids[left], ids[right])): float(score)
        for left, right, score in zip(scored.left, scored.right, scored.score)
    }

    # "Sony Turntable - PSLX350H" vs "Sony PSLX350H Turntable Belt Drive" --
    # same product, reordered. Against an unrelated camera listing.
    duplicate = by_pair[frozenset(("abt_buy:abt:10", "abt_buy:buy:98"))]
    unrelated = by_pair[frozenset(("abt_buy:abt:10", "abt_buy:buy:10"))]
    assert duplicate > unrelated


# ---------------------------------------------------------------------------
# The similarity floor
# ---------------------------------------------------------------------------


def test_the_similarity_floor_prunes_candidates_without_hiding_the_positives():
    # The floor is the memory knob. Raising it must shrink the candidate set
    # and lower the reachable recall -- never quietly shrink the denominator,
    # which would make pruning look free.
    records = make_pairs(12)
    vectorizer = fit_vectorizer(records, include_description=False)

    everything = score_pairs(records, vectorizer, include_description=False, min_similarity=-1.0)
    floored = score_pairs(records, vectorizer, include_description=False, min_similarity=0.99)

    assert floored.score.size < everything.score.size
    assert floored.n_positives_total == everything.n_positives_total
    assert floored.recall_ceiling < 1.0
    assert everything.recall_ceiling == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_run_baseline_ranks_duplicates_above_every_non_duplicate():
    records = make_pairs(60)
    report = run_baseline(records, dataset="synthetic", seed=0)
    variant = report.variants[0]

    assert report.n_records == 120
    assert report.n_entities == 60
    assert report.n_true_pairs == 60
    assert report.n_train_records + report.n_test_records == 120
    assert [v.name for v in report.variants] == ["title", "title + description"]

    # These duplicates are separable by construction, so the *ranking* is
    # perfect: every true pair outscores every false one.
    assert variant.test_pr_auc == pytest.approx(1.0)
    assert variant.test_r_precision == pytest.approx(1.0)
    assert variant.test_oracle.f1 == pytest.approx(1.0)


def test_argmax_f1_threshold_selection_is_fragile_on_separable_data():
    # Pins the gap between a perfect ranking and what one global threshold
    # delivers. With train perfectly separable, every cut between the lowest
    # train positive and the highest train negative scores F1 = 1.0, and
    # argmax-F1 takes the highest of them -- sitting exactly on the lowest
    # train positive. Test positives below that line are simply lost, so
    # precision stays at 1.0 while recall falls.
    #
    # This is not a defect in the baseline; it is the reason CLAUDE.md's
    # invariant says real thresholds come from expected cost rather than
    # argmax F1. Pinned so that replacing it can be shown to help.
    variant = run_baseline(make_pairs(60), dataset="synthetic", seed=0).variants[0]

    assert variant.test_point.precision == pytest.approx(1.0)
    assert variant.test_point.recall < 1.0
    assert variant.test_point.f1 < variant.test_oracle.f1


def test_the_reported_threshold_is_the_one_chosen_on_train():
    records = make_pairs(60)
    train, test = split_by_entity(records, test_fraction=0.3, seed=0)
    variant = run_variant(train, test, name="title", include_description=False)

    # The honest number can only be at or below the test-set oracle; if it
    # ever exceeded it, the threshold had seen the test labels.
    assert variant.test_point.threshold == variant.train_point.threshold
    assert variant.test_point.f1 <= variant.test_oracle.f1 + 1e-12


def test_render_markdown_states_the_protocol_and_the_caveats():
    report = run_baseline(make_pairs(60), dataset="synthetic", seed=0)
    markdown = render_markdown(report)

    # The report is the deliverable, so the parts that stop it being
    # misread are asserted rather than left to survive an edit by luck.
    assert "PR-AUC, not ROC-AUC" in markdown
    assert "Oracle F1" in markdown
    assert "fit on the train split" in markdown
    assert "| title |" in markdown


# ---------------------------------------------------------------------------
# Against the real benchmark (skipped unless it has been downloaded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Abt-Buy not downloaded; see CLAUDE.md > Data")
def test_the_published_baseline_number_still_reproduces():
    # Pins reports/baseline_tfidf.md. Every later stage is justified against
    # this figure, so it silently drifting is a problem in itself.
    records = load_abt_buy(REAL_DATA)
    train, test = split_by_entity(records, test_fraction=0.3, seed=0)
    variant = run_variant(train, test, name="title", include_description=False)

    low, high = PUBLISHED_TITLE_F1
    assert low < variant.test_point.f1 < high
    assert variant.recall_ceiling == pytest.approx(1.0)
