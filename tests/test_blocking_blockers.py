"""Per-blocker contract tests.

Every blocker, however it works internally, owes the same guarantees: unique
upper-triangle pairs, no self-pairs, and the same answer twice for the same
input. Those are asserted for all of them by parametrization, so a blocker
added later inherits the checks by being listed.

The behaviour tests below are about *which* pairs each family catches, chosen
to pin the reason each blocker exists rather than its exact score.
"""

import numpy as np
import pytest

from dedup.blocking.ann import AnnBlocker
from dedup.blocking.lsh import MinHashLSHBlocker
from dedup.blocking.pairs import unpack
from dedup.blocking.sorted_neighborhood import SortedNeighborhoodBlocker
from dedup.blocking.standard import (
    StandardBlocker,
    code_token_keys,
    model_number_keys,
    rare_token_keys,
)
from dedup.normalize import normalize
from dedup.schema import Record


def rec(record_id: str, title: str, entity_id: str = "synthetic:e1"):
    return normalize(
        Record(record_id=record_id, source="synthetic", entity_id=entity_id, title=title)
    )


# A catalog with enough variety that every blocker family has something to
# find: shared vendor codes, a punctuation variant, a brand typo, and rows
# that share nothing at all.
CATALOG = [
    rec("s:1", "Panasonic 2-Line Integrated Telephone System - KXTS208W", "e1"),
    rec("s:2", "Panasonic KX-TS208W Corded Phone", "e1"),
    rec("s:3", "Bose 161 Bookshelf Pair Speakers In White - 161WH", "e2"),
    rec("s:4", "Boss 161 Speaker", "e2"),
    rec("s:5", "Canon PowerShot SD1100 Digital Camera", "e3"),
    rec("s:6", "Canon SD1100 IS PowerShot Camera Silver", "e3"),
    rec("s:7", "Cuisinart Food Processor DLC2009CHB", "e4"),
    rec("s:8", "Weber Genesis Gas Grill 3841001", "e5"),
]


def pair_ids(records, keys) -> set[frozenset[str]]:
    left, right = unpack(keys, len(records))
    return {
        frozenset((records[int(a)].raw.record_id, records[int(b)].raw.record_id))
        for a, b in zip(left, right)
    }


ALL_BLOCKERS = [
    StandardBlocker("model", model_number_keys),
    StandardBlocker("codes", code_token_keys),
    StandardBlocker("rare", rare_token_keys(30)),
    SortedNeighborhoodBlocker(window=4),
    MinHashLSHBlocker(threshold=0.3, shingles="token"),
    MinHashLSHBlocker(threshold=0.3, shingles="char"),
    AnnBlocker(neighbours=3),
]


# ---------------------------------------------------------------------------
# The contract every blocker owes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocker", ALL_BLOCKERS, ids=lambda b: f"{b.name}:{b.params}")
def test_every_blocker_emits_unique_upper_triangle_pairs(blocker):
    keys = blocker.run(CATALOG).keys
    left, right = unpack(keys, len(CATALOG))

    assert bool((left < right).all()), "pairs must be upper-triangle, no self-pairs"
    assert len(set(keys.tolist())) == keys.size, "pairs must be unique"
    assert keys.tolist() == sorted(keys.tolist()), "keys must be sorted"


@pytest.mark.parametrize("blocker", ALL_BLOCKERS, ids=lambda b: f"{b.name}:{b.params}")
def test_every_blocker_is_deterministic(blocker):
    # A report that changes between runs of the same command is not a result.
    first = blocker.run(CATALOG).keys
    second = blocker.run(CATALOG).keys
    assert np.array_equal(first, second)


@pytest.mark.parametrize("blocker", ALL_BLOCKERS, ids=lambda b: f"{b.name}:{b.params}")
def test_every_blocker_reports_its_params(blocker):
    # The report reproduces this string verbatim; a row nobody can tie back to
    # a configuration is not reproducible.
    run = blocker.run(CATALOG)
    assert run.params and run.name


# ---------------------------------------------------------------------------
# Why each family exists
# ---------------------------------------------------------------------------


def test_model_number_blocking_survives_punctuation():
    # KXTS208W vs KX-TS208W -- the same Panasonic phone in two house styles.
    # This pair is the entire reason normalize grew `model_number_key`, and an
    # exact blocker on `model_number` does not catch it.
    found = pair_ids(CATALOG, StandardBlocker("model", model_number_keys).run(CATALOG).keys)
    assert frozenset(("s:1", "s:2")) in found


def test_code_token_blocking_catches_a_code_the_extractor_did_not_choose():
    # Both Canon rows contain SD1100, but extraction commits to one token per
    # title and may pick a different one. Indexing every code-shaped token
    # recovers the pair.
    found = pair_ids(CATALOG, StandardBlocker("codes", code_token_keys).run(CATALOG).keys)
    assert frozenset(("s:5", "s:6")) in found


def test_ann_catches_a_pair_with_a_brand_typo_that_no_key_can_match():
    # "Bose" vs "Boss" share no usable key -- different brand string, no code
    # on the Buy side. Only fuzzy similarity finds this one, which is why the
    # union needs a vector blocker and not just better keys.
    found = pair_ids(CATALOG, AnnBlocker(neighbours=3).run(CATALOG).keys)
    assert frozenset(("s:3", "s:4")) in found


def test_a_record_sharing_nothing_is_not_paired_by_key_blockers():
    # The Weber grill shares no code and no rare token with anything.
    found = pair_ids(CATALOG, StandardBlocker("codes", code_token_keys).run(CATALOG).keys)
    assert not any("s:8" in pair for pair in found)


# ---------------------------------------------------------------------------
# The runaway-block guard
# ---------------------------------------------------------------------------


def test_a_block_larger_than_the_cap_is_dropped_not_expanded():
    # One key shared by many records is quadratic and non-discriminating: a
    # 5,000-record block is 12.5M pairs on its own and separates nothing.
    shared = [rec(f"s:{i}", f"widget xy123z item {i}", f"e{i}") for i in range(12)]
    blocker = StandardBlocker("codes", code_token_keys, max_block_size=5)
    run = blocker.run(shared)

    assert run.n_candidates == 0
    assert blocker.dropped_blocks == 1

    # A dropped block removes true pairs with no other trace, so it has to
    # reach the report. An invisible cap is a silently lowered ceiling --
    # the same class of problem as a hidden recall denominator.
    assert run.warnings, "dropping a block must be surfaced, not silent"
    assert "max_block_size" in run.warnings[0]

    # The same block under a larger cap is kept, so the cap is what changed.
    kept = StandardBlocker("codes", code_token_keys, max_block_size=50).run(shared)
    assert kept.n_candidates > 0
    assert kept.warnings == ()


def test_max_block_size_below_two_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        StandardBlocker("codes", code_token_keys, max_block_size=1)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_rare_token_cutoff_controls_how_many_pairs_survive():
    # The cutoff is the whole design of this blocker: raising it admits more
    # common tokens, which means more candidates.
    tight = StandardBlocker("rare", rare_token_keys(1)).run(CATALOG).n_candidates
    loose = StandardBlocker("rare", rare_token_keys(30)).run(CATALOG).n_candidates
    assert loose >= tight


def test_sorted_neighborhood_window_must_be_at_least_two():
    with pytest.raises(ValueError, match="window must be at least 2"):
        SortedNeighborhoodBlocker(window=1)


def test_lsh_rejects_an_unknown_shingle_mode():
    with pytest.raises(ValueError, match="'token' or 'char'"):
        MinHashLSHBlocker(shingles="ngram")


def test_lsh_rejects_a_threshold_outside_the_unit_interval():
    with pytest.raises(ValueError, match="threshold"):
        MinHashLSHBlocker(threshold=1.5)


def test_ann_is_deterministic_on_a_catalog_large_enough_to_expose_it():
    # The parametrized determinism test above runs on 8 records, which is far
    # too small to build a multi-level HNSW graph and so never catches this.
    # At real sizes faiss parallelizes construction and the order threads link
    # nodes changes the graph: on Abt-Buy, three runs of identical input gave
    # 14,608 / 14,603 / 14,602 candidates before the build was pinned to one
    # thread. Completeness barely moved, but a committed report whose numbers
    # drift between runs is not reproducible.
    catalog = [
        rec(f"s:{i}", f"acme widget model xq{i:04d}zp stainless finish", f"e{i}")
        for i in range(400)
    ]
    blocker = AnnBlocker(neighbours=5)
    counts = {blocker.run(catalog).n_candidates for _ in range(3)}
    assert len(counts) == 1, f"ann candidate count drifted across runs: {sorted(counts)}"


def test_ann_restores_the_thread_count_it_borrowed():
    # Pinning threads is process-global, so leaving it pinned would silently
    # single-thread every later faiss call in the same process.
    import faiss

    before = faiss.omp_get_max_threads()
    AnnBlocker(neighbours=2).run(CATALOG)
    assert faiss.omp_get_max_threads() == before


def test_ann_asking_for_more_neighbours_than_records_is_clamped():
    # k+1 exceeds N on a tiny catalog; faiss must not be asked for more
    # neighbours than it holds.
    run = AnnBlocker(neighbours=50).run(CATALOG[:3])
    assert run.n_candidates <= 3
