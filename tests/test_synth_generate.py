"""Tests for catalog assembly: ground truth, reproducibility, and the leakage groups.

A synthetic catalog is only worth evaluating on if its labels are right and it
regenerates exactly, so those are asserted structurally here rather than inferred
from a report that happens to look plausible.
"""

import pytest

from dedup.blocking.union import ground_truth
from dedup.eval.splits import count_true_pairs, kfold_by_entity, split_by_entity
from dedup.normalize import code_key, normalize
from dedup.schema import Record
from dedup.synth.generate import SynthConfig, family_id, generate, verify_seed_provenance

SEEDS = [
    ("seed:e01", ["Panasonic 2-Line White Phone - KXTS208W", "Panasonic KX-TS208W Corded Phone"]),
    ("seed:e02", ['Samsung 32" LCD TV - LN32A330', "Samsung LN32A330 LCD HDTV"]),
    ("seed:e03", ["Canon PowerShot Digital Camera - SD1100", "Canon PowerShot SD1100 Camera"]),
    ("seed:e04", ["Weber Genesis Gas Grill - GNS320LP"]),
    ("seed:e05", ["Garmin Nuvi GPS Navigator - NUVI255W", "Garmin nuvi 255W Navigator"]),
    ("seed:e06", ['Sharp 32" Widescreen TV']),  # no vendor code at all
]

CONFIG = SynthConfig(siblings_per_family=3.0, seed=0)


def make_seeds(spec=SEEDS):
    return [
        Record(
            record_id=f"{entity}:r{index}",
            source="synthetic",
            entity_id=entity,
            title=title,
            brand=title.split()[0],
            description=f"{title} with accessories",
            price=100.0 + index,
        )
        for entity, titles in spec
        for index, title in enumerate(titles)
    ]


def as_json(records):
    return [record.model_dump_json() for record in records]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_byte_identical_records():
    assert as_json(generate(make_seeds(), CONFIG).records) == as_json(
        generate(make_seeds(), CONFIG).records
    )


def test_the_order_seeds_arrive_in_does_not_matter():
    assert as_json(generate(make_seeds(), CONFIG).records) == as_json(
        generate(list(reversed(make_seeds())), CONFIG).records
    )


def test_a_different_seed_gives_a_different_catalog():
    assert as_json(generate(make_seeds(), CONFIG).records) != as_json(
        generate(make_seeds(), SynthConfig(siblings_per_family=3.0, seed=1)).records
    )


def test_removing_one_family_leaves_every_other_family_unchanged():
    """Families draw from their own generators, so nothing else moves.

    These seeds' codes are too far apart for any sibling to be diverted around
    another family's, which is the one case the generator documents as an
    exception.
    """
    full = generate(make_seeds(), CONFIG)
    without = generate([r for r in make_seeds() if r.entity_id != "seed:e03"], CONFIG)
    removed = family_id("seed:e03")

    def signature(entities):
        return [(entity.entity_id, as_json(entity.records)) for entity in entities]

    assert signature(e for e in full.entities if e.family != removed) == signature(without.entities)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_every_record_is_a_labelled_synthetic_record():
    catalog = generate(make_seeds(), CONFIG)
    ids = [record.record_id for record in catalog.records]
    assert len(ids) == len(set(ids))
    for entity in catalog.entities:
        assert entity.records
        for record in entity.records:
            assert Record.model_validate(record.model_dump()) == record
            assert record.source == "synthetic"
            assert record.entity_id == entity.entity_id
            assert record.split_group == f"synthetic:{entity.family}"
            assert record.raw_attributes["kind"] == entity.product.kind


def test_ground_truth_enumeration_agrees_with_the_pair_count():
    records = generate(make_seeds(), CONFIG).records
    truth, n_true = ground_truth([normalize(record) for record in records])
    assert truth.size == n_true == count_true_pairs(records)


def test_no_two_entities_share_a_vendor_code():
    catalog = generate(make_seeds(), SynthConfig(siblings_per_family=8.0))
    keys = [code_key(entity.product.code) for entity in catalog.entities if entity.product.code]
    assert len(keys) == len(set(keys))


def test_a_family_never_straddles_a_split_or_a_fold():
    records = generate(make_seeds(), CONFIG).records
    train, test = split_by_entity(records, test_fraction=0.3, seed=0)
    assert {r.split_group for r in train}.isdisjoint({r.split_group for r in test})

    groups = [{r.split_group for r in fold} for fold in kfold_by_entity(records, n_folds=3)]
    for index, group in enumerate(groups):
        for other in groups[index + 1 :]:
            assert group.isdisjoint(other)


# ---------------------------------------------------------------------------
# Seed provenance
# ---------------------------------------------------------------------------
#
# data/synthetic.py's loader only compares the manifest's recorded seed_split
# numbers against what a report expects -- it never re-derives the split from the
# seed dataset's own records, so a `split_by_entity` change could move an entity
# from train to test while those two numbers stay the same. verify_seed_provenance
# closes that by recomputing the split fresh and checking real entity membership.


def test_verify_seed_provenance_accepts_a_catalog_seeded_only_from_train():
    seeds = make_seeds()
    train, test = split_by_entity(seeds, test_fraction=0.3, seed=0)
    assert test  # the split must actually hold something out, or this proves nothing
    catalog = generate(train, CONFIG)
    verify_seed_provenance(catalog.records, seeds, test_fraction=0.3, seed=0)  # must not raise


def test_verify_seed_provenance_refuses_a_test_side_entity():
    seeds = make_seeds()
    train, test = split_by_entity(seeds, test_fraction=0.3, seed=0)
    assert test
    catalog = generate(train, CONFIG)
    leaked_entity_id = test[0].entity_id
    tampered = catalog.records[0].model_copy(
        update={
            "raw_attributes": {**catalog.records[0].raw_attributes, "seed_entity_id": leaked_entity_id}
        }
    )
    records = [tampered, *catalog.records[1:]]
    with pytest.raises(ValueError, match="not on the train side"):
        verify_seed_provenance(records, seeds, test_fraction=0.3, seed=0)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_the_target_record_count_is_approximately_reached():
    coded = make_seeds(SEEDS[:5])
    catalog = generate(coded, SynthConfig(target_records=600, seed=0))
    assert 0.8 * 600 <= len(catalog.records) <= 1.2 * 600


def test_entities_larger_than_three_are_generated():
    """The p = 0 under-merge question needs them, and Abt-Buy has none."""
    catalog = generate(make_seeds(), SynthConfig(siblings_per_family=10.0, seed=1))
    assert max(len(entity.records) for entity in catalog.entities) >= 4


def test_siblings_of_both_kinds_are_derived():
    catalog = generate(make_seeds(), SynthConfig(siblings_per_family=6.0))
    assert {entity.product.kind for entity in catalog.entities} == {"root", "near", "far"}


def test_the_catalog_is_sized_one_way_or_the_other():
    with pytest.raises(ValueError, match="exactly one"):
        SynthConfig()
    with pytest.raises(ValueError, match="exactly one"):
        SynthConfig(target_records=10, siblings_per_family=1.0)
    with pytest.raises(ValueError, match="sum to 1"):
        SynthConfig(target_records=10, entity_sizes=((1, 0.5),))
