"""Assemble a synthetic catalog: families of distinct products, each listed several times.

`python -m dedup.synth.generate --seed-dataset abt-buy --records 20000 --out data/synth/abt-buy-train-20k`

The input is `Record`s with `entity_id` -- a seed catalog, dataset-agnostic -- and
the output is `Record`s with `source="synthetic"`, a ground-truth `entity_id`, and
a `split_group` naming the seed family every record was derived from. That last
field is what keeps a family on one side of every split and fold
(`eval/splits.py`): siblings built from one seed title on both sides of a split
would leak that title as surely as a shared entity would.

Three properties the generator guarantees, and `tests/test_synth_generate.py` pins:

  * **Reproducible.** Output depends only on the seed records and the config --
    never on the order records arrive in, and never on Python's per-process
    `hash()`. Each family draws from its own generator, seeded by a SHA-256 of
    the config seed and the seed entity's id.
  * **Local.** Because families draw independently, removing a seed changes no
    other family's output -- unless a sibling of another family had been diverted
    around one of its codes, which the catalog-wide reservations resolve in seed
    order. That is rare, and the alternative is two entities sharing a code.
  * **Labelled.** Every record carries `entity_id` and `split_group`, and its
    `raw_attributes` record the seed it came from and the operators that made it.

The CLI seeds only from the **train** side of the seed dataset's entity-grouped
split, at the same `test_fraction` and `seed` every report on that dataset uses,
so no record of its test split ever reaches a synthetic one.

The entity-size distribution is a judgment call, not a measurement: Abt-Buy has
only pairs and triples, and deciding whether pricing unemitted pairs at p = 0
under-merges larger entities needs entities larger than three.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from dedup.data import DATASETS, load_dataset
from dedup.data.synthetic import read_manifest, write_catalog
from dedup.eval.splits import (
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    group_by_entity,
    split_by_entity,
)
from dedup.schema import Record
from dedup.synth.corrupt import CorruptionProfile, Listing, corrupt, hide_code
from dedup.synth.families import Product, Reservations, alternate_code, base_product, derive_family
from dedup.synth.realism import pair_statistics, render_realism

SOURCE = "synthetic"

DEFAULT_ENTITY_SIZES: tuple[tuple[int, float], ...] = (
    (1, 0.40),
    (2, 0.30),
    (3, 0.15),
    (4, 0.06),
    (5, 0.04),
    (6, 0.03),
    (7, 0.01),
    (8, 0.01),
)


@dataclass(frozen=True)
class SynthConfig:
    """How large a catalog to derive, and how its entities are shaped.

    Size it with exactly one of `target_records` -- the catalog a CLI asks for --
    or `siblings_per_family`, which does not depend on how many seeds there are
    and so keeps families independent of each other.
    """

    target_records: int | None = None
    siblings_per_family: float | None = None
    seed: int = 0
    entity_sizes: tuple[tuple[int, float], ...] = DEFAULT_ENTITY_SIZES
    near_sibling_share: float = 0.5
    corruption: CorruptionProfile = field(default_factory=CorruptionProfile)

    def __post_init__(self) -> None:
        if (self.target_records is None) == (self.siblings_per_family is None):
            raise ValueError(
                "size the catalog with exactly one of target_records or siblings_per_family"
            )
        if self.target_records is not None and self.target_records < 1:
            raise ValueError(f"target_records must be positive, got {self.target_records}")
        if self.siblings_per_family is not None and self.siblings_per_family < 0:
            raise ValueError(
                f"siblings_per_family must be non-negative, got {self.siblings_per_family}"
            )
        if not self.entity_sizes or any(size < 1 or weight < 0 for size, weight in self.entity_sizes):
            raise ValueError("entity_sizes must be (size >= 1, weight >= 0) pairs")
        total = sum(weight for _, weight in self.entity_sizes)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"entity_sizes weights must sum to 1, got {total}")
        if not 0.0 <= self.near_sibling_share <= 1.0:
            raise ValueError(f"near_sibling_share must lie in [0, 1], got {self.near_sibling_share}")

    @property
    def mean_entity_size(self) -> float:
        return sum(size * weight for size, weight in self.entity_sizes)

    def siblings_for(self, n_families: int) -> float:
        """Mean siblings per family, derived from the target when one was given."""
        if self.siblings_per_family is not None:
            return self.siblings_per_family
        assert self.target_records is not None
        return max(self.target_records / self.mean_entity_size / n_families - 1.0, 0.0)


@dataclass(frozen=True)
class SynthEntity:
    """One distinct product and its listings."""

    entity_id: str
    family: str
    product: Product
    records: tuple[Record, ...]


@dataclass(frozen=True)
class SynthCatalog:
    config: SynthConfig
    entities: tuple[SynthEntity, ...]
    n_families: int
    n_dropped_siblings: int  # requested siblings that found no free code or title

    @property
    def records(self) -> list[Record]:
        return [record for entity in self.entities for record in entity.records]


def family_id(seed_entity_id: str) -> str:
    """A stable id for the family derived from one seed entity."""
    return "f" + hashlib.sha256(seed_entity_id.encode("utf-8")).hexdigest()[:12]


def family_rng(seed: int, seed_entity_id: str) -> np.random.Generator:
    """The family's own generator: a function of the config seed and the seed entity only."""
    digest = hashlib.sha256(f"{seed}:{seed_entity_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _round_stochastically(value: float, rng: np.random.Generator) -> int:
    whole = int(np.floor(value))
    return whole + int(rng.random() < value - whole)


def _record(
    product: Product,
    listing: Listing,
    *,
    entity_id: str,
    family: str,
    seed_entity_id: str,
    n: int,
    code_hidden: bool,
) -> Record:
    return Record(
        record_id=f"{entity_id}:r{n}",
        source=SOURCE,
        entity_id=entity_id,
        split_group=f"{SOURCE}:{family}",
        title=listing.title,
        description=listing.description,
        brand=listing.brand,
        price=listing.price,
        raw_attributes={
            "seed_entity_id": seed_entity_id,
            "seed_record_id": product.seed_record_id,
            "kind": product.kind,
            "operators": ["code_hidden", *listing.operators] if code_hidden else list(listing.operators),
        },
    )


def generate(seeds: Sequence[Record], config: SynthConfig) -> SynthCatalog:
    """Derive a labelled synthetic catalog from seed records carrying `entity_id`."""
    grouped = group_by_entity(list(seeds))
    if not grouped:
        raise ValueError("no seed records to derive a catalog from")
    seed_ids = sorted(grouped)
    roots = {seed_id: base_product(grouped[seed_id]) for seed_id in seed_ids}

    # Every seed product is real, so each takes its code and title before any family
    # derives anything: no sibling may become another seed's product.
    reservations = Reservations()
    for root in roots.values():
        reservations.reserve(root)

    mean_siblings = config.siblings_for(len(seed_ids))
    sizes = np.array([size for size, _ in config.entity_sizes], dtype=np.int64)
    weights = np.array([weight for _, weight in config.entity_sizes], dtype=np.float64)

    entities: list[SynthEntity] = []
    dropped = 0
    for seed_id in seed_ids:
        rng = family_rng(config.seed, seed_id)
        family = family_id(seed_id)
        n_siblings = _round_stochastically(mean_siblings, rng)
        n_near = int(rng.binomial(n_siblings, config.near_sibling_share))
        products, lost = derive_family(
            roots[seed_id],
            n_near=n_near,
            n_far=n_siblings - n_near,
            rng=rng,
            reservations=reservations,
        )
        dropped += lost

        for index, product in enumerate(products):
            alt = alternate_code(rng)
            while not reservations.claim_code(alt):
                alt = alternate_code(rng)
            product = replace(product, alt_code=alt)
            listed = product
            if rng.random() < config.corruption.code_hidden:
                listed = hide_code(product)
            entity_id = f"{SOURCE}:{family}:e{index:03d}"
            size = int(rng.choice(sizes, p=weights))
            records = tuple(
                _record(
                    product,
                    corrupt(listed, rng, config.corruption),
                    entity_id=entity_id,
                    family=family,
                    seed_entity_id=seed_id,
                    n=n,
                    code_hidden=listed is not product,
                )
                for n in range(size)
            )
            entities.append(SynthEntity(entity_id, family, product, records))

    return SynthCatalog(
        config=config,
        entities=tuple(entities),
        n_families=len(seed_ids),
        n_dropped_siblings=dropped,
    )


def catalog_manifest(
    catalog: SynthCatalog,
    *,
    seed_dataset: str,
    split_test_fraction: float,
    split_seed: int,
    n_seed_records: int,
) -> dict[str, object]:
    """What a catalog was derived from and how -- enough to regenerate it exactly."""
    return {
        "generator": "dedup.synth.generate",
        "seed_dataset": seed_dataset,
        "seed_split": {"side": "train", "test_fraction": split_test_fraction, "seed": split_seed},
        "n_seed_records": n_seed_records,
        "n_families": catalog.n_families,
        "n_entities": len(catalog.entities),
        "n_dropped_siblings": catalog.n_dropped_siblings,
        "config": asdict(catalog.config),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive a synthetic catalog from a registered dataset's train split."
    )
    parser.add_argument("--seed-dataset", default="abt-buy", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=None, help="override the seed dataset directory")
    parser.add_argument("--records", type=int, required=True, help="target record count")
    parser.add_argument("--seed", type=int, default=0, help="generator seed")
    parser.add_argument(
        "--out", type=Path, required=True, help="directory for records.jsonl and manifest.json"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="also write the realism report here -- seeds against synthetic; its sibling "
        "statistic is quadratic in a brand's entities, so keep it to mid-scale catalogs",
    )
    args = parser.parse_args(argv)

    records = load_dataset(args.seed_dataset, args.root)
    # Always the report split, never a flag: a catalog seeded from any other split
    # could hold records the seed dataset's reports test on, and data/synthetic.py
    # refuses to load one.
    train, _ = split_by_entity(records, test_fraction=DEFAULT_TEST_FRACTION, seed=DEFAULT_SEED)
    catalog = generate(train, SynthConfig(target_records=args.records, seed=args.seed))
    digest = write_catalog(
        args.out,
        catalog.records,
        catalog_manifest(
            catalog,
            seed_dataset=args.seed_dataset,
            split_test_fraction=DEFAULT_TEST_FRACTION,
            split_seed=DEFAULT_SEED,
            n_seed_records=len(train),
        ),
    )
    print(
        f"wrote {len(catalog.records):,} records in {len(catalog.entities):,} entities from "
        f"{catalog.n_families:,} seed families ({catalog.n_dropped_siblings:,} requested "
        f"siblings found no free code) to {args.out}\nrecords sha256 {digest}",
        file=sys.stderr,
    )

    if args.report is not None:
        markdown = render_realism(
            pair_statistics(train),
            pair_statistics(catalog.records),
            read_manifest(args.out),
            catalog=args.out.as_posix(),
            out=args.report.as_posix(),
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown + "\n", encoding="utf-8")
        print(f"wrote {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
