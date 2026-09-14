"""How hard a catalog's duplicates are, measured identically on real seeds and synthetic output.

A synthetic catalog is only evidence if its duplicates are as hard as real ones.
This module computes, for any labelled catalog, the statistics that were measured
on Abt-Buy's train split before `synth/` was written -- title token overlap between
listings of one product, how often their vendor codes agree, how far their prices
disagree, how often a product has a near-code sibling -- so the generator's rates
can be calibrated against the seeds and the result reported side by side.

Dataset-agnostic: it reads `Record`s and `entity_id`, and derives every
comparison through `normalize`, the same canonical form the pipeline sees.
"""

from __future__ import annotations

import textwrap
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from dedup.eval.splits import DEFAULT_SEED, DEFAULT_TEST_FRACTION, group_by_entity
from dedup.normalize import NormalizedRecord, normalize
from dedup.schema import Record

# Same-brand vendor codes at most this many edits apart make two products siblings.
SIBLING_DISTANCE = 2


@dataclass(frozen=True)
class Spread:
    mean: float
    p10: float
    p50: float
    p90: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Spread | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        p10, p50, p90 = np.percentile(array, [10, 50, 90])
        return cls(float(array.mean()), float(p10), float(p50), float(p90))


@dataclass(frozen=True)
class PairStatistics:
    """Difficulty of a catalog's true pairs. Fractions are of true pairs unless stated."""

    n_records: int
    n_entities: int
    n_true_pairs: int
    entity_sizes: dict[int, int]
    title_jaccard: Spread | None
    token_count_ratio: float | None
    code_equal: float
    code_one_missing: float
    code_both_missing: float
    code_differ: float  # both listings print a code, and the codes differ
    price_both_present: float
    price_gap: Spread | None  # |a - b| / max(a, b), over pairs where both prices exist
    brand_both_present: float
    description_empty: float  # of records: None or ""
    sibling_rate: float | None  # of entities with a code: a same-brand code <= 2 edits away


def _sibling_rate(records: Sequence[Record], normalized: dict[str, NormalizedRecord]) -> float | None:
    """Share of coded entities with a same-brand entity whose code is within two edits.

    The brand is the leading token of the normalized title, which is how it was
    measured on Abt-Buy -- its Abt side has no brand column to use instead.
    """
    codes: dict[str, set[str]] = defaultdict(set)
    brand: dict[str, str] = {}
    for record in sorted(records, key=lambda r: r.record_id):
        view = normalized[record.record_id]
        brand.setdefault(record.entity_id, (view.normalized_title.split() or [""])[0])
        if view.model_number_key:
            codes[record.entity_id].add(view.model_number_key)
    if not codes:
        return None

    by_brand: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for entity_id, keys in codes.items():
        by_brand[brand[entity_id]].extend((entity_id, key) for key in sorted(keys))

    with_sibling: set[str] = set()
    for items in by_brand.values():
        if len({entity_id for entity_id, _ in items}) < 2:
            continue
        keys = [key for _, key in items]
        distances = process.cdist(
            keys,
            keys,
            scorer=Levenshtein.distance,
            score_cutoff=SIBLING_DISTANCE,
            dtype=np.int32,
            workers=1,
        )
        for i, j in zip(*np.nonzero(distances <= SIBLING_DISTANCE)):
            if items[i][0] != items[j][0]:
                with_sibling.add(items[i][0])
    return len(with_sibling) / len(codes)


def pair_statistics(records: Sequence[Record], *, siblings: bool = True) -> PairStatistics:
    """Measure a labelled catalog. `siblings=False` skips the one quadratic statistic."""
    grouped = group_by_entity(list(records))
    normalized = {record.record_id: normalize(record) for record in records}

    jaccard: list[float] = []
    ratio: list[float] = []
    gaps: list[float] = []
    equal = one_missing = both_missing = differ = price_both = brand_both = 0
    n_pairs = 0
    for members in grouped.values():
        for a, b in combinations(members, 2):
            n_pairs += 1
            view_a, view_b = normalized[a.record_id], normalized[b.record_id]
            tokens_a, tokens_b = view_a.normalized_title.split(), view_b.normalized_title.split()
            set_a, set_b = set(tokens_a), set(tokens_b)
            jaccard.append(len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 1.0)
            longest = max(len(tokens_a), len(tokens_b))
            ratio.append(min(len(tokens_a), len(tokens_b)) / longest if longest else 1.0)

            key_a, key_b = view_a.model_number_key, view_b.model_number_key
            if key_a and key_b:
                equal += key_a == key_b
                differ += key_a != key_b
            elif key_a or key_b:
                one_missing += 1
            else:
                both_missing += 1

            if a.price is not None and b.price is not None:
                price_both += 1
                top = max(a.price, b.price)
                gaps.append(abs(a.price - b.price) / top if top > 0 else 0.0)
            brand_both += bool(a.brand) and bool(b.brand)

    def share(count: int) -> float:
        return count / n_pairs if n_pairs else 0.0

    return PairStatistics(
        n_records=len(records),
        n_entities=len(grouped),
        n_true_pairs=n_pairs,
        entity_sizes=dict(sorted(Counter(len(m) for m in grouped.values()).items())),
        title_jaccard=Spread.of(jaccard),
        token_count_ratio=float(np.mean(ratio)) if ratio else None,
        code_equal=share(equal),
        code_one_missing=share(one_missing),
        code_both_missing=share(both_missing),
        code_differ=share(differ),
        price_both_present=share(price_both),
        price_gap=Spread.of(gaps),
        brand_both_present=share(brand_both),
        description_empty=(
            sum(not record.description for record in records) / len(records) if records else 0.0
        ),
        sibling_rate=_sibling_rate(records, normalized) if siblings else None,
    )


# ---------------------------------------------------------------------------
# The report: seeds against synthetic, side by side
# ---------------------------------------------------------------------------

# How close a calibrated statistic must land to the seeds'. The corruption rates
# were tuned by hand until every calibrated row fell inside it, against the seed
# dataset's train split only.
CALIBRATION_TOLERANCE = 0.05

CALIBRATED = frozenset(
    {
        "title token Jaccard, mean",
        "code keys equal",
        "code on one listing only",
        "code on neither listing",
        "both print a code, and they differ",
        "price gap, median",
    }
)


def _table(stats: PairStatistics) -> dict[str, float | None]:
    jaccard, gap = stats.title_jaccard, stats.price_gap
    return {
        "title token Jaccard, mean": None if jaccard is None else jaccard.mean,
        "title token Jaccard, p10": None if jaccard is None else jaccard.p10,
        "title token Jaccard, median": None if jaccard is None else jaccard.p50,
        "title token Jaccard, p90": None if jaccard is None else jaccard.p90,
        "token-count ratio, mean": stats.token_count_ratio,
        "code keys equal": stats.code_equal,
        "code on one listing only": stats.code_one_missing,
        "code on neither listing": stats.code_both_missing,
        "both print a code, and they differ": stats.code_differ,
        "price on both listings": stats.price_both_present,
        "price gap, median": None if gap is None else gap.p50,
        "price gap, p90": None if gap is None else gap.p90,
        "brand on both listings": stats.brand_both_present,
        "description empty (share of records)": stats.description_empty,
        "same-brand code within 2 edits (share of coded entities)": stats.sibling_rate,
    }


def calibration_gaps(seed: PairStatistics, synthetic: PairStatistics) -> dict[str, float]:
    """Synthetic minus seed, for every calibrated statistic both catalogs define."""
    ours, theirs = _table(seed), _table(synthetic)
    return {
        name: theirs[name] - ours[name]
        for name in ours
        if name in CALIBRATED and ours[name] is not None and theirs[name] is not None
    }


def _bullet(text: str) -> str:
    return textwrap.fill(
        text,
        width=98,
        initial_indent="- ",
        subsequent_indent="  ",
        break_on_hyphens=False,
        break_long_words=False,
    )


def _reading_notes(
    seed: PairStatistics, synthetic: PairStatistics, manifest: Mapping[str, object]
) -> list[str]:
    """Each interpretive sentence chosen from the measurement, as in every other report."""
    tolerance = CALIBRATION_TOLERANCE
    gaps = calibration_gaps(seed, synthetic)
    notes = []

    outside = [name for name, gap in gaps.items() if abs(gap) > tolerance]
    if gaps and not outside:
        widest = max(gaps, key=lambda name: abs(gaps[name]))
        notes.append(
            f"**Every calibrated statistic is within ±{tolerance:.2f} of the seeds.** The widest "
            f"gap is {widest} at {gaps[widest]:+.3f}. The corruption rates were tuned by hand "
            f"until this held, against the seeds themselves — the seed dataset's train split — so "
            f"it shows the generator can reproduce these statistics, not that its duplicates are "
            f"right in every way that matters."
        )
    elif outside:
        listed = ", ".join(f"{name} ({gaps[name]:+.3f})" for name in outside)
        notes.append(
            f"**{len(outside)} calibrated statistic(s) fall outside ±{tolerance:.2f} of the "
            f"seeds:** {listed}. Every number measured on this catalog inherits that gap."
        )

    split = manifest.get("seed_split", {})
    if (
        isinstance(split, Mapping)
        and split.get("side") == "train"
        and split.get("test_fraction") == DEFAULT_TEST_FRACTION
        and split.get("seed") == DEFAULT_SEED
    ):
        notes.append(
            f"**The seeds are a train split, so their test split never reached this catalog.** "
            f"`test_fraction={split.get('test_fraction')}`, `seed={split.get('seed')}` is the "
            f"split every report on `{manifest.get('seed_dataset')}` is scored on, so a model "
            f"trained here can still be scored on that test split without having seen it."
        )

    brand_gap = synthetic.brand_both_present - seed.brand_both_present
    if abs(brand_gap) > tolerance:
        more = "more" if brand_gap > 0 else "less"
        notes.append(
            f"**Brand is on both listings of {synthetic.brand_both_present:.1%} of synthetic "
            f"pairs against {seed.brand_both_present:.1%} of the seeds', by design.** The "
            f"generator withholds a brand per listing, independently. A seed catalog assembled "
            f"from sources with different columns makes brand co-occurrence a property of which "
            f"sources a pair spans, and that structure is not reproduced — so a brand column "
            f"carries {more} signal here than on the seeds."
        )

    if seed.token_count_ratio is not None and synthetic.token_count_ratio is not None:
        ratio_gap = synthetic.token_count_ratio - seed.token_count_ratio
        if abs(ratio_gap) > tolerance:
            direction = "less" if ratio_gap > 0 else "more"
            notes.append(
                f"**Listings of one product differ {direction} in length than the seeds' do** "
                f"(token-count ratio {synthetic.token_count_ratio:.3f} against "
                f"{seed.token_count_ratio:.3f}). Corruption drops and borrows words per listing; it "
                f"does not reproduce one source writing systematically longer titles than another."
            )

    if seed.sibling_rate is not None and synthetic.sibling_rate is not None:
        sibling_gap = synthetic.sibling_rate - seed.sibling_rate
        rates = f"{synthetic.sibling_rate:.3f} against the seeds' {seed.sibling_rate:.3f}"
        if sibling_gap > tolerance:
            notes.append(
                f"**Siblings are denser than in the seeds** ({rates}). Every near sibling is drawn "
                f"within two edits of its seed's code, so hard negatives are over-represented: a "
                f"model scored here meets more of them than a real catalog would hold."
            )
        elif sibling_gap < -tolerance:
            notes.append(
                f"**Siblings are sparser than in the seeds** ({rates}). Far siblings dilute the "
                f"near ones, so a model scored here meets fewer hard negatives than the seeds hold."
            )
        else:
            notes.append(
                f"**Sibling density matches the seeds** ({rates}), counted as a same-brand code "
                f"within two edits. That holds for the rate, not the shape: every synthetic sibling "
                f"is derived from one seed, where the seeds' siblings are independent products."
            )

    largest_seed, largest = max(seed.entity_sizes, default=0), max(synthetic.entity_sizes, default=0)
    if largest > largest_seed:
        n_larger = sum(count for size, count in synthetic.entity_sizes.items() if size > largest_seed)
        notes.append(
            f"**Entities are larger than any the seeds have.** The seeds' largest entity holds "
            f"{largest_seed} records; here {n_larger:,} entities hold more, up to {largest}. "
            f"That is deliberate — whether pricing unemitted pairs at p = 0 under-merges large "
            f"entities cannot be asked of pairs and triples — and the size distribution is a "
            f"judgment call, not a measurement."
        )
    return [_bullet(note) for note in notes]


def render_realism(
    seed: PairStatistics,
    synthetic: PairStatistics,
    manifest: Mapping[str, object],
    *,
    catalog: str,
    out: str,
) -> str:
    """The reports/ artifact: seed and synthetic difficulty side by side, and what it means."""
    config = manifest.get("config", {})
    split = manifest.get("seed_split", {})
    target = config.get("target_records") if isinstance(config, Mapping) else None
    seed_rows, synthetic_rows = _table(seed), _table(synthetic)
    gaps = calibration_gaps(seed, synthetic)

    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    rows = []
    for name, ours in seed_rows.items():
        theirs = synthetic_rows[name]
        gap = "—" if ours is None or theirs is None else f"{theirs - ours:+.3f}"
        if name in gaps:
            verdict = "within" if abs(gaps[name]) <= CALIBRATION_TOLERANCE else "**outside**"
            calibrated = f"±{CALIBRATION_TOLERANCE:.2f}, {verdict}"
        else:
            calibrated = "—"
        rows.append(f"| {name} | {cell(ours)} | {cell(theirs)} | {gap} | {calibrated} |")

    sizes = sorted(set(seed.entity_sizes) | set(synthetic.entity_sizes))
    size_header = " | ".join(str(size) for size in sizes)
    size_rule = " | ".join("---:" for _ in sizes)
    seed_sizes = " | ".join(f"{seed.entity_sizes.get(size, 0):,}" for size in sizes)
    synthetic_sizes = " | ".join(f"{synthetic.entity_sizes.get(size, 0):,}" for size in sizes)

    dropped = int(manifest.get("n_dropped_siblings", 0))
    dropped_sentence = (
        f"{dropped:,} requested siblings found no free code or title of either kind and were "
        f"dropped"
        if dropped
        else "Every requested sibling found a free code and title"
    )
    target_sentence = (
        f", so the catalog holds {synthetic.n_records / target:.1%} of its {target:,}-record target"
        if target
        else ""
    )
    notes = "\n".join(_reading_notes(seed, synthetic, manifest))

    return f"""# Synthetic catalog: how hard its duplicates are

A catalog generated by `synth/` is only evidence if its duplicates are as hard as
real ones. Every statistic below is computed identically — by `synth/realism.py`,
through `normalize` — on the seeds and on the generated catalog.

Regenerate with:

```bash
python -m dedup.synth.generate --seed-dataset {manifest.get("seed_dataset")} --records {target} --seed {config.get("seed")} --out {catalog} --report {out}
```

## Setup

- Seeds: the {split.get("side")} side of `{manifest.get("seed_dataset")}`'s entity-grouped \
split — {seed.n_records:,} records in {seed.n_entities:,} entities, {seed.n_true_pairs:,} true pairs.
- Catalog: {synthetic.n_records:,} records in {synthetic.n_entities:,} entities derived from \
{int(manifest.get("n_families", 0)):,} seed families, {synthetic.n_true_pairs:,} true pairs. \
{dropped_sentence}{target_sentence}.
- `records.jsonl` SHA-256: `{manifest.get("records_sha256")}`.

Records per entity:

| catalog | {size_header} |
| --- | {size_rule} |
| seeds | {seed_sizes} |
| synthetic | {synthetic_sizes} |

## Duplicate difficulty, seeds against synthetic

Over every true pair — every two listings of one entity — unless the row says
otherwise. **Calibrated** marks the rows the corruption rates were tuned to match.

| statistic | seeds | synthetic | gap | calibrated |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

## Reading this honestly

{notes}
"""
