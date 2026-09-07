"""Canonical record model.

Every dataset (Abt-Buy, Amazon-Google, later DBLP-* if a second domain is
ever needed, synthetic corruptions) is mapped into `Record` before touching
any later stage -- that is what keeps normalize/blocking/features/model/
cluster dataset-agnostic (CLAUDE.md, Layout).

Scoped to the product domain only for now (name/description/brand/price),
per CLAUDE.md's Data section treating DBLP as deferred. `raw_attributes` is
the escape hatch for source columns that don't map onto a canonical field.
"""

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

# Sources a loader is allowed to claim. Extend when a new loader is added
# (e.g. a DBLP source, if that work ever starts) -- keep this in sync with
# any loader module's own literal checks.
SourceName = Literal["abt_buy", "amazon_google", "synthetic"]

# min_length=1 alone admits " ": a whitespace-only id or title is a
# malformed source row, not a short one. StringConstraints strips first and
# then applies the length floor, so the blank case fails validation instead
# of travelling downstream. Stripping is safe at this boundary because
# surrounding whitespace carries no information -- normalize() would fold it
# away regardless, and an untrimmed record_id silently fails to join against
# a ground-truth pair file.
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    """One product listing, normalized to a single cross-source shape.

    Every dataset loader (benchmark or synth) must produce instances of this
    type before anything touches normalize/blocking/features/model/cluster.
    """

    # extra="forbid", not the default "ignore". A loader that passes a
    # misspelled or unmapped column ("titel=", or a source column nobody
    # mapped) should fail at the boundary rather than construct a Record
    # missing that data -- under "ignore" the field is silently dropped and
    # only shows up as unexplained blocking recall several stages later.
    # Columns with no canonical home go in raw_attributes, explicitly.
    model_config = ConfigDict(extra="forbid")

    # f"{source}:{raw_source_id}" by convention -- raw per-source ids collide
    # across sources (Abt id "10" and Buy id "10" are unrelated), so
    # record_id is never just the bare source id. Not enforced by a regex
    # here (over-constrains loaders); documented and exercised by a test.
    record_id: NonBlankStr

    source: SourceName

    # Ground-truth cluster/entity id. Present when loaded from a labeled
    # benchmark or synth-corrupted dataset; None for a genuinely new record
    # arriving at serve time with no known identity yet. This is what a
    # future entity-level (never pair-level) train/test splitter keys off --
    # that splitter does not exist yet; only the field is reserved here.
    entity_id: str | None = None

    # Required: every in-scope source always has a name/title. A title-less
    # record can't be scored by the TF-IDF baseline or any string feature,
    # so this is the one thing schema.py refuses to let through empty.
    title: NonBlankStr

    # Convention: None = "source did not have this column populated";
    # "" = "column present but empty in source". Loaders must preserve this
    # distinction, not collapse one into the other -- it's what
    # features/missingness.py keys off later, with no extra flag fields
    # needed on this model.
    description: str | None = None
    brand: str | None = None
    category: str | None = None

    # float, not Decimal: nothing downstream does currency-exact arithmetic
    # -- features/numeric.py computes relative differences/jitter tolerance,
    # and float is what pandas/DuckDB/LightGBM want natively. None is a
    # legitimate, expected value (Abt-Buy price is populated on well under
    # half of records).
    price: float | None = None

    # Deliberately absent. Model-number extraction is normalize.py's job
    # (CLAUDE.md: "the single highest-value signal") -- it's a derived
    # value, not something a raw source column maps onto directly. It
    # belongs on a NormalizedRecord type that normalize.py defines when
    # that module is built, not on this raw canonical shape.

    # Escape hatch for unmapped source columns (debugging/lineage only).
    # No pipeline stage reads this yet.
    raw_attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("price")
    @classmethod
    def _price_is_finite_and_non_negative(cls, v: float | None) -> float | None:
        # NaN and inf both survive a bare `v < 0` check -- every comparison
        # against NaN is False, and inf is genuinely greater than 0. Either
        # one poisons features/numeric.py's relative-difference arithmetic
        # (NaN propagates through the whole feature vector) with no error to
        # trace it back to. A missing price is None; it is never NaN.
        if v is None:
            return v
        if not math.isfinite(v):
            raise ValueError("price must be a finite number (got NaN or infinity)")
        if v < 0:
            raise ValueError("price must be non-negative")
        return v
