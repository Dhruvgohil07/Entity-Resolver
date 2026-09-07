"""Shared normalize step: canonical text form + model-number extraction.

Runs identically in the batch and serve paths (CLAUDE.md train/serve-parity
invariant) -- normalize() is a pure function of a Record, with no branching
on record.source, so it behaves the same for a source it has never seen
before as for one it was designed against.

Model-number extraction (_extract_model_number) is grounded in three
conventions found in real rows from the actual benchmarks (Abt.csv/Buy.csv
via dchud/ddbench, GoogleProducts.csv via git.iti.cs.ovgu.de) rather than an
assumed shape -- see the approved plan for the source examples. There is no
single reliable delimiter convention across sources, which is why this is
one generic heuristic applied uniformly instead of per-source parsing.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel

from dedup.schema import Record

# ---------------------------------------------------------------------------
# Text folding
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _fold(text: str) -> str:
    """Unicode NFKD + casefold + collapsed whitespace.

    casefold(), not lower() -- casefold is the correct primitive for
    case-insensitive comparison. NFKD brings differently-composed Unicode
    strings that represent the same text to the same form; it does not strip
    accents (that would need filtering the combining-mark category too,
    which nothing here claims to do).
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.casefold()
    return _WHITESPACE_RE.sub(" ", text).strip()


def _fold_optional(text: str | None) -> str | None:
    return None if text is None else _fold(text)


# ---------------------------------------------------------------------------
# Unit spelling canonicalization
# ---------------------------------------------------------------------------

# Spelling normalization only (not conversion) -- makes "4.2 cu ft" and
# "4.2 cu. ft." compare equal as text for later string-similarity features.
# Seeded from real Abt-Buy titles seen while designing this module (e.g.
# "4.2 Cu. Ft.", "24' White..."); extend as more sources/spellings turn up
# once the actual benchmark CSVs are downloaded.
_UNIT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcu\.?\s*ft\.?\b"), "cu ft"),
    (re.compile(r"(?<=\d)\s*['′]"), " in"),  # digit + ' -> inches
    (re.compile(r'(?<=\d)\s*["″]'), " in"),  # digit + " -> inches
    (re.compile(r"\bin\.\b|\binches?\b"), "in"),
    (re.compile(r"\boz\.\b"), "oz"),
    (re.compile(r"\blbs?\.\b|\bpounds?\b"), "lb"),
]


def _canonicalize_units(text: str) -> str:
    for pattern, replacement in _UNIT_RULES:
        text = pattern.sub(replacement, text)
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Brand alias resolution
# ---------------------------------------------------------------------------

# Folded variant -> canonical brand. Seeded, not exhaustive: every brand seen
# in the real Abt/Buy/Google rows researched for this module (sony, bose,
# panasonic, denon, linksys, netgear, belkin, canon, lacie, d-link, logitech,
# kensington, cuisinart, kitchenaid, frigidaire, tripp lite, ...) is already
# a single canonical token needing no entry here. Grow this table once real
# brand columns are in hand.
_BRAND_ALIASES: dict[str, str] = {
    "hewlett-packard": "hp",
    "hewlett packard": "hp",
    "lg electronics": "lg",
}


def _normalize_brand(folded_brand: str) -> str:
    return _BRAND_ALIASES.get(folded_brand, folded_brand)


# ---------------------------------------------------------------------------
# Model-number extraction
# ---------------------------------------------------------------------------

# A "model-number-shaped" token: letters/digits/hyphens only, 4-20 chars,
# with at least one digit AND at least one letter. That mix is what excludes
# plain words ("quickbooks"), years, and price-like digit runs while still
# catching "PSLX350H", "KX-FA83", "cd384-aisk9".
_MODEL_TOKEN_RE = re.compile(r"^[A-Za-z0-9-]+$")
_TRAILING_PUNCT_RE = re.compile(r"[,./=]+$")


def _clean_token(token: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", token)


def _qualifies_as_model_number(token: str) -> bool:
    if not (4 <= len(token) <= 20):
        return False
    if not _MODEL_TOKEN_RE.match(token):
        return False
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def _extract_model_number(title: str) -> str | None:
    """Best-effort vendor-code extraction. Preference order, each grounded
    in a convention actually observed in real titles:

      (a) trailing "<description> - <CODE>"       (Abt/Buy)
      (b) leading  "<CODE> <description>"          (Google)
      (c) an undelimited code somewhere mid-title  (Buy, no trailing code)

    Heuristic, not a guarantee -- false positives/negatives are possible.
    Revisit precision/recall against real labeled data once it's downloaded.
    """
    tokens = [_clean_token(t) for t in title.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None

    # (a) trailing convention: the last token is the code in every real
    # example seen where " - " precedes it.
    if " - " in title and _qualifies_as_model_number(tokens[-1]):
        return tokens[-1].upper()

    # (b) leading convention.
    if _qualifies_as_model_number(tokens[0]):
        return tokens[0].upper()

    # (c) fallback: first qualifying token anywhere.
    for token in tokens:
        if _qualifies_as_model_number(token):
            return token.upper()

    return None


# ---------------------------------------------------------------------------
# NormalizedRecord + normalize()
# ---------------------------------------------------------------------------


class NormalizedRecord(BaseModel):
    """Derived, normalized view of a Record.

    Never constructed directly by a loader -- always produced by
    normalize(), so the batch and serve paths can't diverge (CLAUDE.md
    train/serve-parity invariant). Composition over inheritance: holds the
    untouched raw Record plus derived fields, rather than extending Record
    in place -- later stages that want the original price, say, don't have
    to guess whether a field was normalized in place.
    """

    raw: Record

    normalized_title: str
    normalized_description: str | None
    normalized_brand: str | None
    model_number: str | None


def normalize(record: Record) -> NormalizedRecord:
    """Pure function: Record -> NormalizedRecord.

    No I/O, no branching on record.source -- must behave identically for any
    source, including one this function has never seen, per CLAUDE.md's
    train/serve-parity invariant.
    """
    normalized_title = _canonicalize_units(_fold(record.title))

    normalized_description = _fold_optional(record.description)
    if normalized_description is not None:
        normalized_description = _canonicalize_units(normalized_description)

    folded_brand = _fold_optional(record.brand)
    normalized_brand = None if folded_brand is None else _normalize_brand(folded_brand)

    model_number = _extract_model_number(record.title)

    return NormalizedRecord(
        raw=record,
        normalized_title=normalized_title,
        normalized_description=normalized_description,
        normalized_brand=normalized_brand,
        model_number=model_number,
    )
