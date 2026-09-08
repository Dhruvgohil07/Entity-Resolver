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
#
# Trailing-boundary rule: a pattern must never end in `\.\b`. A `\b` after a
# literal "." needs a word character next, so `\bin\.\b` cannot match "in."
# followed by a space or end-of-string -- exactly the position the rule
# exists for. Anchor the boundary on the letters (`ft\b\.?`) or state it
# explicitly as a lookahead (`\.(?=\s|$)`).
#
# Spelled-out units are anchored to a preceding digit so an ordinary word is
# never rewritten: "6 feet" -> "6 ft", but "Foot Massager" is left alone.
_UNIT_RULES: list[tuple[re.Pattern[str], str]] = [
    # `ft\b\.?`, not `ft\.?\b` -- the boundary is checked after the letters,
    # so the optional period is consumed instead of being left stranded as
    # "cu ft.".
    (re.compile(r"\bcu\.?\s*ft\b\.?"), "cu ft"),
    # ' is the foot mark typographically, but this catalog uses it for
    # inches and never for feet -- measured, not assumed: all 249 digit+'
    # occurrences across Abt.csv and Buy.csv are screen and driver sizes
    # ("3.0' LCD Display", "32' to 50' LCD", "4' x 6' Print Paper",
    # "1-1/8' Dome Tweeter"), and zero are lengths in feet. Normalize exists
    # to make duplicates compare equal, so it follows the source convention
    # rather than the typographic one; revisit if a source that really does
    # sell by the foot is added. The lookahead protects possessives --
    # without it "1980's" folds to "1980 in s". " needs no such guard, and
    # 5"W (5 inches wide) is a real spelling, so that rule takes none.
    (re.compile(r"(?<=\d)\s*['′](?![A-Za-z])"), " in"),
    (re.compile(r'(?<=\d)\s*["″]'), " in"),
    (re.compile(r"\bin\.(?=\s|$)"), "in"),
    (re.compile(r"(?<=\d)\s*(?:inches|inch)\b"), " in"),
    (re.compile(r"\bft\.(?=\s|$)"), "ft"),
    (re.compile(r"(?<=\d)\s*(?:feet|foot)\b"), " ft"),
    (re.compile(r"\boz\.(?=\s|$)"), "oz"),
    (re.compile(r"(?<=\d)\s*(?:ounces|ounce)\b"), " oz"),
    (re.compile(r"\blbs?\.(?=\s|$)"), "lb"),
    (re.compile(r"(?<=\d)\s*(?:lbs|pounds|pound)\b"), " lb"),
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

# A spec is not a model number. "1200W", "12MP" and "500GB" all pass the
# digit-and-letter shape test above, but they describe the product instead of
# identifying it. model_number is the highest-weight blocking key, so a spec
# admitted here puts every 1200-watt product from every brand in one block --
# a false key is much more expensive than a missing one.
#
# Matched as <digits><suffix> against a curated list. The list is deliberately
# conservative, since a suffix that collides with a real vendor code costs
# recall on the strongest signal there is: "wh" is excluded (Bose 161WH is a
# model, not 161 watt-hours) and so is "a" (HP Officejet 8500A/6500A).
# Single-letter suffixes only matter above the 4-character floor anyway, so
# "12V", "55W" and "4K" never reach this check.
_SPEC_UNIT_SUFFIXES = frozenset(
    {
        "w", "kw", "v", "ma", "mah", "ah",
        "mp", "kb", "mb", "gb", "tb",
        "hz", "khz", "mhz", "ghz",
        "mm", "cm", "m", "in", "ft", "yd",
        "g", "kg", "oz", "lb", "lbs",
        "ml", "l", "qt", "gal",
        "p", "k", "fps", "dpi", "ppi", "rpm", "btu", "hp",
        "pk", "ct", "pc",
    }
)
# No decimal branch: _MODEL_TOKEN_RE already rejects any token containing ".",
# so a spec is only ever reached here in its digits-then-letters form.
_SPEC_TOKEN_RE = re.compile(r"^\d+([A-Za-z]+)$")


def _clean_token(token: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", token)


def _is_spec_token(token: str) -> bool:
    match = _SPEC_TOKEN_RE.match(token)
    return match is not None and match.group(1).lower() in _SPEC_UNIT_SUFFIXES


def _qualifies_as_model_number(token: str) -> bool:
    if not (4 <= len(token) <= 20):
        return False
    if not _MODEL_TOKEN_RE.match(token):
        return False
    if not (any(c.isdigit() for c in token) and any(c.isalpha() for c in token)):
        return False
    return not _is_spec_token(token)


# Separators are the difference between "KXTS208W" (Abt) and "KX-TS208W" (Buy)
# -- the same Panasonic phone, written to two house styles. Both are correct as
# printed vendor codes, so `model_number` keeps whichever the source used; this
# is the form used to decide whether two codes are *the same code*.
#
# Measured, not assumed: on Abt-Buy an exact model_number blocker reaches pair
# completeness 0.3354, and the same blocker keyed on this stripped form reaches
# 0.5349. Roughly a fifth of achievable recall on the strongest key there is,
# lost to punctuation.
_MODEL_KEY_STRIP_RE = re.compile(r"[^a-z0-9]+")


def code_key(token: str) -> str:
    """Comparison form of a vendor code or code-shaped token.

    Public because `blocking/` needs the identical rule for the code-shaped
    tokens it indexes, and two private copies that must agree is how
    train/serve skew starts: a serve-time inverted index built from a second
    implementation cannot reproduce the batch blocking key.

    Note this is deliberately *more* aggressive than `_fold`. Discarding
    everything outside `[a-z0-9]` also discards the combining marks NFKD
    leaves behind, so "S8UX390" and "S8ÜX390" collapse to one key -- accent
    folding that `_fold` explicitly declines to do. That is wanted for a
    blocking key, where the cost of a missed match is a lost true pair and
    the cost of an extra one is a cheap comparison. It is stated here because
    it is a real difference in behaviour between the two functions, not an
    accident of the regex.

    Decomposes internally rather than assuming the caller already did. Its
    two callers feed it differently-prepared strings -- `_model_number_key`
    passes a code extracted from the *raw* title, `blocking` passes tokens
    from the already-folded one -- and without the NFKD here the same code
    keys two ways: a precomposed "Ü" is dropped whole ("s8x390") while a
    decomposed one loses only its mark ("s8ux390"). A key function whose
    answer depends on how far its input was already normalized is not usable
    as a shared rule.
    """
    decomposed = unicodedata.normalize("NFKD", token).casefold()
    return _MODEL_KEY_STRIP_RE.sub("", decomposed)


def _model_number_key(model_number: str | None) -> str | None:
    if model_number is None:
        return None
    # A code of nothing but punctuation cannot key a block -- `_qualifies_as_
    # model_number` already requires a letter and a digit, so this is
    # defensive rather than reachable today.
    return code_key(model_number) or None


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

    # (a) trailing convention. The candidate has to be the token that actually
    # follows the final " - ", not merely the last token of a title that has a
    # " - " somewhere earlier: "Canon PowerShot Camera - 12MP Black" satisfies
    # the looser test, and the value it yields is a spec, not a code.
    head_and_tail = title.rsplit(" - ", 1)
    if len(head_and_tail) == 2:
        trailing = head_and_tail[1].split()
        if len(trailing) == 1:
            candidate = _clean_token(trailing[0])
            if _qualifies_as_model_number(candidate):
                return candidate.upper()

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

    # The vendor code as the source printed it ("KX-TS208W"), for display and
    # for a human reading the review queue.
    model_number: str | None

    # The same code reduced to a comparison form ("kxts208w"). Two fields
    # rather than one because they answer different questions, and collapsing
    # them would throw away the printed form. This one lives here rather than
    # in blocking/ so that batch and serve compute it identically, and so
    # features/ can key on it later without importing from a sibling stage.
    model_number_key: str | None


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
        model_number_key=_model_number_key(model_number),
    )
