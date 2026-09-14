"""Duplicate listings of one product -- the operators, and the profile that sets their rates.

A duplicate is not a noisy copy of a clean string. Measured on Abt-Buy's train
split, two listings of the *same* product share only 43% of their title tokens on
average (median 40%), carry the same vendor code on only 56% of pairs, and
disagree on price by a median 18% when both show one. A synthetic catalog that
misses that difficulty makes every model trained and scored on it look better
than it is.

Each operator is a pure function of its input and a numpy Generator, and
`CorruptionProfile` holds the rate each one fires at. `corrupt` applies them in a
fixed order and records which fired, so every synthetic record carries its own
lineage.

What the operators deliberately do not do is anticipate `normalize.py`. Some unit
spellings they produce are absorbed by normalization (`32"`, `32 in.`) and some
are not (`32-inch`); that split is the realism, and `tests/test_synth_corrupt.py`
pins which is which.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace

import numpy as np

from dedup.normalize import code_key
from dedup.synth.families import Product

# Shorthand vendors and marketplaces actually print, word -> abbreviation.
ABBREVIATIONS: dict[str, str] = {
    "accessory": "acc",
    "adapter": "adptr",
    "and": "&",
    "black": "blk",
    "cartridge": "cart",
    "digital": "dig",
    "headphones": "hdphns",
    "microwave": "micro",
    "portable": "port",
    "professional": "pro",
    "refrigerator": "fridge",
    "silver": "slvr",
    "speaker": "spkr",
    "speakers": "spkrs",
    "stainless": "stnls",
    "system": "sys",
    "television": "tv",
    "white": "wht",
    "wireless": "wrls",
    "with": "w/",
}

# Printed after a vendor code for a region or revision: `YP-S2ZG/XAA`.
REGION_SUFFIXES = ("/XAA", "/XAC", "/US", "-B", "-R")

# How a brand gets written when it is not written plainly. normalize.py folds case
# and knows a handful of aliases, and none of these.
BRAND_FORMS = ("{} Inc.", "{} Corporation", "{} Electronics", "{} Corp")

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_REPEATED_DASH = re.compile(r"(?:\s+-){2,}\s+")
_DANGLING_DASH = re.compile(r"^(?:-\s+)+|(?:\s+-)+$")

_INCH = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)"
    r"(?:\s*(?:\"|″|'(?![A-Za-z]))|\s*-\s*inch(?:es)?\b|\s+inch(?:es)?\b|\s+in\b\.?)",
    re.IGNORECASE,
)
_INCH_FORMS = ('{n}"', "{n}'", "{n} in.", "{n} inch", "{n}-inch")
_CU_FT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*cu\b\.?\s*-?\s*ft\b\.?", re.IGNORECASE)
_CU_FT_FORMS = ("{n} Cu. Ft.", "{n} cu ft", "{n} cu.ft.", "{n}-cu-ft")
_STORAGE = re.compile(r"(?<![\w.])(\d+)\s?(GB|TB|MB)\b", re.IGNORECASE)

# Rates that choose among mutually exclusive outcomes for one field.
_EXCLUSIVE = (
    ("code_drop", "code_alternate", "code_region_suffix"),
    ("brand_none", "brand_empty", "brand_alias"),
    ("description_empty", "description_truncate"),
)


@dataclass(frozen=True)
class CorruptionProfile:
    """The rate each operator fires at, per listing unless stated.

    Defaults are calibrated by hand against Abt-Buy **train** pair statistics --
    never test -- and `reports/synth/realism.md` shows where they land.
    """

    code_hidden: float = 0.08  # per entity: none of its listings prints the code
    code_drop: float = 0.08
    code_alternate: float = 0.06
    code_region_suffix: float = 0.05
    code_punctuation: float = 0.20
    code_to_front: float = 0.15
    token_drop: float = 0.22  # per token other than the code
    description_tokens: float = 0.45
    abbreviation: float = 0.30  # per token that has an abbreviation
    unit_spelling: float = 0.50  # per size or capacity mention
    typo: float = 0.25
    brand_none: float = 0.45
    brand_empty: float = 0.03
    brand_alias: float = 0.10
    price_none: float = 0.44
    price_jitter_sigma: float = 0.19  # of log price; not a probability
    description_empty: float = 0.20
    description_truncate: float = 0.40

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{spec.name} must be finite and non-negative, got {value!r}")
            if spec.name != "price_jitter_sigma" and value > 1:
                raise ValueError(f"{spec.name} is a probability, got {value!r}")
        for group in _EXCLUSIVE:
            total = sum(getattr(self, name) for name in group)
            if total > 1 + 1e-12:
                raise ValueError(
                    f"{' + '.join(group)} choose one outcome for a field and sum to {total:.3f}, "
                    f"which is more than 1"
                )


@dataclass(frozen=True)
class Listing:
    """One corrupted listing of a product, and the operators that fired to make it."""

    title: str
    description: str | None
    brand: str | None
    price: float | None
    operators: tuple[str, ...]


def punctuate_code(code: str, rng: np.random.Generator) -> str:
    """Add or remove separators without changing the code: `KXTS208W` <-> `KX-TS208W`."""
    if "-" in code or "/" in code:
        return code.replace("-", "").replace("/", "")
    boundaries = [match.start() for match in _BOUNDARY.finditer(code)]
    if not boundaries:
        return code
    at = boundaries[int(rng.integers(len(boundaries)))]
    return code[:at] + "-" + code[at:]


def typo(token: str, rng: np.random.Generator) -> str:
    """One character deleted, inserted, substituted or transposed; tokens under 4 characters kept."""
    if len(token) < 4:
        return token
    at = int(rng.integers(1, len(token) - 1))
    letter = _LETTERS[int(rng.integers(len(_LETTERS)))]
    if token.isupper():
        letter = letter.upper()
    operation = int(rng.integers(4))
    if operation == 0:
        return token[:at] + token[at + 1 :]
    if operation == 1:
        return token[:at] + letter + token[at:]
    if operation == 2:
        return token[:at] + letter + token[at + 1 :]
    return token[: at - 1] + token[at] + token[at - 1] + token[at + 1 :]


def respell_units(title: str, rng: np.random.Generator, rate: float) -> tuple[str, bool]:
    """Each size, capacity and storage mention respelled with probability `rate`."""
    changed = False

    def respell(pattern: re.Pattern[str], render) -> None:
        nonlocal title

        def replace_match(match: re.Match[str]) -> str:
            nonlocal changed
            if rng.random() >= rate:
                return match.group(0)
            new = render(match)
            changed = changed or new != match.group(0)
            return new

        title = pattern.sub(replace_match, title)

    def pick(forms: tuple[str, ...]) -> str:
        return forms[int(rng.integers(len(forms)))]

    respell(_INCH, lambda match: pick(_INCH_FORMS).format(n=match.group(1)))
    respell(_CU_FT, lambda match: pick(_CU_FT_FORMS).format(n=match.group(1)))
    respell(
        _STORAGE,
        lambda match: pick(("{n}{u}", "{n} {u}", "{n}{l}")).format(
            n=match.group(1), u=match.group(2).upper(), l=match.group(2).lower()
        ),
    )
    return title, changed


def _find_code(tokens: list[str], code: str | None) -> int | None:
    if not code:
        return None
    key = code_key(code)
    return next((index for index, token in enumerate(tokens) if code_key(token) == key), None)


def _tidy(title: str) -> str:
    title = _REPEATED_DASH.sub(" - ", title)
    title = _DANGLING_DASH.sub("", title)
    return " ".join(title.split())


def hide_code(product: Product) -> Product:
    """The product as some source lists it everywhere: with no vendor code at all.

    Applied to a whole entity rather than one listing. It is the truncated-title
    failure CLAUDE.md records (`LG 1.6 cu.ft. Over the Range`), and a per-listing
    drop cannot produce it often enough: 8% of Abt-Buy train pairs carry a code on
    neither side.
    """
    tokens = product.title.split()
    at = _find_code(tokens, product.code)
    if at is None:
        return product
    del tokens[at]
    title = _tidy(" ".join(tokens))
    if not any(char.isalnum() for char in title):
        return product
    return replace(product, title=title, code=None, alt_code=None)


def corrupt(product: Product, rng: np.random.Generator, profile: CorruptionProfile) -> Listing:
    """One listing of `product`, as a different vendor or marketplace might have written it."""
    fired: list[str] = []
    # (token, is_code) pairs, so the vendor code is never dropped, abbreviated or
    # typo'd by a word-level operator -- code corruption has operators of its own.
    parts = [(token, False) for token in product.title.split()]
    at = _find_code([token for token, _ in parts], product.code)

    if at is not None:
        roll = rng.random()
        if roll < profile.code_drop:
            del parts[at]
            at = None
            fired.append("code_drop")
        else:
            token = parts[at][0]
            if roll < profile.code_drop + profile.code_alternate and product.alt_code:
                token = product.alt_code
                fired.append("code_alternate")
            elif roll < profile.code_drop + profile.code_alternate + profile.code_region_suffix:
                token = product.code + REGION_SUFFIXES[int(rng.integers(len(REGION_SUFFIXES)))]
                fired.append("code_region_suffix")
            if "code_alternate" not in fired and rng.random() < profile.code_punctuation:
                token = punctuate_code(token, rng)
                fired.append("code_punctuation")
            parts[at] = (token, True)
            if at > 0 and rng.random() < profile.code_to_front:
                parts.insert(0, parts.pop(at))
                fired.append("code_to_front")

    if profile.token_drop > 0:
        kept = [part for part in parts if part[1] or rng.random() >= profile.token_drop]
        words_left = sum(1 for token, is_code in kept if not is_code and token != "-")
        if len(kept) < len(parts) and words_left:
            parts = kept
            fired.append("token_drop")

    abbreviated = False
    for index, (token, is_code) in enumerate(parts):
        short = None if is_code else ABBREVIATIONS.get(token.lower())
        if short is not None and rng.random() < profile.abbreviation:
            parts[index] = (short, False)
            abbreviated = True
    if abbreviated:
        fired.append("abbreviation")

    if product.description and rng.random() < profile.description_tokens:
        codes = {code_key(code) for code in (product.code, product.alt_code) if code}
        words = [word for word in _WORD.findall(product.description) if code_key(word) not in codes]
        if words:
            start = int(rng.integers(len(words)))
            parts.extend((word, False) for word in words[start : start + int(rng.integers(1, 5))])
            fired.append("description_tokens")

    if rng.random() < profile.typo:
        candidates = [i for i, (token, is_code) in enumerate(parts) if not is_code and len(token) >= 4]
        if candidates:
            index = candidates[int(rng.integers(len(candidates)))]
            parts[index] = (typo(parts[index][0], rng), False)
            fired.append("typo")

    title, respelled = respell_units(_tidy(" ".join(token for token, _ in parts)), rng, profile.unit_spelling)
    if respelled:
        fired.append("unit_spelling")
    if not any(char.isalnum() for char in title):
        title = product.title
        fired.append("title_restored")

    brand = product.brand
    roll = rng.random()
    if product.brand is not None:
        if roll < profile.brand_none:
            brand = None
            fired.append("brand_none")
        elif roll < profile.brand_none + profile.brand_empty:
            brand = ""
            fired.append("brand_empty")
        elif roll < profile.brand_none + profile.brand_empty + profile.brand_alias and product.brand:
            brand = BRAND_FORMS[int(rng.integers(len(BRAND_FORMS)))].format(product.brand)
            fired.append("brand_alias")

    price = product.price
    if price is not None:
        if rng.random() < profile.price_none:
            price = None
            fired.append("price_none")
        else:
            jitter = float(np.exp(rng.normal(0.0, profile.price_jitter_sigma)))
            price = round(max(price * jitter, 0.01), 2)

    description = product.description
    if description is not None:
        roll = rng.random()
        if roll < profile.description_empty:
            description = ""
            fired.append("description_empty")
        elif roll < profile.description_empty + profile.description_truncate:
            description = " ".join(description.split()[: int(rng.integers(3, 16))])
            fired.append("description_truncate")

    return Listing(
        title=title,
        description=description,
        brand=brand,
        price=price,
        operators=tuple(fired),
    )
