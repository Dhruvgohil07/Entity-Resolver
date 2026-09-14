"""Distinct products derived from one seed listing -- the hard negatives `synth/` exists for.

A catalog of duplicates alone is easy: every near-identical pair is a match. Real
catalogs are full of *siblings* instead -- distinct products whose vendor codes
differ by a character or two. Measured on Abt-Buy's train split, 389 of the 699
entities carrying a code have a same-brand sibling within edit distance 2
(`nnh965wf`/`nnh965bf`, `mx350`/`mx850`, `9763a001`/`9764a001`), and the fusions
that survive correlation clustering are exactly that shape (Weber 3780001 /
3880001, Samsung YP-S2ZG / YP-S2ZW).

So a family is a seed product plus siblings of two kinds:

  * **near** -- the vendor code one or two characters away: a digit changed in its
    last digit run, a version number bumped, or a colour or region suffix swapped,
    with the title's colour word swapped to match;
  * **far** -- a fresh code of the same shape behind the same series prefix, brand
    and category words: another product from the same line.

Distinctness is guaranteed rather than hoped for. A sibling whose code key or
normalized title is already taken anywhere in the catalog is redrawn, and dropped
if no free one turns up in a bounded number of tries: two entities sharing a code
would be ground truth that contradicts itself, and no metric computed on it could
show that.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from rapidfuzz.distance import Levenshtein

from dedup.normalize import code_key, normalize
from dedup.schema import Record

NEAR_MUTATIONS = ("digit", "version", "suffix")

# The colour a suffix names, used only to keep a sibling's title honest: a phone
# listed as White whose code moves from ...208W to ...208B is listed as Black.
COLOUR_OF_SUFFIX: dict[str, str] = {
    "B": "black",
    "BK": "black",
    "W": "white",
    "WH": "white",
    "R": "red",
    "RD": "red",
    "SL": "silver",
    "SS": "stainless",
    "BL": "blue",
    "GY": "gray",
}

# Suffixes a swap may choose from, by length. Colour codes plus the region and
# finish letters vendors print (`nnh965wf` / `nnh965bf`).
SUFFIX_POOL: dict[int, tuple[str, ...]] = {
    1: ("A", "B", "E", "K", "R", "S", "U", "W"),
    2: ("BF", "BK", "BL", "GY", "RD", "SL", "SS", "WF", "WH"),
}

MAX_TRIES = 20

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_COLOUR_WORD = re.compile(r"\b(black|white|silver|stainless|red|blue|gray|grey)\b", re.IGNORECASE)
_DIGIT_RUN = re.compile(r"\d+")
_LETTER_SUFFIX = re.compile(r"(?<=\d)([A-Za-z]{1,2})$")
_SERIES_PREFIX = re.compile(r"^[A-Za-z]+")
# A number followed by a unit: the sizes and capacities that distinguish products
# in one line (32" against 40", 1.6 cu ft against 2.0). The lookbehind keeps digits
# inside a vendor code out of it.
_SIZE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)(?=\s*(?:\"|'(?![A-Za-z])|-?\s?inch|in\b|cu\b|gb\b|tb\b|mp\b))",
    re.IGNORECASE,
)
_SIZE_FACTORS = (0.5, 0.75, 1.25, 1.5, 2.0)


@dataclass(frozen=True)
class Product:
    """One distinct product: the clean listing every duplicate of it is corrupted from."""

    title: str
    description: str | None
    brand: str | None
    price: float | None
    code: str | None  # the vendor code exactly as `title` prints it
    alt_code: str | None = None  # a part number from a different numbering scheme
    kind: str = "root"  # root, near or far
    seed_record_id: str = ""


def base_product(records: Sequence[Record]) -> Product:
    """A seed entity's listing to derive from, with fields the benchmark split across listings.

    Prefers a record whose title carries a vendor code, so there is something for
    siblings to vary; ties break on `record_id`. Brand, description and price are
    taken from the first record holding one, because a benchmark can print them on
    one side only -- Abt-Buy's brand exists only on Buy.
    """
    if not records:
        raise ValueError("a seed entity needs at least one record")
    ordered = sorted(records, key=lambda record: record.record_id)
    normalized = [normalize(record) for record in ordered]
    chosen = next((i for i, n in enumerate(normalized) if n.model_number_key), 0)
    record, key = ordered[chosen], normalized[chosen].model_number_key

    code = None
    if key:
        token = next((t for t in record.title.split() if code_key(t) == key), None)
        code = None if token is None else token.strip(",.;:=()[]")
    return Product(
        title=" ".join(record.title.split()),
        description=next((r.description for r in ordered if r.description), record.description),
        brand=next((r.brand for r in ordered if r.brand), None),
        price=next((r.price for r in ordered if r.price is not None), None),
        code=code or None,
        seed_record_id=record.record_id,
    )


def mutate_code(code: str, mutation: str, rng: np.random.Generator) -> str | None:
    """One near mutation of a vendor code, or None when the code has nothing it can change."""
    runs = list(_DIGIT_RUN.finditer(code))
    if mutation == "digit":
        if not runs:
            return None
        run = runs[-1]
        position = int(rng.integers(run.start(), run.end()))
        digit = str((int(code[position]) + int(rng.integers(1, 10))) % 10)
        return code[:position] + digit + code[position + 1 :]
    if mutation == "version":
        if not runs:
            return None
        run = runs[int(rng.integers(len(runs)))]
        bumped = str(int(run.group()) + int(rng.integers(1, 4))).zfill(len(run.group()))
        if len(bumped) != len(run.group()):
            return None
        return code[: run.start()] + bumped + code[run.end() :]
    if mutation == "suffix":
        match = _LETTER_SUFFIX.search(code)
        if match is None:
            return None
        suffix = match.group(1)
        pool = [s for s in SUFFIX_POOL[len(suffix)] if s != suffix.upper()]
        chosen = pool[int(rng.integers(len(pool)))]
        return code[: match.start(1)] + (chosen.lower() if suffix.islower() else chosen)
    raise ValueError(f"unknown mutation {mutation!r}; known: {NEAR_MUTATIONS}")


def recolour(title: str, suffix: str) -> str:
    """The title with its colour word swapped for the one `suffix` names, if both exist."""
    colour = COLOUR_OF_SUFFIX.get(suffix.upper())
    if colour is None:
        return title

    def swap(match: re.Match[str]) -> str:
        return colour.capitalize() if match.group(0)[0].isupper() else colour

    return _COLOUR_WORD.sub(swap, title, count=1)


def resize(title: str, rng: np.random.Generator) -> str | None:
    """The title with one size or capacity changed, or None when it names none."""
    matches = list(_SIZE.finditer(title))
    if not matches:
        return None
    match = matches[int(rng.integers(len(matches)))]
    text = match.group(1)
    value = float(text) * _SIZE_FACTORS[int(rng.integers(len(_SIZE_FACTORS)))]
    decimals = len(text.split(".")[1]) if "." in text else 0
    new = f"{value:.{decimals}f}" if decimals else str(max(round(value), 1))
    if new == text:
        return None
    return title[: match.start(1)] + new + title[match.end(1) :]


def fresh_code(code: str, rng: np.random.Generator) -> str:
    """A code of the same shape behind the same series prefix: KX-TS208W -> KX-QD731M."""
    prefix = _SERIES_PREFIX.match(code)
    keep = prefix.end() if prefix is not None and prefix.end() < len(code) else 0
    out = list(code)
    for index in range(keep, len(out)):
        char = out[index]
        if char.isdigit():
            out[index] = str(int(rng.integers(10)))
        elif char.isalpha():
            letter = _LETTERS[int(rng.integers(len(_LETTERS)))]
            out[index] = letter.upper() if char.isupper() else letter
    return "".join(out)


def alternate_code(rng: np.random.Generator) -> str:
    """A part number from another scheme -- the `0617B002` against `CL41CL` case."""

    def digits(n: int) -> str:
        return "".join(str(int(rng.integers(10))) for _ in range(n))

    return f"{digits(4)}{_LETTERS[int(rng.integers(len(_LETTERS)))].upper()}{digits(3)}"


def _reprice(price: float | None, rng: np.random.Generator, low: float, high: float) -> float | None:
    if price is None:
        return None
    return round(max(price * float(rng.uniform(low, high)), 0.01), 2)


def _swap_code(text: str | None, old: str, new: str) -> str | None:
    return text if text is None else text.replace(old, new)


def near_sibling(
    product: Product,
    rng: np.random.Generator,
    *,
    mutations: Sequence[str] = NEAR_MUTATIONS,
    resize_rate: float = 0.3,
) -> Product | None:
    """A distinct product one or two characters of code away, or None if this draw found none.

    A product with no code can only become a different product through its size.
    """
    if product.code is None:
        title = resize(product.title, rng)
        if title is None:
            return None
        return replace(
            product, title=title, price=_reprice(product.price, rng, 0.75, 1.35), kind="near"
        )

    mutation = mutations[int(rng.integers(len(mutations)))]
    code = mutate_code(product.code, mutation, rng)
    if code is None or not 1 <= Levenshtein.distance(code_key(product.code), code_key(code)) <= 2:
        return None
    title = product.title.replace(product.code, code)
    suffix = _LETTER_SUFFIX.search(code)
    if mutation == "suffix" and suffix is not None:
        title = recolour(title, suffix.group(1))
    if rng.random() < resize_rate:
        title = resize(title, rng) or title
    return replace(
        product,
        title=title,
        description=_swap_code(product.description, product.code, code),
        code=code,
        alt_code=None,
        price=_reprice(product.price, rng, 0.75, 1.35),
        kind="near",
    )


def far_sibling(product: Product, rng: np.random.Generator) -> Product | None:
    """Another product from the same line, or None if this draw landed too close to the root."""
    if product.code is None:
        return None
    code = fresh_code(product.code, rng)
    if Levenshtein.distance(code_key(product.code), code_key(code)) < 3:
        return None
    title = product.title.replace(product.code, code)
    if rng.random() < 0.5:
        title = resize(title, rng) or title
    return replace(
        product,
        title=title,
        description=_swap_code(product.description, product.code, code),
        code=code,
        alt_code=None,
        price=_reprice(product.price, rng, 0.5, 2.0),
        kind="far",
    )


def normalized_title(product: Product) -> str:
    """The comparison form of a product's title, through the pipeline's own `normalize`."""
    return normalize(Record(record_id="synthetic:probe", source="synthetic", title=product.title))


@dataclass
class Reservations:
    """Code keys and normalized titles already taken anywhere in the catalog."""

    codes: set[str] = field(default_factory=set)
    titles: set[str] = field(default_factory=set)

    @staticmethod
    def _keys(product: Product) -> set[str]:
        return {code_key(code) for code in (product.code, product.alt_code) if code}

    def reserve(self, product: Product) -> None:
        """Take a product's code and title unconditionally -- for seed products, which are real."""
        self.codes.update(self._keys(product))
        self.titles.add(normalized_title(product).normalized_title)

    def claim(self, product: Product) -> bool:
        """Take a derived product's code and title, or refuse if either is already taken."""
        keys = self._keys(product)
        title = normalized_title(product).normalized_title
        if title in self.titles or keys & self.codes:
            return False
        self.codes.update(keys)
        self.titles.add(title)
        return True

    def claim_code(self, code: str) -> bool:
        key = code_key(code)
        if key in self.codes:
            return False
        self.codes.add(key)
        return True


def derive_family(
    root: Product,
    *,
    n_near: int,
    n_far: int,
    rng: np.random.Generator,
    reservations: Reservations,
    max_tries: int = MAX_TRIES,
) -> tuple[list[Product], int]:
    """The root followed by its siblings, and how many requested siblings found nothing free.

    Every sibling is derived from the root, never from another sibling, so a near
    sibling is always within two characters of the seed's own code. A root with
    no code has no line to draw a far sibling from, so its far requests become
    near ones. A request that finds nothing free of its own kind is drawn as the
    other kind before it is dropped: a short code has only a few dozen near
    variants, and dropping the rest would shrink the catalog below its target.
    """
    if root.code is None:
        n_near, n_far = n_near + n_far, 0

    def draw(derive) -> Product | None:
        for _ in range(max_tries):
            sibling = derive(root, rng)
            if sibling is not None and reservations.claim(sibling):
                return sibling
        return None

    products = [root]
    dropped = 0
    for derive, fallback, count in (
        (near_sibling, far_sibling, n_near),
        (far_sibling, near_sibling, n_far),
    ):
        for _ in range(count):
            sibling = draw(derive) or draw(fallback)
            if sibling is None:
                dropped += 1
            else:
                products.append(sibling)
    return products, dropped
