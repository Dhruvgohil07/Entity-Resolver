"""Abt-Buy benchmark loader: raw CSV -> canonical `Record`.

Source files, verified reachable on 2026-09-07:

    https://raw.githubusercontent.com/dchud/ddbench/HEAD/data/abt-buy/Abt.csv
    https://raw.githubusercontent.com/dchud/ddbench/HEAD/data/abt-buy/Buy.csv
    .../data/abt-buy/abt_buy_perfectMapping.csv

The Leipzig original, https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip,
carries the same three files -- note the /file/ -> /files/datasets/ redirect.

The facts below are measured against those files rather than assumed. Each
one would otherwise surface much later as a silent defect:

  * Abt.csv is cp1252, not UTF-8 -- it carries 0xad/0xae/0xb0/0xb1/0xba
    (soft hyphen, (R), degree, +/-, masculine ordinal). A default utf-8 read
    raises on it outright. Buy.csv and the mapping are pure ASCII today, so
    decoding them as cp1252 too is a no-op, and one encoding for the dataset
    beats a per-file table. latin-1 is deliberately not used: it decodes
    every byte, so a genuine encoding change would mojibake silently instead
    of raising.
  * The two sides do not share a schema. Abt has no brand column at all;
    Buy has `manufacturer`, populated on all but 6 of its 1092 rows. So
    `brand` is None on every Abt record (column absent) while a blank Buy
    cell is "" (column present, value empty). schema.py's None-vs-""
    convention is load-bearing from the very first dataset -- and a brand
    equality feature is unusable on this benchmark, since it is missing on
    every single pair.
  * Price is written "$399.00", with a thousands comma on 97 of the 1008
    populated values ("$1,999.00"). 61% of Abt rows and 46% of Buy rows
    leave it empty.
  * Row counts: 1081 Abt, 1092 Buy, 1097 ground-truth pairs.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from dedup.schema import Record

SOURCE = "abt_buy"

ABT_FILE = "Abt.csv"
BUY_FILE = "Buy.csv"
MAPPING_FILE = "abt_buy_perfectMapping.csv"

# See the module docstring: cp1252 covers Abt.csv's high bytes and is a no-op
# for the two ASCII files.
ENCODING = "cp1252"

_ABT_COLUMNS = {"id", "name", "description", "price"}
_BUY_COLUMNS = {"id", "name", "description", "manufacturer", "price"}
_MAPPING_COLUMNS = {"idAbt", "idBuy"}

# "$1,999.00", "$49.00", "14.95". Anchored on purpose: a price this does not
# match is a loader bug and is raised, never folded into None. None means
# "the source left this cell empty", and quietly widening it to also mean
# "we could not read this" would feed features/missingness.py a lie.
_PRICE_RE = re.compile(r"^\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)$")


def _read_rows(path: Path, expected_columns: set[str]) -> list[dict[str, str]]:
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.DictReader(handle)
        actual = set(reader.fieldnames or ())
        if actual != expected_columns:
            raise ValueError(
                f"{path.name}: expected columns {sorted(expected_columns)}, got {sorted(actual)}"
            )
        return list(reader)


def _parse_price(raw: str | None) -> float | None:
    """Parse "$1,999.00" -> 1999.0, "" -> None, anything else -> ValueError."""
    text = (raw or "").strip()
    if not text:
        return None
    match = _PRICE_RE.match(text)
    if match is None:
        raise ValueError(f"unparseable price: {raw!r}")
    return float(match.group(1).replace(",", ""))


def _record(row: dict[str, str], side: str, *, has_brand_column: bool) -> Record:
    raw_id = (row["id"] or "").strip()
    if not raw_id:
        raise ValueError(f"{side} row with empty id: {row!r}")

    # Three parts, not schema.py's usual f"{source}:{raw_id}". Abt and Buy are
    # two sides of one source whose ids collide independently -- Abt 552 and
    # Buy 552 are unrelated products -- so the side has to be part of the key.
    record_id = f"{SOURCE}:{side}:{raw_id}"

    # .strip() on the text fields only trims; it never turns a populated cell
    # into an empty one, so the ""-means-present convention survives intact.
    # brand is None when the column is absent (Abt) and "" when the column
    # exists but the cell is blank (6 Buy rows).
    return Record(
        record_id=record_id,
        source=SOURCE,
        title=row["name"],
        description=(row["description"] or "").strip(),
        brand=(row["manufacturer"] or "").strip() if has_brand_column else None,
        category=None,  # neither side ships one
        price=_parse_price(row.get("price")),
        raw_attributes={"side": side, "raw_id": raw_id},
    )


def load_products(root: Path) -> list[Record]:
    """Both sides of the catalog, with `entity_id` still unset."""
    root = Path(root)
    abt = [
        _record(row, "abt", has_brand_column=False)
        for row in _read_rows(root / ABT_FILE, _ABT_COLUMNS)
    ]
    buy = [
        _record(row, "buy", has_brand_column=True)
        for row in _read_rows(root / BUY_FILE, _BUY_COLUMNS)
    ]
    return abt + buy


def load_pairs(root: Path) -> list[tuple[str, str]]:
    """Ground-truth matches, as pairs of `record_id`."""
    rows = _read_rows(Path(root) / MAPPING_FILE, _MAPPING_COLUMNS)
    return [
        (f"{SOURCE}:abt:{row['idAbt'].strip()}", f"{SOURCE}:buy:{row['idBuy'].strip()}")
        for row in rows
    ]


def assign_entity_ids(records: list[Record], pairs: list[tuple[str, str]]) -> list[Record]:
    """Turn ground-truth pairs into cluster ids, by union-find over the graph.

    The benchmark ships ground truth as pairs, but `Record.entity_id` wants a
    cluster id, because that is what an entity-level (never pair-level) split
    keys off. Getting there means connected components, which is the chaining
    hazard CLAUDE.md flags -- so it was measured before being relied on. On
    Abt-Buy it yields 1076 clusters: 1055 of size 2 and 21 of size 3, with no
    runaway component. The size-3 clusters are one Abt product against two
    duplicate Buy listings, which is genuine. Re-check this distribution
    before trusting the same construction on a noisier dataset.

    A record in no pair becomes its own singleton entity: on a labeled
    benchmark an unmatched record is a known distinct product, not an unknown
    one. (Abt-Buy has none -- every id appears in the mapping.)
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    known = {record.record_id for record in records}
    for left, right in pairs:
        missing = {left, right} - known
        if missing:
            raise KeyError(f"ground-truth pair references unknown record_id: {sorted(missing)}")
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    members: dict[str, list[str]] = defaultdict(list)
    for record in records:
        members[find(record.record_id)].append(record.record_id)

    # Numbered by each cluster's smallest record_id so the ids are stable from
    # one load to the next. An entity_id that shifted between runs would
    # silently reshuffle any cached entity-level train/test split.
    entity_of: dict[str, str] = {}
    for index, group in enumerate(sorted(members.values(), key=min), start=1):
        entity_id = f"{SOURCE}:e{index:05d}"
        for record_id in group:
            entity_of[record_id] = entity_id

    return [
        record.model_copy(update={"entity_id": entity_of[record.record_id]})
        for record in records
    ]


def load_abt_buy(root: Path) -> list[Record]:
    """The whole benchmark: both sides, with ground-truth `entity_id` set."""
    records = load_products(root)
    return assign_entity_ids(records, load_pairs(root))
