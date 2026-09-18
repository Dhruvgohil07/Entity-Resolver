"""Synthetic catalogs on disk: `records.jsonl` plus `manifest.json`, written and read back.

`synth/` generates a catalog; this module is how it becomes a dataset like any
other -- registered in `DATASETS`, loaded by name, handed onward as `Record`s. It
is materialized rather than regenerated on every load, so the five report CLIs
read the same bytes, and a catalog can be inspected, hashed and diffed.

**JSONL, not CSV.** The schema distinguishes a field a source never had (`None`)
from one it had and left empty (`""`), and `features/missingness.py` keys off
exactly that. A CSV cell cannot say which it holds; pydantic's JSON round-trips
both.

**The manifest is required, and its hash is checked.** A catalog edited, truncated
or regenerated with other settings after its reports were written would otherwise
load silently and describe a different dataset than the one measured.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from dedup.data.notes import DatasetNotes
from dedup.eval.splits import DEFAULT_SEED, DEFAULT_TEST_FRACTION
from dedup.schema import Record

RECORDS_FILE = "records.jsonl"
MANIFEST_FILE = "manifest.json"


def write_catalog(root: Path, records: Sequence[Record], manifest: Mapping[str, object]) -> str:
    """Write `records` and a manifest carrying their count and SHA-256; return the hash."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = "".join(record.model_dump_json() + "\n" for record in records).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    (root / RECORDS_FILE).write_bytes(payload)
    body = {**manifest, "n_records": len(records), "records_sha256": digest}
    (root / MANIFEST_FILE).write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return digest


def read_manifest(root: Path) -> dict[str, object]:
    path = Path(root) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{root} has no {MANIFEST_FILE}, so it is not a generated synthetic catalog. "
            f"Generate one with `python -m dedup.synth.generate` (CLAUDE.md > Commands)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_synthetic(root: Path, *, seed_dataset: str | None = None) -> list[Record]:
    """A generated catalog, refused if it no longer matches its manifest.

    `seed_dataset`, when given, must match the manifest's recorded `seed_dataset` --
    otherwise nothing ties a registry entry (e.g. `synth-20k`) to the benchmark it is
    supposed to be derived from, and a catalog seeded from the wrong one would load
    silently. This is a cheap self-consistency check against the manifest's own
    claim; it does not verify that claim against the seed dataset's real records --
    `synth.generate.verify_seed_provenance` does that, deliberately not from here,
    so a normal load never needs the seed dataset present on disk.
    """
    root = Path(root)
    manifest = read_manifest(root)
    if seed_dataset is not None and manifest.get("seed_dataset") != seed_dataset:
        raise ValueError(
            f"{root} was seeded from {manifest.get('seed_dataset')!r}, not {seed_dataset!r} "
            f"as this registry entry expects"
        )
    report_split = {"side": "train", "test_fraction": DEFAULT_TEST_FRACTION, "seed": DEFAULT_SEED}
    if manifest.get("seed_split") != report_split:
        raise ValueError(
            f"{root} was seeded from {manifest.get('seed_split')!r}, not the train side of the "
            f"split reports are scored on ({report_split!r}); its seeds could include records "
            f"those reports test on"
        )
    payload = (root / RECORDS_FILE).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["records_sha256"]:
        raise ValueError(
            f"{root / RECORDS_FILE} does not match the hash in its manifest -- it was edited, "
            f"truncated or regenerated without its manifest"
        )
    records = [Record.model_validate_json(line) for line in payload.decode("utf-8").splitlines()]
    if len(records) != manifest["n_records"]:
        raise ValueError(f"{root} holds {len(records)} records; its manifest says {manifest['n_records']}")

    foreign = sorted({record.source for record in records} - {"synthetic"})
    if foreign:
        raise ValueError(f"{root} is a synthetic catalog but holds records from {foreign}")
    unlabelled = [r.record_id for r in records if r.entity_id is None or r.split_group is None]
    if unlabelled:
        raise ValueError(
            f"{len(unlabelled)} record(s) lack entity_id or split_group, e.g. {unlabelled[:3]}; "
            f"without the group a split could put one seed's siblings on both sides"
        )
    return records


# ---------------------------------------------------------------------------
# What reports may say about the registered synthetic catalogs. Only what holds by
# construction is stated here; measured properties live in reports/synth/realism.md.
# ---------------------------------------------------------------------------

_PROVENANCE = """\
- **This catalog is synthetic.** `synth/` derived it from Abt-Buy's train split, as its
  `manifest.json` records, so no record of Abt-Buy's test split reached it and its ground
  truth is exact by construction. How its duplicates compare with real ones is measured in
  `reports/synth/realism.md` — read that before trusting any number here."""

_BLOCKING_TUNING = """\
- **The blocker parameters were tuned on Abt-Buy, not on this catalog.** The window,
  neighbour count and document-frequency cutoff were swept over Abt-Buy and applied here
  unchanged, so these figures measure how those settings transfer, not what tuning on this
  catalog would reach."""

_FEATURES_TUNING = """\
- **The blocker parameters were tuned on Abt-Buy, not on this catalog**, so the PC figures
  inherited here measure how those settings transfer rather than a best-of-sweep ceiling."""

_PASSAGES = {
    "blocking.provenance": _PROVENANCE,
    "blocking.tuning": _BLOCKING_TUNING,
    "features.tuning": _FEATURES_TUNING,
}

# The mid-scale catalog is small enough for the exhaustive baseline with a raised
# similarity floor, so it registers where that report is written. The 200k catalog
# is not, and registers none: it is compared against nothing.
NOTES_20K = DatasetNotes(passages=_PASSAGES, baseline_report=Path("reports/synth/baseline_tfidf.md"))
NOTES_200K = DatasetNotes(passages=_PASSAGES)
