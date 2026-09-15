"""Persisted, incrementally queryable inverted index over two of `standard.py`'s
key functions -- `service/`'s online-lookup half of exact-key blocking.

Only `model_number_keys` and `code_token_keys`: both compute a record's keys
from that record alone. `rare_token_keys` needs corpus-wide document
frequency, which every insert changes -- a record whose token was rare at
df=29 becomes common at df=31, so every *existing* record's keys could
change on every insert. That is not an index that can be incrementally
maintained; it would need a full rebuild on every insert, defeating the
point of having one. Excluded, not deferred quietly: a real, unmeasured
recall gap against the batch path (CLAUDE.md, Open questions).

No vocabulary-freeze problem here, unlike `blocking/ann.py`'s `AnnIndex` --
both key functions are computed fresh per record, so a genuinely new code
value works immediately on the very next lookup.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from dedup.blocking.standard import code_token_keys, model_number_keys
from dedup.normalize import NormalizedRecord

MANIFEST_FILE = "manifest.json"

# name -> the key function it indexes. Both take Sequence[NormalizedRecord] and
# return one Iterable[str] of keys per record, computed independently of the
# rest of the corpus -- the property that makes them safe to index online.
KEY_FUNCTIONS = {
    "model_number": model_number_keys,
    "code_token": code_token_keys,
}


def _digest(indexes: dict[str, dict[str, list[str]]]) -> str:
    return hashlib.sha256(json.dumps(indexes, sort_keys=True).encode("utf-8")).hexdigest()


class InvertedIndex:
    """`dict[key_function_name, dict[key, list[record_id]]]`, incrementally
    maintained. Persistence mirrors `model.train.PairScorer.save`/`.load` and
    `blocking.ann.AnnIndex.save`/`.load`: a directory, a manifest carrying an
    `artifact_sha256`, refuse on hash mismatch or a missing manifest -- the
    same pattern, JSON instead of pickle since nothing here is a scikit-learn
    object and a plain `dict[str, list[str]]` is JSON-native and
    human-inspectable (useful for debugging a bad candidate set).
    """

    def __init__(self, indexes: dict[str, dict[str, list[str]]]) -> None:
        self._indexes = indexes

    @classmethod
    def build(
        cls, records: Sequence[NormalizedRecord], record_ids: Sequence[str]
    ) -> InvertedIndex:
        if len(records) != len(record_ids):
            raise ValueError(
                f"records ({len(records)}) and record_ids ({len(record_ids)}) must match"
            )
        indexes: dict[str, dict[str, list[str]]] = {name: {} for name in KEY_FUNCTIONS}
        for name, key_function in KEY_FUNCTIONS.items():
            for record_id, keys in zip(record_ids, key_function(records)):
                for key in keys:
                    indexes[name].setdefault(key, []).append(record_id)
        return cls(indexes)

    def query_one(self, record: NormalizedRecord) -> set[str]:
        """Union of `record_id`s sharing any key with `record`, across both
        key functions."""
        candidates: set[str] = set()
        for name, key_function in KEY_FUNCTIONS.items():
            (keys,) = key_function([record])
            for key in keys:
                candidates.update(self._indexes[name].get(key, ()))
        return candidates

    def add(self, record: NormalizedRecord, record_id: str) -> None:
        """Grow every key function's index by one record, so a later
        `query_one` finds it."""
        for name, key_function in KEY_FUNCTIONS.items():
            (keys,) = key_function([record])
            for key in keys:
                self._indexes[name].setdefault(key, []).append(record_id)

    def save(self, root: Path) -> str:
        """Write this index to `root`, return the hash `load` will verify."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        digest = _digest(self._indexes)
        for name, index in self._indexes.items():
            (root / f"{name}.json").write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        manifest = {
            "artifact_sha256": digest,
            "key_functions": sorted(self._indexes),
            "created_at": datetime.now(UTC).isoformat(),
        }
        (root / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return digest

    @classmethod
    def load(cls, root: Path) -> InvertedIndex:
        """The inverse of `save`, refused if the artifact no longer matches
        its manifest."""
        root = Path(root)
        manifest_path = root / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{root} has no {MANIFEST_FILE}, so it is not a saved InvertedIndex. Save one "
                f"with InvertedIndex.save or `python -m dedup.service.build_index` "
                f"(CLAUDE.md > Commands)."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        indexes = {
            name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
            for name in manifest["key_functions"]
        }
        digest = _digest(indexes)
        if digest != manifest["artifact_sha256"]:
            raise ValueError(
                f"{root} does not match the hash in its manifest -- it was edited, truncated, "
                f"or regenerated without its manifest"
            )
        return cls(indexes)
