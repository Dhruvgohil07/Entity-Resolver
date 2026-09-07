"""Dataset loaders: raw source files -> canonical `Record` objects.

One module per dataset. Each benchmark has its own column names, encoding,
price spelling and ground-truth format, and keeping them apart is what stops
those quirks leaking into a shared loader full of per-source branching --
the same reason `blocking/` is a package rather than one module.

Every loader here has the same contract: it returns `Record` objects with
`entity_id` populated from the dataset's ground truth, so nothing downstream
needs to know which benchmark a record came from.

`DATASETS` is the registry that keeps it that way. A CLI that has to name a
dataset ("--dataset abt-buy") resolves it here and passes `Record` objects
onward, so `eval/`, `blocking/` and `model/` never import a loader module or
learn a benchmark's name -- CLAUDE.md's rule that `data/` is the only place
that may know one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dedup.data.abt_buy import load_abt_buy
from dedup.schema import Record


@dataclass(frozen=True)
class DatasetSpec:
    """A loader plus where its files live when nobody says otherwise."""

    name: str
    default_root: Path
    load: Callable[[Path], list[Record]]


DATASETS: dict[str, DatasetSpec] = {
    "abt-buy": DatasetSpec(
        name="abt-buy",
        # Relative to the repo root, matching the download snippet in
        # CLAUDE.md's Data section. data/ is gitignored, so this path is
        # empty in a fresh clone and `load_dataset` says so plainly.
        default_root=Path("data/raw/abt-buy"),
        load=load_abt_buy,
    ),
}


def load_dataset(name: str, root: Path | None = None) -> list[Record]:
    """Load a registered benchmark by name, with `entity_id` populated."""
    try:
        spec = DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}") from None

    root = Path(spec.default_root if root is None else root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"dataset {name!r} not found at {root} -- data/ is gitignored, so a fresh "
            f"clone starts empty. See CLAUDE.md > Data for the download commands."
        )
    return spec.load(root)


__all__ = ["DATASETS", "DatasetSpec", "load_dataset"]
