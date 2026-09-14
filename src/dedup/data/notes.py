"""What only `data/` may say about a dataset, handed to the report renderers.

CLAUDE.md: `data/` is the only place that may know a dataset's name. The report
renderers broke that rule in prose rather than in code. They printed "Abt-Buy
is pre-blocked", its baseline F1 of 0.5204 and its Abt-against-Buy description
lengths whatever `--dataset` named, so a report on any other catalog would have
inherited those sentences as false claims about itself.

So each renderer names the passages that make a dataset-specific claim -- its
`PASSAGE_SLOTS` -- and prints generic text there unless the dataset's notes
supply their own. A renderer never branches on a dataset's name; it prints what
it is handed. `tests/test_data_registry.py` rejects a passage under a slot no
renderer declares, because a misspelled slot would silently print the generic
text and drop the dataset's caveat with no error anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetNotes:
    """Report passages and the committed baseline report for one dataset."""

    # Slot id -> markdown printed in place of the renderer's generic text.
    passages: Mapping[str, str] = field(default_factory=dict)

    # The dataset's committed TF-IDF baseline report, relative to the repo root.
    # None when no baseline has been run on this dataset, in which case reports
    # compare against nothing rather than borrowing another dataset's number.
    baseline_report: Path | None = None

    def passage(self, slot: str, default: str, **fields: object) -> str:
        """The dataset's passage for `slot`, or `default`, with any `fields` formatted in.

        Both branches are formatted, so a slot's generic text and a dataset's
        replacement name the same fields.
        """
        text = self.passages.get(slot, default)
        return text.format(**fields) if fields else text


NO_NOTES = DatasetNotes()
