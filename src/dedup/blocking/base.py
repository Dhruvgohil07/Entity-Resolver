"""The contract every blocker satisfies.

A blocker is a filter, not a decision. It answers "is this pair worth
looking at", cheaply, and it is expected to be wrong in one direction all the
time: its output is ~99% non-duplicates by construction. That is the job, not
an error, which is why nothing in this package computes precision, F1 or
accuracy for a blocker (see `evaluate.py`).

`params` is a short human-readable string reproduced verbatim in the report
table, so a row can be tied back to the configuration that produced it. A
blocking table whose rows cannot be reproduced is not a result.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dedup.normalize import NormalizedRecord


@dataclass(frozen=True)
class BlockerRun:
    """One blocker's candidate set, plus what it cost to produce."""

    name: str
    params: str
    keys: np.ndarray  # packed pair keys, sorted and deduplicated

    # None on the union row: build and query times are not additive in any way
    # a reader could act on, and a summed number would invite comparing it
    # against a single blocker's.
    build_seconds: float | None
    query_seconds: float | None

    # Anything the blocker did that silently changes its ceiling, surfaced in
    # the report. A block dropped for being oversized removes true pairs with
    # no other trace -- a lowered ceiling nobody can see is the same class of
    # problem as a hidden denominator.
    warnings: tuple[str, ...] = ()

    @property
    def n_candidates(self) -> int:
        return int(self.keys.size)


class Blocker(Protocol):
    """Emit candidate pairs from normalized records."""

    name: str
    params: str

    def run(self, records: Sequence[NormalizedRecord]) -> BlockerRun: ...


@dataclass
class Elapsed:
    seconds: float = field(default=0.0)


@contextmanager
def timed() -> Iterator[Elapsed]:
    """Measure one phase in seconds.

    Build and query are reported separately because they scale differently
    and the distinction drives real decisions: an ANN index that takes 60s to
    build and 11s to query is fine for a batch pass and unusable for a
    serve-time lookup, and one number would hide that.
    """
    elapsed = Elapsed()
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.seconds = time.perf_counter() - start
