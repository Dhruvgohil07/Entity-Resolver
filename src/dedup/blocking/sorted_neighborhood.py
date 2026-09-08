"""Sorted-neighborhood blocking: sort on a key, slide a window over the order.

The point is to survive a bad key. Exact-key blocking is all-or-nothing --
"sony psl x350h" and "sony pslx350h" share no key and are never compared.
Sorting puts them adjacent anyway, so a window of `w` catches pairs that
differ in a character, as long as they differ *late* in the string.

That last clause is the weakness, and it is why this blocker measures poorly
here (pair completeness 0.6190 at w=20, the lowest of the five, for the
largest candidate count of any single blocker). A title whose first token
differs -- "Bose 161WH" against "Boss 161 Speaker" -- sorts far apart no
matter how large the window grows. Widening `w` costs candidates linearly and
buys very little, so this blocker earns its place through the union or not at
all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from dedup.blocking.base import BlockerRun, timed
from dedup.blocking.pairs import pack
from dedup.normalize import NormalizedRecord

SortKey = Callable[[NormalizedRecord], str]

DEFAULT_WINDOW = 20


def title_sort_key(record: NormalizedRecord) -> str:
    return record.normalized_title


def model_then_title_sort_key(record: NormalizedRecord) -> str:
    """Sort by vendor code where one exists, falling back to the title.

    Records with a code cluster by code; records without stay in title order
    rather than piling up under a single empty key.
    """
    return record.model_number_key or record.normalized_title


class SortedNeighborhoodBlocker:
    def __init__(
        self,
        name: str = "sorted_neighborhood",
        sort_key: SortKey = title_sort_key,
        *,
        window: int = DEFAULT_WINDOW,
        params: str = "",
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        self.name = name
        self.params = params or f"w={window}"
        self._sort_key = sort_key
        self._window = window

    def run(self, records: Sequence[NormalizedRecord]) -> BlockerRun:
        n = len(records)

        with timed() as build:
            # Ties broken by index so the order -- and therefore the candidate
            # set -- does not depend on the order the loader emitted rows in.
            order = np.array(
                sorted(range(n), key=lambda i: (self._sort_key(records[i]), i)), dtype=np.int64
            )

        with timed() as query:
            lefts, rights = [], []
            for offset in range(1, self._window):
                if offset >= n:
                    break
                lefts.append(order[:-offset])
                rights.append(order[offset:])
            if lefts:
                keys = pack(np.concatenate(lefts), np.concatenate(rights), n)
            else:
                keys = np.empty(0, dtype=np.int64)

        return BlockerRun(
            name=self.name,
            params=self.params,
            keys=keys,
            build_seconds=build.seconds,
            query_seconds=query.seconds,
        )
