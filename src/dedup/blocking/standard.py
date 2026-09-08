"""Exact-key blocking: records sharing a key become candidates.

The cheapest blocker there is, and on clean vendor codes the most precise. Its
weakness is that it is exact -- `KXTS208W` and `KX-TS208W` are the same
Panasonic phone and do not share a key unless something has already reduced
them to a common form. That reduction is `normalize.model_number_key`, and it
is worth pair completeness 0.3354 -> 0.5349 on Abt-Buy by itself.

A key function takes the whole record sequence rather than one record, because
the most productive key here is corpus-dependent: "a token rare enough to be
discriminative" cannot be computed from a single title.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from dedup.blocking.base import BlockerRun, timed
from dedup.blocking.pairs import pack_block, union
from dedup.normalize import NormalizedRecord, code_key

KeyFunction = Callable[[Sequence[NormalizedRecord]], list[Iterable[str]]]

# A block of m records is m*(m-1)/2 pairs, so one runaway bucket can cost more
# than every other blocker combined -- a single 5,000-record block is 12.5M
# pairs. Such a block is also worthless: a key shared by thousands of records
# is not discriminating between them. Dropping it loses the few true pairs it
# held and saves the quadratic blow-up, which is the right trade at the
# reduction ratio blocking exists to deliver.
DEFAULT_MAX_BLOCK_SIZE = 100

def model_number_keys(records: Sequence[NormalizedRecord]) -> list[Iterable[str]]:
    """The extracted vendor code, in comparison form. Precise, low coverage."""
    return [(record.model_number_key,) if record.model_number_key else () for record in records]


def code_token_keys(records: Sequence[NormalizedRecord]) -> list[Iterable[str]]:
    """Every code-shaped token in the title, not just the one extraction chose.

    `_extract_model_number` commits to a single token per title, and when it
    picks the wrong one the pair is lost. Indexing every token that *looks*
    like a code recovers those: measured pair completeness 0.6708 against
    0.5349 for the single extracted code, at 2,072 candidates.

    Keys go through `normalize.code_key`, the same rule the model-number key
    uses, so the two cannot drift apart and a serve-time index can reproduce
    either without importing from this package.
    """
    out: list[Iterable[str]] = []
    for record in records:
        keys = set()
        for token in record.normalized_title.split():
            stripped = code_key(token)
            if (
                len(stripped) >= 4
                and any(c.isdigit() for c in stripped)
                and any(c.isalpha() for c in stripped)
            ):
                keys.add(stripped)
        out.append(keys)
    return out


def rare_token_keys(df_cutoff: int = 30) -> KeyFunction:
    """Block on tokens appearing in at most `df_cutoff` records.

    Common words ("black", "digital", "cable") put every unrelated product in
    one block; rare ones behave like accidental identifiers. The cutoff is the
    whole design: 3 gives pair completeness 0.5608, 10 gives 0.7084, 30 gives
    0.8694 -- and the candidate count grows with it, which is the trade the
    report exists to show.
    """

    def keys(records: Sequence[NormalizedRecord]) -> list[Iterable[str]]:
        document_frequency: Counter[str] = Counter()
        tokenized = [set(record.normalized_title.split()) for record in records]
        for tokens in tokenized:
            document_frequency.update(tokens)
        return [{t for t in tokens if document_frequency[t] <= df_cutoff} for tokens in tokenized]

    return keys


class StandardBlocker:
    """Group records by exact key; every pair inside a block is a candidate."""

    def __init__(
        self,
        name: str,
        key_function: KeyFunction,
        *,
        params: str = "",
        max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    ) -> None:
        if max_block_size < 2:
            raise ValueError(f"max_block_size must be at least 2, got {max_block_size}")
        self.name = name
        self.params = params or f"max_block={max_block_size}"
        self._key_function = key_function
        self._max_block_size = max_block_size
        self.dropped_blocks = 0

    def run(self, records: Sequence[NormalizedRecord]) -> BlockerRun:
        n = len(records)

        with timed() as build:
            blocks: dict[str, list[int]] = defaultdict(list)
            for index, keys in enumerate(self._key_function(records)):
                for key in keys:
                    blocks[key].append(index)

        with timed() as query:
            self.dropped_blocks = 0
            parts = []
            for members in blocks.values():
                if len(members) < 2:
                    continue
                if len(members) > self._max_block_size:
                    self.dropped_blocks += 1
                    continue
                parts.append(pack_block(np.array(members, dtype=np.int64), n))
            keys_out = union(*parts)

        warnings: tuple[str, ...] = ()
        if self.dropped_blocks:
            message = (
                f"{self.dropped_blocks} block(s) exceeded max_block_size="
                f"{self._max_block_size} and were dropped, so any true pair held only by "
                f"one of them is not in this candidate set"
            )
            warnings = (message,)

        return BlockerRun(
            name=self.name,
            params=self.params,
            keys=keys_out,
            build_seconds=build.seconds,
            query_seconds=query.seconds,
            warnings=warnings,
        )


def default_blockers(max_block_size: int = DEFAULT_MAX_BLOCK_SIZE) -> list[StandardBlocker]:
    """The three exact-key blockers measurement justified on Abt-Buy."""
    return [
        StandardBlocker(
            "standard (model number)",
            model_number_keys,
            params=f"key=model_number_key, max_block={max_block_size}",
            max_block_size=max_block_size,
        ),
        StandardBlocker(
            "standard (code tokens)",
            code_token_keys,
            params=f"key=code-shaped title tokens, max_block={max_block_size}",
            max_block_size=max_block_size,
        ),
        StandardBlocker(
            "standard (rare tokens)",
            rare_token_keys(30),
            params=f"key=title tokens with df<=30, max_block={max_block_size}",
            max_block_size=max_block_size,
        ),
    ]
