"""Recording a human decision on one review-queue pair.

Kept out of `app.py` so the not-found/already-decided rules are testable
against a bare DuckDB connection, with no `TestClient` needed.
"""

from __future__ import annotations

import duckdb

from dedup.service import store


def record_decision(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    review_id: str,
    *,
    is_match: bool,
    reviewer_id: str | None,
) -> store.ReviewQueueRow:
    """Apply one decision, refusing an unknown pair or a second decision.

    Write-once by design: a decided row may already have been read as a
    training label, so overwriting it silently would mutate a label out from
    under whatever read it. Correcting a mistake is future work (a
    superseding row, or an admin endpoint), not a plain update.
    """
    item = store.get_review_item(conn, run_id, review_id)
    if item is None:
        raise LookupError(f"no review item {review_id!r} in run {run_id!r}")
    if item.is_match is not None:
        raise ValueError(
            f"review item {review_id!r} was already decided ({item.is_match}) at "
            f"{item.decided_at} by {item.reviewer_id!r}"
        )
    store.decide_review(conn, run_id, review_id, is_match=is_match, reviewer_id=reviewer_id)
    decided = store.get_review_item(conn, run_id, review_id)
    assert decided is not None  # just written, under the same connection
    return decided
