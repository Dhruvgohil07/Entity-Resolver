"""FastAPI request/response models. No business logic, no DuckDB -- `app.py`
maps `store.py` rows into these, and `review.py`'s exceptions into status codes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

from dedup.schema import SourceName

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """One pagination wrapper, shared by every listing route."""

    items: list[T]
    total: int
    limit: int
    offset: int


class RecordOut(BaseModel):
    record_id: str
    cluster_id: str
    source: str
    title: str
    brand: str | None
    category: str | None
    price: float | None


class ClusterSummary(BaseModel):
    cluster_id: str
    size: int


class ClusterDetail(BaseModel):
    cluster_id: str
    size: int
    members: list[RecordOut]


class RunSummary(BaseModel):
    run_id: str
    dataset: str
    created_at: datetime
    n_records: int
    n_clusters: int
    n_auto_merge: int
    n_review: int
    n_auto_reject: int


class RunDetail(RunSummary):
    scorer_root: str
    scorer_sha256: str
    clusterer: str
    cost_false_merge: float
    cost_false_split: float
    cost_review: float
    n_candidates: int


class ReviewQueueItem(BaseModel):
    """Both records inline -- what a decision UI needs, with no second request."""

    review_id: str
    left_record: RecordOut
    right_record: RecordOut
    probability: float
    band: str
    created_at: datetime
    is_match: bool | None
    reviewer_id: str | None
    decided_at: datetime | None


class ReviewDecisionRequest(BaseModel):
    is_match: bool
    reviewer_id: str | None = None


class LookupRequest(BaseModel):
    """`schema.Record`'s fields, minus `record_id`/`entity_id` -- the service
    assigns the first (or accepts a caller-supplied one) and always sets the
    second to `None`, since resolving that is the whole question a lookup
    answers."""

    source: SourceName
    title: str
    description: str | None = None
    brand: str | None = None
    category: str | None = None
    price: float | None = None
    record_id: str | None = None  # generated if omitted


class LookupResult(BaseModel):
    decision: Literal["auto_merge", "review", "new_cluster"]
    record: RecordOut
    cluster: ClusterSummary
    secondary_reviews: list[ReviewQueueItem]
