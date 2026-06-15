from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class Review(BaseModel):
    review_id: str
    text: str
    rating: int
    published_at: datetime


class ActionIdea(BaseModel):
    title: str
    rationale: str


class Theme(BaseModel):
    rank: int
    name: str
    summary: str
    cluster_size: int
    avg_rating: float
    quotes: List[str]
    action_ideas: List[ActionIdea]


class Cluster(BaseModel):
    """A cluster of related reviews produced by HDBSCAN.

    Fields:
        cluster_id: HDBSCAN label (non-negative integer).
        review_ids: Stable review IDs belonging to this cluster.
        member_indices: Index positions of cluster members in the original review list.
        representative_snippets: Up to 5 verbatim review snippets that best
            represent the cluster (closest to centroid in embedding space).
        cluster_size: Number of reviews in this cluster.
        avg_rating: Mean star rating of reviews in this cluster.
        rating_std: Standard deviation of ratings.
        date_range: (earliest, latest) review dates in this cluster.
        recency_score: Normalised score [0,1] where newer clusters score higher.
        rank_score: Composite score used for ranking (size * rating_weight * recency).
        rank: Ordinal rank after sorting by rank_score (1 = highest).
    """

    cluster_id: int
    review_ids: List[str] = Field(default_factory=list)
    member_indices: List[int] = Field(default_factory=list)
    representative_snippets: List[str] = Field(default_factory=list)
    cluster_size: int = 0
    avg_rating: float = 0.0
    rating_std: float = 0.0
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None
    recency_score: float = 0.0
    rank_score: float = 0.0
    rank: int = 0


class PulseReportStats(BaseModel):
    total_reviews_fetched: int
    reviews_after_dedupe: int
    reviews_clustered: int
    clusters_found: int
    top_themes_selected: int

class PulseReportPeriod(BaseModel):
    start_date: str
    end_date: str
    window_weeks: int

class AudienceNotes(BaseModel):
    product: str
    support: str
    leadership: str

class PulseReport(BaseModel):
    product: str
    iso_week: str
    period: PulseReportPeriod
    stats: PulseReportStats
    themes: List[Theme]
    audience_notes: AudienceNotes
    generated_at: str

class DocDeliveryInfo(BaseModel):
    document_id: str
    heading_text: str
    heading_anchor: str
    revision_id: str
    appended: bool

class EmailDeliveryInfo(BaseModel):
    mode: str
    message_id: str
    recipients: List[str]
    sent_at: str

class DeliveryRecord(BaseModel):
    doc: Optional[DocDeliveryInfo] = None
    email: Optional[EmailDeliveryInfo] = None

class IngestRecord(BaseModel):
    review_count: int
    mcp_fetch_at: str

class AnalysisRecord(BaseModel):
    model: str
    embedding_model: str
    token_usage: Dict[str, int]

class StageStatus(str, Enum):
    """Status of a single pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunState(str, Enum):
    """Top-level run status."""
    PENDING = "pending"
    INGESTING = "ingesting"
    ANALYZING = "analyzing"
    SUMMARIZING = "summarizing"
    RENDERING = "rendering"
    DELIVERING = "delivering"
    COMPLETED = "completed"
    FAILED = "failed"


class StageRecord(BaseModel):
    """Checkpoint for a single pipeline stage."""
    status: str = StageStatus.PENDING.value
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Ordered pipeline stages (must match orchestrator execution order)
PIPELINE_STAGES: List[str] = [
    "ingest",
    "analyze",
    "summarize",
    "render",
    "deliver",
]

# Map stage name -> RunState set while that stage is running
STAGE_TO_STATE: Dict[str, str] = {
    "ingest": RunState.INGESTING.value,
    "analyze": RunState.ANALYZING.value,
    "summarize": RunState.SUMMARIZING.value,
    "render": RunState.RENDERING.value,
    "deliver": RunState.DELIVERING.value,
}


class RunRecord(BaseModel):
    """Persisted record for a single (product, iso_week) run.

    Stored as JSON at ``data/runs/{product}/{iso_week}/run.json``.
    Acts as the sole source of truth for idempotency and audit.
    """
    # --- Identity ---
    run_id: str
    product: str
    iso_week: str
    status: str = RunState.PENDING.value

    # --- Timestamps ---
    started_at: str
    completed_at: Optional[str] = None
    updated_at: str = ""

    # --- Stage checkpoints ---
    stages: Dict[str, StageRecord] = Field(default_factory=dict)

    # --- Pipeline outputs ---
    delivery: Optional[DeliveryRecord] = None
    ingest: Optional[IngestRecord] = None
    analysis: Optional[AnalysisRecord] = None

    # --- Error ---
    error: Optional[Dict[str, Any]] = None

    def is_completed(self) -> bool:
        return self.status == RunState.COMPLETED.value

    def is_failed(self) -> bool:
        return self.status == RunState.FAILED.value

    def last_completed_stage(self) -> Optional[str]:
        """Return the name of the last stage that completed successfully."""
        for stage in reversed(PIPELINE_STAGES):
            sr = self.stages.get(stage)
            if sr and sr.status == StageStatus.COMPLETED.value:
                return stage
        return None

    def next_stage(self) -> Optional[str]:
        """Return the next stage to execute (first non-completed stage)."""
        for stage in PIPELINE_STAGES:
            sr = self.stages.get(stage)
            if sr is None or sr.status != StageStatus.COMPLETED.value:
                return stage
        return None
