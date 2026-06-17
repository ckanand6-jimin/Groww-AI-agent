"""End-to-end run orchestration — Phase 7.

Wires all pipeline stages (ingest -> analyze -> summarize -> render -> deliver)
into a single idempotent run with:

  - **Ledger-based idempotency**: completed runs are no-ops.
  - **Resume from failure**: failed runs pick up from the last failed stage.
  - **Partial delivery recovery**: doc appended but email failed -> skip doc.
  - **Stage checkpoints**: ledger persisted after every stage transition.
  - **File locking**: concurrent runs for the same (product, iso_week) are rejected.
  - **Dry-run mode**: runs pipeline stages 1-4 but skips delivery.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pulse.audit.ledger import Ledger
from pulse.audit.lock import LockError, RunLock
from pulse.config import ProductConfig, load_product_config, get_date_window_from_iso_week, validate_runtime_config
from pulse.models.models import (
    AnalysisRecord,
    DeliveryRecord,
    IngestRecord,
    PIPELINE_STAGES,
    PulseReport,
    RunRecord,
    RunState,
    StageRecord,
    StageStatus,
    STAGE_TO_STATE,
)

logger = logging.getLogger(__name__)

# IST timezone for timestamps
_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> str:
    """Current datetime as ISO-8601 string in IST."""
    return datetime.now(_IST).isoformat()


def _current_iso_week() -> str:
    """Current ISO week string, e.g. '2026-W24'."""
    now = datetime.now(_IST)
    return f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class OrchestrationError(Exception):
    """Raised when a pipeline stage fails during orchestration."""

    def __init__(
        self, stage: str, message: str, cause: Optional[Exception] = None
    ) -> None:
        self.stage = stage
        self.message = message
        self.cause = cause
        super().__init__(f"[{stage}] {message}")


# ---------------------------------------------------------------------------
# Stage function types (for dependency injection in tests)
# ---------------------------------------------------------------------------

# ingest(cache_dir, iso_week, product) -> (reviews, stats_dict)
IngestFn = Callable[..., Tuple[List, Dict[str, Any]]]

# analyze(reviews, ...) -> (clusters, stats_dict)
AnalyzeFn = Callable[..., Tuple[List, Dict[str, Any]]]

# summarize(clusters, ...) -> (PulseReport, token_usage_dict)
SummarizeFn = Callable[..., Tuple[PulseReport, Dict[str, int]]]

# render(report, ...) -> dict
RenderFn = Callable[..., Dict[str, Any]]


# ---------------------------------------------------------------------------
# Default cache directory resolver
# ---------------------------------------------------------------------------


def _resolve_cache_dir(product: str, iso_week: str) -> str:
    """Resolve the review cache directory for a given (product, iso_week).

    Convention: ``data/cache/{product}/{end_date_iso}/``

    Respects ``PULSE_DATA_DIR`` env var for overriding the base data directory
    (useful in CI where the CWD may differ from the repo root).
    """
    _, end_date = get_date_window_from_iso_week(iso_week, window_weeks=10)
    end_date_str = end_date.strftime("%Y-%m-%d")

    data_dir = os.environ.get("PULSE_DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "cache", product, end_date_str)

    # Repo root: 3 levels up from pulse-agent/src/pulse/
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "data" / "cache" / product / end_date_str)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(
    *,
    product: str = "groww",
    iso_week: Optional[str] = None,
    dry_run: bool = False,
    email_mode: Optional[str] = None,
    config: Optional[ProductConfig] = None,
    ledger_base_dir: Optional[Path] = None,
    # --- Dependency injection (tests) ---
    ingest_fn: Optional[IngestFn] = None,
    analyze_fn: Optional[AnalyzeFn] = None,
    summarize_fn: Optional[SummarizeFn] = None,
    render_fn: Optional[RenderFn] = None,
    deliver_fn: Optional[Callable] = None,
) -> RunRecord:
    """Execute the full pipeline for one ISO week.

    Args:
        product:        Product identifier (default: ``"groww"``).
        iso_week:       ISO week string (default: current week).
        dry_run:        If True, skip the deliver stage.
        email_mode:     Override for ``delivery.email.default_mode``.
        config:         Pre-loaded ``ProductConfig`` (default: load from file).
        ledger_base_dir: Override for the ledger base directory (tests).
        ingest_fn:      Override for the ingest stage function.
        analyze_fn:     Override for the analyze stage function.
        summarize_fn:   Override for the summarize stage function.
        render_fn:      Override for the render stage function.
        deliver_fn:     Override for the deliver stage function (async).

    Returns:
        The final ``RunRecord`` (status = completed or failed).

    Raises:
        OrchestrationError: If a pipeline stage fails.
        LockError:          If a concurrent run is already in progress.
    """
    # --- Resolve parameters ---
    resolved_week = iso_week or _current_iso_week()
    cfg = config or load_product_config()

    # --- Runtime config validation (catches placeholders before pipeline starts) ---
    config_errors = validate_runtime_config(cfg)
    if config_errors:
        raise OrchestrationError(
            stage="config",
            message="; ".join(config_errors),
        )

    ledger = Ledger(base_dir=ledger_base_dir)
    run_dir = ledger.run_dir(product, resolved_week)

    # --- Acquire lock ---
    with RunLock(run_dir):
        return await _execute_pipeline(
            product=product,
            iso_week=resolved_week,
            dry_run=dry_run,
            email_mode=email_mode,
            cfg=cfg,
            ledger=ledger,
            ingest_fn=ingest_fn,
            analyze_fn=analyze_fn,
            summarize_fn=summarize_fn,
            render_fn=render_fn,
            deliver_fn=deliver_fn,
        )


async def _execute_pipeline(
    *,
    product: str,
    iso_week: str,
    dry_run: bool,
    email_mode: Optional[str],
    cfg: ProductConfig,
    ledger: Ledger,
    ingest_fn: Optional[IngestFn],
    analyze_fn: Optional[AnalyzeFn],
    summarize_fn: Optional[SummarizeFn],
    render_fn: Optional[RenderFn],
    deliver_fn: Optional[Callable],
) -> RunRecord:
    """Internal pipeline execution (called within lock context)."""

    # --- Load or create RunRecord ---
    record = ledger.load(product, iso_week)
    is_resume = False

    if record is not None:
        if record.is_completed():
            logger.info(
                "Run already completed for %s %s (run_id=%s). Skipping.",
                product, iso_week, record.run_id,
            )
            return record
        if record.is_failed():
            logger.info(
                "Resuming failed run for %s %s (last stage: %s).",
                product, iso_week, record.last_completed_stage(),
            )
            is_resume = True
        # Clear previous error for retry
        record.error = None
    else:
        record = RunRecord(
            run_id=str(uuid.uuid4()),
            product=product,
            iso_week=iso_week,
            status=RunState.PENDING.value,
            started_at=_now_ist(),
            updated_at=_now_ist(),
        )

    # ------------------------------------------------------------------
    # Stage execution loop
    # ------------------------------------------------------------------
    reviews = None
    clusters = None
    report = None
    rendered = None
    ingest_stats: Dict[str, Any] = {}
    analyze_stats: Dict[str, Any] = {}

    for stage in PIPELINE_STAGES:
        # NOTE: We always re-run all stages (even on resume).  Computation
        # stages (ingest/analyze/summarize/render) are fast and idempotent.
        # Only the deliver stage skips external I/O via existing_delivery.
        # This ensures downstream stages always have the variables they need.

        # --- Stage start checkpoint ---
        stage_start = time.monotonic()
        record.status = STAGE_TO_STATE[stage]
        record.stages[stage] = StageRecord(
            status=StageStatus.RUNNING.value,
            started_at=_now_ist(),
        )
        record.updated_at = _now_ist()
        ledger.save(record)

        try:
            # --- Execute stage ---
            if stage == "ingest":
                reviews, ingest_stats = _run_ingest(
                    product, iso_week, cfg, ingest_fn,
                )
                record.ingest = IngestRecord(
                    review_count=ingest_stats.get("final_count", 0),
                    mcp_fetch_at=ingest_stats.get("ingested_at", _now_ist()),
                )

            elif stage == "analyze":
                clusters, analyze_stats = _run_analyze(reviews, cfg, analyze_fn)
                record.analysis = AnalysisRecord(
                    model=cfg.analysis.llm_model,
                    embedding_model=cfg.analysis.embedding_model,
                    token_usage={"prompt_tokens": 0, "completion_tokens": 0},
                )

            elif stage == "summarize":
                report, token_usage = _run_summarize(
                    clusters, iso_week, cfg, summarize_fn,
                    ingest_stats=ingest_stats,
                    analyze_stats=analyze_stats,
                )
                if record.analysis:
                    record.analysis.token_usage = token_usage

            elif stage == "render":
                rendered = _run_render(report, cfg, render_fn)

            elif stage == "deliver":
                if dry_run:
                    record.stages[stage] = StageRecord(
                        status=StageStatus.SKIPPED.value,
                        started_at=_now_ist(),
                        completed_at=_now_ist(),
                        duration_ms=0,
                        metadata={"reason": "dry_run"},
                    )
                    continue
                delivery_result = await _run_deliver(
                    report, cfg, email_mode, record.delivery, deliver_fn,
                )
                record.delivery = delivery_result

            # --- Stage success checkpoint ---
            elapsed_ms = int((time.monotonic() - stage_start) * 1000)
            record.stages[stage] = StageRecord(
                status=StageStatus.COMPLETED.value,
                started_at=record.stages[stage].started_at,
                completed_at=_now_ist(),
                duration_ms=elapsed_ms,
                metadata=_stage_metadata(stage, locals()),
            )
            record.updated_at = _now_ist()
            ledger.save(record)
            logger.info(
                "Stage '%s' completed in %dms for %s %s",
                stage, elapsed_ms, product, iso_week,
            )

        except Exception as exc:
            # --- Stage failure checkpoint ---
            elapsed_ms = int((time.monotonic() - stage_start) * 1000)
            record.stages[stage] = StageRecord(
                status=StageStatus.FAILED.value,
                started_at=record.stages[stage].started_at,
                completed_at=_now_ist(),
                duration_ms=elapsed_ms,
                metadata={"error": str(exc)[:500]},
            )
            record.status = RunState.FAILED.value
            record.error = {
                "stage": stage,
                "message": str(exc),
                "type": type(exc).__name__,
            }
            record.updated_at = _now_ist()
            ledger.save(record)
            raise OrchestrationError(
                stage=stage,
                message=str(exc),
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # All stages passed — mark completed
    # ------------------------------------------------------------------
    record.status = RunState.COMPLETED.value
    record.completed_at = _now_ist()
    record.updated_at = _now_ist()
    ledger.save(record)

    logger.info(
        "Run completed for %s %s (run_id=%s)",
        product, iso_week, record.run_id,
    )
    return record


# ---------------------------------------------------------------------------
# Stage runners (isolated for testability)
# ---------------------------------------------------------------------------


def _run_ingest(
    product: str,
    iso_week: str,
    cfg: ProductConfig,
    fn: Optional[IngestFn],
) -> Tuple[List, Dict[str, Any]]:
    """Execute the ingest stage."""
    cache_dir = _resolve_cache_dir(product, iso_week)

    if fn:
        return fn(cache_dir, iso_week, product)

    from pulse.ingest.adapter import ingest_from_cache

    # Compute date window for auto-fetch when cache is missing (CI / first run)
    start_date_dt, end_date_dt = get_date_window_from_iso_week(
        iso_week, cfg.review_window_weeks,
    )

    return ingest_from_cache(
        cache_dir,
        iso_week,
        product,
        auto_fetch=True,
        app_id=cfg.play_store.app_id,
        start_date=start_date_dt,
        end_date=end_date_dt,
    )


def _run_analyze(
    reviews: Optional[List],
    cfg: ProductConfig,
    fn: Optional[AnalyzeFn],
) -> Tuple[List, Dict[str, Any]]:
    """Execute the analyze stage."""
    if not reviews:
        raise ValueError("No reviews available for analysis.")
    if fn:
        return fn(reviews)

    from pulse.analysis import analyze

    return analyze(
        reviews,
        model_name=cfg.analysis.embedding_model,
        top_k=cfg.analysis.max_themes,
    )


def _run_summarize(
    clusters: Optional[List],
    iso_week: str,
    cfg: ProductConfig,
    fn: Optional[SummarizeFn],
    *,
    ingest_stats: Optional[Dict[str, Any]] = None,
    analyze_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[PulseReport, Dict[str, int]]:
    """Execute the summarize stage."""
    if not clusters:
        raise ValueError("No clusters available for summarization.")

    start_date_dt, end_date_dt = get_date_window_from_iso_week(
        iso_week, cfg.review_window_weeks,
    )

    if fn:
        return fn(clusters)

    from pulse.summarize import summarize

    # Propagate stats from earlier stages so the report displays real counts.
    ingest_stats = ingest_stats or {}
    analyze_stats = analyze_stats or {}

    return summarize(
        clusters,
        model=cfg.analysis.llm_model,
        product="groww",
        iso_week=iso_week,
        start_date=start_date_dt.strftime("%Y-%m-%d"),
        end_date=end_date_dt.strftime("%Y-%m-%d"),
        window_weeks=cfg.review_window_weeks,
        total_reviews_fetched=ingest_stats.get("fetched", 0),
        reviews_after_dedupe=ingest_stats.get("final_count", 0),
        reviews_clustered=analyze_stats.get("reviews_clustered", 0)
            or ingest_stats.get("final_count", 0),
        clusters_found=analyze_stats.get("clusters_found", 0)
            or analyze_stats.get("total_clusters", 0),
    )


def _run_render(
    report: Optional[PulseReport],
    cfg: ProductConfig,
    fn: Optional[RenderFn],
) -> Dict[str, Any]:
    """Execute the render stage."""
    if not report:
        raise ValueError("No PulseReport available for rendering.")
    if fn:
        return fn(report)

    from pulse.render import render

    doc_url = (
        f"https://docs.google.com/document/d/"
        f"{cfg.delivery.google_doc.document_id}/edit"
    )
    return render(
        report,
        doc_url=doc_url,
        recipients=cfg.delivery.email.stakeholders,
    )


async def _run_deliver(
    report: Optional[PulseReport],
    cfg: ProductConfig,
    email_mode: Optional[str],
    existing_delivery: Optional[DeliveryRecord],
    fn: Optional[Callable],
) -> DeliveryRecord:
    """Execute the deliver stage."""
    if not report:
        raise ValueError("No PulseReport available for delivery.")
    if fn:
        return await fn(report, cfg, email_mode, existing_delivery)

    from pulse.delivery.deliver import deliver

    return await deliver(
        report,
        cfg,
        existing_delivery=existing_delivery,
        email_mode_override=email_mode,
    )


def _stage_metadata(stage: str, local_vars: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stage-specific metadata from local variables."""
    if stage == "ingest":
        stats = local_vars.get("ingest_stats", {})
        return {"review_count": stats.get("final_count", 0)}
    elif stage == "analyze":
        stats = local_vars.get("analyze_stats", {})
        return {
            "reviews_embedded": stats.get("reviews_embedded", 0),
            "clusters_found": stats.get("clusters_found", 0),
        }
    elif stage == "summarize":
        usage = local_vars.get("token_usage", {})
        return {
            "themes_generated": len(
                local_vars.get("report", PulseReport.__new__(PulseReport)).themes
            ) if local_vars.get("report") else 0,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    elif stage == "render":
        r = local_vars.get("rendered", {})
        return {"heading_text": r.get("heading_text", "")}
    elif stage == "deliver":
        dr = local_vars.get("delivery_result")
        if isinstance(dr, DeliveryRecord):
            return {
                "doc_appended": dr.doc.appended if dr.doc else False,
                "email_mode": dr.email.mode if dr.email else "N/A",
            }
        return {}
    return {}
