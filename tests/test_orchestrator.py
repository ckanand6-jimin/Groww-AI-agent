"""Phase 7 — Orchestrator unit tests.

All pipeline stages are mocked via dependency injection.  Uses real
``Ledger`` and ``RunLock`` (with ``tmp_path``) to test actual file I/O.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pulse.audit.ledger import Ledger
from pulse.audit.lock import LockError, RunLock
from pulse.config import (
    AnalysisConfig,
    DeliveryConfig,
    EmailConfig,
    EmailMode,
    GoogleDocConfig,
    PlayStoreConfig,
    ProductConfig,
    ScheduleConfig,
)
from pulse.models.models import (
    ActionIdea,
    AudienceNotes,
    DeliveryRecord,
    DocDeliveryInfo,
    EmailDeliveryInfo,
    PIPELINE_STAGES,
    PulseReport,
    PulseReportPeriod,
    PulseReportStats,
    RunRecord,
    RunState,
    StageRecord,
    StageStatus,
    Theme,
)
from pulse.orchestrator import OrchestrationError, run_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def product_config() -> ProductConfig:
    return ProductConfig(
        product="groww",
        display_name="Groww",
        play_store=PlayStoreConfig(app_id="com.nextbillion.groww"),
        review_window_weeks=10,
        analysis=AnalysisConfig(
            max_themes=3,
            embedding_model="BAAI/bge-small-en-v1.5",
            llm_model="llama-3.3-70b-versatile",
            max_tokens_per_run=80000,
        ),
        delivery=DeliveryConfig(
            google_doc=GoogleDocConfig(
                document_id="test-doc-id",
                document_title="Test Doc",
            ),
            email=EmailConfig(
                stakeholders=["test@example.com"],
                default_mode=EmailMode.DRAFT,
            ),
        ),
        schedule=ScheduleConfig(timezone="Asia/Kolkata", cron="0 6 * * 1"),
    )


@pytest.fixture
def fixture_report() -> PulseReport:
    return PulseReport(
        product="groww",
        iso_week="2026-W23",
        period=PulseReportPeriod(
            start_date="2026-03-31",
            end_date="2026-06-08",
            window_weeks=10,
        ),
        stats=PulseReportStats(
            total_reviews_fetched=1240,
            reviews_after_dedupe=1180,
            reviews_clustered=1100,
            clusters_found=18,
            top_themes_selected=3,
        ),
        themes=[
            Theme(
                rank=1,
                name="App performance",
                summary="Lag and crashes.",
                cluster_size=142,
                avg_rating=2.1,
                quotes=["App freezes at market open."],
                action_ideas=[
                    ActionIdea(title="Fix perf", rationale="Scale infra."),
                ],
            ),
        ],
        audience_notes=AudienceNotes(
            product="Fix performance.",
            support="Login tickets.",
            leadership="Retention risk.",
        ),
        generated_at="2026-06-11T06:30:00+05:30",
    )


@pytest.fixture
def fixture_delivery() -> DeliveryRecord:
    return DeliveryRecord(
        doc=DocDeliveryInfo(
            document_id="test-doc-id",
            heading_text="Groww - Week 2026-W23",
            heading_anchor="",
            revision_id="rev-001",
            appended=True,
        ),
        email=EmailDeliveryInfo(
            mode="draft",
            message_id="msg-001",
            recipients=["test@example.com"],
            sent_at="",
        ),
    )


def _make_stage_fns(
    fixture_report: PulseReport,
    fixture_delivery: DeliveryRecord,
    fail_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Create mock stage functions.  Optionally fail at a specific stage."""

    def ingest_fn(cache_dir, iso_week, product):
        if fail_stage == "ingest":
            raise RuntimeError("Ingest failure: cache not found")
        return (["review1", "review2"], {"final_count": 2, "ingested_at": "now"})

    def analyze_fn(reviews):
        if fail_stage == "analyze":
            raise RuntimeError("Analyze failure: OOM")
        return (["cluster1"], {"reviews_embedded": 2, "clusters_found": 1})

    def summarize_fn(clusters):
        if fail_stage == "summarize":
            raise RuntimeError("Summarize failure: LLM timeout")
        return (
            fixture_report,
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

    def render_fn(report):
        if fail_stage == "render":
            raise RuntimeError("Render failure: template error")
        return {"heading_text": "Groww - Week 2026-W23"}

    async def deliver_fn(report, config, email_mode, existing_delivery):
        if fail_stage == "deliver":
            raise RuntimeError("Deliver failure: MCP timeout")
        return fixture_delivery

    return {
        "ingest_fn": ingest_fn,
        "analyze_fn": analyze_fn,
        "summarize_fn": summarize_fn,
        "render_fn": render_fn,
        "deliver_fn": deliver_fn,
    }


# ===========================================================================
# Tests
# ===========================================================================


class TestFullPipelineSuccess:
    """Test: all 5 stages pass, status=completed, ledger persisted."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        fns = _make_stage_fns(fixture_report, fixture_delivery)

        record = await run_pipeline(
            product="groww",
            iso_week="2026-W23",
            config=product_config,
            ledger_base_dir=tmp_path,
            **fns,
        )

        assert record.status == RunState.COMPLETED.value
        assert record.run_id  # UUID was generated
        assert record.completed_at is not None

        # All 5 stages completed
        for stage in PIPELINE_STAGES:
            assert stage in record.stages
            assert record.stages[stage].status == StageStatus.COMPLETED.value
            assert record.stages[stage].duration_ms is not None

        # Ledger persisted
        ledger = Ledger(base_dir=tmp_path)
        loaded = ledger.load("groww", "2026-W23")
        assert loaded is not None
        assert loaded.status == RunState.COMPLETED.value


class TestIdempotency:
    """Test: completed run is a no-op on re-run."""

    @pytest.mark.asyncio
    async def test_completed_run_is_idempotent(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        fns = _make_stage_fns(fixture_report, fixture_delivery)

        # First run
        record1 = await run_pipeline(
            product="groww",
            iso_week="2026-W23",
            config=product_config,
            ledger_base_dir=tmp_path,
            **fns,
        )
        assert record1.status == RunState.COMPLETED.value

        # Second run with all stages set to fail — should never be called
        fail_fns = _make_stage_fns(
            fixture_report, fixture_delivery, fail_stage="ingest"
        )
        record2 = await run_pipeline(
            product="groww",
            iso_week="2026-W23",
            config=product_config,
            ledger_base_dir=tmp_path,
            **fail_fns,
        )
        # Same record returned, no stages re-executed
        assert record2.run_id == record1.run_id
        assert record2.status == RunState.COMPLETED.value


class TestFailedRun:
    """Test: failed stage records error and status=failed."""

    @pytest.mark.asyncio
    async def test_failed_run_records_error(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        fns = _make_stage_fns(
            fixture_report, fixture_delivery, fail_stage="analyze"
        )

        with pytest.raises(OrchestrationError) as exc_info:
            await run_pipeline(
                product="groww",
                iso_week="2026-W23",
                config=product_config,
                ledger_base_dir=tmp_path,
                **fns,
            )

        assert exc_info.value.stage == "analyze"

        # Ledger shows failed status
        ledger = Ledger(base_dir=tmp_path)
        loaded = ledger.load("groww", "2026-W23")
        assert loaded is not None
        assert loaded.status == RunState.FAILED.value
        assert loaded.error is not None
        assert loaded.error["stage"] == "analyze"
        assert "OOM" in loaded.error["message"]

        # Ingest completed, analyze failed
        assert loaded.stages["ingest"].status == StageStatus.COMPLETED.value
        assert loaded.stages["analyze"].status == StageStatus.FAILED.value


class TestResume:
    """Test: resume from failed stage."""

    @pytest.mark.asyncio
    async def test_resume_from_failed_stage(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        # First run: fail at deliver
        fail_fns = _make_stage_fns(
            fixture_report, fixture_delivery, fail_stage="deliver"
        )
        with pytest.raises(OrchestrationError):
            await run_pipeline(
                product="groww",
                iso_week="2026-W23",
                config=product_config,
                ledger_base_dir=tmp_path,
                **fail_fns,
            )

        # Second run: succeed — should resume from deliver only
        success_fns = _make_stage_fns(fixture_report, fixture_delivery)
        record = await run_pipeline(
            product="groww",
            iso_week="2026-W23",
            config=product_config,
            ledger_base_dir=tmp_path,
            **success_fns,
        )

        assert record.status == RunState.COMPLETED.value
        # Ingest/analyze/summarize/render were completed in first run
        for stage in ["ingest", "analyze", "summarize", "render"]:
            assert record.stages[stage].status == StageStatus.COMPLETED.value
        # Deliver now completed
        assert record.stages["deliver"].status == StageStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_resume_skips_successful_doc(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        """Doc appended but email failed -> retry skips doc, only creates email."""
        # Pre-populate a failed record where doc succeeded but email failed
        ledger = Ledger(base_dir=tmp_path)
        failed_record = RunRecord(
            run_id=str(uuid.uuid4()),
            product="groww",
            iso_week="2026-W23",
            status=RunState.FAILED.value,
            started_at="2026-06-11T06:30:00+05:30",
            updated_at="2026-06-11T06:32:00+05:30",
            stages={
                "ingest": StageRecord(
                    status=StageStatus.COMPLETED.value,
                    started_at="2026-06-11T06:30:00+05:30",
                    completed_at="2026-06-11T06:30:10+05:30",
                    duration_ms=10000,
                ),
                "analyze": StageRecord(
                    status=StageStatus.COMPLETED.value,
                    started_at="2026-06-11T06:30:11+05:30",
                    completed_at="2026-06-11T06:30:40+05:30",
                    duration_ms=29000,
                ),
                "summarize": StageRecord(
                    status=StageStatus.COMPLETED.value,
                    started_at="2026-06-11T06:30:41+05:30",
                    completed_at="2026-06-11T06:31:20+05:30",
                    duration_ms=39000,
                ),
                "render": StageRecord(
                    status=StageStatus.COMPLETED.value,
                    started_at="2026-06-11T06:31:21+05:30",
                    completed_at="2026-06-11T06:31:21+05:30",
                    duration_ms=50,
                ),
                "deliver": StageRecord(
                    status=StageStatus.FAILED.value,
                    started_at="2026-06-11T06:31:22+05:30",
                    completed_at="2026-06-11T06:32:00+05:30",
                    duration_ms=38000,
                ),
            },
            delivery=DeliveryRecord(
                doc=DocDeliveryInfo(
                    document_id="test-doc-id",
                    heading_text="Groww - Week 2026-W23",
                    heading_anchor="",
                    revision_id="rev-existing",
                    appended=True,
                ),
                email=None,  # Email failed
            ),
            error={"stage": "deliver", "message": "Email timeout"},
        )
        ledger.save(failed_record)

        # Track whether deliver_fn was called and what existing_delivery was
        deliver_calls = []

        async def tracking_deliver(report, config, email_mode, existing_delivery):
            deliver_calls.append({
                "existing_delivery": existing_delivery,
                "email_mode": email_mode,
            })
            return DeliveryRecord(
                doc=existing_delivery.doc if existing_delivery and existing_delivery.doc else fixture_delivery.doc,
                email=EmailDeliveryInfo(
                    mode="draft",
                    message_id="msg-retry-001",
                    recipients=["test@example.com"],
                    sent_at="",
                ),
            )

        fns = _make_stage_fns(fixture_report, fixture_delivery)
        fns["deliver_fn"] = tracking_deliver

        record = await run_pipeline(
            product="groww",
            iso_week="2026-W23",
            config=product_config,
            ledger_base_dir=tmp_path,
            **fns,
        )

        assert record.status == RunState.COMPLETED.value
        # deliver_fn was called with existing_delivery containing the doc
        assert len(deliver_calls) == 1
        assert deliver_calls[0]["existing_delivery"] is not None
        assert deliver_calls[0]["existing_delivery"].doc.appended is True
        # Email now has message_id
        assert record.delivery.email.message_id == "msg-retry-001"


class TestConcurrentRun:
    """Test: concurrent runs are rejected."""

    @pytest.mark.asyncio
    async def test_concurrent_run_rejected(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        """Second concurrent run for same (product, iso_week) raises LockError."""
        run_dir = tmp_path / "groww" / "2026-W23"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write a lock file with the current PID
        lock_file = run_dir / ".lock"
        lock_file.write_text(f"{os.getpid()}|2020-01-01T00:00:00", encoding="utf-8")

        # Mock _is_process_alive to return True (process is alive)
        with patch("pulse.audit.lock._is_process_alive", return_value=True):
            fns = _make_stage_fns(fixture_report, fixture_delivery)
            raised = False
            try:
                await run_pipeline(
                    product="groww",
                    iso_week="2026-W23",
                    config=product_config,
                    ledger_base_dir=tmp_path,
                    **fns,
                )
            except LockError:
                raised = True
            assert raised, "Expected LockError but no exception was raised"

    @pytest.mark.asyncio
    async def test_stale_lock_cleanup(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        """Dead-process lock is cleaned up automatically."""
        run_dir = tmp_path / "groww" / "2026-W23"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write a lock file with a PID that definitely doesn't exist
        lock_file = run_dir / ".lock"
        lock_file.write_text("999999999|2020-01-01T00:00:00", encoding="utf-8")

        fns = _make_stage_fns(fixture_report, fixture_delivery)

        # Mock _is_process_alive to always return False (dead process)
        with patch("pulse.audit.lock._is_process_alive", return_value=False):
            record = await run_pipeline(
                product="groww",
                iso_week="2026-W23",
                config=product_config,
                ledger_base_dir=tmp_path,
                **fns,
            )

        # Stale lock was cleaned up and pipeline completed
        assert record.status == RunState.COMPLETED.value
        # Lock file should be removed
        assert not lock_file.exists()


class TestDryRun:
    """Test: dry_run=True skips delivery."""

    @pytest.mark.asyncio
    async def test_dry_run_skips_delivery(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        deliver_called = False

        async def mock_deliver(report, config, email_mode, existing_delivery):
            nonlocal deliver_called
            deliver_called = True
            return fixture_delivery

        fns = _make_stage_fns(fixture_report, fixture_delivery)
        fns["deliver_fn"] = mock_deliver

        record = await run_pipeline(
            product="groww",
            iso_week="2026-W23",
            dry_run=True,
            config=product_config,
            ledger_base_dir=tmp_path,
            **fns,
        )

        assert record.status == RunState.COMPLETED.value
        assert deliver_called is False
        assert record.delivery is None
        # Deliver stage should be "skipped"
        assert record.stages["deliver"].status == StageStatus.SKIPPED.value


class TestStageTiming:
    """Test: each stage records timing."""

    @pytest.mark.asyncio
    async def test_stage_timing_recorded(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        fns = _make_stage_fns(fixture_report, fixture_delivery)

        record = await run_pipeline(
            product="groww",
            iso_week="2026-W23",
            config=product_config,
            ledger_base_dir=tmp_path,
            **fns,
        )

        for stage in PIPELINE_STAGES:
            sr = record.stages[stage]
            assert sr.started_at is not None, f"{stage} missing started_at"
            assert sr.completed_at is not None, f"{stage} missing completed_at"
            assert sr.duration_ms is not None, f"{stage} missing duration_ms"
            assert sr.duration_ms >= 0, f"{stage} duration_ms is negative"


class TestEmailModeOverride:
    """Test: email_mode param overrides config default."""

    @pytest.mark.asyncio
    async def test_email_mode_override(
        self,
        tmp_path: Path,
        product_config: ProductConfig,
        fixture_report: PulseReport,
        fixture_delivery: DeliveryRecord,
    ):
        captured_mode = []

        async def tracking_deliver(report, config, email_mode, existing_delivery):
            captured_mode.append(email_mode)
            return fixture_delivery

        fns = _make_stage_fns(fixture_report, fixture_delivery)
        fns["deliver_fn"] = tracking_deliver

        await run_pipeline(
            product="groww",
            iso_week="2026-W23",
            email_mode="send",
            config=product_config,
            ledger_base_dir=tmp_path,
            **fns,
        )

        assert captured_mode == ["send"]


class TestRunRecordHelpers:
    """Test RunRecord helper methods used by the orchestrator."""

    def test_next_stage_fresh_record(self):
        record = RunRecord(
            run_id="x",
            product="groww",
            iso_week="2026-W23",
            status=RunState.PENDING.value,
            started_at="now",
        )
        assert record.next_stage() == "ingest"

    def test_next_stage_after_ingest(self):
        record = RunRecord(
            run_id="x",
            product="groww",
            iso_week="2026-W23",
            status=RunState.ANALYZING.value,
            started_at="now",
            stages={
                "ingest": StageRecord(status=StageStatus.COMPLETED.value),
            },
        )
        assert record.next_stage() == "analyze"

    def test_next_stage_all_completed(self):
        record = RunRecord(
            run_id="x",
            product="groww",
            iso_week="2026-W23",
            status=RunState.COMPLETED.value,
            started_at="now",
            stages={
                s: StageRecord(status=StageStatus.COMPLETED.value)
                for s in PIPELINE_STAGES
            },
        )
        assert record.next_stage() is None

    def test_last_completed_stage(self):
        record = RunRecord(
            run_id="x",
            product="groww",
            iso_week="2026-W23",
            status=RunState.FAILED.value,
            started_at="now",
            stages={
                "ingest": StageRecord(status=StageStatus.COMPLETED.value),
                "analyze": StageRecord(status=StageStatus.COMPLETED.value),
                "summarize": StageRecord(status=StageStatus.FAILED.value),
            },
        )
        assert record.last_completed_stage() == "analyze"
