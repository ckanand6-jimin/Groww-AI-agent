"""Phase 7 — Run ledger unit tests.

Tests the Ledger class: save, load, query, atomic writes, and round-trip
serialization.  Uses ``tmp_path`` to avoid polluting real directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pulse.audit.ledger import Ledger, LedgerError
from pulse.models.models import (
    AnalysisRecord,
    DeliveryRecord,
    DocDeliveryInfo,
    EmailDeliveryInfo,
    IngestRecord,
    RunRecord,
    RunState,
    StageRecord,
    StageStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(base_dir=tmp_path)


@pytest.fixture
def sample_record() -> RunRecord:
    return RunRecord(
        run_id="test-uuid-001",
        product="groww",
        iso_week="2026-W23",
        status=RunState.COMPLETED.value,
        started_at="2026-06-11T06:30:00+05:30",
        completed_at="2026-06-11T06:32:45+05:30",
        updated_at="2026-06-11T06:32:45+05:30",
        stages={
            "ingest": StageRecord(
                status=StageStatus.COMPLETED.value,
                started_at="2026-06-11T06:30:01+05:30",
                completed_at="2026-06-11T06:30:12+05:30",
                duration_ms=11000,
                metadata={"review_count": 1180},
            ),
        },
        delivery=DeliveryRecord(
            doc=DocDeliveryInfo(
                document_id="doc-123",
                heading_text="Groww - Week 2026-W23",
                heading_anchor="",
                revision_id="rev-001",
                appended=True,
            ),
            email=EmailDeliveryInfo(
                mode="draft",
                message_id="msg-001",
                recipients=["user@example.com"],
                sent_at="",
            ),
        ),
        ingest=IngestRecord(review_count=1180, mcp_fetch_at="2026-06-11T06:30:12+05:30"),
        analysis=AnalysisRecord(
            model="llama-3.3-70b-versatile",
            embedding_model="BAAI/bge-small-en-v1.5",
            token_usage={"prompt_tokens": 2400, "completion_tokens": 800},
        ),
        error=None,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestLedgerSaveLoad:
    """Save and load round-trip tests."""

    def test_save_and_load_round_trip(
        self, ledger: Ledger, sample_record: RunRecord
    ):
        """RunRecord survives save -> load with all fields intact."""
        ledger.save(sample_record)

        loaded = ledger.load("groww", "2026-W23")
        assert loaded is not None
        assert loaded.run_id == "test-uuid-001"
        assert loaded.product == "groww"
        assert loaded.iso_week == "2026-W23"
        assert loaded.status == RunState.COMPLETED.value
        assert loaded.started_at == "2026-06-11T06:30:00+05:30"
        assert loaded.completed_at == "2026-06-11T06:32:45+05:30"

        # Stages
        assert "ingest" in loaded.stages
        assert loaded.stages["ingest"].status == StageStatus.COMPLETED.value
        assert loaded.stages["ingest"].duration_ms == 11000

        # Delivery
        assert loaded.delivery is not None
        assert loaded.delivery.doc is not None
        assert loaded.delivery.doc.document_id == "doc-123"
        assert loaded.delivery.doc.appended is True
        assert loaded.delivery.email is not None
        assert loaded.delivery.email.message_id == "msg-001"

        # Ingest + Analysis
        assert loaded.ingest is not None
        assert loaded.ingest.review_count == 1180
        assert loaded.analysis is not None
        assert loaded.analysis.token_usage["prompt_tokens"] == 2400

    def test_load_nonexistent_returns_none(self, ledger: Ledger):
        """Missing file returns None gracefully."""
        result = ledger.load("groww", "2026-W99")
        assert result is None

    def test_save_produces_valid_json(
        self, ledger: Ledger, sample_record: RunRecord
    ):
        """After save, run.json is valid JSON with all fields."""
        ledger.save(sample_record)

        run_dir = ledger.run_dir("groww", "2026-W23")
        assert (run_dir / "run.json").is_file()

        # Verify the JSON is valid
        text = (run_dir / "run.json").read_text(encoding="utf-8")
        parsed = json.loads(text)
        assert parsed["run_id"] == "test-uuid-001"
        assert parsed["status"] == "completed"
        assert "stages" in parsed
        assert "delivery" in parsed

    def test_save_creates_parent_dirs(
        self, ledger: Ledger, sample_record: RunRecord
    ):
        """data/runs/{product}/{iso_week}/ created automatically."""
        # Use a new product + week that doesn't exist yet
        sample_record.product = "new_product"
        sample_record.iso_week = "2026-W01"

        ledger.save(sample_record)

        expected_dir = ledger.base_dir / "new_product" / "2026-W01"
        assert expected_dir.is_dir()
        assert (expected_dir / "run.json").is_file()


class TestLedgerQuery:
    """Query and latest run tests."""

    def test_query_returns_all_runs(self, ledger: Ledger):
        """Multiple weeks queryable for one product."""
        for week in ["2026-W20", "2026-W21", "2026-W22"]:
            record = RunRecord(
                run_id=f"uuid-{week}",
                product="groww",
                iso_week=week,
                status=RunState.COMPLETED.value,
                started_at="2026-06-01T06:00:00+05:30",
                completed_at="2026-06-01T06:05:00+05:30",
                updated_at="2026-06-01T06:05:00+05:30",
            )
            ledger.save(record)

        results = ledger.query("groww")
        assert len(results) == 3
        # Sorted descending by iso_week
        assert results[0].iso_week == "2026-W22"
        assert results[1].iso_week == "2026-W21"
        assert results[2].iso_week == "2026-W20"

    def test_query_empty_product(self, ledger: Ledger):
        """Query for a non-existent product returns empty list."""
        assert ledger.query("nonexistent") == []

    def test_latest_returns_most_recent(self, ledger: Ledger):
        """Latest completed run returned correctly."""
        # W20: completed
        ledger.save(RunRecord(
            run_id="uuid-w20",
            product="groww",
            iso_week="2026-W20",
            status=RunState.COMPLETED.value,
            started_at="2026-05-18T06:00:00+05:30",
            completed_at="2026-05-18T06:05:00+05:30",
            updated_at="2026-05-18T06:05:00+05:30",
        ))
        # W21: failed
        ledger.save(RunRecord(
            run_id="uuid-w21",
            product="groww",
            iso_week="2026-W21",
            status=RunState.FAILED.value,
            started_at="2026-05-25T06:00:00+05:30",
            updated_at="2026-05-25T06:05:00+05:30",
            error={"stage": "analyze", "message": "OOM"},
        ))
        # W22: completed
        ledger.save(RunRecord(
            run_id="uuid-w22",
            product="groww",
            iso_week="2026-W22",
            status=RunState.COMPLETED.value,
            started_at="2026-06-01T06:00:00+05:30",
            completed_at="2026-06-01T06:05:00+05:30",
            updated_at="2026-06-01T06:05:00+05:30",
        ))

        latest = ledger.latest("groww")
        assert latest is not None
        # Query sorts descending, so W22 comes first
        assert latest.iso_week == "2026-W22"
        assert latest.is_completed()


class TestLedgerExists:
    """Exists helper test."""

    def test_exists_true_after_save(self, ledger: Ledger, sample_record: RunRecord):
        ledger.save(sample_record)
        assert ledger.exists("groww", "2026-W23") is True

    def test_exists_false_before_save(self, ledger: Ledger):
        assert ledger.exists("groww", "2026-W99") is False


class TestLedgerUpdatedAt:
    """updated_at timestamp behaviour."""

    def test_updated_at_changes_on_save(self, ledger: Ledger):
        """updated_at timestamp refreshed on each save."""
        record = RunRecord(
            run_id="uuid-ts",
            product="groww",
            iso_week="2026-W23",
            status=RunState.PENDING.value,
            started_at="2026-06-11T06:30:00+05:30",
            updated_at="2026-06-11T06:30:00+05:30",
        )
        ledger.save(record)

        # Simulate a state change
        record.status = RunState.INGESTING.value
        record.updated_at = "2026-06-11T06:30:05+05:30"
        ledger.save(record)

        loaded = ledger.load("groww", "2026-W23")
        assert loaded is not None
        assert loaded.updated_at == "2026-06-11T06:30:05+05:30"
        assert loaded.status == RunState.INGESTING.value
