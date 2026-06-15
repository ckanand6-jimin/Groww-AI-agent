"""Phase 8 — CLI unit tests.

Tests the CLI commands (run, backfill, status, dry-run) with mocked
orchestrator and ledger. Verifies exit codes and argument parsing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pulse.cli import (
    EXIT_ERROR,
    EXIT_LOCK_REJECTED,
    EXIT_OK,
    EXIT_STAGE_FAILURE,
    _build_parser,
    _cmd_status,
    _cmd_run,
    _iso_weeks_between,
    main,
)
from pulse.audit.lock import LockError
from pulse.models.models import (
    DeliveryRecord,
    DocDeliveryInfo,
    EmailDeliveryInfo,
    RunRecord,
    RunState,
    StageRecord,
    StageStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_record() -> RunRecord:
    """A completed RunRecord for testing."""
    return RunRecord(
        run_id="test-run-id",
        product="groww",
        iso_week="2026-W23",
        status=RunState.COMPLETED.value,
        started_at="2026-06-11T06:30:00+05:30",
        completed_at="2026-06-11T06:35:00+05:30",
        stages={
            "ingest": StageRecord(
                status=StageStatus.COMPLETED.value,
                duration_ms=12000,
            ),
            "analyze": StageRecord(
                status=StageStatus.COMPLETED.value,
                duration_ms=29000,
            ),
            "summarize": StageRecord(
                status=StageStatus.COMPLETED.value,
                duration_ms=39000,
            ),
            "render": StageRecord(
                status=StageStatus.COMPLETED.value,
                duration_ms=50,
            ),
            "deliver": StageRecord(
                status=StageStatus.COMPLETED.value,
                duration_ms=38000,
            ),
        },
        delivery=DeliveryRecord(
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
                sent_at="2026-06-11T06:35:00+05:30",
            ),
        ),
    )


@pytest.fixture
def failed_record() -> RunRecord:
    """A failed RunRecord for testing."""
    return RunRecord(
        run_id="failed-run-id",
        product="groww",
        iso_week="2026-W23",
        status=RunState.FAILED.value,
        started_at="2026-06-11T06:30:00+05:30",
        stages={
            "ingest": StageRecord(
                status=StageStatus.COMPLETED.value,
                duration_ms=12000,
            ),
            "analyze": StageRecord(
                status=StageStatus.FAILED.value,
                duration_ms=5000,
                metadata={"error": "OOM"},
            ),
        },
        error={"stage": "analyze", "message": "Out of memory"},
    )


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestIsoWeeksBetween:
    """Test _iso_weeks_between helper."""

    def test_single_week(self):
        result = _iso_weeks_between("2026-W23", "2026-W23")
        assert result == ["2026-W23"]

    def test_multiple_weeks(self):
        result = _iso_weeks_between("2026-W20", "2026-W23")
        assert result == ["2026-W20", "2026-W21", "2026-W22", "2026-W23"]

    def test_empty_range(self):
        result = _iso_weeks_between("2026-W25", "2026-W23")
        assert result == []

    def test_cross_year_boundary(self):
        # Week 52 of 2025 to week 2 of 2026
        result = _iso_weeks_between("2025-W52", "2026-W02")
        assert "2025-W52" in result
        assert "2026-W01" in result
        assert "2026-W02" in result


# ===========================================================================
# Parser tests
# ===========================================================================


class TestParser:
    """Test argument parser construction."""

    def test_parser_creation(self):
        parser = _build_parser()
        assert parser is not None
        assert parser.prog == "pulse"

    def test_run_command_parsing(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--product", "groww", "--iso-week", "2026-W23"])
        assert args.command == "run"
        assert args.product == "groww"
        assert args.iso_week == "2026-W23"
        assert args.dry_run is False

    def test_run_with_dry_run_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--dry-run"])
        assert args.dry_run is True

    def test_run_with_email_mode(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--email-mode", "send"])
        assert args.email_mode == "send"

    def test_backfill_command_parsing(self):
        parser = _build_parser()
        args = parser.parse_args(["backfill", "--from", "2026-W20", "--to", "2026-W23"])
        assert args.command == "backfill"
        assert args.from_week == "2026-W20"
        assert args.to_week == "2026-W23"

    def test_status_command_parsing(self):
        parser = _build_parser()
        args = parser.parse_args(["status", "--product", "groww", "--iso-week", "2026-W23"])
        assert args.command == "status"
        assert args.product == "groww"
        assert args.iso_week == "2026-W23"

    def test_dry_run_command_parsing(self):
        parser = _build_parser()
        args = parser.parse_args(["dry-run", "--product", "groww"])
        assert args.command == "dry-run"
        assert args.product == "groww"


# ===========================================================================
# Status command tests
# ===========================================================================


class TestStatusCommand:
    """Test 'pulse status' command."""

    def test_status_no_run_found(self, tmp_path: Path, capsys):
        """Status returns OK when no run exists."""
        parser = _build_parser()
        args = parser.parse_args(["status", "--product", "groww", "--iso-week", "2026-W99"])

        with patch("pulse.audit.ledger.Ledger") as MockLedger:
            mock_ledger = MagicMock()
            mock_ledger.load.return_value = None
            MockLedger.return_value = mock_ledger

            result = _cmd_status(args)

        assert result == EXIT_OK
        captured = capsys.readouterr()
        assert "No run found" in captured.out

    def test_status_completed_run(self, tmp_path: Path, mock_record: RunRecord, capsys):
        """Status displays completed run information."""
        parser = _build_parser()
        args = parser.parse_args(["status", "--product", "groww", "--iso-week", "2026-W23"])

        with patch("pulse.audit.ledger.Ledger") as MockLedger:
            mock_ledger = MagicMock()
            mock_ledger.load.return_value = mock_record
            MockLedger.return_value = mock_ledger

            result = _cmd_status(args)

        assert result == EXIT_OK
        captured = capsys.readouterr()
        assert "groww" in captured.out
        assert "2026-W23" in captured.out
        assert "completed" in captured.out
        assert "test-run-id" in captured.out

    def test_status_failed_run_shows_error(self, tmp_path: Path, failed_record: RunRecord, capsys):
        """Status displays error information for failed runs."""
        parser = _build_parser()
        args = parser.parse_args(["status", "--product", "groww", "--iso-week", "2026-W23"])

        with patch("pulse.audit.ledger.Ledger") as MockLedger:
            mock_ledger = MagicMock()
            mock_ledger.load.return_value = failed_record
            MockLedger.return_value = mock_ledger

            result = _cmd_status(args)

        assert result == EXIT_OK
        captured = capsys.readouterr()
        assert "failed" in captured.out
        assert "analyze" in captured.out.lower() or "Error" in captured.out


# ===========================================================================
# Run command tests (with mocked orchestrator)
# ===========================================================================


class TestRunCommand:
    """Test 'pulse run' command."""

    def test_run_success(self, mock_record: RunRecord, capsys):
        """Run returns EXIT_OK on success."""
        parser = _build_parser()
        args = parser.parse_args(["run", "--product", "groww", "--iso-week", "2026-W23"])

        with patch("pulse.orchestrator.run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_record

            result = _cmd_run(args)

        assert result == EXIT_OK
        captured = capsys.readouterr()
        assert "completed" in captured.out.lower()

    def test_run_lock_error(self, capsys):
        """Run returns EXIT_LOCK_REJECTED on LockError."""
        parser = _build_parser()
        args = parser.parse_args(["run", "--product", "groww", "--iso-week", "2026-W23"])

        with patch("pulse.orchestrator.run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = LockError("Another run in progress")

            result = _cmd_run(args)

        assert result == EXIT_LOCK_REJECTED
        captured = capsys.readouterr()
        assert "rejected" in captured.err.lower()

    def test_run_orchestration_error(self, capsys):
        """Run returns EXIT_STAGE_FAILURE on OrchestrationError."""
        from pulse.orchestrator import OrchestrationError

        parser = _build_parser()
        args = parser.parse_args(["run", "--product", "groww", "--iso-week", "2026-W23"])

        with patch("pulse.orchestrator.run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = OrchestrationError("analyze", "OOM error")

            result = _cmd_run(args)

        assert result == EXIT_STAGE_FAILURE
        captured = capsys.readouterr()
        assert "analyze" in captured.err

    def test_run_generic_error(self, capsys):
        """Run returns EXIT_ERROR on unexpected exception."""
        parser = _build_parser()
        args = parser.parse_args(["run", "--product", "groww", "--iso-week", "2026-W23"])

        with patch("pulse.orchestrator.run_pipeline", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected failure")

            result = _cmd_run(args)

        assert result == EXIT_ERROR


# ===========================================================================
# Main entry point tests
# ===========================================================================


class TestMain:
    """Test main() entry point."""

    def test_main_version(self, capsys):
        """main() handles version command."""
        with pytest.raises(SystemExit) as exc_info:
            main(["version"])
        assert exc_info.value.code == EXIT_OK

    def test_main_no_command(self, capsys):
        """main() prints help when no command given."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == EXIT_OK

    def test_main_unknown_command(self, capsys):
        """main() returns EXIT_ERROR for unknown command."""
        # argparse will reject unknown subcommand
        with pytest.raises(SystemExit) as exc_info:
            main(["unknown-command"])
        # argparse exits with 2 for invalid arguments
        assert exc_info.value.code == 2


# ===========================================================================
# Exit code tests
# ===========================================================================


class TestExitCodes:
    """Test exit code constants."""

    def test_exit_ok_value(self):
        assert EXIT_OK == 0

    def test_exit_error_value(self):
        assert EXIT_ERROR == 1

    def test_exit_stage_failure_value(self):
        assert EXIT_STAGE_FAILURE == 2

    def test_exit_lock_rejected_value(self):
        assert EXIT_LOCK_REJECTED == 3
