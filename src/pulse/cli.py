"""CLI entrypoint — Phase 8.

Commands:
  pulse run       Execute full pipeline for a product/week.
  pulse backfill  Run pipeline for a range of ISO weeks.
  pulse status    Print RunRecord for a product/week.
  pulse dry-run   Run pipeline locally without delivery.
  pulse render    Render a PulseReport JSON into preview files.
  pulse export-frontend  Write run data as static JSON for the frontend.
  pulse version   Print version and exit.

Exit codes:
  0  Success (or no-op for idempotent re-runs).
  1  General error.
  2  Pipeline stage failure (stage name printed to stderr).
  3  Concurrent run rejected (lock held).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from pulse import __version__


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STAGE_FAILURE = 2
EXIT_LOCK_REJECTED = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_weeks_between(start_week: str, end_week: str) -> list[str]:
    """Return list of ISO week strings from start_week to end_week inclusive."""
    start_dt = datetime.strptime(f"{start_week}-1", "%G-W%V-%u")
    end_dt = datetime.strptime(f"{end_week}-1", "%G-W%V-%u")
    if end_dt < start_dt:
        return []
    weeks = []
    current = start_dt
    while current <= end_dt:
        iso = current.isocalendar()
        weeks.append(f"{iso.year}-W{iso.week:02d}")
        current += timedelta(weeks=1)
    return weeks


def _current_iso_week() -> str:
    """Current ISO week string in IST (UTC+5:30)."""
    from datetime import timezone
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle 'pulse run' — execute full pipeline."""
    from pulse.audit.lock import LockError
    from pulse.orchestrator import OrchestrationError, run_pipeline

    iso_week = args.iso_week or _current_iso_week()
    product = args.product
    dry_run = getattr(args, "dry_run", False)
    email_mode = getattr(args, "email_mode", None)

    print(f"Running pipeline: product={product} week={iso_week} dry_run={dry_run}")

    try:
        record = asyncio.run(
            run_pipeline(
                product=product,
                iso_week=iso_week,
                dry_run=dry_run,
                email_mode=email_mode,
            )
        )
    except LockError as exc:
        print(f"ERROR: Concurrent run rejected — {exc}", file=sys.stderr)
        return EXIT_LOCK_REJECTED
    except OrchestrationError as exc:
        print(
            f"ERROR: Pipeline failed at stage '{exc.stage}': {exc.message}",
            file=sys.stderr,
        )
        return EXIT_STAGE_FAILURE
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"Status: {record.status}")
    if record.is_completed():
        print(f"Run ID: {record.run_id}")
        if record.delivery:
            if record.delivery.doc:
                print(f"Doc appended: {record.delivery.doc.appended}")
            if record.delivery.email:
                print(f"Email mode: {record.delivery.email.mode}")
    return EXIT_OK


def _cmd_backfill(args: argparse.Namespace) -> int:
    """Handle 'pulse backfill' — run pipeline for a range of ISO weeks."""
    from pulse.audit.lock import LockError
    from pulse.orchestrator import OrchestrationError, run_pipeline

    product = args.product
    weeks = _iso_weeks_between(args.from_week, args.to_week)
    if not weeks:
        print("ERROR: No weeks in range (from > to).", file=sys.stderr)
        return EXIT_ERROR

    email_mode = getattr(args, "email_mode", None)
    print(f"Backfill: product={product} weeks={len(weeks)} ({weeks[0]}..{weeks[-1]})")

    failed = []
    for week in weeks:
        print(f"\n--- {week} ---")
        try:
            record = asyncio.run(
                run_pipeline(
                    product=product,
                    iso_week=week,
                    dry_run=False,
                    email_mode=email_mode,
                )
            )
            print(f"  Status: {record.status}")
        except LockError as exc:
            print(f"  SKIPPED (lock): {exc}", file=sys.stderr)
            failed.append((week, "lock"))
        except OrchestrationError as exc:
            print(
                f"  FAILED at stage '{exc.stage}': {exc.message}",
                file=sys.stderr,
            )
            failed.append((week, exc.stage))
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed.append((week, str(exc)[:50]))

    print(f"\nBackfill complete: {len(weeks) - len(failed)}/{len(weeks)} succeeded.")
    if failed:
        for week, reason in failed:
            print(f"  Failed: {week} ({reason})")
        return EXIT_STAGE_FAILURE
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    """Handle 'pulse status' — print RunRecord for product/week."""
    from pulse.audit.ledger import Ledger

    product = args.product
    iso_week = args.iso_week or _current_iso_week()

    ledger = Ledger()
    record = ledger.load(product, iso_week)

    if record is None:
        print(f"No run found for {product} {iso_week}.")
        return EXIT_OK

    print(f"Product:   {record.product}")
    print(f"Week:      {record.iso_week}")
    print(f"Run ID:    {record.run_id}")
    print(f"Status:    {record.status}")
    print(f"Started:   {record.started_at}")
    print(f"Completed: {record.completed_at or '(pending)'}")

    if record.stages:
        print("\nStages:")
        for stage, sr in record.stages.items():
            duration = f"{sr.duration_ms}ms" if sr.duration_ms is not None else "-"
            print(f"  {stage:12s} {sr.status:10s} ({duration})")

    if record.delivery:
        print("\nDelivery:")
        if record.delivery.doc:
            print(f"  Doc appended: {record.delivery.doc.appended}")
            print(f"  Doc ID:       {record.delivery.doc.document_id}")
        if record.delivery.email:
            print(f"  Email mode:   {record.delivery.email.mode}")
            print(f"  Recipients:   {', '.join(record.delivery.email.recipients)}")

    if record.error:
        print(f"\nError: [{record.error.get('stage')}] {record.error.get('message')}")

    return EXIT_OK


def _cmd_dry_run(args: argparse.Namespace) -> int:
    """Handle 'pulse dry-run' — run pipeline without delivery."""
    args.dry_run = True
    return _cmd_run(args)


def _cmd_render(args: argparse.Namespace) -> int:
    """Handle 'pulse render' — write preview.doc.json and preview.email.html."""
    from pulse.models.models import PulseReport
    from pulse.render import preview

    # Load report from JSON file.
    report_path = os.path.abspath(args.report_file)
    if not os.path.isfile(report_path):
        print(f"ERROR: Report file not found: {report_path}", file=sys.stderr)
        return 1

    with open(report_path, "r", encoding="utf-8") as f:
        report_dict = json.load(f)

    report = PulseReport.model_validate(report_dict)

    # Write preview files.
    output_dir = os.path.abspath(args.output_dir)
    doc_path, email_path = preview(
        report,
        output_dir=output_dir,
        doc_url=args.doc_url or "",
        recipients=args.recipients.split(",") if args.recipients else None,
    )

    print(f"Doc preview:   {doc_path}")
    print(f"Email preview: {email_path}")
    print(f"Heading:       {report.product.title()} — Week {report.iso_week}")
    print(f"Themes:        {len(report.themes)}")
    return 0


def _cmd_export_frontend(args: argparse.Namespace) -> int:
    """Handle 'pulse export-frontend' — write run data to frontend/public/data/."""
    from pulse.audit.ledger import Ledger
    from pulse.models.models import PulseReport

    product = args.product
    ledger = Ledger()
    records = ledger.query(product)

    if not records:
        print(f"No runs found for product '{product}'.")
        return EXIT_OK

    # Resolve output directory
    repo_root = Path(__file__).resolve().parents[3]
    frontend_data = Path(args.output_dir) if args.output_dir else (
        repo_root / "frontend" / "public" / "data"
    )
    runs_dir = frontend_data / "runs"
    reports_dir = frontend_data / "reports"
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Write runs-index.json (array of all RunRecords, newest first)
    runs_list = [json.loads(r.model_dump_json()) for r in records]
    index_path = frontend_data / "runs-index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(runs_list, f, indent=2, ensure_ascii=False)
    print(f"  runs-index.json: {len(runs_list)} runs")

    # Write individual run files
    for record in records:
        run_path = runs_dir / f"{record.iso_week}.json"
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=2))

    # Try to load reports from the render stage preview files
    report_count = 0
    preview_dir = repo_root / "data" / "preview"
    fixture_path = repo_root / "data" / "fixture_report.json"

    # Check if fixture report exists and use it for the latest completed run
    if fixture_path.is_file():
        with open(fixture_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        # Write to the week matching the report's iso_week
        iso_week = report_data.get("iso_week", "")
        if iso_week:
            report_path = reports_dir / f"{iso_week}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            report_count += 1

    print(f"  runs/: {len(records)} files")
    print(f"  reports/: {report_count} files")
    print(f"Output: {frontend_data}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pulse",
        description="Groww Weekly Review Pulse",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print version and exit")

    # --- pulse run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Execute full pipeline for a product/week",
    )
    run_parser.add_argument(
        "--product", "-p",
        default="groww",
        help="Product identifier (default: groww)",
    )
    run_parser.add_argument(
        "--iso-week", "-w",
        default=None,
        help="ISO week string, e.g. 2026-W23 (default: current week)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without delivery (skip deliver stage)",
    )
    run_parser.add_argument(
        "--email-mode",
        choices=["draft", "send", "skip"],
        default=None,
        help="Override email mode (default: from config)",
    )

    # --- pulse backfill ---
    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Run pipeline for a range of ISO weeks",
    )
    backfill_parser.add_argument(
        "--product", "-p",
        default="groww",
        help="Product identifier (default: groww)",
    )
    backfill_parser.add_argument(
        "--from",
        dest="from_week",
        required=True,
        help="Start ISO week (inclusive), e.g. 2026-W20",
    )
    backfill_parser.add_argument(
        "--to",
        dest="to_week",
        required=True,
        help="End ISO week (inclusive), e.g. 2026-W23",
    )
    backfill_parser.add_argument(
        "--email-mode",
        choices=["draft", "send", "skip"],
        default=None,
        help="Override email mode (default: from config)",
    )

    # --- pulse status ---
    status_parser = subparsers.add_parser(
        "status",
        help="Print RunRecord for a product/week",
    )
    status_parser.add_argument(
        "--product", "-p",
        default="groww",
        help="Product identifier (default: groww)",
    )
    status_parser.add_argument(
        "--iso-week", "-w",
        default=None,
        help="ISO week string (default: current week)",
    )

    # --- pulse dry-run ---
    dryrun_parser = subparsers.add_parser(
        "dry-run",
        help="Run pipeline locally without delivery",
    )
    dryrun_parser.add_argument(
        "--product", "-p",
        default="groww",
        help="Product identifier (default: groww)",
    )
    dryrun_parser.add_argument(
        "--iso-week", "-w",
        default=None,
        help="ISO week string (default: current week)",
    )

    # --- pulse render ---
    render_parser = subparsers.add_parser(
        "render",
        help="Render a PulseReport JSON into Doc + Email preview files",
    )
    render_parser.add_argument(
        "report_file",
        help="Path to a PulseReport JSON file",
    )
    render_parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory for preview files (default: current dir)",
    )
    render_parser.add_argument(
        "--doc-url",
        default="",
        help="Google Doc URL for email deep link (optional)",
    )
    render_parser.add_argument(
        "--recipients",
        default="",
        help="Comma-separated email recipients (optional)",
    )

    # --- pulse export-frontend ---
    export_parser = subparsers.add_parser(
        "export-frontend",
        help="Write run data as static JSON for the frontend dashboard",
    )
    export_parser.add_argument(
        "--product", "-p",
        default="groww",
        help="Product identifier (default: groww)",
    )
    export_parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: frontend/public/data/)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "version" or args.command is None:
        print(f"pulse {__version__}")
        if args.command is None:
            parser.print_help()
        sys.exit(EXIT_OK)

    dispatch = {
        "run": _cmd_run,
        "backfill": _cmd_backfill,
        "status": _cmd_status,
        "dry-run": _cmd_dry_run,
        "render": _cmd_render,
        "export-frontend": _cmd_export_frontend,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(EXIT_ERROR)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
