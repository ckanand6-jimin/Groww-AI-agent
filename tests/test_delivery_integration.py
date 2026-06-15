"""Phase 6 — REAL integration test against Railway MCP server.

This test makes live HTTP calls to:
  POST https://mcp-server-production-f90c.up.railway.app/append_to_doc
  POST https://mcp-server-production-f90c.up.railway.app/create_email_draft

Prerequisites:
  - .env file has PULSE_MCP_SERVER_URL set
  - groww.yaml has real document_id and stakeholders

Run:
  python -m pytest tests/test_delivery_integration.py -v -s --tb=long

Or as a standalone script:
  python tests/test_delivery_integration.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import dotenv
import pytest
from pulse.config import load_product_config
from pulse.delivery.docs_client import DocsClient
from pulse.delivery.gmail_client import GmailClient
from pulse.delivery.mcp_http_client import McpHttpClient, McpHttpClientError
from pulse.models.models import (
    ActionIdea,
    AudienceNotes,
    PulseReport,
    PulseReportPeriod,
    PulseReportStats,
    Theme,
)

# ---------------------------------------------------------------------------
# Load .env (so os.environ picks up PULSE_MCP_SERVER_URL etc.)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.is_file():
    dotenv.load_dotenv(_ENV_PATH)
    print(f"[env] Loaded {_ENV_PATH}")
else:
    print(f"[env] WARNING: .env not found at {_ENV_PATH}")


# ---------------------------------------------------------------------------
# Fixture: same PulseReport used by Phase 5 snapshot tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_report() -> PulseReport:
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
                name="App performance & bugs",
                summary="Lag, crashes during trading hours; login/session timeouts.",
                cluster_size=142,
                avg_rating=2.1,
                quotes=[
                    "\u201cThe app freezes exactly when the market opens, very frustrating.\u201d",
                    "\u201cLogin keeps failing during peak hours.\u201d",
                ],
                action_ideas=[
                    ActionIdea(
                        title="Stabilize peak-time performance",
                        rationale="Scale infra during market hours; improve crash visibility.",
                    ),
                ],
            ),
            Theme(
                rank=2,
                name="Customer support friction",
                summary="Slow responses; unresolved tickets.",
                cluster_size=98,
                avg_rating=1.8,
                quotes=[
                    "\u201cSupport takes days to reply and doesn\u2019t solve the issue.\u201d",
                    "\u201cChat support is useless, no one responds.\u201d",
                ],
                action_ideas=[
                    ActionIdea(
                        title="Improve support SLA visibility",
                        rationale="Expected response time in-app; ticket status tracking.",
                    ),
                ],
            ),
            Theme(
                rank=3,
                name="UX & feature gaps",
                summary="Confusing navigation for portfolio insights; missing advanced analytics.",
                cluster_size=76,
                avg_rating=2.5,
                quotes=[
                    "\u201cGood for beginners but lacks detailed analysis tools.\u201d",
                    "\u201cPortfolio view is hard to find and understand.\u201d",
                ],
                action_ideas=[
                    ActionIdea(
                        title="Enhance power-user features",
                        rationale="Advanced portfolio analytics; clearer investments navigation.",
                    ),
                ],
            ),
        ],
        audience_notes=AudienceNotes(
            product="Focus on app stability during trading hours and UX improvements for portfolio navigation.",
            support="Prepare for increased ticket volume around login issues and brokerage queries.",
            leadership="User sentiment is declining due to performance and support gaps; addressing top 3 themes could improve retention by 15-20%.",
        ),
        generated_at="2026-06-11T06:30:00+05:30",
    )


# ===========================================================================
# Test: Doc append (LIVE)
# ===========================================================================


class TestRealDocAppend:
    """Live integration test: POST /append_to_doc"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_append_to_doc_real(self, sample_report: PulseReport):
        """Append the weekly section to the staging Google Doc."""
        config = load_product_config()
        doc_id = config.delivery.google_doc.document_id
        server_url = os.environ["PULSE_MCP_SERVER_URL"]

        print("\n" + "=" * 72)
        print("  LIVE TEST: POST /append_to_doc")
        print("=" * 72)
        print(f"  Server:  {server_url}")
        print(f"  Doc ID:  {doc_id}")
        print(f"  Heading: {sample_report.product.title()} — Week {sample_report.iso_week}")
        print("-" * 72)

        http = McpHttpClient(base_url=server_url)
        client = DocsClient(http)

        result = await client.append_to_doc(
            document_id=doc_id,
            report=sample_report,
        )

        print(f"\n  RESPONSE:")
        print(f"    appended:      {result.appended}")
        print(f"    document_id:   {result.document_id}")
        print(f"    heading_text:  {result.heading_text}")
        print(f"    heading_anchor:{result.heading_anchor!r}")
        print(f"    revision_id:   {result.revision_id!r}")
        print("=" * 72 + "\n")

        assert result.appended is True, (
            f"Doc append failed! revision_id={result.revision_id!r}"
        )
        assert result.document_id == doc_id


# ===========================================================================
# Test: Email draft (LIVE)
# ===========================================================================


class TestRealEmailDraft:
    """Live integration test: POST /create_email_draft"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_create_email_draft_real(self, sample_report: PulseReport):
        """Create a real Gmail draft with the weekly report teaser."""
        config = load_product_config()
        doc_id = config.delivery.google_doc.document_id
        recipients = config.delivery.email.stakeholders
        server_url = os.environ["PULSE_MCP_SERVER_URL"]

        print("\n" + "=" * 72)
        print("  LIVE TEST: POST /create_email_draft")
        print("=" * 72)
        print(f"  Server:     {server_url}")
        print(f"  To:         {recipients}")
        print(f"  Subject:    Groww Weekly Review Pulse — Week 2026-W23")
        print("-" * 72)

        http = McpHttpClient(base_url=server_url)
        client = GmailClient(http)

        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

        result = await client.create_draft(
            report=sample_report,
            recipients=recipients,
            doc_url=doc_url,
            mode="draft",
        )

        print(f"\n  RESPONSE:")
        print(f"    message_id:  {result.message_id!r}")
        print(f"    mode:        {result.mode}")
        print(f"    recipients:  {result.recipients}")
        print(f"    sent_at:     {result.sent_at!r}")
        print("=" * 72 + "\n")

        assert result.message_id, (
            f"Email draft creation failed — empty message_id!"
        )
        assert result.mode == "draft"


# ===========================================================================
# Standalone runner (python tests/test_delivery_integration.py)
# ===========================================================================

if __name__ == "__main__":
    import asyncio

    async def _main():
        # Build the fixture report directly
        report = PulseReport(
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
                    name="App performance & bugs",
                    summary="Lag, crashes during trading hours; login/session timeouts.",
                    cluster_size=142,
                    avg_rating=2.1,
                    quotes=["The app freezes exactly when the market opens.", "Login keeps failing during peak hours."],
                    action_ideas=[ActionIdea(title="Stabilize peak-time performance", rationale="Scale infra during market hours.")],
                ),
                Theme(
                    rank=2,
                    name="Customer support friction",
                    summary="Slow responses; unresolved tickets.",
                    cluster_size=98,
                    avg_rating=1.8,
                    quotes=["Support takes days to reply.", "Chat support is useless."],
                    action_ideas=[ActionIdea(title="Improve support SLA visibility", rationale="Expected response time in-app.")],
                ),
                Theme(
                    rank=3,
                    name="UX & feature gaps",
                    summary="Confusing navigation; missing analytics.",
                    cluster_size=76,
                    avg_rating=2.5,
                    quotes=["Good for beginners but lacks analysis tools.", "Portfolio view is hard to find."],
                    action_ideas=[ActionIdea(title="Enhance power-user features", rationale="Advanced analytics; clearer navigation.")],
                ),
            ],
            audience_notes=AudienceNotes(
                product="Focus on app stability.",
                support="Prepare for ticket volume.",
                leadership="Addressing top themes could improve retention.",
            ),
            generated_at="2026-06-11T06:30:00+05:30",
        )

        config = load_product_config()
        server_url = os.environ.get("PULSE_MCP_SERVER_URL", "")
        if not server_url:
            print("ERROR: PULSE_MCP_SERVER_URL not set in environment or .env")
            sys.exit(1)

        http = McpHttpClient(base_url=server_url)
        doc_id = config.delivery.google_doc.document_id
        recipients = config.delivery.email.stakeholders

        # ── Doc append ──────────────────────────────────────────────
        print("=" * 72)
        print("  PHASE 6 INTEGRATION TEST — LIVE RAILWAY MCP")
        print("=" * 72)
        print(f"  Server: {server_url}")
        print(f"  Doc ID: {doc_id}")
        print()

        docs = DocsClient(http)
        doc_failed = False
        try:
            # Patch call_tool to capture raw response
            original_call = http.call_tool
            async def _doc_call(endpoint, payload):
                result = await original_call(endpoint, payload)
                print(f"  [DOC] Raw response JSON: {json.dumps(result, indent=4, ensure_ascii=False)}")
                return result
            http.call_tool = _doc_call  # type: ignore[method-assign]
            
            doc_result = await docs.append_to_doc(document_id=doc_id, report=report)
            print("  [DOC] append_to_doc RESPONSE:")
            print(f"    appended:       {doc_result.appended}")
            print(f"    document_id:    {doc_result.document_id}")
            print(f"    heading_text:   {doc_result.heading_text}")
            print(f"    heading_anchor: {doc_result.heading_anchor!r}")
            print(f"    revision_id:    {doc_result.revision_id!r}")
            print()
            print(f"  >>> Open your Doc to verify section:")
            print(f"  >>> https://docs.google.com/document/d/{doc_id}/edit")
            print()
        except McpHttpClientError as exc:
            print(f"  [DOC] FAILED: {exc}")
            print()
            doc_failed = True
        finally:
            http.call_tool = original_call  # restore

        # ── Email draft ─────────────────────────────────────────────
        email_http = McpHttpClient(base_url=server_url)
        gmail = GmailClient(email_http)
        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
        try:
            # Patch call_tool to capture raw response
            email_original = email_http.call_tool
            async def _email_call(endpoint, payload):
                result = await email_original(endpoint, payload)
                print(f"  [EMAIL] Raw response JSON: {json.dumps(result, indent=4, ensure_ascii=False)}")
                return result
            email_http.call_tool = _email_call  # type: ignore[method-assign]
            
            email_result = await gmail.create_draft(
                report=report,
                recipients=recipients,
                doc_url=doc_url,
                mode="draft",
            )
            print("  [EMAIL] create_email_draft RESPONSE:")
            print(f"    message_id:  {email_result.message_id!r}")
            print(f"    mode:        {email_result.mode}")
            print(f"    recipients:  {email_result.recipients}")
            print(f"    sent_at:     {email_result.sent_at!r}")
            print()
            print(f"  >>> Check Gmail drafts for: {recipients}")
            print()
        except McpHttpClientError as exc:
            print(f"  [EMAIL] FAILED: {exc}")
            print()
            sys.exit(1)

        print("=" * 72)
        if doc_failed:
            print("  DOC TEST FAILED (see error above)")
        else:
            print("  BOTH TESTS PASSED")
        print("=" * 72)

    asyncio.run(_main())
