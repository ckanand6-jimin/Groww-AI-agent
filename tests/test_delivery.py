"""Phase 6 delivery integration tests (mocked HTTP).

All tests mock ``httpx.AsyncClient`` so no real MCP server is required.
The tests verify payload transformation, response parsing, send guard
behaviour, and error propagation.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pulse.config import EmailMode, ProductConfig
from pulse.delivery.deliver import DeliveryError, deliver
from pulse.delivery.docs_client import DocsClient
from pulse.delivery.gmail_client import GmailClient
from pulse.delivery.mcp_http_client import McpHttpClient, McpHttpClientError
from pulse.models.models import (
    ActionIdea,
    AudienceNotes,
    EmailDeliveryInfo,
    PulseReport,
    PulseReportPeriod,
    PulseReportStats,
    Theme,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_report() -> PulseReport:
    """A realistic PulseReport matching the Phase 5 test fixture."""
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
                    "The app freezes exactly when the market opens, very frustrating.",
                    "Login keeps failing during peak hours.",
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
                    "Support takes days to reply and doesn't solve the issue.",
                    "Chat support is useless, no one responds.",
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
                    "Good for beginners but lacks detailed analysis tools.",
                    "Portfolio view is hard to find and understand.",
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
            product="Focus on app stability during trading hours and UX improvements.",
            support="Prepare for increased ticket volume around login issues.",
            leadership="Addressing top 3 themes could improve retention by 15-20%.",
        ),
        generated_at="2026-06-08T06:30:00+05:30",
    )


@pytest.fixture
def product_config() -> ProductConfig:
    """A minimal ProductConfig suitable for delivery tests."""
    return ProductConfig(
        product="groww",
        display_name="Groww",
        play_store={"app_id": "com.nextbillion.groww"},
        review_window_weeks=10,
        analysis={
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": "gpt-4o-mini",
            "max_tokens_per_run": 80_000,
            "max_themes": 5,
        },
        delivery={
            "google_doc": {
                "document_id": "1AbCdEfGhIjKlMnOpQrStUv",
                "document_title": "Weekly Review Pulse — Groww",
            },
            "email": {
                "stakeholders": ["product@example.com", "support@example.com"],
                "default_mode": "draft",
            },
        },
        schedule={"timezone": "Asia/Kolkata", "cron": "0 6 * * 1"},
    )


# ---------------------------------------------------------------------------
# Mock helper
# ---------------------------------------------------------------------------


def _mock_httpx_response(status_code: int = 200, json_data: dict | None = None):
    """Return a MagicMock that simulates an httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {"status": "success"}

    if status_code >= 400:
        mock_resp.text = json.dumps(json_data or {"error": "bad request"})
        # httpx raises HTTPStatusError from raise_for_status()
        import httpx

        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"Client error '{status_code}'",
            request=MagicMock(),
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status = MagicMock()  # no-op

    return mock_resp


# ===========================================================================
# McpHttpClient
# ===========================================================================


class TestMcpHttpClient:
    """Tests for the generic HTTP MCP client."""

    def test_requires_base_url(self, monkeypatch):
        """Client must raise if PULSE_MCP_SERVER_URL is not set."""
        monkeypatch.delenv("PULSE_MCP_SERVER_URL", raising=False)
        with pytest.raises(McpHttpClientError, match="PULSE_MCP_SERVER_URL"):
            McpHttpClient()

    def test_explicit_base_url_overrides_env(self, monkeypatch):
        """Constructor argument takes precedence over env var."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://wrong.example.com")
        client = McpHttpClient(base_url="https://right.example.com")
        assert client.base_url == "https://right.example.com"

    def test_trailing_slash_stripped(self):
        client = McpHttpClient(base_url="https://mcp.example.com/")
        assert client.base_url == "https://mcp.example.com"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_call_tool_success(self, mock_async_client, monkeypatch):
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")
        mock_resp = _mock_httpx_response(
            200, {"status": "success", "revision_id": "abc123"}
        )
        mock_async_client.return_value.__aenter__.return_value.post.return_value = (
            mock_resp
        )

        client = McpHttpClient(base_url="https://mcp.example.com")
        result = await client.call_tool("append_to_doc", {"document_id": "X"})

        assert result["status"] == "success"
        assert result["revision_id"] == "abc123"

    async def test_call_tool_http_error(self, monkeypatch):
        """Non-2xx status raises McpHttpClientError."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")

        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Server error '500'",
            request=MagicMock(),
            response=mock_resp,
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = (
                mock_resp
            )

            client = McpHttpClient(base_url="https://mcp.example.com")
            with pytest.raises(McpHttpClientError, match="500"):
                await client.call_tool("append_to_doc", {})


# ===========================================================================
# DocsClient
# ===========================================================================


class TestDocsClient:
    """Tests for the DocsClient (POST /append_to_doc wrapper)."""

    @pytest.mark.asyncio
    async def test_append_to_doc_payload_shape(
        self, sample_report: PulseReport, monkeypatch
    ):
        """Verify the payload sent to /append_to_doc has the expected structure."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")
        http = McpHttpClient(base_url="https://mcp.example.com")

        captured_payload = {}

        async def fake_call(endpoint, payload):
            nonlocal captured_payload
            captured_payload = payload
            return {
                "status": "success",
                "result": {
                    "writeControl": {"requiredRevisionId": "rev-001"},
                    "documentId": "1AbCdEfGhIj",
                },
            }

        http.call_tool = fake_call  # type: ignore[method-assign]

        client = DocsClient(http)
        result = await client.append_to_doc(
            document_id="1AbCdEfGhIj",
            report=sample_report,
        )

        # --- Assertions on the payload ---
        assert captured_payload["doc_id"] == "1AbCdEfGhIj"
        assert "content" in captured_payload
        # content is a JSON string (the batch-update requests serialised)
        batch = json.loads(captured_payload["content"])
        assert "requests" in batch
        assert len(batch["requests"]) > 0

        # First request must be insertText
        first_req = batch["requests"][0]
        assert "insertText" in first_req
        assert "Groww" in first_req["insertText"]["text"]
        assert "2026-W23" in first_req["insertText"]["text"]

        # --- Assertions on the result ---
        assert result.document_id == "1AbCdEfGhIj"
        assert result.appended is True
        assert result.revision_id == "rev-001"
        assert "Groww" in result.heading_text

    @pytest.mark.asyncio
    async def test_append_to_doc_error_status(
        self, sample_report: PulseReport, monkeypatch
    ):
        """When the server reports an error, appended should be False."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")
        http = McpHttpClient(base_url="https://mcp.example.com")

        async def fake_call(endpoint, payload):
            return {"status": "error", "message": "Document not found"}

        http.call_tool = fake_call  # type: ignore[method-assign]

        client = DocsClient(http)
        result = await client.append_to_doc(
            document_id="bad-id",
            report=sample_report,
        )
        assert result.appended is False

    @pytest.mark.asyncio
    async def test_missing_heading_text_in_batch(
        self, sample_report: PulseReport, monkeypatch
    ):
        """The heading_text used for idempotency is part of the inserted text."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")
        http = McpHttpClient(base_url="https://mcp.example.com")

        async def fake_call(endpoint, payload):
            # content is a JSON string — parse it to inspect the batch
            batch = json.loads(payload["content"])
            text = batch["requests"][0]["insertText"]["text"]
            return {
                "status": "success",
                "result": {
                    "writeControl": {"requiredRevisionId": "rev-found"},
                    "documentId": "X",
                },
                "heading_found": "Groww" in text,
            }

        http.call_tool = fake_call  # type: ignore[method-assign]

        client = DocsClient(http)
        result = await client.append_to_doc(
            document_id="X", report=sample_report
        )
        assert result.appended is True
        assert result.revision_id == "rev-found"
        # heading_anchor is empty since find_heading is not called
        assert result.heading_anchor == ""


# ===========================================================================
# GmailClient
# ===========================================================================


class TestGmailClient:
    """Tests for the GmailClient (POST /create_email_draft wrapper)."""

    @pytest.mark.asyncio
    async def test_create_draft_payload_shape(
        self, sample_report: PulseReport, monkeypatch
    ):
        """Verify the payload sent to /create_email_draft."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")
        http = McpHttpClient(base_url="https://mcp.example.com")

        captured_payload = {}

        async def fake_call(endpoint, payload):
            nonlocal captured_payload
            captured_payload = payload
            return {
                "status": "success",
                "result": {"message": {"id": "msg-001", "threadId": "thread-001"}},
            }

        http.call_tool = fake_call  # type: ignore[method-assign]

        client = GmailClient(http)
        result = await client.create_draft(
            report=sample_report,
            recipients=["product@example.com"],
            doc_url="https://docs.google.com/document/d/abc/edit",
            mode="draft",
        )

        # --- Assertions on the payload ---
        assert captured_payload["to"] == "product@example.com"
        assert "Groww Weekly Review Pulse" in captured_payload["subject"]
        assert "2026-W23" in captured_payload["subject"]
        assert "<!DOCTYPE html>" in captured_payload["body"]
        assert "Groww" in captured_payload["body"]

        # --- Assertions on the result ---
        assert result.message_id == "msg-001"
        assert result.mode == "draft"
        assert result.recipients == ["product@example.com"]

    async def test_send_mode_blocked_without_env(
        self, sample_report: PulseReport, monkeypatch
    ):
        """send mode must raise ValueError without PULSE_ALLOW_SEND=true."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")
        monkeypatch.delenv("PULSE_ALLOW_SEND", raising=False)

        http = McpHttpClient(base_url="https://mcp.example.com")
        client = GmailClient(http)

        with pytest.raises(ValueError, match="PULSE_ALLOW_SEND"):
            await client.create_draft(
                report=sample_report,
                recipients=["p@e.com"],
                mode="send",
            )

    @pytest.mark.asyncio
    async def test_send_mode_passes_with_env_flag(
        self, sample_report: PulseReport, monkeypatch
    ):
        """send mode succeeds when PULSE_ALLOW_SEND=true."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")
        monkeypatch.setenv("PULSE_ALLOW_SEND", "true")

        http = McpHttpClient(base_url="https://mcp.example.com")

        async def fake_call(endpoint, payload):
            return {
                "status": "success",
                "result": {"message": {"id": "msg-sent", "threadId": "thread-sent"}},
            }

        http.call_tool = fake_call  # type: ignore[method-assign]

        client = GmailClient(http)
        result = await client.create_draft(
            report=sample_report,
            recipients=["p@e.com"],
            mode="send",
        )
        assert result.message_id == "msg-sent"
        assert result.mode == "send"

    @pytest.mark.asyncio
    async def test_send_mode_blocked_async(
        self, sample_report: PulseReport, monkeypatch
    ):
        """send mode raises ValueError async when PULSE_ALLOW_SEND is not 'true'."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")
        monkeypatch.setenv("PULSE_ALLOW_SEND", "false")

        http = McpHttpClient(base_url="https://mcp.example.com")
        client = GmailClient(http)

        with pytest.raises(ValueError, match="PULSE_ALLOW_SEND"):
            await client.create_draft(
                report=sample_report,
                recipients=["p@e.com"],
                mode="send",
            )


# ===========================================================================
# deliver()
# ===========================================================================


class TestDeliver:
    """Tests for the deliver() orchestrator."""

    @pytest.mark.asyncio
    async def test_deliver_full_flow(
        self, sample_report: PulseReport, product_config: ProductConfig, monkeypatch
    ):
        """End-to-end mocked delivery: doc append + email draft."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")

        call_order: list[str] = []

        async def fake_call(endpoint, payload):
            call_order.append(endpoint)
            if endpoint == "append_to_doc":
                return {
                    "status": "success",
                    "result": {"writeControl": {"requiredRevisionId": "rev-xyz"}},
                }
            if endpoint == "create_email_draft":
                return {
                    "status": "success",
                    "result": {"message": {"id": "msg-xyz"}},
                }
            return {"status": "error"}

        with patch.object(McpHttpClient, "call_tool", side_effect=fake_call):
            result = await deliver(sample_report, product_config)

        # --- Assertions ---
        assert call_order == ["append_to_doc", "create_email_draft"]
        assert result.doc is not None
        assert result.doc.appended is True
        assert result.doc.revision_id == "rev-xyz"

        assert result.email is not None
        assert result.email.message_id == "msg-xyz"
        assert result.email.mode == "draft"
        assert result.email.recipients == [
            "product@example.com",
            "support@example.com",
        ]

    @pytest.mark.asyncio
    async def test_deliver_skip_email(
        self, sample_report: PulseReport, monkeypatch
    ):
        """When email_mode is skip, no email call is made."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")

        config = ProductConfig(
            product="groww",
            display_name="Groww",
            play_store={"app_id": "com.nextbillion.groww"},
            review_window_weeks=10,
            analysis={
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "llm_model": "gpt-4o-mini",
                "max_tokens_per_run": 80_000,
                "max_themes": 5,
            },
            delivery={
                "google_doc": {
                    "document_id": "1AbCdEfGhIj",
                    "document_title": "Weekly Review Pulse — Groww",
                },
                "email": {
                    "stakeholders": [],
                    "default_mode": "skip",
                },
            },
            schedule={"timezone": "Asia/Kolkata", "cron": "0 6 * * 1"},
        )

        call_endpoints: list[str] = []

        async def fake_call(endpoint, payload):
            call_endpoints.append(endpoint)
            return {
                "status": "success",
                "result": {"writeControl": {"requiredRevisionId": "rev-skip"}},
            }

        with patch.object(McpHttpClient, "call_tool", side_effect=fake_call):
            result = await deliver(sample_report, config)

        assert call_endpoints == ["append_to_doc"]  # email not called
        assert result.doc is not None
        assert result.email is None

    @pytest.mark.asyncio
    async def test_deliver_doc_failure_propagates(
        self, sample_report: PulseReport, product_config: ProductConfig, monkeypatch
    ):
        """When doc append fails, the error is surfaced as DeliveryError."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")

        async def fake_call(endpoint, payload):
            raise McpHttpClientError("Connection refused")

        with patch.object(McpHttpClient, "call_tool", side_effect=fake_call):
            with pytest.raises(DeliveryError, match="Doc append failed"):
                await deliver(sample_report, product_config)

    @pytest.mark.asyncio
    async def test_deliver_email_failure_propagates(
        self, sample_report: PulseReport, product_config: ProductConfig, monkeypatch
    ):
        """When email creation fails, it surfaces as DeliveryError."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")

        call_count = 0

        async def fake_call(endpoint, payload):
            nonlocal call_count
            call_count += 1
            if endpoint == "append_to_doc":
                return {
                    "status": "success",
                    "result": {"writeControl": {"requiredRevisionId": "rev-ok"}},
                }
            if endpoint == "create_email_draft":
                raise McpHttpClientError("Gmail API timeout")
            return {"status": "error"}

        with patch.object(McpHttpClient, "call_tool", side_effect=fake_call):
            with pytest.raises(DeliveryError, match="Email draft creation failed"):
                await deliver(sample_report, product_config)

    @pytest.mark.asyncio
    async def test_deliver_doc_url_in_email(
        self, sample_report: PulseReport, product_config: ProductConfig, monkeypatch
    ):
        """The email payload includes a working Google Docs URL."""
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")

        captured_email_payload = {}

        async def fake_call(endpoint, payload):
            if endpoint == "create_email_draft":
                nonlocal captured_email_payload
                captured_email_payload = payload
                return {
                    "status": "success",
                    "result": {"message": {"id": "msg-url"}},
                }
            return {"status": "success"}

        with patch.object(McpHttpClient, "call_tool", side_effect=fake_call):
            await deliver(sample_report, product_config)

        doc_id = product_config.delivery.google_doc.document_id
        expected_url = f"https://docs.google.com/document/d/{doc_id}/edit"
        assert expected_url in captured_email_payload["body"]
        assert "Read full report" in captured_email_payload["body"]


# ===========================================================================
# Delivery integration (Phase 5 + Phase 6)
# ===========================================================================


class TestDeliveryIntegration:
    """End-to-end test: Phase 5 render → Phase 6 deliver."""

    @pytest.mark.asyncio
    async def test_render_then_deliver_chain(
        self, sample_report: PulseReport, product_config: ProductConfig, monkeypatch
    ):
        """The full pipeline: render a PulseReport, then deliver via mocked MCP."""
        from pulse.render import render

        # Phase 5: render
        rendered = render(sample_report)
        assert "doc_batch_update" in rendered
        assert "email_payload" in rendered

        doc_requests = rendered["doc_batch_update"]["requests"]
        assert len(doc_requests) > 0
        assert doc_requests[0]["insertText"]["text"]  # has content

        email = rendered["email_payload"]
        assert email["subject"]
        assert email["body_html"]

        # Phase 6: deliver (mocked)
        monkeypatch.setenv("PULSE_MCP_SERVER_URL", "https://mcp.example.com")

        async def fake_call(endpoint, payload):
            if endpoint == "append_to_doc":
                # Verify doc payload matches what render produced
                assert payload["doc_id"] == product_config.delivery.google_doc.document_id
                assert "content" in payload
                return {
                    "status": "success",
                    "result": {"writeControl": {"requiredRevisionId": "rev-int"}},
                }
            if endpoint == "create_email_draft":
                assert "Groww Weekly Review Pulse" in payload["subject"]
                return {
                    "status": "success",
                    "result": {"message": {"id": "msg-int"}},
                }
            return {"status": "error"}

        with patch.object(McpHttpClient, "call_tool", side_effect=fake_call):
            result = await deliver(sample_report, product_config)

        assert result.doc is not None and result.doc.appended
        assert result.email is not None and result.email.message_id == "msg-int"
