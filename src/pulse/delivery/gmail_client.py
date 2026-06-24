"""Gmail MCP client — wraps POST /create_email_draft.

Transforms a Phase-5 rendered email payload into the request body
expected by the Railway MCP server's ``POST /create_email_draft``
endpoint.
"""

from __future__ import annotations

import os
from typing import List

from pulse.models.models import EmailDeliveryInfo, PulseReport
from pulse.render.email import build_email_payload
from pulse.delivery.mcp_http_client import McpHttpClient


class GmailClient:
    """Client for creating Gmail drafts via the Railway MCP server.

    Uses ``POST /create_email_draft`` exclusively.  No direct Gmail API
    calls.  The ``send`` mode is gated behind the ``PULSE_ALLOW_SEND``
    environment variable (must be ``"true"``).
    """

    def __init__(self, http: McpHttpClient) -> None:
        self._http = http

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_draft(
        self,
        *,
        report: PulseReport,
        recipients: List[str],
        doc_url: str = "",
        mode: str = "draft",
    ) -> EmailDeliveryInfo:
        """Create a Gmail draft with the weekly report teaser.

        Args:
            report:     ``PulseReport`` to render into a teaser email.
            recipients: List of recipient email addresses.
            doc_url:    Google Doc URL for the "Read full report" CTA.
            mode:       ``"draft"`` (default), ``"send"``, or ``"skip"``.
                        ``"send"`` requires ``PULSE_ALLOW_SEND=true``.

        Returns:
            ``EmailDeliveryInfo`` with ``message_id`` and metadata.

        Raises:
            McpHttpClientError: If the MCP server returns an error.
            ValueError:         If ``mode="send"`` but the send guard is
                                not satisfied.
        """
        # --- Send guard ---
        if mode == "send":
            if os.environ.get("PULSE_ALLOW_SEND", "").lower() != "true":
                raise ValueError(
                    "PULSE_ALLOW_SEND must be set to 'true' to send emails. "
                    "Use mode='draft' for development and staging."
                )

        # --- Build email payload (Phase 5 renderer) ---
        email = build_email_payload(report, recipients=recipients, doc_url=doc_url)

        # --- Transform to MCP request shape ---
        payload = {
            "to": email["recipients"][0] if email["recipients"] else "",
            "subject": email["subject"],
            "body": email["body_html"],
        }

        response = await self._http.call_tool("create_email_draft", payload)

        # Parse the actual MCP server response shape:
        # { "status": "success", "result": { "message": { "id": "...", "threadId": "..." } } }
        result = response.get("result", {})
        msg = result.get("message", {})
        return EmailDeliveryInfo(
            mode=mode,
            message_id=msg.get("id", ""),
            recipients=recipients,
            sent_at=response.get("sent_at", ""),
        )
