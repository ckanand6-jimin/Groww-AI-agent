"""Phase 6 delivery orchestrator — doc append + email draft.

Orchestrates a single end-to-end delivery run: renders the PulseReport,
POSTs to /append_to_doc, then POSTs to /create_email_draft.

Phase 7 adds ``existing_delivery`` support: the run ledger passes
previously-completed delivery info so that already-succeeded steps
(doc append or email draft) are skipped on resume.
"""

from __future__ import annotations

from typing import Optional

from pulse.config import EmailMode, ProductConfig
from pulse.delivery.docs_client import DocsClient
from pulse.delivery.gmail_client import GmailClient
from pulse.delivery.mcp_http_client import McpHttpClient, McpHttpClientError
from pulse.models.models import DeliveryRecord, PulseReport


class DeliveryError(Exception):
    """Raised when a delivery step fails after exhausting retries."""

    pass


async def deliver(
    report: PulseReport,
    config: ProductConfig,
    *,
    mcp_base_url: Optional[str] = None,
    existing_delivery: Optional[DeliveryRecord] = None,
    email_mode_override: Optional[str] = None,
) -> DeliveryRecord:
    """Deliver a weekly PulseReport to Google Workspace.

    The flow:

    1. **Doc**: Calls ``DocsClient.append_to_doc()`` which renders the
       Phase 5 batch-update payload and POSTs it to
       ``/append_to_doc``.  Skipped if ``existing_delivery.doc.appended``
       is True (resume after partial failure).
    2. **Email**: Unless ``email_mode`` is ``"skip"``, calls
       ``GmailClient.create_draft()`` to create a Gmail draft with the
       teaser and a "Read full report" CTA link.  Skipped if
       ``existing_delivery.email.message_id`` is non-empty.

    Args:
        report:       Complete ``PulseReport`` to deliver.
        config:       ``ProductConfig`` with ``delivery.*`` settings.
        mcp_base_url: Optional override for ``PULSE_MCP_SERVER_URL``
                      (useful in tests).
        existing_delivery: Optional pre-existing ``DeliveryRecord`` from
                      a failed/in-progress run.  Steps that already
                      succeeded are skipped.
        email_mode_override: Optional override for the email mode
                      (e.g. from orchestrator CLI param).

    Returns:
        ``DeliveryRecord`` with ``doc`` and ``email`` delivery info.

    Raises:
        DeliveryError: If a non-skippable delivery step fails.
        McpHttpClientError: If the MCP server URL is not configured.
    """
    http = McpHttpClient(base_url=mcp_base_url)
    docs = DocsClient(http)
    gmail = GmailClient(http)

    doc_config = config.delivery.google_doc
    email_config = config.delivery.email

    # ------------------------------------------------------------------
    # 1. Append Doc section (skip if already succeeded on a prior attempt)
    # ------------------------------------------------------------------
    doc_info = None
    if existing_delivery and existing_delivery.doc and existing_delivery.doc.appended:
        doc_info = existing_delivery.doc
    else:
        try:
            doc_info = await docs.append_to_doc(
                document_id=doc_config.document_id,
                report=report,
            )
        except McpHttpClientError as exc:
            raise DeliveryError(f"Doc append failed: {exc}") from exc

    # ------------------------------------------------------------------
    # 2. Create email draft (unless skip or already succeeded)
    # ------------------------------------------------------------------
    email_mode: str = (
        email_mode_override
        or email_config.default_mode.value
    )  # "draft" | "send" | "skip"

    if email_mode == EmailMode.SKIP.value:
        return DeliveryRecord(doc=doc_info, email=None)

    # Skip email if already created in a prior attempt
    if existing_delivery and existing_delivery.email and existing_delivery.email.message_id:
        return DeliveryRecord(doc=doc_info, email=existing_delivery.email)

    doc_url = f"https://docs.google.com/document/d/{doc_config.document_id}/edit"

    try:
        email_info = await gmail.create_draft(
            report=report,
            recipients=email_config.stakeholders,
            doc_url=doc_url,
            mode=email_mode,
        )
    except (McpHttpClientError, ValueError) as exc:
        raise DeliveryError(f"Email draft creation failed: {exc}") from exc

    return DeliveryRecord(doc=doc_info, email=email_info)
