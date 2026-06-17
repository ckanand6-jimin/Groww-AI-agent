"""Google Docs MCP client — wraps POST /append_to_doc.

Transforms a Phase-5 rendered PulseReport into the request body expected
by the Railway MCP server's ``POST /append_to_doc`` endpoint.

find_heading and get_document are NOT called — idempotency is delegated
to the run ledger in Phase 7.
"""

from __future__ import annotations

from pulse.models.models import DocDeliveryInfo, PulseReport
from pulse.render.docs import build_heading_text
from pulse.delivery.mcp_http_client import McpHttpClient


def _build_plain_text_content(report: PulseReport) -> str:
    """Build a human-readable plain-text version of the report for Google Docs.

    The MCP server inserts ``content`` as raw text into the document, so we
    produce pre-formatted text with headings and bullets rather than Google
    Docs batch-update API JSON.
    """
    lines: list[str] = []

    # H2 heading
    lines.append(build_heading_text(report))
    lines.append("")

    # Intro paragraph
    lines.append(
        f"Period covered: {report.period.start_date} to {report.period.end_date} "
        f"({report.period.window_weeks}-week rolling window). "
        f"Reviews fetched: {report.stats.total_reviews_fetched}, "
        f"after deduplication: {report.stats.reviews_after_dedupe}, "
        f"clustered into {report.stats.clusters_found} themes, "
        f"top {report.stats.top_themes_selected} selected for this report. "
        f"Generated at {report.generated_at}."
    )
    lines.append("")

    # Top themes
    lines.append("Top themes")
    for theme in report.themes:
        lines.append(
            f"  \u2022 {theme.name} \u2014 {theme.summary} "
            f"(size: {theme.cluster_size}, avg rating: {theme.avg_rating:.1f})"
        )
    lines.append("")

    # Real user quotes
    lines.append("Real user quotes")
    for theme in report.themes:
        for quote in theme.quotes:
            lines.append(f"  \u2022 \u201c{quote}\u201d")
    lines.append("")

    # Action ideas
    lines.append("Action ideas")
    for theme in report.themes:
        for action in theme.action_ideas:
            lines.append(f"  \u2022 {action.title}: {action.rationale}")
    lines.append("")

    # Who this helps
    lines.append("Who this helps")
    notes = report.audience_notes
    lines.append(f"  \u2022 Product team: {notes.product}")
    lines.append(f"  \u2022 Support team: {notes.support}")
    lines.append(f"  \u2022 Leadership: {notes.leadership}")
    lines.append("")

    return "\n".join(lines)


class DocsClient:
    """Client for appending weekly report sections to a Google Doc.

    Uses the Railway MCP server's ``POST /append_to_doc`` HTTP endpoint
    exclusively.  No direct Google Docs API calls.
    """

    def __init__(self, http: McpHttpClient) -> None:
        self._http = http

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def append_to_doc(
        self, *, document_id: str, report: PulseReport
    ) -> DocDeliveryInfo:
        """Append a full weekly section to the end of the target document.

        The method:
          1. Calls Phase 5 ``build_batch_update_requests(report)`` to
             produce the structured Google Docs batch-update payload.
          2. Wraps it with ``document_id`` and POSTs to
             ``/append_to_doc``.

        Args:
            document_id: Google Doc ID (from ``groww.yaml``).
            report:      Complete ``PulseReport`` to render and append.

        Returns:
            A ``DocDeliveryInfo`` populated with the MCP server response
            fields (or empty strings on missing keys).

        Raises:
            McpHttpClientError: If the MCP server returns a non-2xx
                status or the connection fails.
        """
        heading_text = build_heading_text(report)
        plain_text = _build_plain_text_content(report)

        payload = {
            "doc_id": document_id,
            "content": plain_text,
        }

        response = await self._http.call_tool("append_to_doc", payload)

        # Parse the actual MCP server response shape:
        # { "status": "success", "result": { "writeControl": { "requiredRevisionId": "..." }, "documentId": "..." } }
        result = response.get("result", {})
        return DocDeliveryInfo(
            document_id=document_id,
            heading_text=heading_text,
            heading_anchor=response.get("heading_anchor", ""),
            revision_id=(
                result.get("writeControl", {}).get("requiredRevisionId", "")
            ),
            appended=response.get("status") == "success",
        )
