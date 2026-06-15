"""Google Docs MCP client — wraps POST /append_to_doc.

Transforms a Phase-5 rendered PulseReport into the request body expected
by the Railway MCP server's ``POST /append_to_doc`` endpoint.

find_heading and get_document are NOT called — idempotency is delegated
to the run ledger in Phase 7.
"""

from __future__ import annotations

import json

from pulse.models.models import DocDeliveryInfo, PulseReport
from pulse.render.docs import build_batch_update_requests, build_heading_text
from pulse.delivery.mcp_http_client import McpHttpClient


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
        batch = build_batch_update_requests(report)
        heading_text = build_heading_text(report)

        payload = {
            "doc_id": document_id,
            "content": json.dumps(batch),  # batch-update requests as JSON string
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
