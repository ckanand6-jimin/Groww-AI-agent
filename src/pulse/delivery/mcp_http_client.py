"""Generic HTTP client for the Railway-deployed Google MCP server.

Phase 6 — provides the transport layer that docs_client and gmail_client
use to call the two deployed endpoints:

    POST /append_to_doc
    POST /create_email_draft

No direct Google APIs. No OAuth. HTTP only.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


class McpHttpClientError(Exception):
    """Raised when an MCP HTTP call fails (network, status, or parse error)."""

    pass


class McpHttpClient:
    """Thin HTTP wrapper around the Railway Google MCP server.

    Reads the server URL from the ``PULSE_MCP_SERVER_URL`` environment
    variable.  Every tool call is a ``POST`` to ``{base_url}/{endpoint}``
    with a JSON body.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        resolved = (base_url or os.environ.get("PULSE_MCP_SERVER_URL", "")).rstrip("/")
        if not resolved:
            raise McpHttpClientError(
                "PULSE_MCP_SERVER_URL environment variable is not set. "
                "Set it to the Railway MCP server URL (e.g. https://google-mcp.railway.app)."
            )
        self.base_url: str = resolved

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def call_tool(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST a JSON payload to an MCP server endpoint.

        Args:
            endpoint: Path segment (e.g. ``"append_to_doc"``).
            payload: JSON-serialisable request body.

        Returns:
            Parsed JSON response dict.

        Raises:
            McpHttpClientError: On any transport, HTTP-status, or
                JSON-parse failure.
        """
        url = f"{self.base_url}/{endpoint}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                raise McpHttpClientError(
                    f"MCP server returned {exc.response.status_code} "
                    f"for POST {url}: {exc.response.text[:500]}"
                ) from exc
            except httpx.RequestError as exc:
                raise McpHttpClientError(
                    f"Failed to connect to MCP server at {url}: {exc}"
                ) from exc
            except ValueError as exc:
                raise McpHttpClientError(
                    f"Invalid JSON response from {url}: {exc}"
                ) from exc
