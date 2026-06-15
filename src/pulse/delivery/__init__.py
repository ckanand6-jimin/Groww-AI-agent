"""MCP delivery clients — Phase 6.

HTTP-based integration with the Railway-deployed Google MCP server.
No direct Google APIs. No OAuth.
"""

from pulse.delivery.deliver import DeliveryError, deliver
from pulse.delivery.docs_client import DocsClient
from pulse.delivery.gmail_client import GmailClient
from pulse.delivery.mcp_http_client import McpHttpClient, McpHttpClientError

__all__ = [
    "deliver",
    "DeliveryError",
    "DocsClient",
    "GmailClient",
    "McpHttpClient",
    "McpHttpClientError",
]
