import asyncio
from typing import Dict, Any
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

class PlayStoreClient:
    def __init__(self, mcp_server_command: str, mcp_server_args: list[str], cwd: str):
        self.server_params = StdioServerParameters(
            command=mcp_server_command,
            args=mcp_server_args,
            cwd=cwd
        )

    async def fetch_reviews(self, app_id: str, start_date_iso: str, end_date_iso: str, max_reviews: int = 10000) -> Dict[str, Any]:
        """
        Invokes the fetch_reviews tool on the play-store-reviews MCP server.
        """
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "fetch_reviews",
                    arguments={
                        "app_id": app_id,
                        "start_date_iso": start_date_iso,
                        "end_date_iso": end_date_iso,
                        "max_reviews": max_reviews
                    }
                )
                
                # result.content contains the response from the tool. 
                # According to standard MCP, the content is a list of TextContent/ImageContent objects.
                import json
                try:
                    data = json.loads(result.content[0].text)
                    return data
                except Exception as e:
                    return {"status": "error", "message": f"Failed to parse MCP response: {str(e)}"}
