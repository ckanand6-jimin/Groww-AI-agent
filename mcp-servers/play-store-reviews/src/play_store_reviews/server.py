"""MCP server entrypoint — Phase 1 will register Play Store tools here."""

from datetime import datetime
from typing import Dict, Any, Optional
import os

from mcp.server.fastmcp import FastMCP
from google_play_scraper import app as get_app_details

from .scraper import PlayStoreScraper
from .cache import CacheManager

mcp = FastMCP("play-store-reviews")
scraper = PlayStoreScraper()
cache_manager = CacheManager()


@mcp.tool()
def health_check() -> dict[str, str]:
    """Return server status."""
    return {"status": "ok", "phase": "1"}


@mcp.tool()
def get_app_metadata(app_id: str) -> dict[str, Any]:
    """Get metadata for a specific Play Store app."""
    try:
        details = get_app_details(app_id)
        return {
            "title": details.get("title"),
            "appId": details.get("appId"),
            "url": details.get("url"),
            "score": details.get("score"),
            "reviews": details.get("reviews")
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def fetch_reviews(app_id: str, start_date_iso: str, end_date_iso: str, max_reviews: int = 10000, lang: str = "en", country: str = "in") -> dict[str, Any]:
    """
    Fetch reviews from the Play Store and cache them locally.
    Dates must be in ISO format (YYYY-MM-DDTHH:MM:SS) or just YYYY-MM-DD.
    Returns the path to the cache directory and fetch statistics.
    """
    try:
        if "T" in start_date_iso:
            start_date = datetime.fromisoformat(start_date_iso)
        else:
            start_date = datetime.strptime(start_date_iso, "%Y-%m-%d")
            
        if "T" in end_date_iso:
            end_date = datetime.fromisoformat(end_date_iso)
        else:
            end_date = datetime.strptime(end_date_iso, "%Y-%m-%d")
            
        reviews = scraper.fetch_reviews_in_window(
            app_id=app_id,
            start_date=start_date,
            end_date=end_date,
            lang=lang,
            country=country,
            max_reviews=max_reviews
        )
        
        truncated = len(reviews) >= max_reviews
        fetch_meta = {
            "app_id": app_id,
            "fetched_at": datetime.now(),
            "start_date": start_date,
            "end_date": end_date,
            "truncated": truncated
        }
        
        cache_dir = cache_manager.save_cache(
            start_date=start_date,
            end_date=end_date,
            reviews=reviews,
            fetch_meta=fetch_meta
        )
        
        return {
            "status": "success",
            "cache_dir": cache_dir,
            "reviews_fetched": len(reviews),
            "truncated": truncated
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
