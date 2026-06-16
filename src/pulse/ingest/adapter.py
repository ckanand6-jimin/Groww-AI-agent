import json
import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from pulse.models.models import Review
from pulse.ingest.pii import scrub_pii

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Resolve the repository root (4 parents up from this file).

    __file__ = .../pulse-agent/src/pulse/ingest/adapter.py
    parents[0] = .../pulse-agent/src/pulse/ingest/
    parents[1] = .../pulse-agent/src/pulse/
    parents[2] = .../pulse-agent/src/
    parents[3] = .../pulse-agent/
    parents[4] = repo root
    """
    return Path(__file__).resolve().parents[4]


def fetch_reviews_to_cache(
    app_id: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: str,
) -> int:
    """Fetch reviews from the Play Store and write normalized cache files.

    This is called automatically by ``ingest_from_cache`` when the cache
    directory does not yet contain ``reviews_normalized.json`` (e.g. in CI
    where no prior MCP server run has populated the cache).

    Uses ``google_play_scraper`` and the normalisation logic from the
    ``play-store-reviews`` MCP server package directly, avoiding the need
    to spawn an MCP stdio process.

    Returns:
        Number of normalised reviews written to cache.

    Raises:
        RuntimeError: If the scraper or cache write fails.
    """
    try:
        from google_play_scraper import Sort, reviews as gps_reviews
        import emoji
    except ImportError as exc:
        raise RuntimeError(
            "google-play-scraper and emoji packages are required for "
            "auto-fetch. Install with: pip install -e ./mcp-servers/play-store-reviews"
        ) from exc

    logger.info(
        "Cache miss at %s — fetching reviews for %s (%s .. %s)",
        cache_dir, app_id, start_date.date(), end_date.date(),
    )

    # --- Scrape reviews (mirrors PlayStoreScraper logic) ---
    fetched: list[dict] = []
    continuation_token = None
    max_reviews = 10_000

    while len(fetched) < max_reviews:
        try:
            result, continuation_token = gps_reviews(
                app_id,
                lang="en",
                country="in",
                sort=Sort.NEWEST,
                count=199,
                continuation_token=continuation_token,
            )
        except Exception as exc:
            logger.warning("Scraper error: %s", exc)
            break

        if not result:
            break

        oldest = datetime.max
        for item in result:
            review_date = item["at"]
            oldest = min(oldest, review_date)
            if start_date <= review_date <= end_date:
                fetched.append(item)

        if oldest < start_date or not continuation_token:
            break

    # --- Normalise (mirrors CacheManager._is_valid_normalized + _normalize_review) ---
    normalised: list[dict] = []
    for r in fetched:
        text = (r.get("content") or "").strip()
        if not text:
            continue
        if len(text.split()) < 8:
            continue
        try:
            if emoji.emoji_count(text) > 0:
                continue
        except Exception:
            pass  # If emoji check fails, include the review
        normalised.append({
            "text": text,
            "rating": r.get("score", 0),
            "published_at": r["at"].isoformat() if hasattr(r["at"], "isoformat") else str(r["at"]),
        })

    # --- Write cache files ---
    os.makedirs(cache_dir, exist_ok=True)

    raw_path = os.path.join(cache_dir, "reviews_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(fetched, f, ensure_ascii=False, indent=2, default=str)

    norm_path = os.path.join(cache_dir, "reviews_normalized.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(normalised, f, ensure_ascii=False, indent=2)

    manifest = {
        "app_id": app_id,
        "fetched_at": datetime.now().isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "raw_count": len(fetched),
        "normalized_count": len(normalised),
        "cache_date": end_date.strftime("%Y-%m-%d"),
    }
    manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Fetched %d reviews, %d after normalisation → %s", len(fetched), len(normalised), cache_dir)
    return len(normalised)


def ingest_from_cache(
    cache_dir: str,
    iso_week: str,
    product: str = "groww",
    *,
    auto_fetch: bool = True,
    app_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[List[Review], dict]:
    """
    Read reviews from the normalized cache.
    Apply defensive validation (check for empty texts), deduplicate based on content,
    and scrub PII.
    Writes an ingest snapshot metadata to the data/runs folder.
    Returns the deduplicated, scrubbed Review models and ingest stats.

    If the cache file does not exist and ``auto_fetch`` is True, attempts to
    fetch reviews from the Play Store directly (requires ``app_id``,
    ``start_date``, and ``end_date`` to be provided).
    """
    normalized_file = os.path.join(cache_dir, "reviews_normalized.json")

    if not os.path.exists(normalized_file):
        if auto_fetch and app_id and start_date and end_date:
            logger.warning(
                "Normalized cache not found at %s — attempting auto-fetch.",
                cache_dir,
            )
            fetch_reviews_to_cache(app_id, start_date, end_date, cache_dir)
        else:
            raise FileNotFoundError(
                f"Normalized cache file not found at {normalized_file}. "
                f"Run the Play Store MCP server first, or pass app_id/start_date/"
                f"end_date to enable auto-fetch."
            )

    with open(normalized_file, "r", encoding="utf-8") as f:
        raw_reviews = json.load(f)

    # Dedup map: hash -> Review
    deduped_reviews = {}
    validation_failures = 0

    for idx, r in enumerate(raw_reviews):
        text = r.get("text", "").strip()
        rating = r.get("rating", 0)
        published_at_str = r.get("published_at")

        # Defensive validation
        if not text:
            logger.warning(f"Skipping review {idx}: Text is empty.")
            validation_failures += 1
            continue
            
        try:
            if "T" in published_at_str:
                published_at = datetime.fromisoformat(published_at_str)
            else:
                published_at = datetime.strptime(published_at_str, "%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Skipping review {idx}: Invalid date format {published_at_str}. {e}")
            validation_failures += 1
            continue

        # Scrub PII
        scrubbed_text = scrub_pii(text)

        # Generate a stable ID since normalized data doesn't have it
        # We hash the original text + rating + published_at
        id_material = f"{text}_{rating}_{published_at_str}".encode('utf-8')
        review_id_hash = hashlib.md5(id_material).hexdigest()

        review = Review(
            review_id=review_id_hash,
            text=scrubbed_text,
            rating=rating,
            published_at=published_at
        )

        if review_id_hash not in deduped_reviews:
            deduped_reviews[review_id_hash] = review

    final_reviews = list(deduped_reviews.values())

    # Create snapshot metadata
    stats = {
        "fetched": len(raw_reviews),
        "validation_failures": validation_failures,
        "deduped_out": len(raw_reviews) - validation_failures - len(final_reviews),
        "final_count": len(final_reviews),
        "ingested_at": datetime.now().isoformat()
    }

    # Save ingest snapshot — use repo-root-relative path
    run_dir = _repo_root() / "data" / "runs" / product / iso_week
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / "ingest.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return final_reviews, stats
