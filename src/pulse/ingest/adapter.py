import json
import os
import hashlib
import logging
from datetime import datetime
from typing import List, Tuple

from pulse.models.models import Review
from pulse.ingest.pii import scrub_pii

logger = logging.getLogger(__name__)

def ingest_from_cache(cache_dir: str, iso_week: str, product: str = "groww") -> Tuple[List[Review], dict]:
    """
    Read reviews from the normalized cache.
    Apply defensive validation (check for empty texts), deduplicate based on content,
    and scrub PII.
    Writes an ingest snapshot metadata to the data/runs folder.
    Returns the deduplicated, scrubbed Review models and ingest stats.
    """
    normalized_file = os.path.join(cache_dir, "reviews_normalized.json")
    if not os.path.exists(normalized_file):
        raise FileNotFoundError(f"Normalized cache file not found at {normalized_file}")

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

    # Save ingest snapshot
    run_dir = os.path.join("data", "runs", product, iso_week)
    os.makedirs(run_dir, exist_ok=True)
    
    snapshot_path = os.path.join(run_dir, "ingest.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return final_reviews, stats
