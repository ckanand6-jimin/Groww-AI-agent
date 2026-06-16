import json
import os
import emoji
from datetime import datetime
from typing import List, Dict

from .models import Review

class CacheManager:
    def __init__(self, base_cache_dir: str = "data/cache/groww"):
        # Default points to repo root/data/cache/groww since MCP is run from there or relatively
        self.base_cache_dir = base_cache_dir

    def _normalize_review(self, review: Review) -> Dict:
        """
        Convert to normalized format: only text, rating, published_at.
        """
        return {
            "text": review.content,
            "rating": review.score,
            "published_at": review.at.isoformat()
        }

    def _is_valid_normalized(self, review: Review) -> bool:
        """
        Validation rules:
        - Remove reviews with less than 8 words
        - Remove reviews containing emojis
        - Remove empty reviews (handled by word count)
        - Remove non-English reviews (Assuming lang='en' handles this mostly, but we can check if it's purely empty or ascii/valid. Actually, since we request lang='en' from scraper, we rely on Play Store's lang detection).
        """
        content = review.content.strip()
        
        # Empty check
        if not content:
            return False
            
        # Word count check
        words = content.split()
        if len(words) < 8:
            return False
            
        # Emoji check
        if emoji.emoji_count(content) > 0:
            return False
            
        return True

    def save_cache(self, start_date: datetime, end_date: datetime, reviews: List[Review], fetch_meta: Dict) -> str:
        """
        Save reviews to cache directory.
        Path format: data/cache/groww/{date}/
        where {date} could be end_date or a formatted range.
        Using end_date format: YYYY-MM-DD
        """
        date_str = end_date.strftime("%Y-%m-%d")
        cache_dir = os.path.join(self.base_cache_dir, date_str)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 1. Save raw
        raw_path = os.path.join(cache_dir, "reviews_raw.json")
        raw_data = [r.model_dump(mode='json') for r in reviews]
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        # 2. Save normalized
        normalized_path = os.path.join(cache_dir, "reviews_normalized.json")
        valid_reviews = [r for r in reviews if self._is_valid_normalized(r)]
        normalized_data = [self._normalize_review(r) for r in valid_reviews]
        
        with open(normalized_path, "w", encoding="utf-8") as f:
            json.dump(normalized_data, f, ensure_ascii=False, indent=2)

        # 3. Save manifest
        manifest_path = os.path.join(cache_dir, "manifest.json")
        fetch_meta["raw_count"] = len(reviews)
        fetch_meta["normalized_count"] = len(normalized_data)
        fetch_meta["cache_date"] = date_str
        
        # Convert any datetimes in fetch_meta
        for k, v in fetch_meta.items():
            if isinstance(v, datetime):
                fetch_meta[k] = v.isoformat()
                
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(fetch_meta, f, ensure_ascii=False, indent=2)

        return cache_dir
