import os
import json
import tempfile
import pytest

from pulse.ingest.pii import scrub_pii
from pulse.ingest.adapter import ingest_from_cache

def test_scrub_pii():
    text1 = "Contact me at user@example.com for details."
    assert scrub_pii(text1) == "Contact me at [EMAIL] for details."
    
    text2 = "Call 1-800-555-1234 or +91 9876543210"
    assert scrub_pii(text2) == "Call [PHONE] or [PHONE]"
    
    text3 = "Here is my URL: https://example.com/token=123"
    assert scrub_pii(text3) == "Here is my URL: [URL]"
    
    text4 = "This text has no PII."
    assert scrub_pii(text4) == text4

def test_ingest_adapter():
    # Setup temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_cache_data = [
            {
                "text": "This is a great app! email@me.com",
                "rating": 5,
                "published_at": "2026-06-08T16:25:26"
            },
            {
                "text": "   ", # empty should be filtered out
                "rating": 1,
                "published_at": "2026-06-08T16:25:26"
            },
            {
                "text": "Duplicate text",
                "rating": 4,
                "published_at": "2026-06-08T10:00:00"
            },
            {
                "text": "Duplicate text",
                "rating": 4,
                "published_at": "2026-06-08T10:00:00"
            }
        ]
        
        with open(os.path.join(temp_dir, "reviews_normalized.json"), "w") as f:
            json.dump(test_cache_data, f)
            
        reviews, stats = ingest_from_cache(temp_dir, "2026-W23", "test_groww")
        
        # 1 valid, 1 empty (filtered), 2 duplicates (1 deduped out) -> total 2
        assert len(reviews) == 2
        
        # Check PII scrubbed
        assert any(r.text == "This is a great app! [EMAIL]" for r in reviews)
        
        # Check deduplication
        assert any(r.text == "Duplicate text" for r in reviews)
        
        # Check stats
        assert stats["fetched"] == 4
        assert stats["validation_failures"] == 1
        assert stats["deduped_out"] == 1
        assert stats["final_count"] == 2
