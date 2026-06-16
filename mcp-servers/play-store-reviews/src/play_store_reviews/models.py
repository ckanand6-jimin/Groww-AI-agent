import hashlib
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class Review(BaseModel):
    # Original scraper fields
    reviewId: str
    userName: str
    userImage: str
    content: str
    score: int
    thumbsUpCount: int
    reviewCreatedVersion: Optional[str] = None
    at: datetime
    replyContent: Optional[str] = None
    repliedAt: Optional[datetime] = None
    appVersion: Optional[str] = None

    # Stable review ID generated post-fetch
    stable_review_id: str = ""

    def generate_stable_id(self, app_id: str) -> str:
        """
        Generate a stable review ID based on app_id, original reviewId, reviewer name, date, and text hash.
        This handles cases where the store reviewId might not be perfectly stable or we want a cross-store format.
        """
        base_string = f"{app_id}_{self.reviewId}_{self.userName}_{self.at.isoformat()}"
        text_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()[:8]
        return f"{base_string}_{text_hash}"


class FetchRequest(BaseModel):
    app_id: str
    start_date: datetime
    end_date: datetime
    max_reviews: Optional[int] = None
    locale: str = "en"
    country: str = "in"


class FetchResult(BaseModel):
    app_id: str
    fetched_at: datetime
    start_date: datetime
    end_date: datetime
    reviews_count: int
    truncated: bool
