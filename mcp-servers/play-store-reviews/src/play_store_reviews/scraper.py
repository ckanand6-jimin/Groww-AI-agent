import time
import logging
from datetime import datetime
from typing import List

from google_play_scraper import Sort, reviews

from .models import Review

logger = logging.getLogger(__name__)

class PlayStoreScraper:
    def __init__(self, max_retries: int = 3, base_backoff: float = 2.0):
        self.max_retries = max_retries
        self.base_backoff = base_backoff

    def fetch_reviews_in_window(
        self,
        app_id: str,
        start_date: datetime,
        end_date: datetime,
        lang: str = 'en',
        country: str = 'in',
        max_reviews: int = 10000
    ) -> List[Review]:
        """
        Fetch reviews for the given app_id within the date window.
        Uses google-play-scraper. Sorts by NEWEST and paginates until we pass start_date.
        """
        fetched_reviews = []
        continuation_token = None
        
        while len(fetched_reviews) < max_reviews:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    result, continuation_token = reviews(
                        app_id,
                        lang=lang,
                        country=country,
                        sort=Sort.NEWEST,
                        count=199,  # Fetch in chunks
                        continuation_token=continuation_token
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Failed to fetch reviews (attempt {retry_count}/{self.max_retries}): {e}")
                    if retry_count == self.max_retries:
                        logger.error("Max retries reached. Aborting fetch.")
                        return fetched_reviews
                    time.sleep(self.base_backoff ** retry_count)

            if not result:
                break

            oldest_date_in_batch = datetime.max
            for item in result:
                # 'at' is already a datetime object from google-play-scraper
                review_date = item['at']
                oldest_date_in_batch = min(oldest_date_in_batch, review_date)
                
                # Check if it falls within the window
                if start_date <= review_date <= end_date:
                    r = Review(**item)
                    r.stable_review_id = r.generate_stable_id(app_id)
                    fetched_reviews.append(r)

            # If the oldest review in this batch is older than our start_date, we've gone far enough back.
            if oldest_date_in_batch < start_date:
                break
                
            if not continuation_token:
                break

        return fetched_reviews
