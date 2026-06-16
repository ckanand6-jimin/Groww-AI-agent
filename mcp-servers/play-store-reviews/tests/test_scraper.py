import datetime
from unittest.mock import patch, MagicMock

from play_store_reviews.scraper import PlayStoreScraper
from play_store_reviews.models import Review

@patch("play_store_reviews.scraper.reviews")
def test_fetch_reviews_in_window(mock_reviews):
    mock_reviews.return_value = (
        [
            {
                "reviewId": "1",
                "userName": "test1",
                "userImage": "img",
                "content": "review 1",
                "score": 5,
                "thumbsUpCount": 0,
                "at": datetime.datetime(2023, 1, 5)
            },
            {
                "reviewId": "2",
                "userName": "test2",
                "userImage": "img",
                "content": "review 2",
                "score": 4,
                "thumbsUpCount": 0,
                "at": datetime.datetime(2023, 1, 1)
            }
        ],
        None # no continuation token
    )

    scraper = PlayStoreScraper()
    start_date = datetime.datetime(2023, 1, 2)
    end_date = datetime.datetime(2023, 1, 10)
    
    result = scraper.fetch_reviews_in_window("com.test", start_date, end_date)
    
    # Should only return review 1 because review 2 is before start_date
    assert len(result) == 1
    assert result[0].reviewId == "1"
    
@patch("play_store_reviews.scraper.reviews")
def test_fetch_reviews_max_retries(mock_reviews):
    mock_reviews.side_effect = Exception("Failed")
    scraper = PlayStoreScraper(max_retries=2, base_backoff=0.01)
    
    start_date = datetime.datetime(2023, 1, 2)
    end_date = datetime.datetime(2023, 1, 10)
    
    result = scraper.fetch_reviews_in_window("com.test", start_date, end_date)
    assert len(result) == 0
    assert mock_reviews.call_count == 2
