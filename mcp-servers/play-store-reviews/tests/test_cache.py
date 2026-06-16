from datetime import datetime, timezone
from play_store_reviews.models import Review
from play_store_reviews.cache import CacheManager

def test_is_valid_normalized_valid():
    manager = CacheManager()
    review = Review(
        reviewId="123",
        userName="test",
        userImage="test.png",
        content="This is a valid review with more than eight words in it.",
        score=5,
        thumbsUpCount=0,
        at=datetime.now(timezone.utc)
    )
    assert manager._is_valid_normalized(review) == True

def test_is_valid_normalized_too_short():
    manager = CacheManager()
    review = Review(
        reviewId="123",
        userName="test",
        userImage="test.png",
        content="Too short review.",
        score=5,
        thumbsUpCount=0,
        at=datetime.now(timezone.utc)
    )
    assert manager._is_valid_normalized(review) == False

def test_is_valid_normalized_has_emoji():
    manager = CacheManager()
    review = Review(
        reviewId="123",
        userName="test",
        userImage="test.png",
        content="This is a valid length review but it has an emoji 😊 in it.",
        score=5,
        thumbsUpCount=0,
        at=datetime.now(timezone.utc)
    )
    assert manager._is_valid_normalized(review) == False

def test_normalize_review():
    manager = CacheManager()
    now = datetime.now(timezone.utc)
    review = Review(
        reviewId="123",
        userName="test",
        userImage="test.png",
        content="This is a valid review with more than eight words in it.",
        score=5,
        thumbsUpCount=0,
        at=now
    )
    normalized = manager._normalize_review(review)
    assert "reviewId" not in normalized
    assert "userName" not in normalized
    assert "text" in normalized
    assert "rating" in normalized
    assert "published_at" in normalized
    assert normalized["text"] == review.content
    assert normalized["rating"] == review.score
