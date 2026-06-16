from play_store_reviews.server import health_check

def test_health_check_returns_ok() -> None:
    result = health_check()
    assert result["status"] == "ok"

