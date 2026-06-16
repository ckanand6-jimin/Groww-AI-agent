import datetime
import os
import sys

# Add the parent directory to sys.path if running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.play_store_reviews.server import fetch_reviews

def generate_10_weeks_cache(app_id: str = "com.nextbillion.groww"):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(weeks=10)
    
    print(f"Generating cache for {app_id}")
    print(f"Timeframe: {start_date.isoformat()} to {end_date.isoformat()}")
    
    result = fetch_reviews(
        app_id=app_id,
        start_date_iso=start_date.isoformat(),
        end_date_iso=end_date.isoformat(),
        max_reviews=50000,
        lang="en",
        country="in"
    )
    
    if result.get("status") == "success":
        print(f"Successfully generated cache!")
        print(f"Path: {result.get('cache_dir')}")
        print(f"Reviews fetched: {result.get('reviews_fetched')}")
    else:
        print(f"Error generating cache: {result.get('message')}")

if __name__ == "__main__":
    generate_10_weeks_cache()
