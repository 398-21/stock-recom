import feedparser
from datetime import datetime, timezone
from typing import List, Dict


def fetch_google_news_rss(query: str, max_items: int = 15) -> List[Dict]:
    """
    Free Google News RSS fetcher (no API key).
    """
    q = query.replace(" ", "+")
    url = (
        f"https://news.google.com/rss/search"
        f"?q={q}&hl=en-US&gl=US&ceid=US:en"
    )

    feed = feedparser.parse(url)
    items = []

    for entry in feed.entries[:max_items]:
        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(
                *entry.published_parsed[:6],
                tzinfo=timezone.utc
            ).isoformat()

        items.append({
            "title": entry.title,
            "publishedAt": published,
            "source": entry.source.title if hasattr(entry, "source") else "Google News",
            "url": entry.link,
        })

    return items
