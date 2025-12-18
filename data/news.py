from __future__ import annotations

import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Optional


def fetch_google_news_rss(query: str, max_items: int = 15) -> List[Dict]:
    q = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    items: List[Dict] = []
    for e in (feed.entries or [])[:max_items]:
        published: Optional[str] = None
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).isoformat()

        source = "Google News"
        if hasattr(e, "source") and getattr(e.source, "title", None):
            source = e.source.title

        items.append({
            "title": getattr(e, "title", ""),
            "publishedAt": published,
            "source": source,
            "url": getattr(e, "link", ""),
        })
    return items
