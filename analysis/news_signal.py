from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import math


@dataclass
class NewsSignal:
    intensity: float   # 0..1
    tone: float        # -1..+1 (informational only)
    n_items_used: int
    explanation: str


_POS = {"beats","beat","surge","soar","soars","upgrade","upgraded","approval","approved","positive","win","wins","record","growth","profit","partnership","contract"}
_NEG = {"miss","misses","plunge","falls","fall","downgrade","dilution","offering","atm","pipe","secondary","lawsuit","fraud","bankruptcy","halt","delist","delisting","guidance cut"}


def _age_hours(published_at: str, now_utc: datetime) -> float:
    # Be permissive; if parsing fails treat as moderately old
    if not published_at:
        return 48.0
    try:
        # many feeds give RFC-ish or ISO; easiest safe fallback: ignore exact parse
        # if your feed already normalizes to ISO, adjust here later.
        return 24.0
    except Exception:
        return 48.0


def build_news_signal(
    news_items: List[Dict[str, Any]],
    *,
    now_utc: Optional[datetime] = None,
    max_items: int = 25,
    half_life_hours: float = 36.0,
) -> NewsSignal:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    items = (news_items or [])[:max_items]
    if not items:
        return NewsSignal(0.0, 0.0, 0, "No news items available; using price-only uncertainty.")

    total_w = 0.0
    pos_w = 0.0
    neg_w = 0.0

    for it in items:
        title = (it.get("title") or "").lower()
        src = (it.get("source") or "").lower()
        text = f"{title} {src}"

        age_h = _age_hours(it.get("publishedAt") or it.get("published") or "", now_utc)
        w = math.exp(-math.log(2) * (age_h / max(1e-6, half_life_hours)))

        pos_hits = sum(1 for p in _POS if p in text)
        neg_hits = sum(1 for n in _NEG if n in text)

        pos_w += w * pos_hits
        neg_w += w * neg_hits
        total_w += w

    intensity = 1.0 - math.exp(-total_w / 2.0)
    intensity = max(0.0, min(1.0, intensity))

    denom = (pos_w + neg_w + 1e-9)
    tone = (pos_w - neg_w) / denom
    tone = max(-1.0, min(1.0, tone))

    explanation = (
        f"News intensity={intensity:.2f} (fresh-news pressure). "
        f"Tone={tone:.2f} (informational only). "
        "Forecast uses intensity to widen range; does not use tone for direction."
    )

    return NewsSignal(intensity=intensity, tone=tone, n_items_used=len(items), explanation=explanation)
