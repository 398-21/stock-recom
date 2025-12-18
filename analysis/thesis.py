from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class Thesis:
    thesis_type: str               # CATALYST / FUNDAMENTAL / TECHNICAL
    thesis_text: str               # "Stock should move because X"
    window_start: Optional[str]    # YYYY-MM-DD
    window_end: Optional[str]      # YYYY-MM-DD
    top_risks: List[str]           # max 2 suggested
    invalidation: str              # "fails if Y happens"
    notes: str = ""                # optional


@dataclass
class ThesisPanel:
    headline: str
    window: str
    risks: List[str]
    invalidation: str
    completeness_score: int        # 0..100
    completeness_level: str        # LOW / MED / HIGH


def _clamp(x: int) -> int:
    return max(0, min(100, x))


def build_thesis_panel(thesis: Thesis) -> ThesisPanel:
    # scoring is purely about clarity/completeness, not "correctness"
    score = 0

    if thesis.thesis_type and thesis.thesis_type.strip():
        score += 20
    if thesis.thesis_text and len(thesis.thesis_text.strip()) >= 12:
        score += 30
    if thesis.invalidation and len(thesis.invalidation.strip()) >= 8:
        score += 25
    if thesis.top_risks and any(r.strip() for r in thesis.top_risks):
        score += 15
    if thesis.window_start or thesis.window_end:
        score += 10

    score = _clamp(score)
    if score >= 80:
        lvl = "HIGH"
    elif score >= 50:
        lvl = "MED"
    else:
        lvl = "LOW"

    window = "N/A"
    if thesis.window_start and thesis.window_end:
        window = f"{thesis.window_start} → {thesis.window_end}"
    elif thesis.window_start and not thesis.window_end:
        window = f"From {thesis.window_start}"
    elif thesis.window_end and not thesis.window_start:
        window = f"Until {thesis.window_end}"

    headline = thesis.thesis_text.strip() if thesis.thesis_text else "N/A"

    risks = [r.strip() for r in (thesis.top_risks or []) if r.strip()]
    risks = risks[:2] if risks else ["N/A"]

    invalidation = thesis.invalidation.strip() if thesis.invalidation else "N/A"

    return ThesisPanel(
        headline=headline,
        window=window,
        risks=risks,
        invalidation=invalidation,
        completeness_score=score,
        completeness_level=lvl,
    )
