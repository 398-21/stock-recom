from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple


# Common offering / registration forms (heuristic)
OFFERING_FORMS = {
    "S-1", "S-3", "S-8", "F-1", "F-3",
    "424B1", "424B2", "424B3", "424B4", "424B5", "424B7",
    "POS AM",
}
WATCH_FORMS = {"8-K"}  # financing often disclosed here


@dataclass
class DilutionScore:
    score: int                 # 0-100
    level: str                 # LOW/MED/HIGH
    reasons: List[str]


def score_dilution_risk(filings: List[Dict[str, str]], lookback_n: int = 15) -> DilutionScore:
    """
    Heuristic scoring based on presence/frequency of offering-related filings.
    """
    recent = filings[:lookback_n]
    score = 0
    reasons: List[str] = []

    offering_hits = []
    eightk_hits = []

    for f in recent:
        form = (f.get("form") or "").upper().strip()
        dt = f.get("filingDate") or "unknown date"

        if form in OFFERING_FORMS or form.startswith("424B"):
            offering_hits.append((dt, form))

        if form in WATCH_FORMS:
            eightk_hits.append((dt, form))

    # Weighting logic
    if offering_hits:
        score += 60
        reasons.append(f"Offering-related forms detected in recent filings: {', '.join([x[1] for x in offering_hits[:5]])}")

    if len(offering_hits) >= 2:
        score += 20
        reasons.append("Multiple offering-related filings recently (suggests active capital-raising posture).")

    if eightk_hits:
        score += 10
        reasons.append("Recent 8-K present (could include financing/ATM/PIPE updates; requires manual read).")

    # Cap at 100
    score = min(score, 100)

    if score >= 70:
        level = "HIGH"
    elif score >= 35:
        level = "MED"
    else:
        level = "LOW"

    if not reasons:
        reasons.append("No obvious offering-form signals in recent filings (not a guarantee).")

    return DilutionScore(score=score, level=level, reasons=reasons)
