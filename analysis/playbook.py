from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PostEventPlaybook:
    positive: List[str]
    neutral: List[str]
    negative: List[str]


def build_post_event_playbook(mode: str, dilution_level: str, fin_health_level: str) -> PostEventPlaybook:
    m = (mode or "HYBRID").upper()
    dil = (dilution_level or "LOW").upper()
    fin = (fin_health_level or "MED").upper()

    # baseline actions
    pos = [
        "Scale out into strength: take partial profits into first spike.",
        "Watch volume: if volume fades quickly, treat spike as sellable rather than “new trend.”",
    ]
    neu = [
        "If reaction is muted, avoid forcing trades: wait for structure (BOS) or AVWAP reclaim before adding.",
        "Re-check filings/news: neutral outcomes often precede financing events in small caps.",
    ]
    neg = [
        "Protect capital first: cut if thesis is invalidated; do not average down without a new, explicit thesis.",
        "Wait for stabilization: base + volume return + reclaim key levels before any re-entry.",
    ]

    # mode nuance
    if m == "PURE_CATALYST":
        pos.insert(0, "Treat the catalyst as resolved: decide whether to hold only if there is a second catalyst; otherwise de-risk.")
        neu.insert(0, "Do nothing until new information exists: catalyst edge is gone if outcome is neutral.")
        neg.insert(0, "Assume edge is broken: exit unless you can articulate a new catalyst with a new window.")
    elif m == "PURE_TECHNICAL":
        pos.insert(0, "Hold only while structure holds: trail using swing lows / AVWAP; exit on BOS against you.")
        neu.insert(0, "No edge without follow-through: wait for breakout + retest confirmation.")
        neg.insert(0, "Exit on invalidation level: technical setups fail quickly; respect stops.")
    else:  # HYBRID
        pos.insert(0, "Only add if both: catalyst narrative improves AND price holds above AVWAP with volume.")
        neu.insert(0, "Stay small unless BOS confirms: HYBRID requires confirmation to justify size.")
        neg.insert(0, "Reduce size aggressively if financing/dilution risk increases after the event.")

    # dilution nuance
    if dil == "HIGH":
        pos.append("Assume supply overhead: sell into rallies more aggressively (financing can cap upside).")
        neu.append("Do not chase: high dilution risk makes “pop then fade” more likely.")
        neg.append("Expect second-leg down risk: dilution headlines can accelerate downside.")

    # fundamentals nuance
    if fin == "LOW":
        pos.append("Do not convert a spike into a long-term hold without runway clarity.")
        neu.append("Watch cash/runway signals: low health often implies financing risk even without headlines.")
        neg.append("Avoid “hope holds”: low financial health increases tail-risk during drawdowns.")

    return PostEventPlaybook(positive=pos[:6], neutral=neu[:6], negative=neg[:6])
