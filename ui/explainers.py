from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# ---------- Core UI Labels (emoji + plain) ----------

def emoji_action_label(action: str) -> str:
    a = (action or "").upper()
    if a == "BUY":
        return "🟢 BUY"
    if a == "WAIT":
        return "🟡 WAIT"
    if a == "AVOID":
        return "🔴 AVOID"
    return a


def size_recommendation(mode: str, confidence: str, decision: str) -> Tuple[str, str]:
    """
    Returns: (best_move_line, rationale_one_liner)

    Philosophy:
    - Confidence is about forecast reliability (headline/vol regime).
    - Decision is about setup quality (your rules-based GO/WAIT/AVOID).
    - Mode changes how strict execution should be.
    """
    m = (mode or "HYBRID").upper()
    c = (confidence or "MED").upper()
    d = (decision or "WAIT").upper()

    # Base from decision
    if "AVOID" in d:
        base = "WAIT"
    elif "WAIT" in d:
        base = "WAIT"
    else:
        base = "NORMAL SIZE"  # only if decision is GO/BUY in your system

    # Confidence overlay
    if c == "LOW":
        # Even if a setup is decent, sizing must be conservative
        if base == "NORMAL SIZE":
            base = "SMALL SIZE"
        else:
            base = "WAIT"
        why = "Headline/vol regime is unstable → sizing must be conservative."
    elif c == "MED":
        if base == "NORMAL SIZE":
            base = "SMALL SIZE" if m == "HYBRID" else "NORMAL SIZE"
        why = "Some uncertainty present → prefer smaller exposure unless setup is very clean."
    else:
        why = "Calmer regime → forecast bands are more reliable; execution is easier."

    # Mode strictness
    if m == "HYBRID" and base == "NORMAL SIZE":
        # Hybrid requires both catalyst cleanliness + technical alignment
        base = "SMALL SIZE"
        why = "Hybrid mode is strict → even on good setups, start smaller and scale on confirmation."

    line = f"Best move: **{base}**"
    return line, why


# ---------- Kid-level tooltips / explainers ----------

def tooltip_map() -> Dict[str, str]:
    return {
        "AVWAP": (
            "Anchored VWAP = the average price paid (weighted by volume) since a chosen date.\n"
            "Above it: buyers since the anchor are mostly winning.\n"
            "Below it: buyers since the anchor are mostly losing."
        ),
        "SWEEP": (
            "Sweep = price quickly takes an obvious high/low to trigger stops, then reverses.\n"
            "It often signals big players hunting liquidity."
        ),
        "BOS": (
            "BOS (Break of Structure) = price breaks a previous swing high/low in a clear way.\n"
            "It can signal trend continuation or reversal."
        ),
        "DILUTION": (
            "Dilution risk = the company may issue/sell more shares.\n"
            "More shares can cap price or push it down short-term."
        ),
        "TREND": (
            "Trend regime uses moving averages.\n"
            "If price < SMA20 < SMA50 → downtrend (market memory says it's drifting down)."
        ),
        "RVOL": (
            "Relative volume compares today's volume to normal.\n"
            "Low RVOL = few people paying attention; high RVOL = attention is back."
        ),
        "ATR": (
            "ATR is the typical daily wiggle size.\n"
            "Higher ATR means bigger swings and larger risk per day."
        ),
        "FORECAST_CONF": (
            "Forecast confidence measures how trustworthy the range forecast is.\n"
            "High confidence = calm conditions. Low confidence = headline/event risk or high volatility."
        ),
    }


def kid_explainer_block(snapshot: Dict[str, str]) -> str:
    """
    snapshot keys you can pass (strings are ok):
      avwap_signal, avwap_dist, trend, rvol, vol20, atr, sweep, bos, dilution_level, decision, mode
    """
    avwap_signal = snapshot.get("avwap_signal", "N/A")
    avwap_dist = snapshot.get("avwap_dist", "N/A")
    trend = snapshot.get("trend", "N/A")
    rvol = snapshot.get("rvol", "N/A")
    vol20 = snapshot.get("vol20", "N/A")
    atr = snapshot.get("atr", "N/A")
    sweep = snapshot.get("sweep", "N/A")
    bos = snapshot.get("bos", "N/A")
    dilution_level = snapshot.get("dilution_level", "N/A")
    decision = snapshot.get("decision", "N/A")
    mode = snapshot.get("mode", "N/A")

    return f"""
### Explain like I'm 10 (what these signals mean)

**1) AVWAP**  
- Signal: **{avwap_signal}** (distance: {avwap_dist})  
- Meaning: If price is **below AVWAP**, most buyers since the anchor are **losing**, so bounces often get sold.

**2) Sweep / BOS (big-money clues)**  
- Sweep: **{sweep}**  
- BOS: **{bos}**  
- Meaning: If there is **no sweep and no BOS**, nothing “big” is happening yet — it’s usually a **waiting phase**.

**3) Dilution / Financing Risk**  
- Level: **{dilution_level}**  
- Meaning: Higher risk means the company might **sell more shares**, which can **cap price**.

**4) Trend + Volume**  
- Trend: **{trend}**  
- Rel Volume: **{rvol}**  
- 20D Vol: **{vol20}** | ATR: **{atr}**  
- Meaning: Downtrend + low volume usually = **slow drift**, not explosive moves.

**5) Decision panel**  
- Mode: **{mode}**  
- Decision: **{decision}**  
- Meaning: In HYBRID, you only act when **both** (a) catalyst posture is clean **and** (b) technicals align.
"""


def invalidation_box(mode: str) -> str:
    m = (mode or "HYBRID").upper()

    if m == "PURE_CATALYST":
        return """
### What would invalidate this catalyst thesis?

Stop treating it as “hold through noise” if any of these happen:
- **Dilution before the catalyst** (offering/ATM/PIPE that changes the payoff).
- **Cash runway becomes insufficient** to reach the catalyst window.
- **Official update delays/cancels** the catalyst or changes probability materially.
- **Management guidance** implies funding pressure or shift in priorities.
"""
    if m == "PURE_TECHNICAL":
        return """
### What would invalidate this technical trade?

Treat the setup as broken if:
- Price **fails reclaim** of key level (AVWAP / prior swing) and closes back below.
- A clear **BOS against** your position happens.
- Relative volume stays dead and price chops (no follow-through).
- ATR expands while structure deteriorates (risk up, edge down).
"""
    # HYBRID default
    return """
### What would invalidate this HYBRID setup?

The HYBRID edge fails if either side breaks:

**Catalyst side fails if:**
- New offering/ATM/PIPE appears or financing posture worsens.
- Cash runway risk rises before the catalyst window.

**Technical side fails if:**
- Price cannot reclaim/hold AVWAP and keeps closing below it.
- No BOS + no volume return (no execution signal).
"""
