from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScoreResult:
    mode: str  # PURE_CATALYST / HYBRID / PURE_TECHNICAL
    decision: str  # GO / WAIT / AVOID
    score: int  # 0-100
    reasons: List[str]


def _clamp(x: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, x))


def decide_entry(
    *,
    mode: str,
    dilution_level: str,     # LOW / MED / HIGH
    dilution_score: int,     # 0-100
    abnormal_volume: bool,
    trend: str,              # from technicals.py
    sweep_signal: str,       # from technicals.py
    bos_signal: str,         # from technicals.py
    vol_20d: Optional[float] # std of returns (None allowed)
) -> ScoreResult:
    """
    Rule-based scoring (no subjective judgement):
    - Pure Catalyst: emphasize dilution risk + abnormal volume as "information day"
    - Hybrid: require catalyst safety + technical alignment for execution
    - Pure Technical: ignore filings/news except as risk modifier, prioritize structure
    """

    m = mode.upper().strip()
    reasons: List[str] = []
    score = 50  # neutral baseline

    # --- common risk modifiers ---
    if dilution_level == "HIGH":
        score -= 30
        reasons.append("Dilution/financing risk HIGH: reduce aggressiveness until filings are read.")
    elif dilution_level == "MED":
        score -= 12
        reasons.append("Dilution/financing risk MED: require confirmation / smaller size.")
    else:
        score += 8
        reasons.append("Dilution/financing risk LOW: cleaner catalyst posture (still not guaranteed).")

    if abnormal_volume:
        score += 8
        reasons.append("Abnormal volume: treat as information day (could be positioning/catalyst).")
    else:
        reasons.append("Volume normal: moves more likely liquidity/noise unless supported by filings/news.")

    # Volatility consideration (for sizing; lightly affects decision)
    if vol_20d is not None:
        if vol_20d >= 0.08:
            score -= 6
            reasons.append("Very high vol regime: require smaller size / wider risk bands.")
        elif vol_20d <= 0.03:
            score += 3
            reasons.append("Lower vol regime: technical signals tend to be cleaner.")

    # --- technical alignment signals ---
    trend_up = "Uptrend" in trend
    trend_down = "Downtrend" in trend
    bos_up = "BOS up" in bos_signal
    bos_down = "BOS down" in bos_signal
    swept_low = "Low sweep" in sweep_signal
    swept_high = "High sweep" in sweep_signal

    tech_alignment_score = 0

    # In practice, for long entries:
    # - Low sweep + BOS up is the highest-prob “reversal then continuation” pattern
    # - Uptrend + BOS up indicates continuation
    if swept_low and bos_up:
        tech_alignment_score += 18
        reasons.append("Technical: low sweep + BOS up (strong bullish execution pattern).")
    elif trend_up and bos_up:
        tech_alignment_score += 12
        reasons.append("Technical: uptrend + BOS up (continuation alignment).")
    elif trend_down and bos_down:
        tech_alignment_score -= 10
        reasons.append("Technical: downtrend + BOS down (bearish alignment).")
    elif swept_high and bos_down:
        tech_alignment_score -= 14
        reasons.append("Technical: high sweep + BOS down (strong bearish pattern).")
    else:
        reasons.append("Technical: no strong sweep/BOS alignment detected (default to WAIT unless catalyst demands).")

    # Apply technical score depending on mode
    if m == "PURE_CATALYST":
        # technical is only execution-timing minor influence
        score += int(tech_alignment_score * 0.35)
        reasons.append("Mode PURE_CATALYST: technicals used lightly for entry timing, not thesis.")
    elif m == "HYBRID":
        # require both; technical carries meaningful weight
        score += int(tech_alignment_score * 0.75)
        reasons.append("Mode HYBRID: requires catalyst cleanliness + technical alignment for execution.")
    elif m == "PURE_TECHNICAL":
        # technical dominates; dilution still risk modifier
        score += int(tech_alignment_score * 1.00)
        reasons.append("Mode PURE_TECHNICAL: decisions driven by tape/structure; filings are risk overlays.")
    else:
        # default to hybrid if unknown
        score += int(tech_alignment_score * 0.75)
        reasons.append("Mode defaulted to HYBRID (unknown mode).")

    score = _clamp(score)

    # Decision thresholds tuned for “GO/WAIT/AVOID” clarity
    if score >= 70:
        decision = "GO"
    elif score >= 45:
        decision = "WAIT"
    else:
        decision = "AVOID"

    return ScoreResult(mode=m, decision=decision, score=score, reasons=reasons)
