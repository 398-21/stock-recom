from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


def tooltip_map() -> Dict[str, str]:
    return {
        "DILUTION": "How likely the company raises money by selling more shares (can push price down).",
        "TREND": "A quick trend label using moving averages (who is winning recently).",
        "ATR": "Average True Range: typical daily wiggle size in dollars.",
        "RVOL": "Relative Volume: today’s volume compared with normal. Low = nobody cares; high = attention.",
        "FORECAST_CONF": "Forecast confidence depends on how stable the recent price behavior is (less noise = higher confidence).",
        "FIN_HEALTH": "This is not ‘profit.’ It is a financing stability guess: cash/runway/solvency-type hints.",
    }


def emoji_action_label(action: str) -> str:
    a = (action or "").upper()
    if a == "BUY":
        return "BUY"
    if a == "AVOID":
        return "AVOID"
    return "WAIT"


def _bucket_conf(conf: str) -> str:
    c = (conf or "").upper()
    if c in ("HIGH", "H"):
        return "HIGH"
    if c in ("MED", "MEDIUM", "M"):
        return "MED"
    return "LOW"


def size_recommendation(
    *,
    mode: str,
    confidence: str,
    decision: str,
    fin_health_level: str = "MED",
    runway_months_est: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Returns:
      (best_move_line, reasoning_caption)
    Uses: mode + forecast confidence + decision + financing stability/runway.
    """
    m = (mode or "HYBRID").upper()
    conf = _bucket_conf(confidence)
    dec = (decision or "WAIT").upper()
    fin = (fin_health_level or "MED").upper()

    # Hard safety rail: if runway looks short, force small/wait.
    runway_flag = False
    if runway_months_est is not None:
        if runway_months_est < 6:
            runway_flag = True
        elif runway_months_est < 12 and fin == "LOW":
            runway_flag = True

    if runway_flag:
        return (
            "**Best move: WAIT or VERY SMALL size (runway risk)**",
            "Even if price signals look good, short runway increases the chance of financing/dilution shocks.",
        )

    if dec == "AVOID":
        return ("**Best move: AVOID**", "Rules-based decision says risk is high or thesis/timing is poor.")
    if dec == "WAIT":
        if conf == "HIGH" and m == "PURE_TECHNICAL":
            return ("**Best move: WAIT (but watch for breakout)**", "High-confidence regime but no trigger yet.")
        if conf == "LOW":
            return ("**Best move: WAIT / SMALL**", "Low confidence means the tape is noisy; reduce size.")
        return ("**Best move: WAIT**", "No strong alignment yet.")

    # GO
    if conf == "LOW":
        return ("**Best move: SMALL size**", "Direction is not reliable; treat as a probe, not a full position.")
    if m == "PURE_CATALYST":
        return ("**Best move: NORMAL size (catalyst posture ok)**", "In catalyst mode, timing matters more than precision.")
    return ("**Best move: NORMAL size**", "Decision and confidence support taking the trade.")


# ---------------- Glossary (Kid-mode) ----------------

def glossary_block() -> str:
    """
    Explains each technical term in simple language + what it usually implies.
    """
    g = [
        ("Candle / Candlestick",
         "A candle shows what price did in one time period: open, high, low, close.",
         "Big body = strong move. Long wicks = rejection (price tried, failed)."),

        ("Volume",
         "How many shares traded. Think: how many people showed up to play.",
         "High volume = attention/conviction. Low volume = sleepy market."),

        ("SMA20 / SMA50",
         "Simple Moving Average: average price over last 20/50 days. It is ‘trend memory.’",
         "If price is above them, trend is healthier. If below, trend is weaker."),

        ("Trend Regime (close < SMA20 < SMA50)",
         "A quick label: price is below short-term average, and short-term is below long-term.",
         "Usually means downtrend pressure. Rallies may get sold until it changes."),

        ("AVWAP (Anchored VWAP)",
         "Average price paid by traders since a chosen ‘anchor day’ (like earnings day, low, high).",
         "Above AVWAP: buyers since anchor are winning → support. Below: many are losing → resistance."),

        ("Support / Resistance",
         "Support is a floor where price often bounces. Resistance is a ceiling where price often struggles.",
         "Near support: buyers may defend. Near resistance: sellers may appear."),

        ("Pivot / R1 / S1",
         "Pivot is a reference level from last candle. R1 is first resistance, S1 first support.",
         "Used as ‘map lines’ for where price might pause or bounce."),

        ("ATR(14)",
         "Average True Range: typical daily movement size (in dollars).",
         "Bigger ATR = wilder price swings. Helps set stops/targets realistically."),

        ("20D Volatility",
         "How much price wiggles day-to-day (statistical).",
         "High vol = unpredictable; low vol = calmer and more forecastable."),

        ("RVOL (Relative Volume)",
         "Today’s volume compared to normal volume.",
         "RVOL > 1 = unusual attention. RVOL < 1 = quiet; moves can be weak/fake."),

        ("Abnormal Volume",
         "A flag that volume is unusually high compared to recent history.",
         "High abnormal volume can mean ‘something is happening’ (breakout/news/positioning)."),

        ("Liquidity Sweep",
         "Price quickly takes an obvious high/low, then reverses. Like a ‘trap.’",
         "Often signals stop-hunting before the real move starts."),

        ("BOS (Break of Structure)",
         "Price breaks a recent swing high/low with intent, changing the ‘story.’",
         "BOS up can mean trend shift to bullish; BOS down can mean bearish continuation."),

        ("7D Range (p05–p95)",
         "A probabilistic ‘box’ where price is likely to stay 90% of the time.",
         "Not direction. Just ‘how wide’ the next week could be."),

        ("Expected Close (7D)",
         "A statistical center estimate for 7 days out (not guaranteed).",
         "Use it lightly; the range matters more than the exact number."),

        ("Forecast Confidence: LOW / MED / HIGH",
         "How trustworthy the forecast model is given recent behavior.",
         "LOW means noisy/choppy/unstable → treat predictions cautiously and reduce size."),
    ]

    md = "### Glossary (simple meaning + what it suggests)\n"
    for term, meaning, suggests in g:
        md += f"**{term}**\n\n- Meaning: {meaning}\n- Suggests: {suggests}\n\n"
    return md


def kid_explainer_block(snapshot: dict) -> str:
    """
    snapshot keys you pass:
      avwap_signal, avwap_dist, trend, rvol, vol20, atr, sweep, bos, dilution_level, decision, mode
    """
    av = snapshot.get("avwap_signal", "N/A")
    avd = snapshot.get("avwap_dist", "N/A")
    tr = snapshot.get("trend", "N/A")
    rvol = snapshot.get("rvol", "N/A")
    vol20 = snapshot.get("vol20", "N/A")
    atr = snapshot.get("atr", "N/A")
    sweep = snapshot.get("sweep", "N/A")
    bos = snapshot.get("bos", "N/A")
    dil = snapshot.get("dilution_level", "N/A")
    dec = snapshot.get("decision", "N/A")
    mode = snapshot.get("mode", "N/A")

    md = f"""
### What the dashboard is saying (kid-level)

**1) Decision (top result): {dec}**  
This is the system’s final “go / wait / avoid” based on your chosen mode.

**2) Mode: {mode}**  
- PURE_CATALYST: care more about catalyst cleanliness (financing/dilution) than short-term trend.  
- HYBRID: need both catalyst cleanliness + a decent technical setup.  
- PURE_TECHNICAL: care most about trend/levels/volume.

**3) Trend: {tr}**  
If trend says downtrend, it means price has been sliding and buyers have not taken control yet.

**4) AVWAP signal: {av} (distance: {avd})**  
- Above AVWAP: more buyers are in profit since the anchor → support is more likely.  
- Below AVWAP: more buyers are trapped in loss → selling pressure is more likely.

**5) Volume interest: RVOL {rvol}**  
- Around 1.0 = normal  
- Much less than 1.0 = sleepy (moves can stall)  
- Much more than 1.0 = attention (moves can be real)

**6) Volatility / movement size: 20D vol {vol20}, ATR {atr}**  
This tells how “wild” the stock is. If wild, small size is safer.

**7) Smart-money footprints: Sweep {sweep}, BOS {bos}**  
- Sweep = trap then reverse.  
- BOS = breaks the “story” (trend shift).

**8) Dilution / financing risk: {dil}**  
High dilution risk means the company may create/sell shares; that can cap rallies.

---

""" + glossary_block()

    return md


def invalidation_box(mode: str) -> str:
    m = (mode or "HYBRID").upper()
    if m == "PURE_CATALYST":
        return (
            "**Invalidation examples (Catalyst mode):**\n\n"
            "- New offering / ATM / shelf expands (dilution risk spikes)\n"
            "- Catalyst timeline slips materially\n"
            "- Thesis X no longer true (trial halted, FDA pushback, etc.)\n"
        )
    if m == "PURE_TECHNICAL":
        return (
            "**Invalidation examples (Technical mode):**\n\n"
            "- Breaks key support and does not reclaim\n"
            "- Breakout fails: price returns below the breakout level on high volume\n"
            "- Volatility spikes in wrong direction (stop should be hit quickly)\n"
        )
    return (
        "**Invalidation examples (Hybrid mode):**\n\n"
        "- Dilution risk rises AND price loses AVWAP/support\n"
        "- No BOS confirmation after a sweep\n"
        "- Volume fades on rallies (weak demand)\n"
    )
