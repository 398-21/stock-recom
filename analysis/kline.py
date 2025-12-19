from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class KlineReadout:
    trend: str
    bias: str
    trend_strength: str
    direction_extent: str
    side: str

    primary_entry: Optional[float] = None
    secondary_entry: Optional[float] = None
    aggressive_entry: Optional[float] = None

    primary_stop: Optional[float] = None
    primary_target: Optional[float] = None
    primary_rr: Optional[float] = None

    confidence_score: int = 0
    confidence_label: str = "LOW"

    patterns: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    explanation: str = ""

    # current vs entry
    last_close: Optional[float] = None
    current_vs_entry_flag: str = ""
    current_vs_entry_note: str = ""

    # NEW: dip-buy band for long-only traders
    dip_upper: Optional[float] = None      # top of dip-buy zone (closest to current price)
    dip_lower: Optional[float] = None      # bottom of dip-buy zone (max pain)
    dip_confidence: int = 0                # 0–100 confidence on the zone
    dip_note: str = ""                     # human explanation

    # 3-day scenario probabilities (heuristic, tape-based)
    prob_support_break_3d: int = 0       # % risk support fails on a close
    prob_bounce_3d: int = 0             # % chance of a 3-day bounce / net gain
    prob_resistance_break_3d: int = 0    # % chance of breaking nearby resistance
    scenarios_note: str = ""             # human explanation

    # For debugging / possible display
    rsi_last: Optional[float] = None     # last RSI(14) value, if computed


def _compute_candle_stats(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_wick = h - c.where(c > o, o)
    lower_wick = c.where(c < o, o) - l

    candle_type = np.where(c > o, "BULL", np.where(c < o, "BEAR", "DOJI"))

    with np.errstate(invalid="ignore"):
        doji_mask = (body / rng) <= 0.1
    candle_type = np.where(doji_mask, "DOJI", candle_type)

    return pd.DataFrame(
        {
            "body": body,
            "range": rng,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "candle_type": candle_type,
        },
        index=df.index,
    )


def _detect_patterns(df: pd.DataFrame, stats: pd.DataFrame, lookback: int = 10) -> List[str]:
    patterns: List[str] = []
    if len(df) < 3:
        return patterns

    df_recent = df.iloc[-lookback:]
    stats_recent = stats.iloc[-lookback:]

    last_idx = df_recent.index[-1]
    prev_idx = df_recent.index[-2]

    last = df_recent.loc[last_idx]
    last_s = stats_recent.loc[last_idx]
    prev = df_recent.loc[prev_idx]
    prev_s = stats_recent.loc[prev_idx]

    if last_s["candle_type"] == "DOJI":
        patterns.append("Doji (indecision near recent prices)")

    with np.errstate(invalid="ignore"):
        is_hammer = (
            (last_s["lower_wick"] >= 2 * last_s["body"]) and
            (last_s["upper_wick"] <= last_s["body"]) and
            (last_s["range"] > 0)
        )
    if is_hammer:
        patterns.append("Hammer (buyers absorbed heavy selling)")

    if prev_s["candle_type"] == "BEAR" and last_s["candle_type"] == "BULL":
        prev_body_low = min(prev["Open"], prev["Close"])
        prev_body_high = max(prev["Open"], prev["Close"])
        last_body_low = min(last["Open"], last["Close"])
        last_body_high = max(last["Open"], last["Close"])
        if (last_body_low <= prev_body_low) and (last_body_high >= prev_body_high):
            patterns.append("Bullish Engulfing (buyers overpowered previous selling)")

    return patterns


def _compute_trend_and_bias(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    sma20: Optional[pd.Series],
    sma50: Optional[pd.Series],
    trend_lookback: int = 10,
) -> tuple[str, str, str, List[str]]:
    reasons: List[str] = []

    if len(df) < trend_lookback + 5:
        return "RANGE", "NEUTRAL", "WEAK", ["Too few candles to infer a strong trend."]

    recent = df["Close"].iloc[-trend_lookback:]
    candle_types = stats["candle_type"].iloc[-trend_lookback:]

    bullish_count = int((candle_types == "BULL").sum())
    bearish_count = int((candle_types == "BEAR").sum())

    last_close = float(recent.iloc[-1])

    sma20_last = float(sma20.dropna().iloc[-1]) if sma20 is not None and not sma20.dropna().empty else None
    sma50_last = float(sma50.dropna().iloc[-1]) if sma50 is not None and not sma50.dropna().empty else None

    highs = df["High"].iloc[-(trend_lookback + 2):]
    lows = df["Low"].iloc[-(trend_lookback + 2):]
    recent_highs = highs.rolling(2).max().dropna()
    recent_lows = lows.rolling(2).min().dropna()

    hh = recent_highs.iloc[-1] > recent_highs.iloc[0]
    hl = recent_lows.iloc[-1] > recent_lows.iloc[0]
    lh = recent_highs.iloc[-1] < recent_highs.iloc[0]
    ll = recent_lows.iloc[-1] < recent_lows.iloc[0]

    trend = "RANGE"
    bias = "NEUTRAL"

    if hh and hl and bullish_count >= bearish_count + 2:
        trend = "UP"
        bias = "BULL"
        reasons.append(
            f"Recent candles show higher highs and higher lows with more bullish candles "
            f"({bullish_count}) than bearish ({bearish_count})."
        )
    elif lh and ll and bearish_count >= bullish_count + 2:
        trend = "DOWN"
        bias = "BEAR"
        reasons.append(
            f"Recent candles show lower highs and lower lows with more bearish candles "
            f"({bearish_count}) than bullish ({bullish_count})."
        )
    else:
        trend = "RANGE"
        bias = "NEUTRAL"
        reasons.append(
            f"Candles are mixed ({bullish_count} bullish vs {bearish_count} bearish) "
            f"without clear dominance in highs/lows."
        )

    strength = "WEAK"
    if sma20_last is not None and sma50_last is not None:
        if trend == "UP":
            if last_close > sma20_last > sma50_last:
                strength = "STRONG"
                reasons.append(
                    f"Price ({last_close:.2f}) is above both SMA20 ({sma20_last:.2f}) and "
                    f"SMA50 ({sma50_last:.2f}), confirming a strong bullish regime."
                )
            elif last_close > sma20_last and sma20_last <= sma50_last:
                strength = "MODERATE"
                reasons.append(
                    f"Price ({last_close:.2f}) is above SMA20 ({sma20_last:.2f}) "
                    f"but SMA20 is not yet clearly above SMA50 ({sma50_last:.2f})."
                )
        elif trend == "DOWN":
            if last_close < sma20_last < sma50_last:
                strength = "STRONG"
                reasons.append(
                    f"Price ({last_close:.2f}) is below both SMA20 ({sma20_last:.2f}) and "
                    f"SMA50 ({sma50_last:.2f}), confirming a strong bearish regime."
                )
            elif last_close < sma20_last and sma20_last >= sma50_last:
                strength = "MODERATE"
                reasons.append(
                    f"Price ({last_close:.2f}) is below SMA20 ({sma20_last:.2f}), "
                    f"but SMA20 is not clearly below SMA50 ({sma50_last:.2f})."
                )
        else:
            strength = "WEAK"
            reasons.append("Moving averages are not clearly aligned; regime looks more sideways.")

    return trend, bias, strength, reasons


def _compute_entries(
    df: pd.DataFrame,
    trend: str,
    bias: str,
    sma20: Optional[pd.Series],
    sma50: Optional[pd.Series],
) -> tuple[str, Optional[float], Optional[float], Optional[float], List[str]]:
    reasons: List[str] = []
    last_close = float(df["Close"].iloc[-1])

    sma20_last = float(sma20.dropna().iloc[-1]) if sma20 is not None and not sma20.dropna().empty else None
    sma50_last = float(sma50.dropna().iloc[-1]) if sma50 is not None and not sma50.dropna().empty else None

    window = min(10, len(df))
    recent_lows = df["Low"].iloc[-window:]
    recent_highs = df["High"].iloc[-window:]

    swing_low = float(recent_lows.min())
    swing_high = float(recent_highs.max())

    primary_entry = None
    secondary_entry = None
    aggressive_entry = None
    side = "NONE"

    if trend == "UP" and bias == "BULL":
        side = "LONG"

        if sma20_last is not None:
            primary_entry = sma20_last
            reasons.append(
                f"Primary entry near SMA20 ({sma20_last:.2f}), using it as dynamic support in an uptrend."
            )
        else:
            primary_entry = swing_low
            reasons.append(
                f"Primary entry near recent swing low ({swing_low:.2f}) where buyers previously defended."
            )

        secondary_entry = swing_low
        if secondary_entry != primary_entry:
            reasons.append(
                f"Secondary (deeper) entry near recent swing low ({swing_low:.2f}) for higher reward but more risk."
            )

        if last_close > swing_high * 0.995:
            aggressive_entry = last_close
            reasons.append(
                f"Aggressive entry at current price ({last_close:.2f}) as it is pressing or breaking recent highs "
                f"({swing_high:.2f})."
            )

    elif trend == "DOWN" and bias == "BEAR":
        side = "SHORT"

        if sma20_last is not None:
            primary_entry = sma20_last
            reasons.append(
                f"Primary short entry near SMA20 ({sma20_last:.2f}), using it as dynamic resistance in a downtrend."
            )
        else:
            primary_entry = swing_high
            reasons.append(
                f"Primary short entry near recent swing high ({swing_high:.2f}) where sellers previously defended."
            )

        secondary_entry = swing_high
        if secondary_entry != primary_entry:
            reasons.append(
                f"Secondary short entry near recent swing high ({swing_high:.2f}) for better risk/reward."
            )

        if last_close < swing_low * 1.005:
            aggressive_entry = last_close
            reasons.append(
                f"Aggressive short entry at current price ({last_close:.2f}) as it is pressing or breaking recent lows "
                f"({swing_low:.2f})."
            )

    else:
        side = "NONE"
        reasons.append(
            "Market appears range-bound; better entries are usually near support (bottom of range) or resistance "
            "(top of range) rather than in the middle."
        )

    return side, primary_entry, secondary_entry, aggressive_entry, reasons


def _compute_confidence(
    trend: str,
    bias: str,
    trend_strength: str,
    patterns: List[str],
    df: pd.DataFrame,
) -> tuple[int, str, List[str]]:
    reasons: List[str] = []

    if len(df) < 20:
        base = 35
        reasons.append("Limited history (<20 candles) reduces K-line confidence.")
    else:
        base = 50
        reasons.append("Sufficient candle history for a stable K-line read.")

    if trend in ("UP", "DOWN") and bias != "NEUTRAL":
        base += 10
        reasons.append("Trend and bias are aligned (clear direction in candles).")
    else:
        base -= 5
        reasons.append("Trend or bias is not clearly aligned, suggesting more noise in candles.")

    if trend_strength == "STRONG":
        base += 15
        reasons.append("Moving averages and price strongly support the K-line trend.")
    elif trend_strength == "MODERATE":
        base += 5
        reasons.append("Moving averages somewhat support the K-line trend.")
    else:
        base -= 5
        reasons.append("Moving averages do not strongly support the K-line trend.")

    if patterns:
        base += 10
        reasons.append("Recognized candlestick pattern(s) add conviction to the K-line view.")
    else:
        reasons.append("No strong reversal/continuation candlestick patterns detected.")

    score = max(0, min(100, base))

    if score >= 70:
        label = "HIGH"
    elif score >= 50:
        label = "MED"
    else:
        label = "LOW"

    return score, label, reasons


def analyze_kline(
    df: pd.DataFrame,
    sma20: Optional[pd.Series] = None,
    sma50: Optional[pd.Series] = None,
    allowed_sides: str = "BOTH",   # "BOTH", "LONGS_ONLY", "SHORTS_ONLY"
) -> KlineReadout:
    # Basic guards
    if df is None or df.empty or not all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        return KlineReadout(
            trend="RANGE",
            bias="NEUTRAL",
            trend_strength="WEAK",
            direction_extent="Not enough data to read trend.",
            side="NONE",
            confidence_score=0,
            confidence_label="LOW",
            explanation="No valid candle data available for K-line analysis.",
        )

    if sma20 is None:
        sma20 = df["Close"].rolling(20).mean()
    if sma50 is None:
        sma50 = df["Close"].rolling(50).mean()

    # Core tape read
    stats = _compute_candle_stats(df)
    patterns = _detect_patterns(df, stats, lookback=10)
    trend, bias, trend_strength, trend_reasons = _compute_trend_and_bias(df, stats, sma20, sma50)
    side, primary_entry, secondary_entry, aggressive_entry, entry_reasons = _compute_entries(
        df, trend, bias, sma20, sma50
    )
    confidence_score, confidence_label, conf_reasons = _compute_confidence(
        trend, bias, trend_strength, patterns, df
    )

    closes = df["Close"]
    last_close = float(closes.iloc[-1])

    # --- RSI(14) for overbought/oversold / bounce logic ---
    rsi_last: Optional[float] = None
    if len(closes) >= 15:
        delta = closes.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        rsi_last = float(val) if np.isfinite(val) else None

    # --- Vol-adjusted daily move ---
    vol20 = closes.pct_change().rolling(20).std().iloc[-1]
    if pd.isna(vol20) or vol20 == 0:
        direction_extent = "Typical daily movement cannot be estimated reliably from recent candles."
    else:
        est_move_pct = vol20 * 100
        direction_extent = (
            f"Typical daily swing is about {est_move_pct:.1f}% around the last close ({last_close:.2f}), "
            f"so the current {trend.lower()} can extend but may also see healthy pullbacks within that range."
        )

    # --- Dip band (long-only scout) using recent swing low ---
    dip_upper = dip_lower = None
    dip_conf = 0
    dip_note = ""
    window_for_dip = min(10, len(df))
    recent_lows_for_dip = df["Low"].iloc[-window_for_dip:]
    recent_highs_for_dip = df["High"].iloc[-window_for_dip:]
    swing_low = float(recent_lows_for_dip.min())
    swing_high = float(recent_highs_for_dip.max())

    if len(df) >= 25 and np.isfinite(swing_low):
        base_support = swing_low

        # depth of band scales with volatility, capped 5–15%
        if pd.isna(vol20) or vol20 <= 0:
            depth_pct = 0.08
        else:
            depth_pct = float(vol20) * 1.5
            depth_pct = max(0.05, min(0.15, depth_pct))

        dip_upper = base_support
        dip_lower = base_support * (1.0 - depth_pct)

        dip_conf = 50
        if trend == "DOWN" and trend_strength == "STRONG":
            dip_conf -= 15
        elif trend == "DOWN" and trend_strength == "MODERATE":
            dip_conf -= 5
        elif trend == "RANGE":
            dip_conf += 10
        elif trend == "UP":
            dip_conf += 15
        dip_conf = max(0, min(100, dip_conf))

        dip_note = (
            f"From a long-only perspective, a potential **dip-buy band** sits around "
            f"{dip_lower:.2f}–{dip_upper:.2f}, anchored on the recent swing-low support near "
            f"{base_support:.2f} and adjusted for recent volatility. "
        )
        if trend == "DOWN":
            dip_note += (
                "This is **countertrend** (the main tape is bearish), so auto-buys in this zone should be "
                "smaller size and assume price can temporarily overshoot the band if selling pressure persists."
            )
        elif trend == "RANGE":
            dip_note += (
                "Trend looks more sideways, so this band represents a potential accumulation area inside the range."
            )
        else:
            dip_note += (
                "Trend is already up, so this band is a normal ‘buy the dip’ area inside an existing uptrend."
            )

    # --- Primary stop/target using swing low/high (execution plan) ---
    primary_stop = primary_target = None
    primary_rr = None

    window = min(10, len(df))
    recent_lows = df["Low"].iloc[-window:]
    recent_highs = df["High"].iloc[-window:]
    swing_low_st = float(recent_lows.min())
    swing_high_st = float(recent_highs.max())

    # Fallback: if K-line side is NONE but longs are allowed, create a basic range long plan,
    # but only when confidence is at least 30/100 (avoid pure noise).
    long_side_allowed = (allowed_sides or "BOTH").upper() in ("BOTH", "LONGS_ONLY")
    if (
        side == "NONE"
        and long_side_allowed
        and confidence_score >= 30
        and np.isfinite(swing_low_st)
        and np.isfinite(swing_high_st)
    ):
        side = "LONG"
        primary_entry = swing_low_st
        secondary_entry = None
        aggressive_entry = None
        entry_reasons.append(
            "Structure looks range-like or noisy, so a basic long-only range plan is suggested: "
            "consider buying near recent support (swing low) and aiming toward the top of the range, "
            "instead of forcing a breakout trade."
        )

    # Enrich direction extent with distance to support / resistance
    if np.isfinite(swing_low_st) and np.isfinite(swing_high_st):
        gap_support_pct = max(0.0, (last_close - swing_low_st) / last_close * 100.0)
        gap_res_pct = max(0.0, (swing_high_st - last_close) / last_close * 100.0)
        direction_extent = (
            direction_extent
            + f" Within this recent range, price is about {gap_support_pct:.1f}% above nearby support "
              f"(~{swing_low_st:.2f}) and {gap_res_pct:.1f}% below nearby resistance (~{swing_high_st:.2f})."
        )

    if primary_entry is not None and side in ("LONG", "SHORT"):
        if side == "LONG":
            # clamp down-side to ~20% and/or swing low
            raw_stop_swing = swing_low_st
            raw_stop_pct = primary_entry * 0.80
            raw_stop = max(raw_stop_swing, raw_stop_pct)
            if raw_stop >= primary_entry:
                raw_stop = primary_entry * 0.97
            primary_stop = raw_stop

            risk = primary_entry - primary_stop
            if risk > 0:
                target_rr = primary_entry + 2 * risk
                target_res = swing_high_st
                primary_target = min(target_rr, target_res)
                if primary_target <= primary_entry:
                    primary_target = primary_entry + risk
                primary_rr = (primary_target - primary_entry) / risk

        elif side == "SHORT":
            raw_stop_swing = swing_high_st
            raw_stop_pct = primary_entry * 1.20
            raw_stop = min(raw_stop_swing, raw_stop_pct)
            if raw_stop <= primary_entry:
                raw_stop = primary_entry * 1.05
            primary_stop = raw_stop

            risk = primary_stop - primary_entry
            if risk > 0:
                target_rr = primary_entry - 2 * risk
                target_sup = swing_low_st
                candidate = max(target_rr, target_sup)
                floor = primary_entry * 0.30
                candidate = max(candidate, floor)
                if candidate >= primary_entry:
                    candidate = primary_entry - risk
                primary_target = max(candidate, 0.01)
                primary_rr = (primary_entry - primary_target) / risk

    # --- Apply LONGS_ONLY / SHORTS_ONLY filter ---
    mode = (allowed_sides or "BOTH").upper()
    allow_long = mode in ("BOTH", "LONGS_ONLY")
    allow_short = mode in ("BOTH", "SHORTS_ONLY")

    orig_side = side
    filter_msg = ""

    if orig_side == "LONG" and not allow_long:
        filter_msg = (
            "The tape currently favours a long setup, but the trade-direction mode excludes longs. "
            "No long execution plan is produced; in this mode the zone is treated as an **avoid / wait** area."
        )
        side = "NONE"
        primary_entry = secondary_entry = aggressive_entry = None
        primary_stop = primary_target = primary_rr = None

    elif orig_side == "SHORT" and not allow_short:
        filter_msg = (
            "The tape currently favours a short setup, but the trade-direction mode excludes shorts. "
            "From a long-only perspective this is an **avoid / wait** area, not a buy-the-dip signal."
        )
        side = "NONE"
        primary_entry = secondary_entry = aggressive_entry = None
        primary_stop = primary_target = primary_rr = None

    # --- Is now a good execution or wait? ---
    current_flag = "N/A"
    if primary_entry is not None and side in ("LONG", "SHORT"):
        lower_band = primary_entry * 0.99
        upper_band = primary_entry * 1.01

        if side == "LONG":
            if lower_band <= last_close <= upper_band:
                current_flag = "AT_ZONE"
                current_note = (
                    f"Current close ({last_close:.2f}) is trading **inside** the primary long zone around "
                    f"{primary_entry:.2f}. This is a valid execution area if the risk plan fits."
                )
            elif last_close < lower_band:
                current_flag = "BELOW_ZONE"
                current_note = (
                    f"Current close ({last_close:.2f}) is already **below** the primary long entry zone "
                    f"(~{primary_entry:.2f}). Price is cheaper but may be breaking support; prefer stabilisation or "
                    "a retest rather than blindly adding here."
                )
            else:
                current_flag = "ABOVE_ZONE"
                current_note = (
                    f"Current close ({last_close:.2f}) is **above** the primary long zone (~{primary_entry:.2f}). "
                    "Plan prefers waiting for a pullback closer to the entry instead of chasing higher."
                )

        else:  # SHORT
            if lower_band <= last_close <= upper_band:
                current_flag = "AT_ZONE"
                current_note = (
                    f"Current close ({last_close:.2f}) is trading **inside** the primary short zone around "
                    f"{primary_entry:.2f}. This is a valid execution area if the risk plan fits."
                )
            elif last_close > upper_band:
                current_flag = "ABOVE_ZONE"
                current_note = (
                    f"Current close ({last_close:.2f}) is **above** the primary short entry zone (~{primary_entry:.2f}). "
                    "You would be shorting into a bounce; size carefully and make sure the stop still makes sense."
                )
            else:
                current_flag = "BELOW_ZONE"
                current_note = (
                    f"Current close ({last_close:.2f}) is already **below** the primary short entry zone "
                    f"(~{primary_entry:.2f}). You would be shorting into local lows. The plan prefers waiting for a "
                    "bounce back toward the entry instead of pressing here."
                )
    else:
        if filter_msg:
            current_note = (
                "Because the preferred setup is disabled by the trade-direction mode, the effective guidance is to "
                "**stand aside / wait** rather than forcing a trade at the current price."
            )
        else:
            current_note = (
                "No strong directional edge from K-lines; there is no specific entry guidance at the current price."
            )

    # --- 3-day scenario probabilities (heuristic, tape-based) ---

    # Which levels count as support / resistance for scenarios
    support_level = dip_upper if dip_upper is not None else (swing_low if np.isfinite(swing_low) else None)

    sma20_last = float(sma20.dropna().iloc[-1]) if sma20 is not None and not sma20.dropna().empty else None
    resistance_level = swing_high if np.isfinite(swing_high) else None
    if resistance_level is None:
        if sma20_last is not None:
            resistance_level = sma20_last
        else:
            resistance_level = swing_high_st

    # 3-day support break risk
    prob_support_break_3d = 50
    if trend == "DOWN":
        prob_support_break_3d += 15 if trend_strength == "STRONG" else 8
    elif trend == "UP":
        prob_support_break_3d -= 15 if trend_strength == "STRONG" else 8

    if rsi_last is not None:
        if rsi_last < 30:
            prob_support_break_3d -= 10
        if rsi_last < 20:
            prob_support_break_3d -= 10
        if rsi_last > 70:
            prob_support_break_3d += 5

    if support_level is not None:
        if support_level * 0.99 <= last_close <= support_level * 1.01:
            prob_support_break_3d += 5   # sitting right on it → binary
        elif last_close < support_level:
            prob_support_break_3d = max(prob_support_break_3d, 70)  # already cracking

    prob_support_break_3d = int(max(5, min(95, round(prob_support_break_3d))))

    # 3-day bounce / up-move chance
    prob_bounce_3d = 50
    if trend == "UP":
        prob_bounce_3d += 15 if trend_strength == "STRONG" else 8
    elif trend == "DOWN":
        prob_bounce_3d -= 15 if trend_strength == "STRONG" else 8

    if rsi_last is not None:
        if rsi_last < 30:
            prob_bounce_3d += 10
        if rsi_last < 20:
            prob_bounce_3d += 10
        if rsi_last > 70:
            prob_bounce_3d -= 10

    if support_level is not None and dip_lower is not None:
        if dip_lower <= last_close <= dip_upper:
            prob_bounce_3d += 5   # sitting inside the dip band

    prob_bounce_3d = int(max(5, min(95, round(prob_bounce_3d))))

    # 3-day resistance break chance
    prob_resistance_break_3d = 50
    if trend == "UP":
        prob_resistance_break_3d += 15 if trend_strength == "STRONG" else 8
    elif trend == "DOWN":
        prob_resistance_break_3d -= 10

    if rsi_last is not None:
        if rsi_last > 70:
            prob_resistance_break_3d -= 10  # overbought → breakout more fragile
        elif rsi_last < 30:
            prob_resistance_break_3d -= 5   # deeply oversold below resistance

    if resistance_level is not None:
        if resistance_level * 0.99 <= last_close <= resistance_level * 1.01:
            prob_resistance_break_3d += 5   # current price pressing resistance

    prob_resistance_break_3d = int(max(5, min(95, round(prob_resistance_break_3d))))

    if support_level is not None and resistance_level is not None:
        scenarios_note = (
            f"Looking out over the next ~3 sessions, the tape-based read suggests roughly a "
            f"{prob_bounce_3d}% chance of a short-term **bounce** (3-day net gain), "
            f"about {prob_support_break_3d}% risk that the support zone around {support_level:.2f} fails on a closing basis, "
            f"and around {prob_resistance_break_3d}% chance of a decisive push through the nearby resistance near {resistance_level:.2f}. "
            "These are heuristic odds from candles, volatility, moving averages and RSI – they are **guidance**, "
            "not statistically calibrated probabilities."
        )
    else:
        scenarios_note = (
            "Support/resistance structure is fuzzy here, so short-term scenario odds should be treated as very rough guidance."
        )

    # --- Human explanation paragraph (same style as before) ---
    expl_parts: List[str] = []

    expl_parts.append(
        f"The system reads the recent K-lines as a **{trend_strength.lower()} {trend.lower()} trend** "
        f"with a **{bias.lower()} bias**."
    )
    expl_parts.append(
        "This is deduced by looking at sequences of higher/lower highs and lows, the balance of bullish vs "
        "bearish candles, and where price trades relative to the 20-day and 50-day moving averages."
    )

    if patterns:
        expl_parts.append(
            "The latest candles also show these notable patterns: " + ", ".join(patterns) + "."
        )
    else:
        expl_parts.append(
            "No major hammer, doji, or engulfing patterns are dominating the most recent candles."
        )

    if trend == "UP" and bias == "BULL":
        expl_parts.append(
            "Because the regime leans bullish, the entries focus on pullbacks toward support (like SMA20 or "
            "recent swing lows) rather than chasing extended highs."
        )
    elif trend == "DOWN" and bias == "BEAR":
        expl_parts.append(
            "Because the regime leans bearish, the entries focus on rallies back into resistance (like SMA20 or "
            "recent swing highs) rather than shorting into panic lows."
        )
    else:
        expl_parts.append(
            "Since the structure appears more range-bound or mixed, there is no strong directional side, so entries "
            "are described more cautiously."
        )

    if primary_entry is not None and primary_stop is not None and primary_target is not None and side in ("LONG", "SHORT"):
        expl_parts.append(
            f"For the primary {side.lower()} plan, the module suggests stalking an entry near **{primary_entry:.2f}**, "
            f"cutting risk if price trades beyond **{primary_stop:.2f}**, and taking first profit/trim around "
            f"**{primary_target:.2f}**, which corresponds to roughly a **1:{primary_rr:.1f} reward-to-risk** "
            "based on these levels."
        )

    if filter_msg:
        expl_parts.append(filter_msg)

    expl_parts.append(current_note)

    expl_parts.append(
        f"The K-line confidence score of **{confidence_score}/100 ({confidence_label})** weights trend clarity, "
        "moving-average alignment, pattern strength, and amount of candle history. It does not use news or the quant "
        "forecast; it only reads the candles and moving averages."
    )

    explanation = " ".join(expl_parts)
    all_reasons = trend_reasons + entry_reasons + conf_reasons
    if filter_msg:
        all_reasons.append(filter_msg)

    return KlineReadout(
        trend=trend,
        bias=bias,
        trend_strength=trend_strength,
        direction_extent=direction_extent,
        side=side,
        primary_entry=primary_entry,
        secondary_entry=secondary_entry,
        aggressive_entry=aggressive_entry,
        primary_stop=primary_stop,
        primary_target=primary_target,
        primary_rr=primary_rr,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        patterns=patterns,
        reasons=all_reasons,
        explanation=explanation,
        last_close=last_close,
        current_vs_entry_flag=current_flag,
        current_vs_entry_note=current_note,
        dip_upper=dip_upper,
        dip_lower=dip_lower,
        dip_confidence=dip_conf,
        dip_note=dip_note,
        prob_support_break_3d=prob_support_break_3d,
        prob_bounce_3d=prob_bounce_3d,
        prob_resistance_break_3d=prob_resistance_break_3d,
        scenarios_note=scenarios_note,
        rsi_last=rsi_last,
    )
