from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import math
import pandas as pd


@dataclass
class TechnicalReport:
    last_close: float
    trend: str
    vol_20d: Optional[float]
    atr_14: Optional[float]
    rel_volume: Optional[float]          # today vol / 20d avg vol
    abnormal_volume_flag: bool
    sweep_signal: str
    bos_signal: str


def _atr(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    if len(df) < n + 2:
        return None
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = (high - low).abs().to_frame("hl")
    tr["hc"] = (high - prev_close).abs()
    tr["lc"] = (low - prev_close).abs()
    true_range = tr.max(axis=1)
    atr = true_range.rolling(n).mean().iloc[-1]
    return float(atr) if not math.isnan(atr) else None


def _liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> str:
    if len(df) < lookback + 2:
        return "Insufficient data"
    window = df.iloc[-(lookback+1):-1]
    prev_high = float(window["High"].max())
    prev_low = float(window["Low"].min())

    today = df.iloc[-1]
    close = float(today["Close"])
    high = float(today["High"])
    low = float(today["Low"])

    swept_high = (high > prev_high) and (close < prev_high)
    swept_low = (low < prev_low) and (close > prev_low)

    if swept_high and swept_low:
        return "Both-side sweep (chop / news day)"
    if swept_high:
        return f"High sweep + rejection (took highs > {prev_high:.2f}, closed back below)"
    if swept_low:
        return f"Low sweep + rejection (took lows < {prev_low:.2f}, closed back above)"
    return "No obvious sweep"


def _bos(df: pd.DataFrame, pivot: int = 5) -> str:
    if len(df) < 60:
        return "Insufficient data"
    highs = df["High"].values
    lows = df["Low"].values

    swing_highs = []
    swing_lows = []
    for i in range(pivot, len(df) - pivot):
        if highs[i] == max(highs[i-pivot:i+pivot+1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] == min(lows[i-pivot:i+pivot+1]):
            swing_lows.append((i, float(lows[i])))

    if not swing_highs or not swing_lows:
        return "Unable to identify swings"

    last_close = float(df["Close"].iloc[-1])
    last_sh = swing_highs[-1][1]
    last_sl = swing_lows[-1][1]

    if last_close > last_sh:
        return f"BOS up (close {last_close:.2f} > swing high {last_sh:.2f})"
    if last_close < last_sl:
        return f"BOS down (close {last_close:.2f} < swing low {last_sl:.2f})"
    return "No BOS (inside swing range)"


def analyze_technicals(hist: pd.DataFrame) -> TechnicalReport:
    df = hist.dropna().copy()
    if df.empty:
        raise RuntimeError("No price history available.")

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    last = df.iloc[-1]
    last_close = float(last["Close"])

    sma20 = float(last["SMA20"]) if not math.isnan(last["SMA20"]) else None
    sma50 = float(last["SMA50"]) if not math.isnan(last["SMA50"]) else None

    if sma20 and sma50:
        if last_close > sma20 > sma50:
            trend = "Uptrend (close > SMA20 > SMA50)"
        elif last_close < sma20 < sma50:
            trend = "Downtrend (close < SMA20 < SMA50)"
        else:
            trend = "Range/transition (mixed MAs)"
    else:
        trend = "Unknown (insufficient MA data)"

    rets = df["Close"].pct_change()
    vol_20 = rets.rolling(20).std().iloc[-1]
    vol_20d = float(vol_20) if not math.isnan(vol_20) else None

    atr_14 = _atr(df, 14)

    # Volume analytics
    if "Volume" in df.columns and len(df) >= 25:
        avg20 = df["Volume"].rolling(20).mean().iloc[-2]  # exclude today
        today_vol = df["Volume"].iloc[-1]
        rel_vol = float(today_vol / avg20) if avg20 and not math.isnan(avg20) else None
        abnormal = (rel_vol is not None and rel_vol >= 2.0)
    else:
        rel_vol = None
        abnormal = False

    return TechnicalReport(
        last_close=last_close,
        trend=trend,
        vol_20d=vol_20d,
        atr_14=atr_14,
        rel_volume=rel_vol,
        abnormal_volume_flag=abnormal,
        sweep_signal=_liquidity_sweep(df, 20),
        bos_signal=_bos(df, 5),
    )
