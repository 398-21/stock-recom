from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class AVWAPReport:
    anchor_date: str
    avwap_last: float
    price_last: float
    distance_pct: float
    slope_10: Optional[float]
    signal: str
    explanation: str


def compute_avwap(df: pd.DataFrame, anchor_dt: pd.Timestamp) -> pd.Series:
    """
    Anchored VWAP from anchor_dt to end:
      AVWAP = sum(TypicalPrice * Volume) / sum(Volume)
    TypicalPrice = (H+L+C)/3
    """
    x = df.copy()
    x = x[x.index >= anchor_dt].dropna()
    if x.empty:
        raise RuntimeError("No data after anchor date for AVWAP.")

    if "Volume" not in x.columns:
        raise RuntimeError("Volume column missing; AVWAP requires volume.")

    tp = (x["High"] + x["Low"] + x["Close"]) / 3.0
    pv = tp * x["Volume"].astype(float)
    avwap = pv.cumsum() / x["Volume"].astype(float).cumsum()
    return avwap


def _pick_anchor(df_daily: pd.DataFrame, anchor_mode: str) -> pd.Timestamp:
    """
    Default anchors that are meaningful without needing advanced event calendars.
    """
    anchor_mode = anchor_mode.upper().strip()
    if anchor_mode == "LAST_60D_LOW":
        w = df_daily.tail(60)
        idx = w["Low"].idxmin()
        return pd.Timestamp(idx)
    if anchor_mode == "LAST_60D_HIGH":
        w = df_daily.tail(60)
        idx = w["High"].idxmax()
        return pd.Timestamp(idx)
    if anchor_mode == "LAST_20D_START":
        return pd.Timestamp(df_daily.tail(20).index[0])
    # fallback
    return pd.Timestamp(df_daily.tail(60).index[0])


def analyze_avwap(
    df_daily: pd.DataFrame,
    anchor_mode: str = "LAST_60D_LOW",
    custom_anchor: Optional[str] = None,
) -> Tuple[pd.Series, AVWAPReport]:
    """
    Returns (avwap_series, report).
    """
    df = df_daily.dropna().copy()
    if len(df) < 40:
        raise RuntimeError("Insufficient daily bars for AVWAP (need ~40+).")

    if custom_anchor:
        anchor_dt = pd.to_datetime(custom_anchor)
    else:
        anchor_dt = _pick_anchor(df, anchor_mode)

    avwap = compute_avwap(df, anchor_dt)
    avwap_last = float(avwap.iloc[-1])
    price_last = float(df["Close"].iloc[-1])
    dist_pct = (price_last / avwap_last - 1.0) if avwap_last != 0 else np.nan

    # Slope proxy: change over last 10 bars
    slope_10 = None
    if len(avwap) >= 12:
        slope_10 = float(avwap.iloc[-1] - avwap.iloc[-11])

    # Interpretation logic
    # Core “Wall Street” reading:
    # - Price above rising AVWAP => buyers in control, dips into AVWAP often defended
    # - Price below falling AVWAP => sellers in control, rallies into AVWAP often sold
    # - Reclaim (cross back above) matters more than being above by a tiny amount
    signal = "NEUTRAL"
    if slope_10 is not None and slope_10 > 0 and price_last > avwap_last:
        signal = "BULLISH_BIAS"
    elif slope_10 is not None and slope_10 < 0 and price_last < avwap_last:
        signal = "BEARISH_BIAS"
    elif price_last > avwap_last:
        signal = "ABOVE_AVWAP"
    elif price_last < avwap_last:
        signal = "BELOW_AVWAP"

    explanation = (
        "AVWAP is the volume-weighted average price since the chosen anchor date. "
        "Institutions often treat AVWAP as a 'fair price' reference for positioning. "
        "If price is above a rising AVWAP, the tape is typically constructive: pullbacks into AVWAP often act as support. "
        "If price is below a falling AVWAP, rallies into AVWAP often face supply (resistance). "
        "The most actionable event is a reclaim: price moves from below to above AVWAP and holds."
    )

    rep = AVWAPReport(
        anchor_date=str(anchor_dt.date()),
        avwap_last=avwap_last,
        price_last=price_last,
        distance_pct=float(dist_pct),
        slope_10=slope_10,
        signal=signal,
        explanation=explanation,
    )
    return avwap, rep
