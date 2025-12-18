from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, List
import pandas as pd


@dataclass
class DirectionReport:
    # Auto-selected engine after detecting market regime
    engine: str               # "TREND" | "MEAN_REVERSION"
    regime: str               # "TRENDING" | "CHOP"
    bias: str                 # "UP" | "DOWN" | "RANGE" | "MIXED"
    range_low: float          # likely 7D low bound (5th percentile)
    range_high: float         # likely 7D high bound (95th percentile)
    summary: str              # plain-English explanation
    targets: List[str]        # practical levels/targets to watch


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(n).mean()


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Lightweight ADX implementation (no external libs).
    ADX is a regime tool:
      - Low ADX => chop/range (signals get messy)
      - Higher ADX => trend strength (trend rules work better)
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

    atr = _atr(df, n).replace(0, pd.NA)

    plus_di = 100 * (plus_dm.rolling(n).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(n).sum() / atr)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([pd.NA, pd.NaT], 0)
    adx = dx.rolling(n).mean()
    return adx


def analyze_direction(
    daily: pd.DataFrame,
    avwap_report: Optional[Any],
    forecast: Optional[Any],
) -> Optional[DirectionReport]:
    """
    AUTO Regime Engine (best overall):
      - Detect regime (TRENDING vs CHOP)
      - Pick TREND engine or MEAN_REVERSION engine
      - Output:
          * Bias (UP/DOWN/RANGE)
          * Likely 7D range (prefers forecast 5–95% bands)
          * Practical targets/levels
    """
    if daily is None or daily.empty or len(daily) < 60:
        return None

    df = daily.copy()
    close = df["Close"]

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    atr14 = _atr(df, 14)
    adx14 = _adx(df, 14)

    last = df.index[-1]
    last_close = float(close.loc[last])
    last_sma20 = float(sma20.loc[last]) if pd.notna(sma20.loc[last]) else last_close
    last_sma50 = float(sma50.loc[last]) if pd.notna(sma50.loc[last]) else last_close
    last_atr = float(atr14.loc[last]) if pd.notna(atr14.loc[last]) else 0.0
    last_adx = float(adx14.loc[last]) if pd.notna(adx14.loc[last]) else 0.0

    # Trend slope proxy: compare SMA today vs ~10 trading days ago
    lookback = 10
    if len(df) > lookback + 50 and pd.notna(sma20.iloc[-(lookback + 1)]) and pd.notna(sma50.iloc[-(lookback + 1)]):
        sma20_slope = float(last_sma20 - sma20.iloc[-(lookback + 1)])
        sma50_slope = float(last_sma50 - sma50.iloc[-(lookback + 1)])
    else:
        sma20_slope = 0.0
        sma50_slope = 0.0

    # Regime detection: ADX + aligned slopes is a robust “is this directional?” test
    trending = (last_adx >= 18.0) and (sma20_slope * sma50_slope >= 0)
    regime = "TRENDING" if trending else "CHOP"

    # AVWAP info if available
    avwap_signal = getattr(avwap_report, "signal", None) if avwap_report is not None else None
    avwap_dist = getattr(avwap_report, "distance_pct", None) if avwap_report is not None else None
    avwap_last = getattr(avwap_report, "avwap_last", None) if avwap_report is not None else None

    # Support/resistance proxy = 20D low/high (simple and effective)
    win = 20
    recent_low = float(df["Low"].rolling(win).min().iloc[-1])
    recent_high = float(df["High"].rolling(win).max().iloc[-1])

    targets: List[str] = []

    # Choose engine + bias
    if regime == "TRENDING":
        engine = "TREND"
        if (last_close > last_sma20 > last_sma50) and (sma20_slope > 0) and (sma50_slope > 0):
            bias = "UP"
            targets.append(f"Support: SMA20 ~ {last_sma20:.2f}")
            if avwap_last is not None:
                targets.append(f"Support/Mean: AVWAP ~ {float(avwap_last):.2f}")
            targets.append(f"Next resistance (20D high): {recent_high:.2f}")
        elif (last_close < last_sma20 < last_sma50) and (sma20_slope < 0) and (sma50_slope < 0):
            bias = "DOWN"
            targets.append(f"Resistance: SMA20 ~ {last_sma20:.2f}")
            if avwap_last is not None:
                targets.append(f"Resistance/Mean: AVWAP ~ {float(avwap_last):.2f}")
            targets.append(f"Next support (20D low): {recent_low:.2f}")
        else:
            bias = "MIXED"
            targets.append(f"Range: {recent_low:.2f} – {recent_high:.2f}")
            if avwap_signal:
                targets.append(f"AVWAP: {avwap_signal}")
    else:
        engine = "MEAN_REVERSION"

        # “Stretched” heuristics (bounce/pullback risk)
        stretched_down = False
        stretched_up = False

        if avwap_dist is not None:
            stretched_down = avwap_dist <= -0.04
            stretched_up = avwap_dist >= 0.04
        if last_atr and last_atr > 0:
            stretched_down = stretched_down or (last_close < last_sma20 - 1.5 * last_atr)
            stretched_up = stretched_up or (last_close > last_sma20 + 1.5 * last_atr)

        if stretched_down:
            bias = "RANGE (bounce risk)"
            targets.append(f"Mean: SMA20 ~ {last_sma20:.2f}")
            if avwap_last is not None:
                targets.append(f"Mean: AVWAP ~ {float(avwap_last):.2f}")
            targets.append(f"Support (20D low): {recent_low:.2f}")
        elif stretched_up:
            bias = "RANGE (pullback risk)"
            targets.append(f"Mean: SMA20 ~ {last_sma20:.2f}")
            if avwap_last is not None:
                targets.append(f"Mean: AVWAP ~ {float(avwap_last):.2f}")
            targets.append(f"Resistance (20D high): {recent_high:.2f}")
        else:
            bias = "RANGE"
            targets.append(f"Range: {recent_low:.2f} – {recent_high:.2f}")
            if avwap_signal:
                targets.append(f"AVWAP: {avwap_signal}")

    # Likely 7D range: prefer forecast bands (risk-adjusted), fallback to ATR envelope
    if forecast is not None and getattr(forecast, "bands", None):
        bands = forecast.bands
        range_low = float(bands.get("p05", last_close - 2 * last_atr))
        range_high = float(bands.get("p95", last_close + 2 * last_atr))
    else:
        mult = 2.5 if regime == "TRENDING" else 2.0
        range_low = float(last_close - mult * last_atr) if last_atr else float(recent_low)
        range_high = float(last_close + mult * last_atr) if last_atr else float(recent_high)

    # Plain-English summary
    if regime == "TRENDING":
        summary = (
            f"Regime is TRENDING (ADX≈{last_adx:.1f}). Using TREND engine: "
            f"best results come from following the trend and using SMA20/AVWAP as support/resistance."
        )
    else:
        summary = (
            f"Regime is CHOP/RANGE (ADX≈{last_adx:.1f}). Using MEAN-REVERSION engine: "
            f"best results come from buying near support and selling near resistance unless a real breakout appears."
        )

    return DirectionReport(
        engine=engine,
        regime=regime,
        bias=bias,
        range_low=range_low,
        range_high=range_high,
        summary=summary,
        targets=targets,
    )
