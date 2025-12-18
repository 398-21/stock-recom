from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
import math


@dataclass
class Forecast7D:
    last_close: float
    expected_close: float
    p_up: float
    p_down: float
    bands: Dict[str, float]  # p05,p25,p50,p75,p95
    expected_return: float
    expected_vol_7d: float
    model: str
    confidence: str          # HIGH / MED / LOW
    notes: str


def _ewma_vol(log_returns: np.ndarray, lam: float = 0.94) -> float:
    if len(log_returns) < 20:
        return float(np.std(log_returns, ddof=1))

    var = log_returns[0] ** 2
    for r in log_returns[1:]:
        var = lam * var + (1 - lam) * (r ** 2)
    return float(math.sqrt(var))


def _confidence_badge(event_risk_level: str, news_intensity: float, vol20: float | None) -> str:
    """
    Wall-Street style confidence:
    - LOW when headline/event risk is high or volatility is high
    - MED when moderate
    - HIGH when calm
    """
    lvl = (event_risk_level or "LOW").upper()
    ni = max(0.0, min(1.0, float(news_intensity)))

    # volatility thresholds are rough; adapt later if you want percentile-based
    v = float(vol20) if vol20 is not None else 0.0

    if lvl == "HIGH" or ni >= 0.65 or v >= 0.06:
        return "LOW"
    if lvl == "MED" or ni >= 0.30 or v >= 0.035:
        return "MED"
    return "HIGH"


def forecast_next_7_days_ewma(
    hist: pd.DataFrame,
    *,
    n_days: int = 7,
    lookback_days: int = 180,
    n_sims: int = 8000,
    seed: int = 42,
    lam: float = 0.94,
    event_risk_level: str = "LOW",   # LOW / MED / HIGH
    min_bars: int = 45,
    news_intensity: float = 0.0,     # 0..1 (range widening only)
    vol20: float | None = None,      # optional for confidence badge
) -> Forecast7D:
    """
    Probabilistic 7D forecast (Wall Street style):
      - Price-only drift (mu) from returns
      - EWMA volatility baseline
      - Event risk widens sigma (dilution/filing risk)
      - News intensity widens sigma (range only)
      - NO directional tilt from news (ultra-pure)
    """
    df = hist.dropna().copy()
    if "Close" not in df.columns:
        raise RuntimeError("Forecast requires Close series.")

    if len(df) < min_bars:
        raise RuntimeError(f"Insufficient data for forecast (need ~{min_bars}+ daily bars).")

    close = df["Close"].astype(float)
    close = close.iloc[-lookback_days:] if len(close) > lookback_days else close

    lr = np.log(close).diff().dropna().to_numpy()
    if len(lr) < max(30, min_bars - 1):
        raise RuntimeError("Insufficient return history after cleaning.")

    mu = float(np.mean(lr))
    sigma = _ewma_vol(lr, lam=lam)

    # 1) Event widening
    lvl = (event_risk_level or "LOW").upper()
    if lvl == "HIGH":
        sigma *= 1.8
        event_note = "Event widening HIGH (x1.8)."
    elif lvl == "MED":
        sigma *= 1.35
        event_note = "Event widening MED (x1.35)."
    else:
        event_note = "Event widening LOW (x1.0)."

    # 2) News intensity widening (range only; no drift tilt)
    ni = max(0.0, min(1.0, float(news_intensity)))
    sigma *= (1.0 + 0.50 * ni)
    news_note = f"News intensity widening (x{1.0 + 0.50*ni:.2f}); range-only (no direction tilt)."

    last = float(close.iloc[-1])

    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu, scale=sigma, size=(n_sims, n_days))
    end_prices = last * np.exp(shocks.cumsum(axis=1)[:, -1])

    p05, p25, p50, p75, p95 = np.percentile(end_prices, [5, 25, 50, 75, 95])
    expected = float(np.mean(end_prices))

    p_up = float(np.mean(end_prices > last))
    p_down = float(np.mean(end_prices < last))

    expected_return = (expected / last) - 1.0
    expected_vol_7d = float(np.std(np.log(end_prices / last), ddof=1))

    conf = _confidence_badge(event_risk_level=lvl, news_intensity=ni, vol20=vol20)

    notes = (
        f"{event_note} {news_note} "
        "This is a probabilistic range forecast for the 7D close. "
        "Direction is driven by price history; news only widens/narrows uncertainty."
    )

    return Forecast7D(
        last_close=last,
        expected_close=expected,
        p_up=p_up,
        p_down=p_down,
        bands={"p05": float(p05), "p25": float(p25), "p50": float(p50), "p75": float(p75), "p95": float(p95)},
        expected_return=float(expected_return),
        expected_vol_7d=expected_vol_7d,
        model="EWMA_MC_RANGE_ONLY",
        confidence=conf,
        notes=notes,
    )
