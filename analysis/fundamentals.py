from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math


@dataclass
class FundamentalsReport:
    health_score: int              # 0..100 (higher = healthier)
    health_level: str              # HIGH / MED / LOW
    runway_months_est: Optional[float]
    leverage_flag: bool
    profitability_flag: bool
    notes: list[str]
    raw: Dict[str, Any]


def _safe_num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def analyze_fundamentals_from_yf_info(info: Dict[str, Any]) -> FundamentalsReport:
    """
    Uses yfinance .info fields only (fast, free, available for many tickers).
    This is NOT valuation; it's *financial stability / financing pressure*.
    """
    notes: list[str] = []
    raw = dict(info or {})

    cash = _safe_num(raw.get("totalCash"))
    debt = _safe_num(raw.get("totalDebt"))
    fcf = _safe_num(raw.get("freeCashflow"))          # often available, may be None
    op_cf = _safe_num(raw.get("operatingCashflow"))   # sometimes available
    ebitda = _safe_num(raw.get("ebitda"))             # sometimes available
    net_income = _safe_num(raw.get("netIncomeToCommon"))
    current_ratio = _safe_num(raw.get("currentRatio"))
    quick_ratio = _safe_num(raw.get("quickRatio"))

    # Estimate burn/runway if we can
    # If FCF is negative, burn ~= -FCF/yr. Runway = cash / (burn/12).
    runway_m: Optional[float] = None
    burn_per_year = None

    if fcf is not None and fcf < 0 and cash is not None and cash > 0:
        burn_per_year = -fcf
        runway_m = cash / (burn_per_year / 12.0)
        notes.append(f"Runway estimate uses negative FCF: cash / burn.")
    elif op_cf is not None and op_cf < 0 and cash is not None and cash > 0:
        burn_per_year = -op_cf
        runway_m = cash / (burn_per_year / 12.0)
        notes.append(f"Runway estimate uses negative operating cashflow: cash / burn.")
    else:
        notes.append("Runway estimate unavailable (missing cash + negative cashflow fields).")

    # Flags
    leverage_flag = False
    if cash is not None and debt is not None:
        if debt > cash * 1.5:
            leverage_flag = True
            notes.append("High leverage: debt materially exceeds cash.")
    elif debt is not None and debt > 0:
        notes.append("Debt present but cash unknown; leverage risk uncertain.")

    profitability_flag = False
    if net_income is not None and net_income > 0:
        profitability_flag = True
        notes.append("Net income positive (profitable on trailing basis).")
    elif ebitda is not None and ebitda > 0:
        notes.append("EBITDA positive (operational profitability improving).")
    else:
        notes.append("Profitability weak/unclear (net income and EBITDA not positive or unavailable).")

    # Score: start at 60, adjust based on stability
    score = 60

    # Liquidity health
    if current_ratio is not None:
        if current_ratio >= 2.0:
            score += 10
            notes.append("Strong liquidity: current ratio >= 2.")
        elif current_ratio < 1.0:
            score -= 15
            notes.append("Weak liquidity: current ratio < 1.")
    if quick_ratio is not None and quick_ratio < 1.0:
        score -= 5
        notes.append("Quick ratio < 1 suggests tighter near-term liquidity.")

    # Runway adjustments (key for biotech/event-driven names)
    if runway_m is not None:
        if runway_m >= 18:
            score += 15
            notes.append("Runway >= 18 months (lower near-term financing pressure).")
        elif runway_m >= 9:
            score += 5
            notes.append("Runway 9–18 months (moderate financing pressure).")
        else:
            score -= 20
            notes.append("Runway < 9 months (high financing pressure).")

    # Leverage penalty
    if leverage_flag:
        score -= 15

    # Profitability benefit
    if profitability_flag:
        score += 10

    score = max(0, min(100, score))

    if score >= 75:
        lvl = "HIGH"
    elif score >= 45:
        lvl = "MED"
    else:
        lvl = "LOW"

    return FundamentalsReport(
        health_score=score,
        health_level=lvl,
        runway_months_est=runway_m,
        leverage_flag=leverage_flag,
        profitability_flag=profitability_flag,
        notes=notes,
        raw=raw,
    )
