from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re


@dataclass
class OwnershipReport:
    float_shares: Optional[float]
    shares_outstanding: Optional[float]
    inst_held_pct: Optional[float]      # 0..1
    insider_held_pct: Optional[float]   # 0..1
    short_float_pct: Optional[float]    # 0..1
    atm_or_shelf_flag: bool
    atm_or_shelf_evidence: List[str]
    trapped_holder_hint: str
    notes: List[str]


def _num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except Exception:
        return None


def _pct(x) -> Optional[float]:
    v = _num(x)
    if v is None:
        return None
    # yfinance often provides 0..1 already, keep as is
    if v > 1.0 and v <= 100.0:
        return v / 100.0
    return v


_ATM_PATTERNS = [
    r"\bat[-\s]?the[-\s]?market\b",
    r"\bATM program\b",
    r"\batm offering\b",
    r"\bshelf registration\b",
    r"\bshelf\b",
    r"\bprospectus supplement\b",
    r"\b424B5\b",
    r"\b424B7\b",
    r"\bS-3\b",
    r"\bS-1\b",
]


def detect_atm_or_shelf_overhang(filings: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    evidence: List[str] = []
    if not filings:
        return False, evidence

    for f in filings[:30]:
        form = (f.get("form") or "").upper()
        doc = (f.get("primaryDocument") or "")
        # forms that often indicate financing pathways
        if form in {"S-3", "S-1", "424B5", "424B7", "424B3", "424B4"}:
            evidence.append(f"Form {form} ({f.get('filingDate','')})")
        if isinstance(doc, str) and doc:
            for pat in _ATM_PATTERNS:
                if re.search(pat, doc, flags=re.IGNORECASE):
                    evidence.append(f"Keyword match in document name: {doc}")
                    break

    # de-dup
    evidence = list(dict.fromkeys(evidence))
    return (len(evidence) > 0), evidence[:8]


def analyze_ownership_from_yf_info(info: Dict[str, Any], filings: List[Dict[str, Any]]) -> OwnershipReport:
    raw = info or {}
    notes: List[str] = []

    float_shares = _num(raw.get("floatShares"))
    shares_outstanding = _num(raw.get("sharesOutstanding"))
    inst_held = _pct(raw.get("heldPercentInstitutions"))
    insider_held = _pct(raw.get("heldPercentInsiders"))
    short_float = _pct(raw.get("shortPercentOfFloat"))

    atm_flag, atm_ev = detect_atm_or_shelf_overhang(filings)

    if float_shares is None:
        notes.append("Float shares unavailable (data source may be missing).")
    if inst_held is None:
        notes.append("Institutional ownership % unavailable.")
    if short_float is None:
        notes.append("Short % of float unavailable.")

    # trapped holders heuristic (context only)
    # If high institutional ownership and price under AVWAP often means "defensive selling overhead"
    trapped_hint = "N/A"
    if inst_held is not None:
        if inst_held >= 0.60:
            trapped_hint = "High institutional ownership: flows can dominate; reactions around filings/earnings can be sharp."
        elif inst_held <= 0.15:
            trapped_hint = "Low institutional ownership: price can be more retail-driven and volatile."
        else:
            trapped_hint = "Mixed ownership: both retail and institutions can influence moves."

    if atm_flag:
        notes.append("ATM/shelf overhang detected: potential supply cap during rallies.")

    return OwnershipReport(
        float_shares=float_shares,
        shares_outstanding=shares_outstanding,
        inst_held_pct=inst_held,
        insider_held_pct=insider_held,
        short_float_pct=short_float,
        atm_or_shelf_flag=atm_flag,
        atm_or_shelf_evidence=atm_ev,
        trapped_holder_hint=trapped_hint,
        notes=notes,
    )
