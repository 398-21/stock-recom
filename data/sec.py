from __future__ import annotations

import os
import time
import json
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional

SEC_TICKER_MAP_URLS = [
    "https://www.sec.gov/files/company_tickers_exchange.json",
    "https://www.sec.gov/files/company_tickers.json",
]

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"

# EDGAR company search (ticker-friendly). The ATOM output includes <cik>...</cik>.
SEC_BROWSE_ATOM_URL = "https://www.sec.gov/cgi-bin/browse-edgar?CIK={q}&owner=exclude&action=getcompany&output=atom"

# Local cache so you don’t repeatedly hit SEC
CACHE_PATH = Path("data/sec_cik_cache.json")


def _sec_headers() -> Dict[str, str]:
    ua = (os.getenv("SEC_USER_AGENT") or "").strip()
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT is not set.\n"
            "PowerShell (current): $env:SEC_USER_AGENT = 'App/1.0 (email: you@example.com)'\n"
            "Permanent: setx SEC_USER_AGENT 'App/1.0 (email: you@example.com)' then reopen terminal"
        )
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
    }


def _get(url: str, timeout: int = 25) -> requests.Response:
    r = requests.get(url, headers=_sec_headers(), timeout=timeout)
    return r


def _get_json(url: str, timeout: int = 25) -> Any:
    r = _get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"SEC request failed: {r.status_code} | {url} | {r.text[:300]}")
    return r.json()


def _load_cache() -> Dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: Dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _ticker_to_cik_from_mapping(ticker: str) -> Optional[str]:
    """
    Attempt mapping from SEC files (fast, but can be incomplete).
    Supports both list and dict formats.
    """
    tkr = ticker.upper().strip()

    for url in SEC_TICKER_MAP_URLS:
        try:
            data = _get_json(url)

            # Newer format: list of dicts
            if isinstance(data, list):
                for row in data:
                    if (row.get("ticker") or "").upper() == tkr:
                        cik = str(row.get("cik") or "")
                        if cik.isdigit():
                            return cik.zfill(10)

            # Older format: dict keyed by index
            if isinstance(data, dict):
                for _, row in data.items():
                    if (row.get("ticker") or "").upper() == tkr:
                        cik = str(row.get("cik_str") or row.get("cik") or "")
                        if cik.isdigit():
                            return cik.zfill(10)

        except Exception:
            continue

    return None


def _ticker_to_cik_from_browse_atom(query: str) -> Optional[str]:
    """
    Robust fallback: call browse-edgar with CIK=query (ticker or name),
    parse ATOM XML and extract the <cik> element.
    """
    url = SEC_BROWSE_ATOM_URL.format(q=query)
    r = _get(url)
    if r.status_code != 200:
        return None

    # The response is XML. Parse it.
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return None

    # CIK is usually in: feed/company-info/cik
    # Namespaces may exist; handle by stripping namespace.
    def strip_ns(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for elem in root.iter():
        if strip_ns(elem.tag).lower() == "cik":
            val = (elem.text or "").strip()
            if val.isdigit():
                return val.zfill(10)

    return None


def ticker_to_cik(ticker: str, polite_sleep_s: float = 0.2) -> Optional[str]:
    """
    Auto-resolve ticker -> CIK with multiple strategies + caching.
    User only needs to input ticker.
    """
    tkr = ticker.upper().strip()
    cache = _load_cache()

    # 1) cache
    if tkr in cache:
        return cache[tkr].zfill(10)

    # 2) mapping JSON
    time.sleep(polite_sleep_s)
    cik = _ticker_to_cik_from_mapping(tkr)
    if cik:
        cache[tkr] = cik
        _save_cache(cache)
        return cik

    # 3) browse-edgar ATOM fallback (accepts ticker)
    time.sleep(polite_sleep_s)
    cik = _ticker_to_cik_from_browse_atom(tkr)
    if cik:
        cache[tkr] = cik
        _save_cache(cache)
        return cik

    return None


def fetch_recent_filings(
    ticker: str,
    limit: int = 30,
    polite_sleep_s: float = 0.2,
) -> List[Dict[str, Any]]:
    cik10 = ticker_to_cik(ticker, polite_sleep_s=polite_sleep_s)
    if not cik10:
        return []

    time.sleep(polite_sleep_s)
    sub = _get_json(SEC_SUBMISSIONS_URL.format(cik10=cik10))

    recent = (sub.get("filings") or {}).get("recent") or {}
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    accession = recent.get("accessionNumber", []) or []
    docs = recent.get("primaryDocument", []) or []

    out: List[Dict[str, Any]] = []
    for i in range(min(limit, len(forms), len(dates), len(accession), len(docs))):
        out.append({
            "ticker": ticker.upper(),
            "cik10": cik10,
            "form": forms[i],
            "filingDate": dates[i],
            "accessionNumber": accession[i],
            "primaryDocument": docs[i],
        })
    return out
