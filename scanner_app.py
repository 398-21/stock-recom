from __future__ import annotations

import sys
import os
from datetime import datetime
from typing import List

import yfinance as yf

from data.sec import fetch_recent_filings
from data.news import fetch_google_news_rss
from analysis.dilution import score_dilution_risk
from analysis.technicals import analyze_technicals
from alerts.telegram import send_telegram
from alerts.emailer import send_email


def _fmt_pct(x):
    if x is None:
        return "N/A"
    return f"{x*100:.2f}%"


def _fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def build_report(ticker: str) -> str:
    ticker = ticker.upper().strip()

    # Prices
    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", interval="1d", auto_adjust=False)
    tech = analyze_technicals(hist)

    # SEC filings (catalyst / dilution)
    filings = fetch_recent_filings(ticker, limit=30)
    dil = score_dilution_risk(filings, lookback_n=15)

    # News (free)
    # Query best-effort: ticker + company name if available
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    q = info.get("longName") or info.get("shortName") or ticker
    news = fetch_google_news_rss(f"{q} {ticker}", max_items=10)

    # Compose “Wall Street style” report
    lines = []
    lines.append(f"DAILY CATALYST SCANNER — {ticker}")
    lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("MARKET MICROSTRUCTURE (price/vol/volume)")
    lines.append(f"- Last close: {tech.last_close:.2f}")
    lines.append(f"- Trend regime: {tech.trend}")
    lines.append(f"- 20D volatility (std): {_fmt_pct(tech.vol_20d)}")
    lines.append(f"- ATR(14): {_fmt(tech.atr_14)}")
    lines.append(f"- Relative volume (today/avg20): {_fmt(tech.rel_volume)} {'(ABNORMAL)' if tech.abnormal_volume_flag else ''}")
    lines.append(f"- Liquidity sweep: {tech.sweep_signal}")
    lines.append(f"- BOS: {tech.bos_signal}")
    lines.append("")
    lines.append("CATALYST / FILINGS (SEC EDGAR)")
    if filings:
        lines.append("- Recent filings (top 10):")
        for f in filings[:10]:
            lines.append(f"  • {f['filingDate']} — {f['form']}")
    else:
        lines.append("- No filings found (or ticker->CIK mapping failed).")

    lines.append(f"- Dilution risk score: {dil.score}/100 ({dil.level})")
    for r in dil.reasons:
        lines.append(f"  • {r}")

    lines.append("")
    lines.append("NEWS (Google News RSS)")
    if news:
        for a in news[:8]:
            lines.append(f"- {a.get('publishedAt','')} | {a.get('source','')}: {a.get('title','')}")
    else:
        lines.append("- No news items returned.")

    # Simple actionable interpretation (you can tighten rules later)
    lines.append("")
    lines.append("ACTION LENS (entry-type guidance)")
    if dil.level == "HIGH":
        lines.append("- Catalyst entry caution: dilution/financing risk elevated. Require manual read of latest filings (esp. 8-K / 424B / S-1/S-3).")
    else:
        lines.append("- Catalyst posture: no strong dilution-form signals; focus on upcoming event window and thesis validity.")

    if tech.abnormal_volume_flag:
        lines.append("- Tape note: abnormal volume today; treat as information day (possible catalyst leak / positioning).")
    else:
        lines.append("- Tape note: volume normal; moves likely liquidity/noise unless supported by filings/news.")

    return "\n".join(lines)


def maybe_alert(report: str) -> None:
    # Telegram
    sent_tg = send_telegram(report)

    # Email (subject trimmed)
    subject = "Daily Catalyst Scanner"
    sent_em = send_email(subject, report)

    # Always print what happened (no silent failure)
    if sent_tg:
        print("[ALERT] Telegram sent.")
    if sent_em:
        print("[ALERT] Email sent.")
    if not sent_tg and not sent_em:
        print("[ALERT] No alert channel configured (Telegram/Email env vars not set).")


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scanner_app.py <TICKER> [--alert]")
        print("Example: python scanner_app.py TELO --alert")
        return 1

    ticker = argv[1].strip().upper()
    do_alert = ("--alert" in argv)

    report = build_report(ticker)
    print(report)

    if do_alert:
        maybe_alert(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
