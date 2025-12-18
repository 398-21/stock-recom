from __future__ import annotations

import os
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from data.sec import fetch_recent_filings
from data.news import fetch_google_news_rss
from analysis.dilution import score_dilution_risk
from analysis.technicals import analyze_technicals
from analysis.scoring import decide_entry
from analysis.forecast import forecast_next_7_days_ewma
from analysis.avwap import analyze_avwap
from analysis.news_signal import build_news_signal

from ui.explainers import (
    size_recommendation,
    emoji_action_label,
    tooltip_map,
    kid_explainer_block,
    invalidation_box,
)

st.set_page_config(page_title="Daily Catalyst Scanner", layout="wide")


def fmt_pct(x):
    return "N/A" if x is None else f"{x * 100:.2f}%"


def fmt_float(x, nd=4):
    return "N/A" if x is None else f"{x:.{nd}f}"


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Open" in df.columns and df.index.inferred_type == "datetime64":
        return df.resample("1D").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum" if "Volume" in df.columns else "size",
            }
        ).dropna()
    return df


def make_price_volume_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df.get("Volume", pd.Series([0] * len(df), index=df.index)),
            name="Volume",
            yaxis="y2",
            opacity=0.35,
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_avwap_overlay_chart(df_daily: pd.DataFrame, avwap: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_daily.index,
            open=df_daily["Open"],
            high=df_daily["High"],
            low=df_daily["Low"],
            close=df_daily["Close"],
            name="Price",
        )
    )
    fig.add_trace(go.Scatter(x=avwap.index, y=avwap.values, mode="lines", name="Anchored VWAP"))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_volatility_chart(df: pd.DataFrame) -> go.Figure:
    rets = df["Close"].pct_change()
    vol20 = rets.rolling(20).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=vol20, mode="lines", name="20D Vol (Std)"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def make_gauge(score: int, label: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": label},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [{"range": [0, 35]}, {"range": [35, 70]}, {"range": [70, 100]}],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------- Sidebar ----------------
st.sidebar.title("Scanner Controls")

ticker = st.sidebar.text_input("Ticker", value="TELO").strip().upper()
period = st.sidebar.selectbox("Price window", ["3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

mode = st.sidebar.selectbox("Decision Mode", ["PURE_CATALYST", "HYBRID", "PURE_TECHNICAL"], index=1)
kid_mode = st.sidebar.toggle("Kid-mode explanations", value=True)

st.sidebar.markdown("---")
st.sidebar.write("SEC EDGAR requires `SEC_USER_AGENT` env var.")
st.sidebar.code('setx SEC_USER_AGENT "StockEntryAnalyzer/1.0 (email: you@example.com)"')

st.sidebar.write("If ticker→CIK mapping misses a name (e.g., TELO), set override once:")
st.sidebar.code('setx SEC_CIK_OVERRIDES "TELO=1971532"')

st.sidebar.markdown("---")
anchor_mode = st.sidebar.selectbox("AVWAP Anchor", ["LAST_60D_LOW", "LAST_60D_HIGH", "LAST_20D_START"], index=0)
custom_anchor = st.sidebar.text_input("Custom anchor date (YYYY-MM-DD, optional)", value="").strip() or None

run_btn = st.sidebar.button("Run Scan", type="primary")


# ---------------- Main ----------------
st.title("Daily Catalyst Scanner Dashboard")
st.caption(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")

if not run_btn:
    st.info("Enter a ticker and click **Run Scan**.")
    st.stop()

ua = (os.getenv("SEC_USER_AGENT") or "").strip()
if not ua:
    st.error(
        "SEC_USER_AGENT is not set.\n\n"
        "PowerShell (current session):\n"
        "$env:SEC_USER_AGENT = \"StockEntryAnalyzer/1.0 (email: you@example.com)\"\n\n"
        "Permanent:\n"
        "setx SEC_USER_AGENT \"StockEntryAnalyzer/1.0 (email: you@example.com)\" (reopen terminal)"
    )
    st.stop()

t = yf.Ticker(ticker)
hist = t.history(period=period, interval=interval, auto_adjust=False)

if hist.empty:
    st.error(f"No price history returned for {ticker}. Check ticker symbol.")
    st.stop()

if "Volume" not in hist.columns:
    hist["Volume"] = 0

daily = hist if interval == "1d" else to_daily(hist)

# Technicals
tech = analyze_technicals(daily)

# SEC filings + dilution
filings = fetch_recent_filings(ticker, limit=30)
dil = score_dilution_risk(filings, lookback_n=15)

# News (free RSS)
try:
    info = t.info or {}
except Exception:
    info = {}
q = info.get("longName") or info.get("shortName") or ticker
news = fetch_google_news_rss(f"{q} {ticker}", max_items=12)
ns = build_news_signal(news)

# Decision badge (GO / WAIT / AVOID)
sr = decide_entry(
    mode=mode,
    dilution_level=dil.level,
    dilution_score=dil.score,
    abnormal_volume=tech.abnormal_volume_flag,
    trend=tech.trend,
    sweep_signal=tech.sweep_signal,
    bos_signal=tech.bos_signal,
    vol_20d=tech.vol_20d,
)

decision_label = f"{sr.decision} — Score {sr.score}/100"

# Forecast (two versions: baseline vs headline-risk adjusted)
forecast_err = None
try:
    daily_for_fc = t.history(period="1y", interval="1d", auto_adjust=False)
    if daily_for_fc.empty:
        daily_for_fc = daily.copy()

    fc_base = forecast_next_7_days_ewma(
        daily_for_fc,
        event_risk_level="LOW",
        news_intensity=0.0,
        vol20=tech.vol_20d,
    )
    fc_risk = forecast_next_7_days_ewma(
        daily_for_fc,
        event_risk_level=dil.level,
        news_intensity=getattr(ns, "intensity", 0.0),
        vol20=tech.vol_20d,
    )
except Exception as e:
    fc_base = None
    fc_risk = None
    forecast_err = str(e)

# AVWAP
avwap_err = None
try:
    avwap_series, avrep = analyze_avwap(daily, anchor_mode=anchor_mode, custom_anchor=custom_anchor)
except Exception as e:
    avwap_series, avrep = None, None
    avwap_err = str(e)


# ---------------- Decision Panel ----------------
st.subheader("Decision Panel")
c0, c1, c2, c3 = st.columns([1.2, 1.0, 1.0, 2.5])

with c0:
    if sr.decision == "GO":
        st.success(decision_label)
    elif sr.decision == "WAIT":
        st.warning(decision_label)
    else:
        st.error(decision_label)

with c1:
    st.metric("Mode", sr.mode)

with c2:
    st.metric("Dilution Risk", f"{dil.level} ({dil.score}/100)", help=tooltip_map()["DILUTION"])

with c3:
    st.write("**Why (rules-based):**")
    for r in sr.reasons[:6]:
        st.write(f"- {r}")


# ---------------- Kid-mode explainers + invalidation ----------------
if kid_mode:
    snap = {
        "avwap_signal": getattr(avrep, "signal", "N/A") if avrep else "N/A",
        "avwap_dist": f"{getattr(avrep, 'distance_pct', 0.0) * 100:.2f}%" if avrep else "N/A",
        "trend": str(tech.trend),
        "rvol": f"{tech.rel_volume:.2f}",
        "vol20": fmt_pct(tech.vol_20d),
        "atr": fmt_float(getattr(tech, "atr_14", None), 4),
        "sweep": str(tech.sweep_signal),
        "bos": str(tech.bos_signal),
        "dilution_level": str(dil.level),
        "decision": decision_label,
        "mode": str(mode),
    }

    with st.expander("🧒 Explain these signals (kid-mode)", expanded=True):
        st.markdown(kid_explainer_block(snap))

    with st.expander("🧯 What would invalidate this?", expanded=True):
        st.markdown(invalidation_box(str(mode)))
else:
    st.caption("Kid-mode explanations are off (toggle in sidebar).")


# How-to-read AVWAP (so you understand)
with st.expander("How to read AVWAP (Anchored VWAP) and what it implies"):
    st.write(
        "AVWAP is the Volume-Weighted Average Price starting from a chosen anchor date. "
        "It approximates the average cost basis of participants since that event/anchor.\n\n"
        "**Key implications:**\n"
        "1) **Price above rising AVWAP** → bullish bias. Dips into AVWAP often act as support because buyers are profitable and defend.\n"
        "2) **Price below falling AVWAP** → bearish bias. Rallies into AVWAP often get sold because trapped holders use it to exit.\n"
        "3) **Reclaim** is the highest-signal event: price moves from below to above AVWAP and holds for several bars.\n"
        "4) **Distance from AVWAP** matters: far above can mean extended (pullback risk), far below can mean capitulation or trend down.\n\n"
        "**How to use with your 3 modes:**\n"
        "- PURE_CATALYST: AVWAP is entry-timing and risk control (avoid chasing far above AVWAP).\n"
        "- HYBRID: prefer GO when catalyst risk is acceptable AND price is above/rising AVWAP or reclaiming.\n"
        "- PURE_TECHNICAL: AVWAP becomes a core dynamic support/resistance level."
    )


# ---------------- Main layout ----------------
colA, colB = st.columns([2.2, 1.3])

with colA:
    st.subheader("Price & Volume")
    st.plotly_chart(make_price_volume_chart(hist), width="stretch")

    st.subheader("Anchored VWAP Overlay (Daily)")
    if avwap_series is None or avrep is None:
        st.warning(f"AVWAP unavailable: {avwap_err}")
    else:
        st.plotly_chart(make_avwap_overlay_chart(daily, avwap_series), width="stretch")
        st.write(
            f"- Anchor date: **{avrep.anchor_date}**\n"
            f"- Last AVWAP: **{avrep.avwap_last:.2f}** | Last Close: **{avrep.price_last:.2f}**\n"
            f"- Distance: **{avrep.distance_pct*100:.2f}%** | Signal: **{avrep.signal}**"
        )

with colB:
    st.subheader("Tape / Risk Metrics")
    st.metric("Last Close", f"{tech.last_close:.2f}")
    st.metric("Trend Regime", tech.trend, help=tooltip_map()["TREND"])
    st.metric("20D Volatility", fmt_pct(tech.vol_20d))
    st.metric("ATR(14)", fmt_float(getattr(tech, "atr_14", None), 4), help=tooltip_map()["ATR"])
    st.metric("Rel Volume", fmt_float(tech.rel_volume, 2), help=tooltip_map()["RVOL"])
    st.metric("Abnormal Vol", "YES" if tech.abnormal_volume_flag else "NO")
    st.write("**Liquidity / Structure**")
    st.write(f"- Sweep: {tech.sweep_signal}")
    st.write(f"- BOS: {tech.bos_signal}")
    st.plotly_chart(make_volatility_chart(daily), width="stretch")

    st.subheader("Dilution / Financing Risk")
    st.plotly_chart(make_gauge(dil.score, f"Dilution Risk ({dil.level})"), width="stretch")
    for r in dil.reasons:
        st.write(f"- {r}")

st.markdown("---")


# ---------------- Forecast panel ----------------
st.subheader("Next 7 Market Days Forecast (Probabilistic)")

if fc_risk is None or fc_base is None:
    st.warning(f"Forecast unavailable: {forecast_err}")
else:
    conf = fc_risk.confidence
    if conf == "HIGH":
        st.success("Forecast Confidence: HIGH", icon="✅")
    elif conf == "MED":
        st.warning("Forecast Confidence: MED", icon="⚠️")
    else:
        st.error("Forecast Confidence: LOW", icon="🛑")

    # Best move line (mode + confidence + decision)
    best_line, best_why = size_recommendation(
        mode=str(mode),
        confidence=str(conf),
        decision=str(sr.decision),
    )

    # Emoji action based on decision
    action = "BUY" if sr.decision == "GO" else ("AVOID" if sr.decision == "AVOID" else "WAIT")

    st.markdown(f"### {emoji_action_label(action)}")
    st.markdown(best_line)
    st.caption(best_why)
    st.caption(getattr(ns, "explanation", ""))

    cL, cR = st.columns(2)
    with cL:
        st.markdown("#### Price-only (baseline)")
        a1, a2, a3 = st.columns(3)
        a1.metric("Expected Close (7D)", f"{fc_base.expected_close:.2f}", help=tooltip_map()["FORECAST_CONF"])
        a2.metric("P(Up over 7D)", f"{fc_base.p_up*100:.1f}%")
        a3.metric("Band (5%–95%)", f"{fc_base.bands['p05']:.2f} – {fc_base.bands['p95']:.2f}")
        st.caption(fc_base.notes)

    with cR:
        st.markdown("#### Headline-risk adjusted")
        b1, b2, b3 = st.columns(3)
        b1.metric("Expected Close (7D)", f"{fc_risk.expected_close:.2f}", help=tooltip_map()["FORECAST_CONF"])
        b2.metric("P(Up over 7D)", f"{fc_risk.p_up*100:.1f}%")
        b3.metric("Band (5%–95%)", f"{fc_risk.bands['p05']:.2f} – {fc_risk.bands['p95']:.2f}")
        st.caption(fc_risk.notes)

st.markdown("---")


# ---------------- Filings + News ----------------
col1, col2 = st.columns([1.2, 1.0])

with col1:
    st.subheader("SEC Filings (Recent)")
    if filings:
        df_f = pd.DataFrame(filings[:15])[["filingDate", "form", "accessionNumber", "primaryDocument"]]
        st.dataframe(df_f, width="stretch", height=360)
    else:
        st.warning(
            "No filings returned. If this is a known ticker, it is usually a ticker→CIK mapping miss.\n\n"
            "Fix: set `SEC_CIK_OVERRIDES` e.g. `TELO=1971532` and rerun."
        )

with col2:
    st.subheader("News (Google News RSS)")
    if news:
        for a in news[:10]:
            st.markdown(
                f"- **{a.get('title','')}**  \n"
                f"  {a.get('source','')} | {a.get('publishedAt','')}  \n"
                f"  {a.get('url','')}"
            )
    else:
        st.write("No news items returned.")
