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
from analysis.fundamentals import analyze_fundamentals_from_yf_info

# KEEP: top-tier additions
from analysis.thesis import Thesis, build_thesis_panel
from analysis.ownership import analyze_ownership_from_yf_info
from analysis.playbook import build_post_event_playbook

# RE-ADD: price movement “engine”
from analysis.direction import analyze_direction

# NEW: K-Line / candlestick reader (trend + entries + explanation)
from analysis.kline import analyze_kline

# RE-ADD: chart helpers (range box + level labels + background tint)
from ui.chart_helpers import compute_sr_levels, add_levels_to_chart, bias_tint

from ui.explainers import (
    size_recommendation,
    emoji_action_label,
    tooltip_map,
    kid_explainer_block,
    invalidation_box,
)

def prob_badge(pct: int) -> str:
    """Return a coloured label for LOW / MED / HIGH probability."""
    if pct >= 70:
        label = "HIGH"
        color = "#d90429"   # red
    elif pct >= 40:
        label = "MEDIUM"
        color = "#fca311"   # amber
    else:
        label = "LOW"
        color = "#2b9348"   # green

    return f"<span style='color:{color}; font-weight:600'>{label}</span> ({pct}%)"


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
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()
    return df


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def build_levels_table(
    *,
    last_close: float,
    sma20_last: float | None,
    sma50_last: float | None,
    avwap_last: float | None,
    sr,  # SRLevels or similar
    range_low: float | None,
    range_high: float | None,
) -> pd.DataFrame:
    rows = []

    def add(name: str, price: float | None, note: str = ""):
        if price is None:
            return
        dist_pct = (price / last_close - 1.0) if last_close else None
        rows.append(
            {
                "Level": name,
                "Price": float(price),
                "Distance vs Last Close": f"{dist_pct*100:+.2f}%" if dist_pct is not None else "N/A",
                "Meaning": note,
            }
        )

    add("Resistance", getattr(sr, "resistance", None), "Common ceiling where price may struggle.")
    add("R1", getattr(sr, "r1", None), "Pivot-based resistance level.")
    add("Pivot", getattr(sr, "pivot", None), "Reference level from last candle (map line).")
    add("S1", getattr(sr, "s1", None), "Pivot-based support level.")
    add("Support", getattr(sr, "support", None), "Common floor where price may bounce.")

    add("AVWAP", avwap_last, "Average cost since anchor. Above = buyers winning; below = trapped sellers.")
    add("SMA20", sma20_last, "Short-term trend memory (20 days).")
    add("SMA50", sma50_last, "Medium-term trend memory (50 days).")

    add("7D Range Low (p05)", range_low, "Lower edge of probabilistic 7-day range (not direction).")
    add("7D Range High (p95)", range_high, "Upper edge of probabilistic 7-day range (not direction).")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # sort by price descending so higher levels are at top
    df = df.sort_values("Price", ascending=False).reset_index(drop=True)
    return df


def make_price_volume_chart(
    df: pd.DataFrame,
    *,
    daily_df: pd.DataFrame,
    sma20: pd.Series,
    sma50: pd.Series,
    avwap: pd.Series | None,
    dir_bias: str,
    range_low: float | None,
    range_high: float | None,
) -> go.Figure:
    fig = go.Figure()

    # Background tint = green/red based on bias
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        fillcolor=bias_tint(dir_bias),
        line=dict(width=0),
        layer="below",
    )

    # Likely range box (7D p05–p95)
    if range_low is not None and range_high is not None and range_high > range_low > 0:
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=float(range_low),
            y1=float(range_high),
            fillcolor="rgba(30, 144, 255, 0.12)",
            line=dict(width=1, color="rgba(30, 144, 255, 0.45)"),
            layer="below",
        )

    # Price candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))

    # Volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df.get("Volume", pd.Series([0] * len(df), index=df.index)),
        name="Volume",
        yaxis="y2",
        opacity=0.35
    ))

    # Overlays (trend memory)
    fig.add_trace(go.Scatter(x=sma20.index, y=sma20.values, mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=sma50.index, y=sma50.values, mode="lines", name="SMA50"))
    if avwap is not None and not avwap.empty:
        fig.add_trace(go.Scatter(x=avwap.index, y=avwap.values, mode="lines", name="AVWAP"))

    # Support/Resistance + labels
    sr_levels = compute_sr_levels(daily_df, lookback=60)
    add_levels_to_chart(
        fig,
        sma20=sma20,
        sma50=sma50,
        avwap=avwap,
        sr=sr_levels,
        range_low=range_low,
        range_high=range_high,
    )

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=90, t=40, b=10),
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_avwap_overlay_chart(df_daily: pd.DataFrame, avwap: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_daily.index, open=df_daily["Open"], high=df_daily["High"], low=df_daily["Low"],
        close=df_daily["Close"], name="Price"
    ))
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
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label},
        gauge={"axis": {"range": [0, 100]}, "steps": [{"range": [0, 35]}, {"range": [35, 70]}, {"range": [70, 100]}]},
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig




def posture_meter(mode: str, thesis: Thesis, dilution_level: str, fin_health: str) -> tuple[int, str]:
    m = (mode or "HYBRID").upper()
    dil = (dilution_level or "LOW").upper()
    fin = (fin_health or "MED").upper()

    if m != "PURE_CATALYST":
        return 0, "Posture meter is only relevant in PURE_CATALYST mode."

    score = 55
    why = []

    panel = build_thesis_panel(thesis)
    if panel.completeness_level == "HIGH":
        score += 15
        why.append("Thesis is explicit (clear X/Y).")
    elif panel.completeness_level == "LOW":
        score -= 15
        why.append("Thesis is vague; edge is unclear.")

    if dil == "HIGH":
        score -= 20
        why.append("High dilution pressure can cap rallies pre-catalyst.")
    elif dil == "MED":
        score -= 8
        why.append("Moderate dilution pressure: size carefully.")
    else:
        why.append("Low dilution pressure: less supply risk.")

    if fin == "HIGH":
        score += 8
        why.append("Financing stability is strong (lower forced dilution risk).")
    elif fin == "LOW":
        score -= 10
        why.append("Financing stability is weak (tail-risk elevated).")

    score = max(0, min(100, score))
    return score, " ".join(why[:4])


# ---------------- Sidebar ----------------
st.sidebar.title("Scanner Controls")

ticker = st.sidebar.text_input("Ticker", value="TELO").strip().upper()
period = st.sidebar.selectbox("Price window", ["3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

mode = st.sidebar.selectbox("Decision Mode", ["PURE_CATALYST", "HYBRID", "PURE_TECHNICAL"], index=1)
kid_mode = st.sidebar.toggle("Kid-mode explanations", value=True)

# NEW: which direction the K-Line module is allowed to trade
direction_mode_label = st.sidebar.selectbox(
    "Trade direction",
    ["LONGS ONLY (no shorts)", "SHORTS ONLY (no longs)", "LONGS + SHORTS"],
    index=0,  # default: you are a long-only trader
)

direction_mode = {
    "LONGS ONLY (no shorts)": "LONGS_ONLY",
    "SHORTS ONLY (no longs)": "SHORTS_ONLY",
    "LONGS + SHORTS": "BOTH",
}[direction_mode_label]

st.sidebar.markdown("---")
st.sidebar.write("SEC EDGAR requires `SEC_USER_AGENT` env var (Streamlit secrets).")
st.sidebar.code('SEC_USER_AGENT = "StockEntryAnalyzer/1.0 (email: you@example.com)"', language="toml")
st.sidebar.write("Optional: ticker→CIK override")
st.sidebar.code('SEC_CIK_OVERRIDES = "TELO=1971532"', language="toml")

st.sidebar.markdown("---")
anchor_mode = st.sidebar.selectbox("AVWAP Anchor", ["LAST_60D_LOW", "LAST_60D_HIGH", "LAST_20D_START"], index=0)
custom_anchor = st.sidebar.text_input("Custom anchor date (YYYY-MM-DD, optional)", value="").strip() or None

st.sidebar.markdown("---")
st.sidebar.subheader("Thesis (Decision-First)")
if "thesis_text" not in st.session_state:
    st.session_state.thesis_text = ""
if "invalidation" not in st.session_state:
    st.session_state.invalidation = ""
if "risk1" not in st.session_state:
    st.session_state.risk1 = ""
if "risk2" not in st.session_state:
    st.session_state.risk2 = ""
if "win_start" not in st.session_state:
    st.session_state.win_start = ""
if "win_end" not in st.session_state:
    st.session_state.win_end = ""

thesis_type = st.sidebar.selectbox("Thesis type", ["CATALYST", "FUNDAMENTAL", "TECHNICAL"], index=0)
st.session_state.thesis_text = st.sidebar.text_area(
    "Why should it go up? (X)", value=st.session_state.thesis_text, height=70,
    placeholder="Example: IND acceptance in Q1; market underpricing probability."
)
st.session_state.invalidation = st.sidebar.text_area(
    "Fails if what happens? (Y)", value=st.session_state.invalidation, height=60,
    placeholder="Example: dilution before catalyst, trial delay, negative filing."
)
st.session_state.risk1 = st.sidebar.text_input("Top risk #1", value=st.session_state.risk1, placeholder="Example: financing/dilution")
st.session_state.risk2 = st.sidebar.text_input("Top risk #2", value=st.session_state.risk2, placeholder="Example: catalyst delay")
st.session_state.win_start = st.sidebar.text_input("Window start (YYYY-MM-DD)", value=st.session_state.win_start)
st.session_state.win_end = st.sidebar.text_input("Window end (YYYY-MM-DD)", value=st.session_state.win_end)

run_btn = st.sidebar.button("Run Scan", type="primary")


# ---------------- Main ----------------
st.title("Daily Catalyst Scanner Dashboard (Top-Tier + Price Movement)")
st.caption(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")

if not run_btn:
    st.info("Enter a ticker, define thesis (optional but recommended), and click **Run Scan**.")
    st.stop()

ua = (os.getenv("SEC_USER_AGENT") or "").strip()
if not ua:
    st.error("SEC_USER_AGENT is not set in Streamlit secrets (TOML).")
    st.stop()

t = yf.Ticker(ticker)
hist = t.history(period=period, interval=interval, auto_adjust=False)
if hist.empty:
    st.error(f"No price history returned for {ticker}. Check ticker symbol.")
    st.stop()
if "Volume" not in hist.columns:
    hist["Volume"] = 0

daily = hist if interval == "1d" else to_daily(hist)

# Core analytics
tech = analyze_technicals(daily)
filings = fetch_recent_filings(ticker, limit=30)
dil = score_dilution_risk(filings, lookback_n=15)

try:
    info = t.info or {}
except Exception:
    info = {}

try:
    fin = analyze_fundamentals_from_yf_info(info)
except Exception:
    fin = analyze_fundamentals_from_yf_info({})

q = info.get("longName") or info.get("shortName") or ticker
news = fetch_google_news_rss(f"{q} {ticker}", max_items=12)
ns = build_news_signal(news)

own = analyze_ownership_from_yf_info(info, filings)

thesis = Thesis(
    thesis_type=thesis_type,
    thesis_text=st.session_state.thesis_text.strip(),
    window_start=(st.session_state.win_start.strip() or None),
    window_end=(st.session_state.win_end.strip() or None),
    top_risks=[st.session_state.risk1.strip(), st.session_state.risk2.strip()],
    invalidation=st.session_state.invalidation.strip(),
)
th_panel = build_thesis_panel(thesis)

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

# AVWAP
avwap_err = None
try:
    avwap_series, avrep = analyze_avwap(daily, anchor_mode=anchor_mode, custom_anchor=custom_anchor)
except Exception as e:
    avwap_series, avrep = None, None
    avwap_err = str(e)

# Forecast (needed for range box + direction engine context)
forecast_err = None
fc_risk = None
fc_base = None

try:
    daily_for_fc = t.history(period="1y", interval="1d", auto_adjust=False)
    if daily_for_fc.empty:
        daily_for_fc = daily.copy()

    fc_base = forecast_next_7_days_ewma(
        daily_for_fc,
        event_risk_level="LOW",
        news_intensity=0.0,
        vol20=tech.vol_20d,
        fin_health_level=fin.health_level,
        runway_months=fin.runway_months_est,
    )
    fc_risk = forecast_next_7_days_ewma(
        daily_for_fc,
        event_risk_level=dil.level,
        news_intensity=getattr(ns, "intensity", 0.0),
        vol20=tech.vol_20d,
        fin_health_level=fin.health_level,
        runway_months=fin.runway_months_est,
    )
except Exception as e:
    forecast_err = str(e)

# Direction / path engine (bias + regime + likely range + targets)
dirrep = analyze_direction(daily=daily, avwap_report=avrep, forecast=fc_risk)

range_low = None
range_high = None
if fc_risk is not None and getattr(fc_risk, "bands", None):
    range_low = float(fc_risk.bands.get("p05", 0.0)) if fc_risk.bands.get("p05") else None
    range_high = float(fc_risk.bands.get("p95", 0.0)) if fc_risk.bands.get("p95") else None

# SMAs
sma20 = _sma(daily["Close"], 20)
sma50 = _sma(daily["Close"], 50)

# ---------------- Thesis Panel ----------------
st.subheader("Thesis Panel (Decision-First)")
cT1, cT2, cT3, cT4 = st.columns([2.4, 1.2, 1.2, 1.2])
with cT1:
    st.write(f"**Thesis (X):** {th_panel.headline}")
    st.write(f"**Fails if (Y):** {th_panel.invalidation}")
with cT2:
    st.metric("Window", th_panel.window)
with cT3:
    st.metric("Clarity score", f"{th_panel.completeness_score}/100")
with cT4:
    lvl = th_panel.completeness_level
    if lvl == "HIGH":
        st.success("Thesis clarity: HIGH")
    elif lvl == "MED":
        st.warning("Thesis clarity: MED")
    else:
        st.error("Thesis clarity: LOW")

st.write("**Top risks:**")
for r in th_panel.risks:
    st.write(f"- {r}")

st.markdown("---")

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

st.markdown("---")

# ---------------- Financing Stability ----------------
cF1, cF2, cF3 = st.columns([1.3, 1.3, 2.4])
with cF1:
    st.metric("Financing Stability", f"{fin.health_level} ({fin.health_score}/100)", help=tooltip_map().get("FIN_HEALTH", ""))
with cF2:
    st.metric("Runway (est.)", f"{fin.runway_months_est:.1f} months" if fin.runway_months_est is not None else "N/A")
with cF3:
    st.write("**Notes:**")
    for n in fin.notes[:4]:
        st.write(f"- {n}")

st.markdown("---")

# ---------------- Ownership / Positioning ----------------
st.subheader("Positioning / Ownership (Context)")
o1, o2, o3, o4 = st.columns(4)

float_label = (
    f"{own.float_shares:,.0f}"
    if getattr(own, "float_shares", None)
    else "Not provided by Yahoo for this ticker"
)
inst_label = (
    f"{(own.inst_held_pct*100):.1f}%"
    if getattr(own, "inst_held_pct", None) is not None
    else "Not provided"
)
insider_label = (
    f"{(own.insider_held_pct*100):.1f}%"
    if getattr(own, "insider_held_pct", None) is not None
    else "Not provided"
)
short_label = (
    f"{(own.short_float_pct*100):.1f}%"
    if getattr(own, "short_float_pct", None) is not None
    else "Not provided"
)

o1.metric("Float", float_label)
o2.metric("% Inst Held", inst_label)
o3.metric("% Insider Held", insider_label)
o4.metric("% Short of Float", short_label)

st.write(f"**Flow hint:** {own.trapped_holder_hint}")
if own.atm_or_shelf_flag:
    st.warning("ATM / Shelf overhang detected (potential supply cap).")
    with st.expander("Evidence"):
        for e in own.atm_or_shelf_evidence:
            st.write(f"- {e}")
else:
    st.success("No ATM/shelf overhang detected (from recent filings heuristics).")

st.markdown("---")

# ---------------- Kid-mode explainers ----------------
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
    with st.expander("Explain these signals (kid-mode)", expanded=False):
        st.markdown(kid_explainer_block(snap))

with st.expander("What would invalidate this?", expanded=False):
    st.markdown(invalidation_box(str(mode)))

st.markdown("---")

# ---------------- Price Movement Chart (restored + upgraded) ----------------
st.subheader("Price & Volume (Trend + Levels + Likely Range)")
st.caption("Green/red background = engine bias. Blue box = likely 7-day range. Lines on the right are labeled so you don’t guess.")

st.plotly_chart(
    make_price_volume_chart(
        hist,
        daily_df=daily,
        sma20=sma20,
        sma50=sma50,
        avwap=avwap_series if avwap_series is not None else None,
        dir_bias=(dirrep.bias if dirrep else ""),
        range_low=range_low,
        range_high=range_high,
    ),
    width="stretch",
)

# Optional AVWAP overlay chart (kept)
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

st.subheader("Levels Table (exact numbers)")

# Pull last values safely
last_close = float(tech.last_close)

sma20_last = None
sma50_last = None
try:
    if hasattr(tech, "sma20") and tech.sma20 is not None and not tech.sma20.dropna().empty:
        sma20_last = float(tech.sma20.dropna().iloc[-1])
    if hasattr(tech, "sma50") and tech.sma50 is not None and not tech.sma50.dropna().empty:
        sma50_last = float(tech.sma50.dropna().iloc[-1])
except Exception:
    pass

avwap_last = None
try:
    if avwap_series is not None and not avwap_series.dropna().empty:
        avwap_last = float(avwap_series.dropna().iloc[-1])
except Exception:
    pass

sr_levels = compute_sr_levels(daily, lookback=60)

df_levels = build_levels_table(
    last_close=last_close,
    sma20_last=sma20_last,
    sma50_last=sma50_last,
    avwap_last=avwap_last,
    sr=sr_levels,
    range_low=range_low,
    range_high=range_high,
)

if df_levels.empty:
    st.info("No levels available to display.")
else:
    st.dataframe(df_levels, width="stretch", height=320)


st.markdown("---")

# ---------------- Tape / Risk Metrics (kept) ----------------
st.subheader("Tape / Risk Metrics")
cR = st.columns([1.1, 1.1, 1.1, 1.1])
cR[0].metric("Last Close", f"{tech.last_close:.2f}")
cR[1].metric("Trend Regime", tech.trend, help=tooltip_map()["TREND"])
cR[2].metric("20D Volatility", fmt_pct(tech.vol_20d))
cR[3].metric("ATR(14)", fmt_float(getattr(tech, "atr_14", None), 4), help=tooltip_map()["ATR"])
st.metric("Rel Volume", fmt_float(tech.rel_volume, 2), help=tooltip_map()["RVOL"])
st.metric("Abnormal Vol", "YES" if tech.abnormal_volume_flag else "NO")
st.write("**Liquidity / Structure**")
st.write(f"- Sweep: {tech.sweep_signal}")
st.write(f"- BOS: {tech.bos_signal}")
st.plotly_chart(make_volatility_chart(daily), width="stretch")

st.markdown("---")

# ---------------- NEW: K-Line Readout (Candles + Entries + Confidence) ----------------
st.subheader("K-Line Readout (Candlestick Trend + Entry / Stop / Target Plan)")

krep = analyze_kline(daily, sma20=sma20, sma50=sma50, allowed_sides=direction_mode)

k1, k2, k3, k4, k5 = st.columns([1.1, 1.1, 1.1, 1.2, 1.0])

with k1:
    st.metric("K-Line Trend", krep.trend)

with k2:
    st.metric("Bias", krep.bias)

with k3:
    st.metric("K-Line Confidence", f"{krep.confidence_score}/100 ({krep.confidence_label})")

with k4:
    if krep.last_close is not None:
        st.metric("Current Close", f"{krep.last_close:.4f}")
    st.write("**Direction Extent (vol-adjusted):**")
    st.caption(krep.direction_extent)

with k5:
    if krep.rsi_last is not None:
        if krep.rsi_last < 30:
            rsi_tag = "Oversold"
        elif krep.rsi_last > 70:
            rsi_tag = "Overbought"
        else:
            rsi_tag = "Neutral"
        st.metric("RSI(14)", f"{krep.rsi_last:.1f}", rsi_tag)
    else:
        st.metric("RSI(14)", "N/A")


# Primary execution plan
if (
    krep.side in ("LONG", "SHORT")
    and krep.primary_entry is not None
    and krep.primary_stop is not None
    and krep.primary_target is not None
):
    e1, e2, e3, e4 = st.columns([1.0, 1.1, 1.1, 1.1])
    with e1:
        st.metric("Side", krep.side)
    with e2:
        st.metric("Primary Entry", f"{krep.primary_entry:.4f}")
    with e3:
        st.metric("Stop Loss", f"{krep.primary_stop:.4f}")
    with e4:
        st.metric("First Target", f"{krep.primary_target:.4f}")

    st.caption(
        "Default K-line plan: use the primary entry as the main zone, cut risk if price breaks the stop level, "
        "and start trimming or taking profit around the first target. The numbers are built from recent candles, "
        "moving averages and nearby swing highs/lows."
    )
    # NEW: tell you if now is an entry or a wait
    if krep.current_vs_entry_note:
        st.caption(krep.current_vs_entry_note)
else:
    st.write("No high-conviction primary execution plan; structure is too mixed or range-bound.")

st.write("**Suggested Entry Zones (from K-Lines)**")
if krep.side in ("LONG", "SHORT"):
    st.write(f"- **Side:** {krep.side}")
else:
    st.write("- **Side:** NONE (structure looks more range-bound)")

entry_rows = []
if krep.primary_entry is not None:
    entry_rows.append(
        {
            "Type": "Primary",
            "Entry": round(krep.primary_entry, 4),
            "Stop Loss": round(krep.primary_stop, 4) if krep.primary_stop is not None else None,
            "First Target": round(krep.primary_target, 4) if krep.primary_target is not None else None,
            "Comment": "Most balanced zone based on K-line trend and moving averages.",
        }
    )
if krep.secondary_entry is not None:
    entry_rows.append(
        {
            "Type": "Secondary",
            "Entry": round(krep.secondary_entry, 4),
            "Stop Loss": None,
            "First Target": None,
            "Comment": "Deeper pullback zone (higher reward, higher risk).",
        }
    )
if krep.aggressive_entry is not None:
    entry_rows.append(
        {
            "Type": "Aggressive",
            "Entry": round(krep.aggressive_entry, 4),
            "Stop Loss": None,
            "First Target": None,
            "Comment": "Breakout/continuation-style entry if momentum is strong.",
        }
    )

if entry_rows:
    st.dataframe(pd.DataFrame(entry_rows), width="stretch", height=220)
else:
    st.info("No clear K-line-based entry zones identified. Market may be too noisy or flat.")

st.write("**How this K-Line confidence is deduced (tape-only)**")
st.caption(krep.explanation)

if krep.reasons:
    with st.expander("Detailed K-Line reasoning (for advanced users)"):
        for r in krep.reasons[:15]:
            st.write(f"- {r}")

if kid_mode:
    with st.expander("Explain K-Line terms (kid-mode)", expanded=False):
        st.markdown(
            "- **Trend**: pattern of candles mostly moving up, down, or sideways.\n"
            "- **Entry Zone**: price area where it usually makes more sense to buy (for longs) or sell (for shorts) "
            "instead of clicking randomly.\n"
            "- **K-Line Confidence**: how clean the candles and moving averages look, from 0 to 100, based only on "
            "price action.\n"
            "- **Hammer / Doji / Engulfing**: special candle shapes that show strong fights between buyers and sellers."
        )
st.markdown("### Long-Only Dip Scout (where to pre-place buy orders)")

if krep.dip_upper is not None and krep.dip_lower is not None:
    d1, d2, d3 = st.columns([1.1, 1.1, 1.2])
    with d1:
        st.metric("Dip band upper (nearest support)", f"{krep.dip_upper:.4f}")
    with d2:
        st.metric("Dip band lower (max pain)", f"{krep.dip_lower:.4f}")
    with d3:
        st.metric("Dip-band confidence", f"{krep.dip_confidence}/100")

    st.caption(
        "This band is designed for **auto-buys on the long side**, even if the main K-line side is SHORT or "
        "AVOID. It does not guarantee a bottom; it just marks where the candles and volatility suggest the "
        "current selloff is most likely to exhaust."
    )
    st.caption(krep.dip_note)
else:
    st.caption(
        "Not enough history or structure to suggest a meaningful dip-buy band for long-only orders."
    )

st.markdown("### 3-Day Tape Scenario Guide (heuristic, tape-based)")

c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.3, 1.0])
with c1:
    st.metric("Risk support breaks (3-day)", f"{krep.prob_support_break_3d}%")
with c2:
    st.metric("Chance of 3-day bounce", f"{krep.prob_bounce_3d}%")
with c3:
    st.metric("Chance to break resistance (3-day)", f"{krep.prob_resistance_break_3d}%")
with c4:
    if krep.rsi_last is not None:
        if krep.rsi_last < 30:
            rsi_tag = "Oversold"
        elif krep.rsi_last > 70:
            rsi_tag = "Overbought"
        else:
            rsi_tag = "Neutral"
        st.metric("RSI(14)", f"{krep.rsi_last:.1f}", rsi_tag)
    else:
        st.metric("RSI(14)", "N/A")

# Colour-coded legend under the metrics
st.markdown(
    f"""
- **Support break risk:** {prob_badge(krep.prob_support_break_3d)}  
- **3-day bounce chance:** {prob_badge(krep.prob_bounce_3d)}  
- **Resistance break chance:** {prob_badge(krep.prob_resistance_break_3d)}
""",
    unsafe_allow_html=True,
)

st.caption(
    "These percentages are **scenario odds inferred from the tape** (trend direction/strength, volatility, moving "
    "averages, RSI, and how price sits vs support/resistance). They are heuristics, not statistically calibrated "
    "probabilities, and should be used together with your own judgement and the catalyst/quant modules."
)

if krep.scenarios_note:
    st.caption(krep.scenarios_note)


st.markdown("---")

# ---------------- Forecast / Posture (Mode-aware, with joint confidence) ----------------
st.subheader("Forecast / Posture (Mode-aware)")

if str(mode).upper() == "PURE_CATALYST":
    ps, pwhy = posture_meter(str(mode), thesis, dil.level, fin.health_level)
    st.metric("Pre-Catalyst Posture", f"{ps}/100", help="Readiness to express the catalyst thesis (not direction).")
    st.caption(pwhy)
    show_forecast = st.toggle("Show 7-day forecast (advanced)", value=False)
else:
    show_forecast = True

# Helper: map quant forecast to a 0–100 score using P(up)
def _quant_conf_score(fc) -> int:
    if fc is None or getattr(fc, "p_up", None) is None:
        return 50
    try:
        return int(round(fc.p_up * 100))
    except Exception:
        return 50

def _label_from_score(score: int) -> str:
    if score >= 70:
        return "HIGH"
    elif score >= 50:
        return "MED"
    return "LOW"

if show_forecast:
    if fc_risk is None or fc_base is None:
        st.warning(f"Forecast unavailable: {forecast_err}")
    else:
        # Quant confidence from risk-adjusted forecast
        quant_conf_score = _quant_conf_score(fc_risk)
        quant_conf_label = _label_from_score(quant_conf_score)

        # Joint confidence: average of quant (P(up)) and K-line confidence
        joint_conf_score = int(round(0.5 * quant_conf_score + 0.5 * krep.confidence_score))
        joint_conf_label = _label_from_score(joint_conf_score)

        # Show forecast confidence (model) and then joint
        conf = fc_risk.confidence
        if conf == "HIGH":
            st.success("Forecast Confidence (model): HIGH")
        elif conf == "MED":
            st.warning("Forecast Confidence (model): MED")
        else:
            st.error("Forecast Confidence (model): LOW")

        jc1, jc2, jc3 = st.columns([1.2, 1.2, 2.6])
        with jc1:
            st.metric("Quant P(Up)", f"{fc_risk.p_up*100:.1f}%")
        with jc2:
            st.metric("Quant Conf Score", f"{quant_conf_score}/100 ({quant_conf_label})")
        with jc3:
            st.metric("Joint Confidence (Quant + K-Line)", f"{joint_conf_score}/100 ({joint_conf_label})")

        st.caption(
            "Joint confidence blends the probabilistic model (P(up) from the EWMA forecast) with the tape-based "
            "K-line confidence. It is higher when both the quant model and the candles point in the same direction."
        )

        # IMPORTANT: best move now includes financing stability + runway
        best_line, best_why = size_recommendation(
            mode=str(mode),
            confidence=str(conf),
            decision=str(sr.decision),
            fin_health_level=str(fin.health_level),
            runway_months_est=fin.runway_months_est,
        )

        action = "BUY" if sr.decision == "GO" else ("AVOID" if sr.decision == "AVOID" else "WAIT")
        st.markdown(f"### {emoji_action_label(action)}")
        st.markdown(best_line)
        st.caption(best_why)
        st.caption(getattr(ns, "explanation", ""))

        if dirrep is not None:
            st.markdown("### Direction Engine Readout")
            d1, d2, d3, d4 = st.columns([1.1, 1.1, 1.1, 2.2])
            d1.metric("Engine", dirrep.engine)
            d2.metric("Regime", dirrep.regime)
            d3.metric("Bias", dirrep.bias)
            d4.metric("Likely Range (7D)", f"{dirrep.range_low:.2f} – {dirrep.range_high:.2f}")
            st.caption(dirrep.summary)

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

# ---------------- Post-event playbook (kept) ----------------
st.subheader("Post-Event Playbook (What to do after the catalyst/news)")
pb = build_post_event_playbook(mode=str(mode), dilution_level=dil.level, fin_health_level=fin.health_level)

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown("### If POSITIVE")
    for x in pb.positive:
        st.write(f"- {x}")
with p2:
    st.markdown("### If NEUTRAL")
    for x in pb.neutral:
        st.write(f"- {x}")
with p3:
    st.markdown("### If NEGATIVE")
    for x in pb.negative:
        st.write(f"- {x}")

st.markdown("---")

# ---------------- Filings + News (kept) ----------------
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
