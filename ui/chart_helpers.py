from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import plotly.graph_objects as go


@dataclass
class SRLevels:
    support: float | None
    resistance: float | None
    pivot: float | None
    r1: float | None
    s1: float | None


def compute_sr_levels(daily: pd.DataFrame, lookback: int = 60) -> SRLevels:
    if daily is None or daily.empty:
        return SRLevels(None, None, None, None, None)

    d = daily.tail(max(lookback, 5)).copy()
    if not {"High", "Low", "Close"}.issubset(set(d.columns)):
        return SRLevels(None, None, None, None, None)

    support = float(d["Low"].min())
    resistance = float(d["High"].max())

    last = d.iloc[-1]
    H, L, C = float(last["High"]), float(last["Low"]), float(last["Close"])
    pivot = (H + L + C) / 3.0
    r1 = 2 * pivot - L
    s1 = 2 * pivot - H
    return SRLevels(support=support, resistance=resistance, pivot=pivot, r1=r1, s1=s1)


def bias_tint(bias: str) -> str:
    b = (bias or "").upper()
    if "UP" in b:
        return "rgba(0, 200, 0, 0.08)"
    if "DOWN" in b:
        return "rgba(255, 0, 0, 0.08)"
    return "rgba(0,0,0,0.0)"


def _line(fig: go.Figure, y: float, color: str, dash: str, width: int = 1):
    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=y,
        y1=y,
        line=dict(color=color, width=width, dash=dash),
        layer="above",
    )


def _annot(
    fig: go.Figure,
    *,
    y: float,
    text: str,
    color: str,
    x: float,
    yshift: int,
):
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=x,
        y=y,
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        yshift=yshift,
        font=dict(size=12, color=color),
        bgcolor="rgba(255,255,255,0.82)",
        bordercolor="rgba(0,0,0,0.14)",
        borderwidth=1,
    )


def _add_level_item(items: List[Tuple[float, str, str, str, int]], y: float, label: str, color: str, dash: str, width: int = 1):
    # item: (y, label, color, dash, width)
    items.append((float(y), label, color, dash, width))


def add_levels_to_chart(
    fig: go.Figure,
    *,
    sma20: Optional[pd.Series] = None,
    sma50: Optional[pd.Series] = None,
    avwap: Optional[pd.Series] = None,
    sr: Optional[SRLevels] = None,
    range_low: float | None = None,
    range_high: float | None = None,
):
    """
    Draw horizontal lines + right-side labels.
    Labels are auto-staggered to avoid overlapping.
    """

    # 1) Collect all levels first
    items: List[Tuple[float, str, str, str, int]] = []

    if sma20 is not None and not sma20.dropna().empty:
        y = float(sma20.dropna().iloc[-1])
        _add_level_item(items, y, f"SMA20 {y:.2f}", "rgba(80,80,80,1)", "solid", 1)

    if sma50 is not None and not sma50.dropna().empty:
        y = float(sma50.dropna().iloc[-1])
        _add_level_item(items, y, f"SMA50 {y:.2f}", "rgba(120,120,120,1)", "dot", 1)

    if avwap is not None and not avwap.dropna().empty:
        y = float(avwap.dropna().iloc[-1])
        _add_level_item(items, y, f"AVWAP {y:.2f}", "rgba(30,144,255,1)", "dash", 1)

    if sr is not None:
        if sr.resistance is not None:
            _add_level_item(items, float(sr.resistance), f"Resistance {sr.resistance:.2f}", "rgba(220,0,0,1)", "dash", 1)
        if sr.support is not None:
            _add_level_item(items, float(sr.support), f"Support {sr.support:.2f}", "rgba(0,140,0,1)", "dash", 1)
        if sr.pivot is not None:
            _add_level_item(items, float(sr.pivot), f"Pivot {sr.pivot:.2f}", "rgba(160,100,0,1)", "dot", 1)
        if sr.r1 is not None:
            _add_level_item(items, float(sr.r1), f"R1 {sr.r1:.2f}", "rgba(220,0,0,0.75)", "dot", 1)
        if sr.s1 is not None:
            _add_level_item(items, float(sr.s1), f"S1 {sr.s1:.2f}", "rgba(0,140,0,0.75)", "dot", 1)

    if range_low is not None and range_high is not None and range_high > range_low > 0:
        _add_level_item(items, float(range_high), f"7D Range High {range_high:.2f}", "rgba(30,144,255,0.95)", "solid", 1)
        _add_level_item(items, float(range_low), f"7D Range Low {range_low:.2f}", "rgba(30,144,255,0.95)", "solid", 1)

    if not items:
        return

    # 2) Draw lines first
    for y, _, color, dash, width in items:
        _line(fig, y, color=color, dash=dash, width=width)

    # 3) Auto-stagger labels to avoid overlap
    # Sort by y (price level)
    items_sorted = sorted(items, key=lambda t: t[0])

    # We create vertical offsets (yshift in pixels) when levels are too close.
    # “Too close” in axis units is hard; yshift works in pixels, so we stack using a fixed step.
    min_px_gap = 14  # pixels between labels
    yshifts: List[int] = []
    last_y_px_bucket = None

    # Also spread across 3 “columns” on the right to reduce pile-up
    # x positions slightly outside the plot area.
    xs = [1.01, 1.08, 1.15]

    # We assign label rows: if consecutive levels are close, bump yshift
    # This is heuristic but works well visually.
    current_stack = 0
    last_y = None

    for i, (y, label, color, dash, width) in enumerate(items_sorted):
        if last_y is None:
            current_stack = 0
        else:
            # If y levels are very close relative to the visible range, stack more.
            # Use a relative threshold based on y magnitude (stable across penny stocks vs large caps).
            rel = abs(y - last_y) / max(1e-6, abs(y))
            if rel < 0.004:  # ~0.4% separation threshold
                current_stack += 1
            else:
                current_stack = 0

        # Convert stack index into pixel shift
        yshift = int(current_stack * min_px_gap)

        # Alternate shift direction up/down so it doesn't run off one side only
        if current_stack % 2 == 1:
            yshift = -yshift

        yshifts.append(yshift)
        last_y = y

    # 4) Place annotations with x column staggering too
    for i, (y, label, color, dash, width) in enumerate(items_sorted):
        col = i % len(xs)
        _annot(
            fig,
            y=y,
            text=label,
            color=color,
            x=xs[col],
            yshift=yshifts[i],
        )
