# ─────────────────────────────────────────
#  TAB: CAPM / Beta Analysis
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, GREEN, RED, MUTED, TEXT, FONT
from dashboard.components import card, stat, stat_row, input_field, run_button, graph, section_label
from models.capm import (
    compute_beta_alpha, expected_return_capm, r_squared,
    sharpe_ratio, treynor_ratio, information_ratio, rolling_beta,
)


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                input_field("Stock Ticker",    "cp-stock",  "AAPL", type="text"),
                input_field("Market Index",    "cp-market", "^GSPC", type="text"),
                input_field("Start Date",      "cp-start",  "2018-01-01", type="text"),
                input_field("End Date",        "cp-end",    "2024-01-01", type="text"),
                input_field("Risk-Free Rate",  "cp-rf",     0.05, step=0.001),
                run_button("cp-btn", "COMPUTE"),
            ], width=3),
            dbc.Col([
                html.Div(id="cp-stats", style={"marginBottom": "10px"}),
                card("SECURITY MARKET LINE", graph("cp-chart",  300)),
                card("ROLLING BETA (12M)",   graph("cp-rolling-beta", 220)),
            ], width=9),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("cp-chart",        "figure"),
        Output("cp-rolling-beta", "figure"),
        Output("cp-stats",        "children"),
        Input("cp-btn",           "n_clicks"),
        State("cp-stock",         "value"),
        State("cp-market",        "value"),
        State("cp-start",         "value"),
        State("cp-end",           "value"),
        State("cp-rf",            "value"),
        prevent_initial_call=False,
    )
    def update(_n, stock, market, start, end, rf):
        stock  = (stock  or "AAPL").upper()
        market = (market or "^GSPC").upper()
        rf     = float(rf or 0.05)

        raw_s = yf.Ticker(stock).history( start=start or "2018-01-01", end=end or "2024-01-01")["Close"]
        raw_m = yf.Ticker(market).history(start=start or "2018-01-01", end=end or "2024-01-01")["Close"]

        s_ret = np.log(raw_s / raw_s.shift(1)).resample("ME").sum().dropna()
        m_ret = np.log(raw_m / raw_m.shift(1)).resample("ME").sum().dropna()
        idx   = s_ret.index.intersection(m_ret.index)
        s_ret, m_ret = s_ret[idx], m_ret[idx]

        beta, alpha = compute_beta_alpha(s_ret.values, m_ret.values)
        r2          = r_squared(s_ret.values, m_ret.values)
        ann_ret     = float(s_ret.mean() * 12)
        ann_vol     = float(s_ret.std()  * np.sqrt(12))
        sr          = sharpe_ratio(ann_ret, rf, ann_vol)
        tr          = treynor_ratio(ann_ret, rf, beta)
        ir          = information_ratio(s_ret.values, m_ret.values)

        x_line = np.linspace(m_ret.min(), m_ret.max(), 200)
        y_line = alpha + beta * x_line

        fig_sml = go.Figure()
        fig_sml.add_trace(go.Scatter(
            x=m_ret, y=s_ret, mode="markers",
            marker=dict(color=ACCENT2, size=5, opacity=0.7), name="Monthly Returns",
        ))
        fig_sml.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=ACCENT, width=2), name="SML Fit",
        ))
        fig_sml.update_layout(**PLOTLY_LAYOUT, height=300,
                              xaxis_title="Market Return", yaxis_title=f"{stock} Return")

        # Rolling Beta
        rb = rolling_beta(s_ret.values, m_ret.values, window=12)
        rb_dates = idx[12 - 1:]
        fig_rb = go.Figure()
        fig_rb.add_trace(go.Scatter(
            x=rb_dates, y=rb, mode="lines",
            line=dict(color=ACCENT3, width=2), name="Rolling β",
        ))
        fig_rb.add_hline(y=1, line_dash="dot", line_color=MUTED, annotation_text="β=1")
        fig_rb.update_layout(**PLOTLY_LAYOUT, height=220,
                             xaxis_title="Date", yaxis_title="Beta")

        kpis = stat_row(
            stat("BETA",         f"{beta:.3f}",       ACCENT3 if beta > 1.2 else GREEN),
            stat("ALPHA",        f"{alpha * 100:.3f}%", GREEN if alpha > 0 else RED),
            stat("R²",           f"{r2:.3f}",          ACCENT),
            stat("SHARPE",       f"{sr:.3f}",          ACCENT2),
            stat("TREYNOR",      f"{tr:.3f}",          MUTED),
            stat("INFO RATIO",   f"{ir:.3f}",          MUTED),
        )
        return fig_sml, fig_rb, kpis
