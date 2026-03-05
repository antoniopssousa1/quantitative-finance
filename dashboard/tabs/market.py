# ─────────────────────────────────────────
#  TAB: Market Overview
# ─────────────────────────────────────────

import datetime
import numpy as np
import yfinance as yf
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, RED, GREEN, MUTED, BORDER
from dashboard.components import card, stat, stat_row, input_field, run_button, graph
import dash_bootstrap_components as dbc


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                input_field("Ticker", "mkt-ticker", "AAPL", type="text"),
                input_field("Start",  "mkt-start",  "2020-01-01", type="text"),
                input_field("End",    "mkt-end",    str(datetime.date.today()), type="text"),
                run_button("mkt-btn", "LOAD"),
            ], width=2),
            dbc.Col([
                html.Div(id="mkt-stats", style={"marginBottom": "10px"}),
                card("CANDLESTICK + VOLUME", graph("mkt-candle", 420)),
            ], width=10),
        ]),
        dbc.Row([
            dbc.Col(card("LOG RETURNS DISTRIBUTION",   graph("mkt-dist", 280)), width=6),
            dbc.Col(card("ROLLING VOLATILITY (30d)",   graph("mkt-vol",  280)), width=6),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("mkt-candle", "figure"),
        Output("mkt-dist",   "figure"),
        Output("mkt-vol",    "figure"),
        Output("mkt-stats",  "children"),
        Input("mkt-btn",     "n_clicks"),
        State("mkt-ticker",  "value"),
        State("mkt-start",   "value"),
        State("mkt-end",     "value"),
        prevent_initial_call=False,
    )
    def update(_n, ticker, start, end):
        ticker = ticker or "AAPL"
        df = yf.download(ticker, start or "2020-01-01", end or str(datetime.date.today()), progress=False)
        if df.empty:
            empty = go.Figure()
            return empty, empty, empty, []

        close   = df["Close"].squeeze()
        vol_col = df["Volume"].squeeze()
        returns = np.log(close / close.shift(1)).dropna()
        vol30   = returns.rolling(30).std() * np.sqrt(252)
        pct_chg = close.pct_change()
        bar_colors = [GREEN if v >= 0 else RED for v in pct_chg.fillna(0)]

        # Candlestick + Volume
        fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.75, 0.25], vertical_spacing=0.02)
        fig_c.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"].squeeze(), high=df["High"].squeeze(),
            low=df["Low"].squeeze(),   close=close,
            increasing_line_color=GREEN, decreasing_line_color=RED, name=ticker,
        ), row=1, col=1)
        fig_c.add_trace(go.Bar(x=df.index, y=vol_col, marker_color=bar_colors,
                               name="Volume", opacity=0.7), row=2, col=1)
        fig_c.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
        fig_c.update_xaxes(rangeslider_visible=False)

        # Returns distribution
        mu_r, sigma_r = returns.mean(), returns.std()
        x_r = np.linspace(mu_r - 4 * sigma_r, mu_r + 4 * sigma_r, 200)
        fig_d = go.Figure([
            go.Histogram(x=returns, nbinsx=100, histnorm="probability density",
                         marker_color=ACCENT2, opacity=0.7, name="Returns"),
            go.Scatter(x=x_r, y=norm.pdf(x_r, mu_r, sigma_r),
                       line=dict(color=ACCENT, width=2), name="Normal fit"),
        ])
        fig_d.update_layout(**PLOTLY_LAYOUT, height=280)

        # Rolling volatility
        fig_v = go.Figure([
            go.Scatter(x=vol30.index, y=vol30, line=dict(color=ACCENT3, width=1.5),
                       fill="tozeroy", fillcolor="rgba(245,158,11,0.1)", name="Ann. Vol"),
        ])
        fig_v.update_layout(**PLOTLY_LAYOUT, height=280)

        last  = float(close.iloc[-1])
        chg   = float(pct_chg.iloc[-1]) * 100
        color = GREEN if chg >= 0 else RED
        kpis  = stat_row(
            stat("LAST",      f"${last:.2f}",  color),
            stat("CHG %",     f"{chg:+.2f}%",  color),
            stat("52W HIGH",  f"${float(close.rolling(252).max().iloc[-1]):.2f}", ACCENT),
            stat("52W LOW",   f"${float(close.rolling(252).min().iloc[-1]):.2f}", RED),
            stat("ANN. VOL",  f"{float(vol30.iloc[-1]) * 100:.1f}%", ACCENT3),
            stat("MEAN RET",  f"{mu_r * 252 * 100:.1f}%/yr", ACCENT2),
        )
        return fig_c, fig_d, fig_v, kpis
