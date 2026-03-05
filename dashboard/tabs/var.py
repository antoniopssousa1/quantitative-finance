# ─────────────────────────────────────────
#  TAB: Value at Risk & CVaR
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, GREEN, RED, MUTED, TEXT, FONT
from dashboard.components import card, stat, stat_row, input_field, run_button, graph, section_label
from models.var import (
    var_parametric, var_historical, var_monte_carlo,
    cvar_parametric, cvar_monte_carlo,
)


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                section_label("PORTFOLIO"),
                input_field("Ticker",             "var-ticker", "SPY",  type="text"),
                input_field("Start Date",         "var-start",  "2018-01-01", type="text"),
                input_field("End Date",           "var-end",    "2024-01-01", type="text"),
                input_field("Portfolio Value ($)", "var-S",     1_000_000, step=100_000),
                input_field("Confidence Level",   "var-c",     0.95, step=0.01, min=0.5, max=0.9999),
                input_field("Horizon (days)",     "var-n",     1, step=1, min=1, max=30),
                run_button("var-btn", "COMPUTE VaR"),
            ], width=3),
            dbc.Col([
                html.Div(id="var-stats", style={"marginBottom": "10px"}),
                card("P&L DISTRIBUTION",            graph("var-chart",   340)),
                card("ROLLING VaR (PARAMETRIC)",     graph("var-rolling", 220)),
            ], width=9),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("var-chart",   "figure"),
        Output("var-rolling", "figure"),
        Output("var-stats",   "children"),
        Input("var-btn",      "n_clicks"),
        State("var-ticker",   "value"),
        State("var-start",    "value"),
        State("var-end",      "value"),
        State("var-S",        "value"),
        State("var-c",        "value"),
        State("var-n",        "value"),
        prevent_initial_call=False,
    )
    def update(_n, ticker, start, end, S_val, conf, horizon):
        ticker  = (ticker or "SPY").upper()
        conf    = float(conf    or 0.95)
        horizon = int(horizon   or 1)
        S_val   = float(S_val   or 1_000_000)

        raw  = yf.Ticker(ticker).history(start=start or "2018-01-01",
                                          end=end   or "2024-01-01")["Close"]
        rets = np.log(raw / raw.shift(1)).dropna().values
        dates = raw.index[1:]

        # Compute VaR / CVaR
        mu_d, sig_d = rets.mean(), rets.std()
        var_p  = var_parametric( mu_d, sig_d, conf, horizon)
        var_h  = var_historical( rets, conf, horizon)
        var_mc = var_monte_carlo(mu_d, sig_d, conf, horizon, 100_000)
        cvar_p = cvar_parametric(mu_d, sig_d, conf, horizon)
        cvar_m = cvar_monte_carlo(mu_d, sig_d, conf, horizon, 100_000)

        # P&L distribution
        pnl = rets * S_val
        var_p_dollar  = var_p  * S_val
        cvar_p_dollar = cvar_p * S_val

        fig = go.Figure()
        # Histogram — green / red split
        fig.add_trace(go.Histogram(
            x=pnl[pnl >= -var_p_dollar], nbinsx=80,
            marker_color=GREEN, opacity=0.7, name="Profit / Neutral",
        ))
        fig.add_trace(go.Histogram(
            x=pnl[pnl < -var_p_dollar], nbinsx=30,
            marker_color=RED, opacity=0.9, name="Tail Loss",
        ))
        fig.add_vline(x=-var_p_dollar,  line_dash="dash", line_color=ACCENT3,
                      annotation_text=f"VaR {conf*100:.0f}%", annotation_font_color=ACCENT3)
        fig.add_vline(x=-cvar_p_dollar, line_dash="dot",  line_color=RED,
                      annotation_text="CVaR", annotation_font_color=RED)
        fig.update_layout(**PLOTLY_LAYOUT, height=340, barmode="overlay",
                          xaxis_title="Daily P&L ($)", yaxis_title="Frequency")

        # Rolling parametric VaR (30-day window)
        window = 30
        r_vars = [
            var_parametric(rets[i - window: i].mean(),
                           rets[i - window: i].std(), conf, 1)
            for i in range(window, len(rets))
        ]
        roll_dates = dates[window:]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=roll_dates, y=np.array(r_vars) * 100,
            mode="lines", line=dict(color=ACCENT3, width=1.5), name="Rolling VaR%",
        ))
        fig_r.update_layout(**PLOTLY_LAYOUT, height=220,
                            xaxis_title="Date", yaxis_title="VaR (%)")

        kpis = stat_row(
            stat("PARAM VaR",   f"${var_p_dollar:,.0f}",            RED),
            stat("HIST VaR",    f"${var_h * S_val:,.0f}",           ACCENT3),
            stat("MC VaR",      f"${var_mc * S_val:,.0f}",          ACCENT),
            stat("CVaR (ES)",   f"${cvar_p_dollar:,.0f}",           RED),
            stat("MC CVaR",     f"${cvar_m * S_val:,.0f}",          ACCENT2),
            stat("CONFIDENCE",  f"{conf * 100:.1f}%",               MUTED),
        )
        return fig, fig_r, kpis
