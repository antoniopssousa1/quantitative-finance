# ─────────────────────────────────────────
#  TAB: Markowitz Portfolio Optimisation
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, GREEN, RED, MUTED, BORDER, TEXT, FONT
from dashboard.components import card, stat, stat_row, input_field, run_button, graph, section_label
from models.markowitz import (
    generate_random_portfolios, max_sharpe_portfolio,
    min_volatility_portfolio, efficient_frontier_curve, portfolio_performance,
)


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                input_field("Stocks (comma-separated)", "mk-stocks", "AAPL,MSFT,TSLA,GE,WMT", type="text"),
                input_field("Start Date",  "mk-start", "2018-01-01", type="text"),
                input_field("End Date",    "mk-end",   "2024-01-01", type="text"),
                input_field("Risk-Free Rate", "mk-rf", 0.05, step=0.001),
                input_field("Simulated Portfolios", "mk-n", 2000, step=500, min=500, max=10000),
                run_button("mk-btn", "OPTIMISE"),
                html.Div(id="mk-weights", style={"marginTop": "12px"}),
            ], width=3),
            dbc.Col([
                html.Div(id="mk-stats", style={"marginBottom": "10px"}),
                card("EFFICIENT FRONTIER", graph("mk-frontier", 480)),
            ], width=9),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("mk-frontier", "figure"),
        Output("mk-stats",    "children"),
        Output("mk-weights",  "children"),
        Input("mk-btn",       "n_clicks"),
        State("mk-stocks",    "value"),
        State("mk-start",     "value"),
        State("mk-end",       "value"),
        State("mk-rf",        "value"),
        State("mk-n",         "value"),
        prevent_initial_call=False,
    )
    def update(_n, stocks_str, start, end, rf, n_port):
        stocks  = [s.strip() for s in (stocks_str or "AAPL,MSFT,TSLA,GE,WMT").split(",")]
        rf      = float(rf or 0.05)
        n_port  = int(n_port or 2000)

        data = {s: yf.Ticker(s).history(start=start or "2018-01-01",
                                         end=end   or "2024-01-01")["Close"]
                for s in stocks}
        df      = pd.DataFrame(data).dropna()
        returns = np.log(df / df.shift(1)).dropna()
        mu      = returns.mean()
        cov     = returns.cov()

        rets, vols, sharpes, weights = generate_random_portfolios(mu, cov, n_port)

        # Optimal portfolios
        opt_sharpe = max_sharpe_portfolio(mu, cov, rf)
        opt_minvol = min_volatility_portfolio(mu, cov)
        ef_vols, ef_rets = efficient_frontier_curve(mu, cov, 60)

        opt_r, opt_v = portfolio_performance(opt_sharpe.x, mu, cov)
        min_r, min_v = portfolio_performance(opt_minvol.x, mu, cov)

        fig = go.Figure()
        # Random portfolios coloured by Sharpe
        fig.add_trace(go.Scatter(
            x=vols, y=rets, mode="markers",
            marker=dict(color=sharpes, colorscale="Viridis", size=4, opacity=0.5,
                        colorbar=dict(title="Sharpe", thickness=12,
                                      tickfont=dict(color=TEXT, family=FONT))),
            name="Portfolios",
        ))
        # Efficient frontier curve
        if len(ef_vols):
            fig.add_trace(go.Scatter(
                x=ef_vols, y=ef_rets, mode="lines",
                line=dict(color=ACCENT, width=2), name="Efficient Frontier",
            ))
        # Max-Sharpe star
        fig.add_trace(go.Scatter(
            x=[opt_v], y=[opt_r], mode="markers",
            marker=dict(symbol="star", size=18, color=ACCENT,
                        line=dict(color="white", width=1)),
            name="Max Sharpe",
        ))
        # Min-Vol diamond
        fig.add_trace(go.Scatter(
            x=[min_v], y=[min_r], mode="markers",
            marker=dict(symbol="diamond", size=14, color=ACCENT3,
                        line=dict(color="white", width=1)),
            name="Min Volatility",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=480,
                          xaxis_title="Volatility (Risk)", yaxis_title="Expected Return")

        kpis = stat_row(
            stat("MAX SHARPE RETURN",   f"{opt_r * 100:.2f}%",           GREEN),
            stat("MAX SHARPE VOL",      f"{opt_v * 100:.2f}%",           ACCENT3),
            stat("SHARPE RATIO",        f"{(opt_r - rf) / opt_v:.3f}",   ACCENT),
            stat("MIN VOL RETURN",      f"{min_r * 100:.2f}%",           ACCENT2),
            stat("MIN VOLATILITY",      f"{min_v * 100:.2f}%",           MUTED),
        )

        weight_bars = html.Div([
            section_label("OPTIMAL WEIGHTS"),
            *[html.Div([
                html.Span(s, style={"color": MUTED, "fontSize": "10px", "fontFamily": FONT,
                                    "width": "52px", "display": "inline-block"}),
                html.Div(style={"display": "inline-block", "background": ACCENT,
                                "width": f"{w * 100:.0f}%", "height": "7px", "borderRadius": "2px",
                                "marginLeft": "4px", "verticalAlign": "middle", "maxWidth": "120px"}),
                html.Span(f" {w * 100:.1f}%", style={"color": TEXT, "fontSize": "10px", "fontFamily": FONT}),
            ], style={"marginBottom": "4px"}) for s, w in zip(stocks, opt_sharpe.x)],
        ])
        return fig, kpis, weight_bars
