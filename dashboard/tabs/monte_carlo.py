# ─────────────────────────────────────────
#  TAB: Monte Carlo Simulations
# ─────────────────────────────────────────

import numpy as np
import plotly.graph_objects as go
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, GREEN, RED, MUTED, TEXT, FONT
from dashboard.components import card, stat, stat_row, input_field, run_button, graph, section_label
from models.monte_carlo import stock_price_mc, bs_mc_price


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                section_label("GBM PARAMETERS"),
                input_field("Initial Price (S₀)", "mc-S0",   100.0, step=10.0),
                input_field("Drift (μ, annual)",  "mc-mu",   0.10,  step=0.01),
                input_field("Volatility (σ)",     "mc-sig",  0.20,  step=0.01),
                input_field("Time Steps (N)",     "mc-N",    252,   step=1, min=10),
                input_field("Paths",              "mc-paths", 200,  step=50, min=10, max=2000),
                section_label("BS CALL (MC CHECK)"),
                input_field("Strike (K)",         "mc-K",    100.0, step=1.0),
                input_field("Risk-Free (r)",      "mc-rf",   0.05,  step=0.005),
                input_field("Expiry T (years)",   "mc-T",    1.0,   step=0.25),
                run_button("mc-btn", "SIMULATE"),
            ], width=3),
            dbc.Col([
                html.Div(id="mc-stats", style={"marginBottom": "10px"}),
                card("SIMULATED PRICE PATHS",    graph("mc-chart", 320)),
                card("TERMINAL PRICE DISTRIBUTION", graph("mc-dist", 220)),
            ], width=9),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("mc-chart", "figure"),
        Output("mc-dist",  "figure"),
        Output("mc-stats", "children"),
        Input("mc-btn",    "n_clicks"),
        State("mc-S0",     "value"),
        State("mc-mu",     "value"),
        State("mc-sig",    "value"),
        State("mc-N",      "value"),
        State("mc-paths",  "value"),
        State("mc-K",      "value"),
        State("mc-rf",     "value"),
        State("mc-T",      "value"),
        prevent_initial_call=False,
    )
    def update(_n, S0, mu, sig, N, paths, K, rf, T):
        S0    = float(S0    or 100)
        mu    = float(mu    or 0.10)
        sig   = float(sig   or 0.20)
        N     = int(N       or 252)
        paths = int(paths   or 200)
        K     = float(K     or 100)
        rf    = float(rf    or 0.05)
        T     = float(T     or 1.0)

        # Generate GBM paths — shape (N+1, paths)
        sims     = stock_price_mc(S0, mu, sig, N, paths)
        terminal = sims[-1, :]   # terminal prices row

        # Price charts
        fig_paths = go.Figure()
        t_axis = np.linspace(0, T, sims.shape[0])
        for i in range(min(paths, 150)):  # cap at 150 for perf
            fig_paths.add_trace(go.Scatter(
                x=t_axis, y=sims[:, i], mode="lines",
                line=dict(color=ACCENT2, width=0.6), opacity=0.3,
                showlegend=False,
            ))
        # Mean path
        fig_paths.add_trace(go.Scatter(
            x=t_axis, y=sims.mean(axis=1), mode="lines",
            line=dict(color=ACCENT, width=2), name="Mean Path",
        ))
        fig_paths.add_hline(y=S0, line_dash="dot", line_color=MUTED)
        fig_paths.update_layout(**PLOTLY_LAYOUT, height=320,
                                xaxis_title="Time (years)", yaxis_title="Price")

        # Terminal distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=terminal, nbinsx=60,
            marker_color=ACCENT, opacity=0.8, name="Terminal Price",
        ))
        fig_dist.add_vline(x=float(np.mean(terminal)), line_dash="dash", line_color=GREEN, annotation_text="Mean")
        fig_dist.add_vline(x=S0, line_dash="dot",  line_color=MUTED, annotation_text="S₀")
        fig_dist.update_layout(**PLOTLY_LAYOUT, height=220,
                               xaxis_title="Terminal Price", yaxis_title="Frequency")

        # BS MC call price
        mc_call, mc_put = bs_mc_price(S0, K, T, rf, sig, 100_000)

        kpis = stat_row(
            stat("E[S_T]",      f"${np.mean(terminal):.2f}",       ACCENT),
            stat("STD[S_T]",    f"${np.std(terminal):.2f}",        ACCENT3),
            stat("P(S_T > K)",  f"{(terminal > K).mean() * 100:.1f}%", GREEN),
            stat("MC CALL",     f"${mc_call:.4f}",                 ACCENT2),
            stat("MC PUT",      f"${mc_put:.4f}",                  RED),
        )
        return fig_paths, fig_dist, kpis
