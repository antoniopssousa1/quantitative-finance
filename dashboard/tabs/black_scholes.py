# ─────────────────────────────────────────
#  TAB: Black-Scholes Options
# ─────────────────────────────────────────

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, GREEN, RED, MUTED, TEXT, FONT
from dashboard.components import card, stat, stat_row, input_field, run_button, graph, section_label
from models.black_scholes import call_price, put_price, greeks, implied_volatility, mc_option_price


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                section_label("OPTION PARAMETERS"),
                input_field("Spot Price (S)",     "bs-S",   150.0, step=1.0),
                input_field("Strike Price (K)",   "bs-K",   150.0, step=1.0),
                input_field("Time to Expiry (T, years)", "bs-T", 1.0, step=0.05),
                input_field("Risk-Free Rate (r)", "bs-rf",  0.05,  step=0.005),
                input_field("Volatility (σ)",     "bs-sig", 0.20,  step=0.01),
                section_label("MC CHECK"),
                input_field("MC Simulations",     "bs-mc-n", 50000, step=10000),
                run_button("bs-btn", "PRICE"),
            ], width=3),
            dbc.Col([
                html.Div(id="bs-results", style={"marginBottom": "10px"}),
                card("PRICE vs SPOT",   graph("bs-spot-chart",   280)),
                card("GREEKS vs SPOT",  graph("bs-greeks-chart", 280)),
            ], width=9),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("bs-spot-chart",   "figure"),
        Output("bs-greeks-chart", "figure"),
        Output("bs-results",      "children"),
        Input("bs-btn",           "n_clicks"),
        State("bs-S",             "value"),
        State("bs-K",             "value"),
        State("bs-T",             "value"),
        State("bs-rf",            "value"),
        State("bs-sig",           "value"),
        State("bs-mc-n",          "value"),
        prevent_initial_call=False,
    )
    def update(_n, S, K, T, r, sig, mc_n):
        S, K, T = float(S or 150), float(K or 150), float(T or 1)
        r, sig  = float(r or 0.05), float(sig or 0.20)
        mc_n    = int(mc_n or 50000)

        # Current prices & greeks
        C  = call_price(S, K, T, r, sig)
        P  = put_price( S, K, T, r, sig)
        G  = greeks(S, K, T, r, sig)
        mc_c, mc_p = mc_option_price(S, K, T, r, sig, mc_n)

        # Sweep spot prices
        spots   = np.linspace(S * 0.50, S * 1.50, 200)
        calls   = [call_price(s, K, T, r, sig) for s in spots]
        puts    = [put_price( s, K, T, r, sig) for s in spots]
        intrin  = np.maximum(spots - K, 0)

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=spots, y=calls,  mode="lines", name="Call", line=dict(color=GREEN,  width=2)))
        fig_price.add_trace(go.Scatter(x=spots, y=puts,   mode="lines", name="Put",  line=dict(color=RED,    width=2)))
        fig_price.add_trace(go.Scatter(x=spots, y=intrin, mode="lines", name="Intrinsic", line=dict(color=MUTED, width=1, dash="dot")))
        fig_price.add_vline(x=S, line_dash="dash", line_color=ACCENT, annotation_text="S")
        fig_price.add_vline(x=K, line_dash="dash", line_color=ACCENT3, annotation_text="K")
        fig_price.update_layout(**PLOTLY_LAYOUT, height=280,
                                xaxis_title="Spot Price", yaxis_title="Option Price")

        # Greeks sweep
        d_calls  = [greeks(s, K, T, r, sig)["call_delta"] for s in spots]
        g_vals   = [greeks(s, K, T, r, sig)["gamma"]       for s in spots]
        t_calls  = [greeks(s, K, T, r, sig)["call_theta"]  for s in spots]
        v_vals   = [greeks(s, K, T, r, sig)["vega"]        for s in spots]

        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=spots, y=d_calls, mode="lines", name="Δ Call", line=dict(color=GREEN,  width=2)))
        fig_g.add_trace(go.Scatter(x=spots, y=g_vals,  mode="lines", name="Γ",      line=dict(color=ACCENT, width=2)))
        fig_g.add_trace(go.Scatter(x=spots, y=t_calls, mode="lines", name="Θ Call", line=dict(color=RED,    width=2)))
        fig_g.add_trace(go.Scatter(x=spots, y=v_vals,  mode="lines", name="ν",      line=dict(color=ACCENT3, width=2)))
        fig_g.add_vline(x=S, line_dash="dash", line_color=ACCENT)
        fig_g.update_layout(**PLOTLY_LAYOUT, height=280,
                            xaxis_title="Spot Price", yaxis_title="Greek Value")

        kpis = stat_row(
            stat("BS CALL",    f"${C:.4f}",            GREEN),
            stat("BS PUT",     f"${P:.4f}",            RED),
            stat("MC CALL",    f"${mc_c:.4f}",         ACCENT),
            stat("MC PUT",     f"${mc_p:.4f}",         ACCENT2),
            stat("Δ CALL",     f"{G['call_delta']:.4f}", MUTED),
            stat("GAMMA",      f"{G['gamma']:.5f}",      MUTED),
            stat("VEGA",       f"{G['vega']:.4f}",       MUTED),
            stat("Θ CALL",     f"{G['call_theta']:.4f}", MUTED),
        )
        return fig_price, fig_g, kpis
