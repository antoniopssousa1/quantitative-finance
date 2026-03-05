# ─────────────────────────────────────────
#  TAB: Derivatives (Forwards, IRS, CDS)
# ─────────────────────────────────────────

import numpy as np
import plotly.graph_objects as go
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, GREEN, RED, MUTED, TEXT, FONT
from dashboard.components import card, stat, stat_row, input_field, run_button, graph, section_label
from models.derivatives import (
    forward_price, forward_value,
    irs_fixed_rate, irs_value,
    cds_spread,
)


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                section_label("FORWARD CONTRACT"),
                input_field("Spot Price (S₀)",    "fwd-S",   100.0, step=5.0),
                input_field("Risk-Free Rate",     "fwd-r",   0.05,  step=0.005),
                input_field("Storage Cost (u)",   "fwd-u",   0.0,   step=0.005),
                input_field("Dividend Yield (q)", "fwd-q",   0.02,  step=0.005),
                section_label("INTEREST RATE SWAP"),
                input_field("Notional ($)",       "irs-N",   1_000_000, step=100_000),
                input_field("Market Rate (Rm)",   "irs-Rm",  0.06,  step=0.005),
                input_field("Fixed Payments",     "irs-n",   4,     step=1, min=1, max=40),
                input_field("Discount Rate",      "irs-r",   0.05,  step=0.005),
                section_label("CDS"),
                input_field("Hazard Rate (λ)",    "cds-lam", 0.02,  step=0.005),
                input_field("Recovery Rate (R)",  "cds-R",   0.40,  step=0.05),
                input_field("Maturity (years)",   "cds-T",   5.0,   step=1.0),
                input_field("Risk-Free Rate",     "cds-r",   0.05,  step=0.005),
                run_button("deriv-btn", "PRICE"),
            ], width=3),
            dbc.Col([
                html.Div(id="deriv-stats", style={"marginBottom": "10px"}),
                dbc.Row([
                    dbc.Col(card("FORWARD PRICE vs MATURITY", graph("fwd-chart",  260)), width=6),
                    dbc.Col(card("IRS VALUE vs MARKET RATE",  graph("irs-chart",  260)), width=6),
                ]),
                card("CDS SPREAD vs HAZARD RATE", graph("cds-chart", 270)),
            ], width=9),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("fwd-chart",   "figure"),
        Output("irs-chart",   "figure"),
        Output("cds-chart",   "figure"),
        Output("deriv-stats", "children"),
        Input("deriv-btn",    "n_clicks"),
        State("fwd-S",        "value"),
        State("fwd-r",        "value"),
        State("fwd-u",        "value"),
        State("fwd-q",        "value"),
        State("irs-N",        "value"),
        State("irs-Rm",       "value"),
        State("irs-n",        "value"),
        State("irs-r",        "value"),
        State("cds-lam",      "value"),
        State("cds-R",        "value"),
        State("cds-T",        "value"),
        State("cds-r",        "value"),
        prevent_initial_call=False,
    )
    def update(_n, S, fwd_r, u, q, notional, Rm, irs_n, irs_r, lam, R, cds_T, cds_r):
        S       = float(S       or 100)
        fwd_r   = float(fwd_r   or 0.05)
        u       = float(u       or 0.0)
        q       = float(q       or 0.02)
        notional= float(notional or 1_000_000)
        Rm      = float(Rm      or 0.06)
        irs_n   = int(irs_n     or 4)
        irs_r   = float(irs_r   or 0.05)
        lam     = float(lam     or 0.02)
        R       = float(R       or 0.40)
        cds_T   = float(cds_T   or 5.0)
        cds_r   = float(cds_r   or 0.05)

        # ── Forward price vs maturity ──
        T_range = np.linspace(0.1, 5.0, 120)
        fwd_prices = [forward_price(S, fwd_r, t, u, q) for t in T_range]

        fig_fwd = go.Figure()
        fig_fwd.add_trace(go.Scatter(x=T_range, y=fwd_prices, mode="lines",
                                     line=dict(color=ACCENT, width=2), name="Forward Price"))
        fig_fwd.add_hline(y=S, line_dash="dot", line_color=MUTED, annotation_text="Spot")
        fig_fwd.update_layout(**PLOTLY_LAYOUT, height=260,
                              xaxis_title="Maturity (years)", yaxis_title="Forward Price ($)")

        # ── IRS value vs market rate ──
        fixed_K  = irs_fixed_rate(irs_r, irs_n)
        rm_range = np.linspace(0.01, 0.15, 200)
        irs_vals = [irs_value(notional, fixed_K, rm, irs_n, irs_r) for rm in rm_range]

        fig_irs = go.Figure()
        fig_irs.add_trace(go.Scatter(x=rm_range * 100, y=irs_vals, mode="lines",
                                     line=dict(color=ACCENT2, width=2), name="IRS Value (Pay Fixed)"))
        fig_irs.add_hline(y=0, line_dash="dot", line_color=MUTED)
        fig_irs.add_vline(x=fixed_K * 100, line_dash="dash", line_color=ACCENT3,
                          annotation_text=f"Fixed={fixed_K*100:.2f}%")
        fig_irs.update_layout(**PLOTLY_LAYOUT, height=260,
                              xaxis_title="Market Rate (%)", yaxis_title="Swap Value ($)")

        # ── CDS spread vs hazard rate ──
        lam_range  = np.linspace(0.001, 0.15, 200)
        cds_spreads= [cds_spread(l, R, cds_T, cds_r) * 10_000 for l in lam_range]  # bps

        fig_cds = go.Figure()
        fig_cds.add_trace(go.Scatter(x=lam_range * 100, y=cds_spreads, mode="lines",
                                     line=dict(color=RED, width=2), name="CDS Spread (bps)"))
        current_cds = cds_spread(lam, R, cds_T, cds_r) * 10_000
        fig_cds.add_scatter(x=[lam * 100], y=[current_cds], mode="markers",
                            marker=dict(color=ACCENT, size=10, symbol="circle-open-dot"),
                            name="Current λ")
        fig_cds.update_layout(**PLOTLY_LAYOUT, height=270,
                              xaxis_title="Hazard Rate (%)", yaxis_title="CDS Spread (bps)")

        # KPIs
        fwd_now = forward_price(S, fwd_r, 1.0, u, q)
        irs_now = irs_value(notional, fixed_K, Rm, irs_n, irs_r)

        kpis = stat_row(
            stat("FWD PRICE (1Y)", f"${fwd_now:,.2f}",            ACCENT),
            stat("IRS FIXED RATE", f"{fixed_K * 100:.3f}%",       ACCENT2),
            stat("IRS VALUE",      f"${irs_now:,.0f}",            GREEN if irs_now >= 0 else RED),
            stat("CDS SPREAD",     f"{current_cds:.1f} bps",      ACCENT3),
            stat("CDS SPREAD %",   f"{current_cds / 100:.3f}%",   MUTED),
        )
        return fig_fwd, fig_irs, fig_cds, kpis
