# ─────────────────────────────────────────
#  TAB: Bond Pricing & Yield Analysis
# ─────────────────────────────────────────

import numpy as np
import plotly.graph_objects as go
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, GREEN, RED, MUTED, TEXT, FONT
from dashboard.components import card, stat, stat_row, input_field, run_button, graph, section_label
from models.bonds import (
    zcb_price, zcb_ytm, zcb_duration,
    coupon_bond_price, macaulay_duration, modified_duration, bond_convexity,
    vasicek_bond_mc,
)


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                section_label("ZERO-COUPON BOND"),
                input_field("Face Value ($)",     "zcb-par",   1000.0, step=100.0),
                input_field("Maturity (years)",   "zcb-mat",   10.0,   step=1.0),
                input_field("YTM / Rate",         "zcb-rate",  0.05,   step=0.005),
                section_label("COUPON BOND"),
                input_field("Face Value ($)",     "cb-par",    1000.0, step=100.0),
                input_field("Coupon Rate",        "cb-coupon", 0.06,   step=0.005),
                input_field("Periods / year",     "cb-freq",   2,      step=1, min=1, max=12),
                input_field("Maturity (years)",   "cb-mat",    10.0,   step=1.0),
                input_field("Market YTM",         "cb-ytm",    0.05,   step=0.005),
                section_label("VASICEK (MC)"),
                input_field("r₀",                 "vas-r0",    0.05,   step=0.005),
                input_field("κ (mean reversion)", "vas-kappa", 0.30,   step=0.05),
                input_field("θ (long-run mean)",  "vas-theta", 0.05,   step=0.005),
                input_field("σ (vol)",            "vas-sigma", 0.02,   step=0.005),
                input_field("Maturity T",         "vas-T",     10.0,   step=1.0),
                run_button("bond-btn", "PRICE BONDS"),
            ], width=3),
            dbc.Col([
                html.Div(id="bond-stats", style={"marginBottom": "10px"}),
                dbc.Row([
                    dbc.Col(card("ZCB PRICE vs MATURITY", graph("zcb-chart", 260)), width=6),
                    dbc.Col(card("VASICEK RATE PATHS",     graph("vas-chart", 260)), width=6),
                ]),
                card("COUPON BOND PRICE vs YTM", graph("cb-chart", 260)),
            ], width=9),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("zcb-chart",  "figure"),
        Output("vas-chart",  "figure"),
        Output("cb-chart",   "figure"),
        Output("bond-stats", "children"),
        Input("bond-btn",    "n_clicks"),
        State("zcb-par",     "value"),
        State("zcb-mat",     "value"),
        State("zcb-rate",    "value"),
        State("cb-par",      "value"),
        State("cb-coupon",   "value"),
        State("cb-freq",     "value"),
        State("cb-mat",      "value"),
        State("cb-ytm",      "value"),
        State("vas-r0",      "value"),
        State("vas-kappa",   "value"),
        State("vas-theta",   "value"),
        State("vas-sigma",   "value"),
        State("vas-T",       "value"),
        prevent_initial_call=False,
    )
    def update(_n, zcb_par, zcb_mat, zcb_rate,
               cb_par, cb_coupon, cb_freq, cb_mat, cb_ytm,
               r0, kappa, theta, sigma, vas_T):
        zcb_par  = float(zcb_par  or 1000)
        zcb_mat  = float(zcb_mat  or 10)
        zcb_rate = float(zcb_rate or 0.05)
        cb_par   = float(cb_par   or 1000)
        cb_coupon= float(cb_coupon or 0.06)
        cb_freq  = int(cb_freq    or 2)
        cb_mat   = float(cb_mat   or 10)
        cb_ytm   = float(cb_ytm   or 0.05)
        r0       = float(r0       or 0.05)
        kappa    = float(kappa    or 0.30)
        theta    = float(theta    or 0.05)
        sigma    = float(sigma    or 0.02)
        vas_T    = float(vas_T    or 10)

        # ── ZCB price vs maturity ──
        mats  = np.linspace(0.5, 30, 120)
        zcb_prices = [zcb_price(zcb_par, m, zcb_rate) for m in mats]
        zcb_ytms   = [zcb_ytm(zcb_prices[i], zcb_par, m) for i, m in enumerate(mats)]
        ytm_spreads= [(y - zcb_rate) * 10_000 for y in zcb_ytms]  # bps vs input rate

        fig_zcb = go.Figure()
        fig_zcb.add_trace(go.Scatter(x=mats, y=zcb_prices, mode="lines",
                                     line=dict(color=ACCENT, width=2), name="ZCB Price"))
        fig_zcb.add_vline(x=zcb_mat, line_dash="dash", line_color=ACCENT3,
                          annotation_text=f"T={zcb_mat}")
        fig_zcb.update_layout(**PLOTLY_LAYOUT, height=260,
                              xaxis_title="Maturity (years)", yaxis_title="Price ($)")

        # ── Vasicek MC ──
        paths_vas, _ = vasicek_bond_mc(r0, kappa, theta, sigma, vas_T,
                                        n_steps=int(vas_T * 52), n_paths=200)
        t_axis = np.linspace(0, vas_T, paths_vas.shape[1])
        fig_vas = go.Figure()
        for i in range(min(100, paths_vas.shape[0])):
            fig_vas.add_trace(go.Scatter(
                x=t_axis, y=paths_vas[i], mode="lines",
                line=dict(color=ACCENT2, width=0.5), opacity=0.3, showlegend=False,
            ))
        fig_vas.add_trace(go.Scatter(
            x=t_axis, y=paths_vas.mean(axis=0), mode="lines",
            line=dict(color=ACCENT, width=2), name="Mean Rate",
        ))
        fig_vas.add_hline(y=theta, line_dash="dot", line_color=MUTED, annotation_text="θ")
        fig_vas.update_layout(**PLOTLY_LAYOUT, height=260,
                              xaxis_title="Time (years)", yaxis_title="Short Rate")

        # ── Coupon bond price vs YTM ──
        ytm_range  = np.linspace(0.001, 0.20, 200)
        cb_prices  = [coupon_bond_price(cb_par, cb_coupon, cb_freq, cb_mat, y) for y in ytm_range]
        current_cb = coupon_bond_price(cb_par, cb_coupon, cb_freq, cb_mat, cb_ytm)
        mac_dur    = macaulay_duration( cb_par, cb_coupon, cb_freq, cb_mat, cb_ytm)
        mod_dur    = modified_duration( cb_par, cb_coupon, cb_freq, cb_mat, cb_ytm)
        convex     = bond_convexity(    cb_par, cb_coupon, cb_freq, cb_mat, cb_ytm)

        fig_cb = go.Figure()
        fig_cb.add_trace(go.Scatter(x=ytm_range * 100, y=cb_prices, mode="lines",
                                    line=dict(color=GREEN, width=2), name="Bond Price"))
        fig_cb.add_scatter(x=[cb_ytm * 100], y=[current_cb], mode="markers",
                           marker=dict(color=ACCENT, size=10, symbol="circle-open-dot"),
                           name="Current YTM")
        fig_cb.update_layout(**PLOTLY_LAYOUT, height=260,
                             xaxis_title="YTM (%)", yaxis_title="Price ($)")

        # KPI row
        zcb_p_now  = zcb_price(zcb_par, zcb_mat, zcb_rate)
        zcb_ytm_now = zcb_ytm(zcb_p_now, zcb_par, zcb_mat)
        zcb_dur    = zcb_duration(zcb_mat)

        kpis = stat_row(
            stat("ZCB PRICE",    f"${zcb_p_now:,.2f}",  ACCENT),
            stat("ZCB YTM",      f"{zcb_ytm_now*100:.2f}%", MUTED),
            stat("ZCB DURATION", f"{zcb_dur:.2f} yrs",  ACCENT3),
            stat("CB PRICE",     f"${current_cb:,.2f}", GREEN),
            stat("MAC DURATION", f"{mac_dur:.2f} yrs",  ACCENT),
            stat("MOD DURATION", f"{mod_dur:.2f}",       ACCENT2),
            stat("CONVEXITY",    f"{convex:.2f}",        MUTED),
        )
        return fig_zcb, fig_vas, fig_cb, kpis
