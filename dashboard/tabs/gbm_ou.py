# ─────────────────────────────────────────
#  TAB: Stochastic Processes (GBM / OU / Heston)
# ─────────────────────────────────────────

import numpy as np
import plotly.graph_objects as go
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import PLOTLY_LAYOUT, ACCENT, ACCENT2, ACCENT3, GREEN, RED, MUTED, TEXT, FONT
from dashboard.components import card, stat, stat_row, input_field, run_button, graph, section_label
from models.stochastic import wiener_process, gbm, ornstein_uhlenbeck, vasicek, heston


def layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                section_label("GBM"),
                input_field("S₀",            "gbm-S0",    100.0, step=10.0),
                input_field("μ (drift)",     "gbm-mu",    0.10,  step=0.01),
                input_field("σ (vol)",       "gbm-sig",   0.20,  step=0.01),
                input_field("T (years)",     "gbm-T",     1.0,   step=0.25),
                input_field("Paths",         "gbm-paths", 100,   step=50, min=10, max=500),
                section_label("OU PROCESS"),
                input_field("r₀",            "ou-r0",     0.05,  step=0.01),
                input_field("κ (reversion)", "ou-kappa",  0.50,  step=0.05),
                input_field("θ (long mean)", "ou-theta",  0.05,  step=0.005),
                input_field("σ",             "ou-sigma",  0.02,  step=0.005),
                input_field("T",             "ou-T",      5.0,   step=0.5),
                section_label("HESTON (STOCH VOL)"),
                input_field("ρ (corr)",      "hst-rho",   -0.70, step=0.05, min=-1.0, max=1.0),
                input_field("v₀ (init var)", "hst-v0",    0.04,  step=0.005),
                input_field("κ_v",           "hst-kv",    2.0,   step=0.1),
                input_field("θ_v",           "hst-tv",    0.04,  step=0.005),
                input_field("σ_v",           "hst-sv",    0.30,  step=0.05),
                run_button("gbm-btn", "SIMULATE"),
            ], width=3),
            dbc.Col([
                html.Div(id="gbm-stats", style={"marginBottom": "10px"}),
                dbc.Row([
                    dbc.Col(card("GBM PATHS",        graph("gbm-chart",  260)), width=6),
                    dbc.Col(card("OU / VASICEK",     graph("ou-chart",   260)), width=6),
                ]),
                card("HESTON STOCHASTIC VOLATILITY", graph("heston-chart", 280)),
            ], width=9),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("gbm-chart",    "figure"),
        Output("ou-chart",     "figure"),
        Output("heston-chart", "figure"),
        Output("gbm-stats",    "children"),
        Input("gbm-btn",       "n_clicks"),
        State("gbm-S0",        "value"),
        State("gbm-mu",        "value"),
        State("gbm-sig",       "value"),
        State("gbm-T",         "value"),
        State("gbm-paths",     "value"),
        State("ou-r0",         "value"),
        State("ou-kappa",      "value"),
        State("ou-theta",      "value"),
        State("ou-sigma",      "value"),
        State("ou-T",          "value"),
        State("hst-rho",       "value"),
        State("hst-v0",        "value"),
        State("hst-kv",        "value"),
        State("hst-tv",        "value"),
        State("hst-sv",        "value"),
        prevent_initial_call=False,
    )
    def update(_n, S0, mu, sig, T_gbm, paths,
               r0, kappa, theta, sigma, T_ou,
               rho, v0, kv, tv, sv):
        S0     = float(S0    or 100);  mu    = float(mu    or 0.10)
        sig    = float(sig   or 0.20); T_gbm = float(T_gbm or 1.0)
        paths  = int(paths   or 100)
        r0     = float(r0    or 0.05); kappa = float(kappa or 0.50)
        theta  = float(theta or 0.05); sigma = float(sigma or 0.02)
        T_ou   = float(T_ou  or 5.0)
        rho    = float(rho   or -0.70); v0   = float(v0   or 0.04)
        kv     = float(kv    or 2.0);   tv   = float(tv   or 0.04)
        sv     = float(sv    or 0.30)

        N_gbm = int(T_gbm * 252)
        N_ou  = int(T_ou  * 252)

        # ── GBM — shape (N_gbm, paths) ──
        gbm_paths = gbm(S0, mu, sig, N_gbm, paths)
        t_gbm     = np.linspace(0, T_gbm, N_gbm)
        fig_gbm   = go.Figure()
        for i in range(min(paths, 120)):
            fig_gbm.add_trace(go.Scatter(
                x=t_gbm, y=gbm_paths[:, i], mode="lines",
                line=dict(color=ACCENT2, width=0.5), opacity=0.25, showlegend=False,
            ))
        fig_gbm.add_trace(go.Scatter(
            x=t_gbm, y=gbm_paths.mean(axis=1), mode="lines",
            line=dict(color=ACCENT, width=2), name="Mean",
        ))
        fig_gbm.add_hline(y=S0, line_dash="dot", line_color=MUTED)
        fig_gbm.update_layout(**PLOTLY_LAYOUT, height=260,
                              xaxis_title="Time (years)", yaxis_title="Price")

        # ── OU — shape (N_ou+1, paths) ──
        ou_paths = ornstein_uhlenbeck(r0, kappa, theta, sigma, N_ou, paths)
        t_ou     = np.linspace(0, T_ou, ou_paths.shape[0])
        fig_ou   = go.Figure()
        for i in range(min(paths, 120)):
            fig_ou.add_trace(go.Scatter(
                x=t_ou, y=ou_paths[:, i], mode="lines",
                line=dict(color=ACCENT3, width=0.5), opacity=0.25, showlegend=False,
            ))
        fig_ou.add_trace(go.Scatter(
            x=t_ou, y=ou_paths.mean(axis=1), mode="lines",
            line=dict(color=GREEN, width=2), name="Mean",
        ))
        fig_ou.add_hline(y=theta, line_dash="dot", line_color=MUTED, annotation_text="θ")
        fig_ou.update_layout(**PLOTLY_LAYOUT, height=260,
                             xaxis_title="Time (years)", yaxis_title="Rate")

        # ── Heston — shape (N_hst+1, paths) ──
        N_hst = int(T_gbm * 252)
        try:
            hst_S, hst_v = heston(S0, mu, v0, kv, tv, sv, rho, N_hst, min(paths, 100))
        except Exception:
            hst_S = np.vstack([np.full(min(paths, 100), S0), gbm_paths[:N_hst, :min(paths, 100)]])
            hst_v = np.full_like(hst_S, sig ** 2)

        t_hst = np.linspace(0, T_gbm, hst_S.shape[0])
        fig_hst = go.Figure()
        # Price (left axis)
        for i in range(min(hst_S.shape[1], 60)):
            fig_hst.add_trace(go.Scatter(
                x=t_hst, y=hst_S[:, i], mode="lines",
                line=dict(color=ACCENT2, width=0.4), opacity=0.20, showlegend=False,
            ))
        fig_hst.add_trace(go.Scatter(
            x=t_hst, y=hst_S.mean(axis=1), mode="lines",
            line=dict(color=ACCENT, width=2), name="Mean Price",
        ))
        # Stochastic vol on secondary axis
        mean_vol = np.sqrt(np.maximum(hst_v.mean(axis=1), 0)) * 100
        fig_hst.add_trace(go.Scatter(
            x=t_hst, y=mean_vol, mode="lines",
            line=dict(color=RED, width=1.5, dash="dash"), name="Mean σ %",
            yaxis="y2",
        ))
        fig_hst.update_layout(
            **PLOTLY_LAYOUT, height=280,
            xaxis_title="Time (years)",
            yaxis_title="Price",
            yaxis2=dict(title="Volatility (%)", overlaying="y", side="right",
                        color=RED, gridcolor="rgba(0,0,0,0)"),
        )

        # KPIs
        terminal_gbm = gbm_paths[-1, :]
        kpis = stat_row(
            stat("E[S_T] GBM",   f"${np.mean(terminal_gbm):.2f}", ACCENT),
            stat("STD[S_T]",     f"${np.std(terminal_gbm):.2f}",  ACCENT3),
            stat("OU LONG MEAN", f"{theta:.4f}",                   GREEN),
            stat("OU FINAL",     f"{ou_paths[-1].mean():.4f}",     ACCENT2),
            stat("HESTON E[σ]",  f"{mean_vol[-1]:.2f}%",          RED),
        )
        return fig_gbm, fig_ou, fig_hst, kpis
