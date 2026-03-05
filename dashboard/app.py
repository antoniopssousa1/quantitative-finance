# ═══════════════════════════════════════════════════════════════════════════
#  QF Terminal  —  dashboard/app.py
#  Modular Bloomberg-style quantitative finance dashboard
# ═══════════════════════════════════════════════════════════════════════════

import sys
import os

# Allow running from repo root: python dashboard/app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from dashboard.theme import BG, PANEL, CARD, BORDER, ACCENT, ACCENT2, ACCENT3, RED, GREEN, TEXT, MUTED, FONT
from dashboard.tabs import market, markowitz, capm, black_scholes, monte_carlo, var, bonds, gbm_ou, derivatives

# ── App ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="QF Terminal",
    suppress_callback_exceptions=True,
)
server = app.server  # for Gunicorn / cloud deployment

# ── Register all tab callbacks ────────────────────────────────────────────
for _tab in [market, markowitz, capm, black_scholes, monte_carlo, var, bonds, gbm_ou, derivatives]:
    _tab.register_callbacks(app)

# ── Tab definitions ───────────────────────────────────────────────────────
TABS = [
    ("MARKET",      "tab-market"),
    ("MARKOWITZ",   "tab-markowitz"),
    ("CAPM",        "tab-capm"),
    ("OPTIONS",     "tab-bs"),
    ("MONTE CARLO", "tab-mc"),
    ("VaR / CVaR",  "tab-var"),
    ("BONDS",       "tab-bonds"),
    ("STOCHASTIC",  "tab-gbm"),
    ("DERIVATIVES", "tab-deriv"),
]

_tab_style = {
    "fontFamily": FONT, "fontSize": "11px", "fontWeight": "600",
    "color": MUTED, "background": PANEL, "border": f"1px solid {BORDER}",
    "padding": "6px 14px", "borderRadius": "3px 3px 0 0",
    "letterSpacing": "0.5px",
}
_tab_sel_style = {**_tab_style, "color": ACCENT, "background": CARD,
                  "borderBottom": f"2px solid {ACCENT}"}


def _header():
    return html.Div([
        html.Div([
            html.Span("▣ ", style={"color": ACCENT, "fontSize": "20px"}),
            html.Span("QF TERMINAL", style={
                "fontFamily": FONT, "fontSize": "20px", "fontWeight": "800",
                "color": TEXT, "letterSpacing": "3px",
            }),
            html.Span("  ·  QUANTITATIVE FINANCE DASHBOARD", style={
                "fontFamily": FONT, "fontSize": "11px", "color": MUTED,
                "letterSpacing": "2px", "marginLeft": "8px",
            }),
        ], style={"display": "inline-block", "verticalAlign": "middle"}),
        html.Div(id="qf-clock", style={
            "float": "right", "fontFamily": FONT, "fontSize": "12px",
            "color": ACCENT, "paddingTop": "4px",
        }),
    ], style={
        "background": PANEL, "borderBottom": f"2px solid {ACCENT}",
        "padding": "10px 20px", "marginBottom": "0",
    })


def _tabs_bar():
    return dcc.Tabs(
        id="main-tabs", value="tab-market",
        children=[
            dcc.Tab(label=label, value=val,
                    style=_tab_style, selected_style=_tab_sel_style)
            for label, val in TABS
        ],
        style={"background": PANEL, "borderBottom": f"1px solid {BORDER}"},
        colors={"border": BORDER, "primary": ACCENT, "background": PANEL},
    )


app.layout = html.Div([
    _header(),
    _tabs_bar(),
    html.Div(id="tab-content", style={
        "background": BG, "minHeight": "calc(100vh - 100px)",
        "padding": "16px 20px",
    }),
    dcc.Interval(id="clock-tick", interval=1000, n_intervals=0),
], style={"background": BG, "minHeight": "100vh"})


# ── Tab routing ───────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    mapping = {
        "tab-market":    market.layout,
        "tab-markowitz": markowitz.layout,
        "tab-capm":      capm.layout,
        "tab-bs":        black_scholes.layout,
        "tab-mc":        monte_carlo.layout,
        "tab-var":       var.layout,
        "tab-bonds":     bonds.layout,
        "tab-gbm":       gbm_ou.layout,
        "tab-deriv":     derivatives.layout,
    }
    return mapping.get(tab, market.layout)()


# ── Live clock ────────────────────────────────────────────────────────────
@app.callback(Output("qf-clock", "children"), Input("clock-tick", "n_intervals"))
def update_clock(_):
    return datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ▣  QF Terminal  →  http://127.0.0.1:8050\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
