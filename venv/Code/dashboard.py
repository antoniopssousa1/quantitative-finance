import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm
import scipy.optimize as optimization
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from numpy import log, exp, sqrt
import datetime

# ─────────────────────────────────────────────
#  CORE QUANT FUNCTIONS
# ─────────────────────────────────────────────

# --- Black-Scholes ---
def bs_call(S, E, T, rf, sigma):
    d1 = (log(S / E) + (rf + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - E * exp(-rf * T) * norm.cdf(d2), d1, d2

def bs_put(S, E, T, rf, sigma):
    d1 = (log(S / E) + (rf + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return -S * norm.cdf(-d1) + E * exp(-rf * T) * norm.cdf(-d2), d1, d2

def greeks(S, E, T, rf, sigma):
    d1 = (log(S / E) + (rf + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    delta_call = norm.cdf(d1)
    delta_put  = -norm.cdf(-d1)
    gamma      = norm.pdf(d1) / (S * sigma * sqrt(T))
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - rf * E * exp(-rf * T) * norm.cdf(d2)) / 365
    vega       = S * norm.pdf(d1) * sqrt(T) / 100
    rho_call   = E * T * exp(-rf * T) * norm.cdf(d2) / 100
    return delta_call, delta_put, gamma, theta_call, vega, rho_call

# --- GBM ---
def simulate_gbm(S0, mu, sigma, T=1, N=252, n_paths=10):
    dt = T / N
    paths = []
    for _ in range(n_paths):
        W = np.cumsum(np.random.standard_normal(N)) * sqrt(dt)
        X = (mu - 0.5 * sigma**2) * np.linspace(0, T, N) + sigma * W
        paths.append(S0 * np.exp(X))
    return paths

# --- Monte Carlo Stock Price ---
def mc_stock_price(S0, mu, sigma, N=252, n_sims=500):
    results = []
    for _ in range(n_sims):
        prices = [S0]
        for _ in range(N):
            prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
        results.append(prices)
    df = pd.DataFrame(results).T
    df['mean'] = df.mean(axis=1)
    return df

# --- VaR ---
def calculate_var(position, c, mu, sigma, n=1):
    return position * (mu * n - sigma * sqrt(n) * norm.ppf(1 - c))

def mc_var(S, mu, sigma, c, n=1, iterations=50000):
    rand = np.random.normal(0, 1, iterations)
    stock_price = S * np.exp(n * (mu - 0.5 * sigma**2) + sigma * sqrt(n) * rand)
    stock_price.sort()
    percentile = np.percentile(stock_price, (1 - c) * 100)
    return S - percentile, stock_price

# --- Vasicek Bond Pricing ---
def vasicek_bond_price(x, r0, kappa, theta, sigma, T=1, n_sims=500, n_points=100):
    dt = T / n_points
    result = []
    for _ in range(n_sims):
        rates = [r0]
        for _ in range(n_points):
            dr = kappa * (theta - rates[-1]) * dt + sigma * sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        result.append(rates)
    sim_data = pd.DataFrame(result).T
    integral_sum = sim_data.sum() * dt
    bond_price = x * np.mean(np.exp(-integral_sum))
    return bond_price, sim_data

# --- Markowitz ---
def markowitz_portfolios(returns, n=3000):
    n_stocks = returns.shape[1]
    means, risks, weights_list = [], [], []
    for _ in range(n):
        w = np.random.random(n_stocks)
        w /= w.sum()
        ret = np.sum(returns.mean() * w) * 252
        risk = sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        means.append(ret)
        risks.append(risk)
        weights_list.append(w)
    return np.array(means), np.array(risks), np.array(weights_list)

def optimize_sharpe(returns, rf=0.05):
    n = returns.shape[1]
    def neg_sharpe(w):
        ret = np.sum(returns.mean() * w) * 252
        vol = sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        return -(ret - rf) / vol
    w0 = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    result = optimization.minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# --- Zero Coupon Bond ---
def zcb_price(principal, maturity, rate):
    return principal / (1 + rate) ** maturity

def zcb_ytm(principal, price, maturity):
    return (principal / price) ** (1 / maturity) - 1

# --- CAPM ---
def capm_expected_return(beta, rf, market_return):
    return rf + beta * (market_return - rf)

# ─────────────────────────────────────────────
#  DASHBOARD LAYOUT
# ─────────────────────────────────────────────

DARK_BG    = "#0a0e1a"
PANEL_BG   = "#0d1117"
CARD_BG    = "#111827"
BORDER     = "#1f2937"
ACCENT     = "#00d4aa"
ACCENT2    = "#3b82f6"
ACCENT3    = "#f59e0b"
RED        = "#ef4444"
GREEN      = "#22c55e"
TEXT       = "#e2e8f0"
MUTED      = "#6b7280"
FONT       = "JetBrains Mono, Consolas, monospace"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT, family=FONT, size=11),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(bgcolor=CARD_BG, bordercolor=BORDER),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)

def card(title, children, id=None):
    return dbc.Card([
        dbc.CardHeader(
            html.Span(title, style={"color": ACCENT, "fontFamily": FONT, "fontSize": "11px",
                                    "letterSpacing": "2px", "textTransform": "uppercase"}),
            style={"background": CARD_BG, "borderBottom": f"1px solid {BORDER}", "padding": "8px 14px"}
        ),
        dbc.CardBody(children, style={"background": CARD_BG, "padding": "12px"})
    ], style={"border": f"1px solid {BORDER}", "borderRadius": "6px", "marginBottom": "12px"},
       id=id or "")

def stat_box(label, value, color=ACCENT):
    return html.Div([
        html.Div(label, style={"color": MUTED, "fontSize": "10px", "letterSpacing": "1px",
                               "textTransform": "uppercase", "fontFamily": FONT}),
        html.Div(value, style={"color": color, "fontSize": "20px", "fontWeight": "bold",
                               "fontFamily": FONT, "marginTop": "2px"})
    ], style={"background": PANEL_BG, "border": f"1px solid {BORDER}", "borderRadius": "4px",
              "padding": "10px 14px", "flex": "1", "minWidth": "120px"})

def label_input(label, id, value, type="number", step=None, min=None, max=None):
    return html.Div([
        html.Label(label, style={"color": MUTED, "fontSize": "10px", "fontFamily": FONT,
                                 "letterSpacing": "1px", "display": "block", "marginBottom": "3px"}),
        dcc.Input(id=id, type=type, value=value, step=step, min=min, max=max,
                  style={"background": PANEL_BG, "border": f"1px solid {BORDER}", "color": TEXT,
                         "fontFamily": FONT, "fontSize": "13px", "padding": "5px 8px",
                         "borderRadius": "3px", "width": "100%"})
    ], style={"marginBottom": "8px"})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                title="QF Terminal")

app.layout = html.Div(style={"background": DARK_BG, "minHeight": "100vh", "fontFamily": FONT}, children=[

    # ── HEADER ──
    html.Div([
        html.Div([
            html.Span("▣ ", style={"color": ACCENT, "fontSize": "20px"}),
            html.Span("QUANTITATIVE FINANCE TERMINAL", style={"color": TEXT, "fontSize": "14px",
                                                               "letterSpacing": "4px", "fontWeight": "bold"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
        html.Div(id="clock", style={"color": MUTED, "fontSize": "11px"}),
        dcc.Interval(id="clock-interval", interval=1000),
    ], style={"background": CARD_BG, "borderBottom": f"2px solid {ACCENT}", "padding": "12px 24px",
              "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),

    # ── TABS ──
    dbc.Tabs(id="tabs", active_tab="tab-market", style={"background": PANEL_BG, "borderBottom": f"1px solid {BORDER}",
                                                         "paddingLeft": "16px"}, children=[
        dbc.Tab(label="📈 MARKET",      tab_id="tab-market",     label_style={"color": MUTED, "fontFamily": FONT, "fontSize": "11px"},
                active_label_style={"color": ACCENT,  "fontFamily": FONT, "fontSize": "11px", "borderTop": f"2px solid {ACCENT}"}),
        dbc.Tab(label="⚖️  MARKOWITZ",  tab_id="tab-markowitz",  label_style={"color": MUTED, "fontFamily": FONT, "fontSize": "11px"},
                active_label_style={"color": ACCENT,  "fontFamily": FONT, "fontSize": "11px", "borderTop": f"2px solid {ACCENT}"}),
        dbc.Tab(label="📐 CAPM",        tab_id="tab-capm",       label_style={"color": MUTED, "fontFamily": FONT, "fontSize": "11px"},
                active_label_style={"color": ACCENT,  "fontFamily": FONT, "fontSize": "11px", "borderTop": f"2px solid {ACCENT}"}),
        dbc.Tab(label="🎲 BLACK-SCHOLES",tab_id="tab-bs",        label_style={"color": MUTED, "fontFamily": FONT, "fontSize": "11px"},
                active_label_style={"color": ACCENT,  "fontFamily": FONT, "fontSize": "11px", "borderTop": f"2px solid {ACCENT}"}),
        dbc.Tab(label="🎰 MONTE CARLO", tab_id="tab-mc",         label_style={"color": MUTED, "fontFamily": FONT, "fontSize": "11px"},
                active_label_style={"color": ACCENT,  "fontFamily": FONT, "fontSize": "11px", "borderTop": f"2px solid {ACCENT}"}),
        dbc.Tab(label="🛡️  VAR",         tab_id="tab-var",        label_style={"color": MUTED, "fontFamily": FONT, "fontSize": "11px"},
                active_label_style={"color": ACCENT,  "fontFamily": FONT, "fontSize": "11px", "borderTop": f"2px solid {ACCENT}"}),
        dbc.Tab(label="🏦 BONDS",       tab_id="tab-bonds",      label_style={"color": MUTED, "fontFamily": FONT, "fontSize": "11px"},
                active_label_style={"color": ACCENT,  "fontFamily": FONT, "fontSize": "11px", "borderTop": f"2px solid {ACCENT}"}),
        dbc.Tab(label="〜 GBM / OU",    tab_id="tab-gbm",        label_style={"color": MUTED, "fontFamily": FONT, "fontSize": "11px"},
                active_label_style={"color": ACCENT,  "fontFamily": FONT, "fontSize": "11px", "borderTop": f"2px solid {ACCENT}"}),
    ]),

    html.Div(id="tab-content", style={"padding": "16px 20px"}),
])

# ─────────────────────────────────────────────
#  TAB LAYOUTS
# ─────────────────────────────────────────────

def tab_market():
    return html.Div([
        dbc.Row([
            dbc.Col([
                label_input("Ticker (e.g. AAPL, TSLA, ^GSPC)", "mkt-ticker", "AAPL", type="text"),
                label_input("Start Date", "mkt-start", "2020-01-01", type="text"),
                label_input("End Date",   "mkt-end",   str(datetime.date.today()), type="text"),
                dbc.Button("LOAD", id="mkt-btn", color="success", size="sm",
                           style={"width": "100%", "fontFamily": FONT, "letterSpacing": "2px"}),
            ], width=2),
            dbc.Col([
                html.Div(id="mkt-stats", style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"}),
                card("CANDLESTICK + VOLUME", dcc.Graph(id="mkt-candle", style={"height": "420px"},
                     config={"displayModeBar": False})),
            ], width=10),
        ]),
        dbc.Row([
            dbc.Col(card("LOG RETURNS DISTRIBUTION", dcc.Graph(id="mkt-dist", style={"height": "280px"},
                    config={"displayModeBar": False})), width=6),
            dbc.Col(card("ROLLING VOLATILITY (30d)", dcc.Graph(id="mkt-vol", style={"height": "280px"},
                    config={"displayModeBar": False})), width=6),
        ])
    ])

def tab_markowitz():
    return html.Div([
        dbc.Row([
            dbc.Col([
                label_input("Stocks (comma-sep)", "mk-stocks", "AAPL,MSFT,TSLA,GE,WMT", type="text"),
                label_input("Start Date", "mk-start", "2018-01-01", type="text"),
                label_input("End Date",   "mk-end",   "2024-01-01", type="text"),
                label_input("Risk-Free Rate", "mk-rf", 0.05, step=0.001),
                label_input("Portfolios",     "mk-n",  2000, step=500, min=500, max=10000),
                dbc.Button("OPTIMIZE", id="mk-btn", color="success", size="sm",
                           style={"width": "100%", "fontFamily": FONT, "letterSpacing": "2px"}),
                html.Div(id="mk-weights", style={"marginTop": "12px"}),
            ], width=3),
            dbc.Col([
                html.Div(id="mk-stats", style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"}),
                card("EFFICIENT FRONTIER", dcc.Graph(id="mk-frontier", style={"height": "460px"},
                     config={"displayModeBar": False})),
            ], width=9),
        ]),
    ])

def tab_capm():
    return html.Div([
        dbc.Row([
            dbc.Col([
                label_input("Stock Ticker",   "cp-stock",  "IBM",  type="text"),
                label_input("Market Index",   "cp-market", "^GSPC",type="text"),
                label_input("Start Date",     "cp-start",  "2015-01-01", type="text"),
                label_input("End Date",       "cp-end",    "2024-01-01", type="text"),
                label_input("Risk-Free Rate", "cp-rf",     0.04, step=0.001),
                dbc.Button("COMPUTE", id="cp-btn", color="success", size="sm",
                           style={"width": "100%", "fontFamily": FONT, "letterSpacing": "2px"}),
            ], width=2),
            dbc.Col([
                html.Div(id="cp-stats", style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"}),
                card("CAPM LINEAR REGRESSION  —  Rₐ = β·Rₘ + α", dcc.Graph(id="cp-chart", style={"height": "430px"},
                     config={"displayModeBar": False})),
            ], width=10),
        ]),
    ])

def tab_bs():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("BLACK-SCHOLES", style={"color": ACCENT, "fontSize": "10px", "letterSpacing": "2px",
                                                  "marginBottom": "10px"}),
                label_input("Spot Price S₀",  "bs-S",     100,  step=1),
                label_input("Strike Price K",  "bs-E",     100,  step=1),
                label_input("Time to Expiry T (yrs)", "bs-T", 1.0, step=0.1, min=0.01),
                label_input("Risk-Free Rate r",  "bs-rf",  0.05, step=0.005),
                label_input("Volatility σ",      "bs-sig", 0.20, step=0.01, min=0.01),
                dbc.Button("PRICE", id="bs-btn", color="success", size="sm",
                           style={"width": "100%", "fontFamily": FONT, "letterSpacing": "2px", "marginTop": "4px"}),
                html.Div(id="bs-results", style={"marginTop": "14px"}),
            ], width=3),
            dbc.Col([
                card("OPTION PRICE vs SPOT", dcc.Graph(id="bs-spot-chart", style={"height": "280px"},
                     config={"displayModeBar": False})),
                card("THE GREEKS vs SPOT", dcc.Graph(id="bs-greeks-chart", style={"height": "280px"},
                     config={"displayModeBar": False})),
            ], width=9),
        ]),
    ])

def tab_mc():
    return html.Div([
        dbc.Row([
            dbc.Col([
                label_input("Initial Price S₀", "mc-S0",    100,  step=1),
                label_input("Drift μ (annual)",  "mc-mu",    0.08, step=0.01),
                label_input("Volatility σ",      "mc-sig",   0.20, step=0.01),
                label_input("Days N",            "mc-N",     252,  step=10, min=10),
                label_input("Paths",             "mc-paths", 200,  step=50, min=10, max=1000),
                dbc.Button("SIMULATE", id="mc-btn", color="success", size="sm",
                           style={"width": "100%", "fontFamily": FONT, "letterSpacing": "2px"}),
                html.Div(id="mc-stats", style={"marginTop": "14px"}),
            ], width=3),
            dbc.Col([
                card("MONTE CARLO PATHS + MEAN", dcc.Graph(id="mc-chart", style={"height": "380px"},
                     config={"displayModeBar": False})),
                card("TERMINAL PRICE DISTRIBUTION", dcc.Graph(id="mc-dist", style={"height": "250px"},
                     config={"displayModeBar": False})),
            ], width=9),
        ]),
    ])

def tab_var():
    return html.Div([
        dbc.Row([
            dbc.Col([
                label_input("Investment ($)",    "var-S",   1_000_000, step=10000),
                label_input("Daily Drift μ",     "var-mu",  0.0005,    step=0.0001),
                label_input("Daily Volatility σ","var-sig", 0.015,     step=0.001),
                label_input("Confidence Level",  "var-c",   0.99,      step=0.01, min=0.90, max=0.999),
                label_input("Horizon (days)",    "var-n",   1,         step=1, min=1),
                dbc.Button("CALCULATE", id="var-btn", color="success", size="sm",
                           style={"width": "100%", "fontFamily": FONT, "letterSpacing": "2px"}),
                html.Div(id="var-stats", style={"marginTop": "14px"}),
            ], width=3),
            dbc.Col([
                card("MONTE CARLO VaR — P&L DISTRIBUTION", dcc.Graph(id="var-chart", style={"height": "420px"},
                     config={"displayModeBar": False})),
            ], width=9),
        ]),
    ])

def tab_bonds():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("ZERO COUPON BOND", style={"color": ACCENT, "fontSize": "10px",
                                                     "letterSpacing": "2px", "marginBottom": "6px"}),
                label_input("Face Value ($)",    "zcb-par",  1000, step=100),
                label_input("Maturity (yrs)",    "zcb-mat",  5,    step=1, min=1),
                label_input("Discount Rate",     "zcb-rate", 0.05, step=0.005),
                html.Div(id="zcb-stats", style={"marginBottom": "16px"}),
                html.Hr(style={"borderColor": BORDER}),
                html.Div("VASICEK BOND PRICING", style={"color": ACCENT2, "fontSize": "10px",
                                                         "letterSpacing": "2px", "marginBottom": "6px"}),
                label_input("Face Value ($)",    "vas-par",   1000, step=100),
                label_input("r₀ (initial rate)", "vas-r0",    0.10, step=0.01),
                label_input("κ (mean reversion)","vas-kappa", 0.30, step=0.05),
                label_input("θ (long-run mean)", "vas-theta", 0.30, step=0.01),
                label_input("σ (volatility)",    "vas-sigma", 0.03, step=0.005),
                label_input("T (years)",         "vas-T",     1.0,  step=0.5),
                dbc.Button("PRICE BONDS", id="bond-btn", color="success", size="sm",
                           style={"width": "100%", "fontFamily": FONT, "letterSpacing": "2px"}),
                html.Div(id="vas-stats", style={"marginTop": "12px"}),
            ], width=3),
            dbc.Col([
                card("ZCB PRICE vs MATURITY", dcc.Graph(id="zcb-chart", style={"height": "260px"},
                     config={"displayModeBar": False})),
                card("VASICEK INTEREST RATE PATHS", dcc.Graph(id="vas-chart", style={"height": "300px"},
                     config={"displayModeBar": False})),
            ], width=9),
        ]),
    ])

def tab_gbm():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("GEOMETRIC BROWNIAN MOTION", style={"color": ACCENT, "fontSize": "10px",
                                                              "letterSpacing": "2px", "marginBottom": "6px"}),
                label_input("S₀",           "gbm-S0",    100,  step=10),
                label_input("μ (drift)",    "gbm-mu",    0.10, step=0.01),
                label_input("σ (vol)",      "gbm-sig",   0.20, step=0.01),
                label_input("T (years)",    "gbm-T",     2.0,  step=0.5),
                label_input("Paths",        "gbm-paths", 20,   step=5, min=1, max=100),
                html.Hr(style={"borderColor": BORDER}),
                html.Div("ORNSTEIN-UHLENBECK", style={"color": ACCENT2, "fontSize": "10px",
                                                       "letterSpacing": "2px", "marginBottom": "6px"}),
                label_input("r₀",           "ou-r0",     0.10, step=0.01),
                label_input("κ (speed)",    "ou-kappa",  0.70, step=0.05),
                label_input("θ (mean)",     "ou-theta",  0.10, step=0.01),
                label_input("σ",            "ou-sigma",  0.02, step=0.005),
                label_input("T (years)",    "ou-T",      5.0,  step=1.0),
                label_input("Paths",        "ou-paths",  15,   step=5, min=1, max=50),
                dbc.Button("SIMULATE", id="gbm-btn", color="success", size="sm",
                           style={"width": "100%", "fontFamily": FONT, "letterSpacing": "2px"}),
            ], width=3),
            dbc.Col([
                card("GEOMETRIC BROWNIAN MOTION", dcc.Graph(id="gbm-chart", style={"height": "350px"},
                     config={"displayModeBar": False})),
                card("ORNSTEIN-UHLENBECK PROCESS", dcc.Graph(id="ou-chart", style={"height": "300px"},
                     config={"displayModeBar": False})),
            ], width=9),
        ]),
    ])

# ─────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────

@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(tab):
    return {
        "tab-market":   tab_market(),
        "tab-markowitz":tab_markowitz(),
        "tab-capm":     tab_capm(),
        "tab-bs":       tab_bs(),
        "tab-mc":       tab_mc(),
        "tab-var":      tab_var(),
        "tab-bonds":    tab_bonds(),
        "tab-gbm":      tab_gbm(),
    }.get(tab, tab_market())

@app.callback(Output("clock", "children"), Input("clock-interval", "n_intervals"))
def update_clock(_):
    return datetime.datetime.now().strftime("🕐  %Y-%m-%d  %H:%M:%S")

# ── MARKET ──
@app.callback(
    Output("mkt-candle", "figure"), Output("mkt-dist", "figure"),
    Output("mkt-vol", "figure"),    Output("mkt-stats", "children"),
    Input("mkt-btn", "n_clicks"),
    State("mkt-ticker", "value"), State("mkt-start", "value"), State("mkt-end", "value"),
    prevent_initial_call=False
)
def update_market(_, ticker, start, end):
    ticker = ticker or "AAPL"
    start  = start  or "2020-01-01"
    end    = end    or str(datetime.date.today())
    df = yf.download(ticker, start, end, progress=False)
    if df.empty:
        empty = go.Figure()
        return empty, empty, empty, []
    close = df["Close"].squeeze()
    returns = np.log(close / close.shift(1)).dropna()
    vol30 = returns.rolling(30).std() * np.sqrt(252)

    # Candlestick
    fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
    fig_c.add_trace(go.Candlestick(
        x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(), close=close,
        increasing_line_color=GREEN, decreasing_line_color=RED, name=ticker
    ), row=1, col=1)
    colors = [GREEN if v >= 0 else RED for v in df["Close"].squeeze().pct_change().fillna(0)]
    fig_c.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(), marker_color=colors,
                           name="Volume", opacity=0.7), row=2, col=1)
    fig_c.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=420)
    fig_c.update_xaxes(rangeslider_visible=False)

    # Distribution
    mu_r, sigma_r = returns.mean(), returns.std()
    x_range = np.linspace(mu_r - 4*sigma_r, mu_r + 4*sigma_r, 200)
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=returns, nbinsx=100, histnorm="probability density",
                                  marker_color=ACCENT2, opacity=0.7, name="Returns"))
    fig_d.add_trace(go.Scatter(x=x_range, y=norm.pdf(x_range, mu_r, sigma_r),
                                line=dict(color=ACCENT, width=2), name="Normal fit"))
    fig_d.update_layout(**PLOTLY_TEMPLATE["layout"], height=280, showlegend=True)

    # Rolling Vol
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=vol30.index, y=vol30, line=dict(color=ACCENT3, width=1.5), fill="tozeroy",
                                fillcolor="rgba(245,158,11,0.1)", name="Ann. Vol"))
    fig_v.update_layout(**PLOTLY_TEMPLATE["layout"], height=280)

    last  = float(close.iloc[-1])
    chg   = float(close.pct_change().iloc[-1]) * 100
    color = GREEN if chg >= 0 else RED
    stats_row = [
        stat_box("LAST", f"${last:.2f}", color),
        stat_box("CHG%", f"{chg:+.2f}%", color),
        stat_box("52W HIGH", f"${float(close.rolling(252).max().iloc[-1]):.2f}", ACCENT),
        stat_box("52W LOW",  f"${float(close.rolling(252).min().iloc[-1]):.2f}", RED),
        stat_box("ANN. VOL", f"{float(vol30.iloc[-1])*100:.1f}%", ACCENT3),
        stat_box("MEAN RET", f"{mu_r*252*100:.1f}%/yr", ACCENT2),
    ]
    return fig_c, fig_d, fig_v, stats_row

# ── MARKOWITZ ──
@app.callback(
    Output("mk-frontier", "figure"), Output("mk-stats", "children"), Output("mk-weights", "children"),
    Input("mk-btn", "n_clicks"),
    State("mk-stocks", "value"), State("mk-start", "value"), State("mk-end", "value"),
    State("mk-rf", "value"), State("mk-n", "value"),
    prevent_initial_call=False
)
def update_markowitz(_, stocks_str, start, end, rf, n_port):
    stocks_str = stocks_str or "AAPL,MSFT,TSLA,GE,WMT"
    stocks = [s.strip() for s in stocks_str.split(",")]
    rf = rf or 0.05
    n_port = int(n_port or 2000)
    data = {}
    for s in stocks:
        t = yf.Ticker(s)
        data[s] = t.history(start=start or "2018-01-01", end=end or "2024-01-01")["Close"]
    df = pd.DataFrame(data).dropna()
    returns = np.log(df / df.shift(1)).dropna()
    means, risks, weights = markowitz_portfolios(returns, n_port)
    sharpe = (means - rf) / risks
    opt = optimize_sharpe(returns, rf)
    opt_ret = np.sum(returns.mean() * opt.x) * 252
    opt_vol = np.sqrt(np.dot(opt.x.T, np.dot(returns.cov() * 252, opt.x)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risks, y=means, mode="markers",
        marker=dict(color=sharpe, colorscale="Viridis", size=4, opacity=0.6,
                    colorbar=dict(title="Sharpe", thickness=12, tickfont=dict(color=TEXT))),
        name="Portfolios"
    ))
    fig.add_trace(go.Scatter(
        x=[opt_vol], y=[opt_ret], mode="markers",
        marker=dict(symbol="star", size=18, color=ACCENT, line=dict(color="white", width=1)),
        name="Optimal Portfolio"
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=460,
                      xaxis_title="Volatility (Risk)", yaxis_title="Expected Return")

    stats_row = [
        stat_box("OPT. RETURN",   f"{opt_ret*100:.2f}%", GREEN),
        stat_box("OPT. VOLATILITY", f"{opt_vol*100:.2f}%", ACCENT3),
        stat_box("SHARPE RATIO",  f"{(opt_ret-rf)/opt_vol:.3f}", ACCENT),
    ]
    weight_rows = [html.Div([
        html.Span(s, style={"color": MUTED, "fontSize": "10px", "fontFamily": FONT, "width": "50px", "display": "inline-block"}),
        html.Div(style={"display": "inline-block", "background": ACCENT,
                        "width": f"{w*100:.0f}%", "height": "8px",
                        "borderRadius": "2px", "marginLeft": "4px", "verticalAlign": "middle",
                        "maxWidth": "120px"}),
        html.Span(f" {w*100:.1f}%", style={"color": TEXT, "fontSize": "10px", "fontFamily": FONT})
    ], style={"marginBottom": "4px"}) for s, w in zip(stocks, opt.x)]

    return fig, stats_row, html.Div(weight_rows)

# ── CAPM ──
@app.callback(
    Output("cp-chart", "figure"), Output("cp-stats", "children"),
    Input("cp-btn", "n_clicks"),
    State("cp-stock", "value"), State("cp-market", "value"),
    State("cp-start", "value"), State("cp-end", "value"), State("cp-rf", "value"),
    prevent_initial_call=False
)
def update_capm(_, stock, market, start, end, rf):
    stock  = stock  or "IBM"
    market = market or "^GSPC"
    rf     = rf     or 0.04
    data = {}
    for s in [stock, market]:
        t = yf.download(s, start or "2015-01-01", end or "2024-01-01", progress=False)
        data[s] = t["Close"].squeeze()
    df = pd.DataFrame(data).resample("ME").last()
    df_ret = np.log(df / df.shift(1)).dropna()
    beta, alpha = np.polyfit(df_ret[market], df_ret[stock], deg=1)
    expected_ret = rf + beta * (df_ret[market].mean() * 12 - rf)
    cov = np.cov(df_ret[stock], df_ret[market])
    beta_cov = cov[0, 1] / cov[1, 1]
    r2 = np.corrcoef(df_ret[market], df_ret[stock])[0, 1] ** 2
    x_line = np.linspace(df_ret[market].min(), df_ret[market].max(), 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ret[market], y=df_ret[stock], mode="markers",
                              marker=dict(color=ACCENT2, size=6, opacity=0.7), name="Monthly Returns"))
    fig.add_trace(go.Scatter(x=x_line, y=beta * x_line + alpha,
                              line=dict(color=RED, width=2), name=f"β={beta:.3f}  α={alpha:.4f}"))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=430,
                      xaxis_title=f"Market Return ({market})", yaxis_title=f"Stock Return ({stock})")

    stats_row = [
        stat_box("β (BETA)",      f"{beta:.4f}",     ACCENT),
        stat_box("α (ALPHA)",     f"{alpha:.4f}",    ACCENT3),
        stat_box("R²",            f"{r2:.4f}",       ACCENT2),
        stat_box("E[RETURN]",     f"{expected_ret*100:.2f}%", GREEN),
    ]
    return fig, stats_row

# ── BLACK-SCHOLES ──
@app.callback(
    Output("bs-spot-chart", "figure"), Output("bs-greeks-chart", "figure"), Output("bs-results", "children"),
    Input("bs-btn", "n_clicks"),
    State("bs-S", "value"), State("bs-E", "value"), State("bs-T", "value"),
    State("bs-rf", "value"), State("bs-sig", "value"),
    prevent_initial_call=False
)
def update_bs(_, S, E, T, rf, sigma):
    S, E, T, rf, sigma = S or 100, E or 100, T or 1, rf or 0.05, sigma or 0.20
    call, d1, d2 = bs_call(S, E, T, rf, sigma)
    put, _, _    = bs_put(S, E, T, rf, sigma)
    d_call, d_put, gamma, theta, vega, rho = greeks(S, E, T, rf, sigma)

    S_range = np.linspace(max(1, S * 0.5), S * 1.5, 200)
    calls   = [bs_call(s, E, T, rf, sigma)[0] for s in S_range]
    puts    = [bs_put(s, E, T, rf, sigma)[0]  for s in S_range]

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=S_range, y=calls, line=dict(color=GREEN, width=2),  name="Call"))
    fig_s.add_trace(go.Scatter(x=S_range, y=puts,  line=dict(color=RED,   width=2),  name="Put"))
    fig_s.add_vline(x=S, line_dash="dash", line_color=MUTED)
    fig_s.add_vline(x=E, line_dash="dot",  line_color=ACCENT3, annotation_text="K")
    fig_s.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                        xaxis_title="Spot Price", yaxis_title="Option Price")

    d_calls  = [bs_call(s, E, T, rf, sigma)[0] for s in S_range]
    gammas   = [greeks(s, E, T, rf, sigma)[2]  for s in S_range]
    dc_calls = [greeks(s, E, T, rf, sigma)[0]  for s in S_range]
    vegas    = [greeks(s, E, T, rf, sigma)[4]  for s in S_range]
    thetas   = [greeks(s, E, T, rf, sigma)[3]  for s in S_range]

    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=S_range, y=dc_calls, line=dict(color=GREEN,  width=1.5), name="Δ Call"))
    fig_g.add_trace(go.Scatter(x=S_range, y=gammas,   line=dict(color=ACCENT, width=1.5), name="Γ Gamma"))
    fig_g.add_trace(go.Scatter(x=S_range, y=vegas,    line=dict(color=ACCENT2,width=1.5), name="ν Vega"))
    fig_g.add_trace(go.Scatter(x=S_range, y=thetas,   line=dict(color=RED,    width=1.5), name="Θ Theta"))
    fig_g.add_vline(x=S, line_dash="dash", line_color=MUTED)
    fig_g.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                        xaxis_title="Spot Price", yaxis_title="Greek Value")

    results = html.Div([
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}, children=[
            stat_box("CALL",  f"${call:.4f}", GREEN),
            stat_box("PUT",   f"${put:.4f}",  RED),
        ]),
        html.Div(style={"marginTop": "8px"}),
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}, children=[
            stat_box("Δ CALL", f"{d_call:.4f}",  ACCENT),
            stat_box("Δ PUT",  f"{d_put:.4f}",   ACCENT),
            stat_box("Γ",      f"{gamma:.4f}",   ACCENT2),
            stat_box("Θ/day",  f"{theta:.4f}",   ACCENT3),
            stat_box("ν VEGA", f"{vega:.4f}",    GREEN),
            stat_box("ρ RHO",  f"{rho:.4f}",     MUTED),
        ]),
        html.Div(style={"marginTop": "8px", "color": MUTED, "fontSize": "10px", "fontFamily": FONT}, children=[
            f"d₁ = {d1:.4f}   d₂ = {d2:.4f}"
        ])
    ])
    return fig_s, fig_g, results

# ── MONTE CARLO ──
@app.callback(
    Output("mc-chart", "figure"), Output("mc-dist", "figure"), Output("mc-stats", "children"),
    Input("mc-btn", "n_clicks"),
    State("mc-S0", "value"), State("mc-mu", "value"), State("mc-sig", "value"),
    State("mc-N", "value"), State("mc-paths", "value"),
    prevent_initial_call=False
)
def update_mc(_, S0, mu, sigma, N, paths):
    S0, mu, sigma, N, paths = S0 or 100, mu or 0.08, sigma or 0.20, int(N or 252), int(paths or 200)
    df = mc_stock_price(S0, mu, sigma, N, paths)
    terminal = df.iloc[-1, :-1].values
    predicted = float(df["mean"].iloc[-1])

    fig = go.Figure()
    for i in range(min(paths, df.shape[1] - 1)):
        fig.add_trace(go.Scatter(y=df.iloc[:, i], mode="lines",
                                  line=dict(width=0.4, color=ACCENT2), opacity=0.3, showlegend=False))
    fig.add_trace(go.Scatter(y=df["mean"], mode="lines",
                              line=dict(color=ACCENT, width=2.5), name="Mean Path"))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380,
                      xaxis_title="Days", yaxis_title="Price ($)")

    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=terminal, nbinsx=60, marker_color=ACCENT2, opacity=0.8, name="Terminal Price"))
    fig_d.add_vline(x=predicted, line_dash="dash", line_color=ACCENT, annotation_text=f"Mean: ${predicted:.2f}")
    fig_d.update_layout(**PLOTLY_TEMPLATE["layout"], height=250, xaxis_title="Terminal Price", yaxis_title="Count")

    p5  = np.percentile(terminal, 5)
    p95 = np.percentile(terminal, 95)
    stats_row = [
        stat_box("PREDICTED",  f"${predicted:.2f}", ACCENT),
        stat_box("P5",         f"${p5:.2f}",        RED),
        stat_box("P95",        f"${p95:.2f}",       GREEN),
        stat_box("STD",        f"${np.std(terminal):.2f}", ACCENT3),
    ]
    return fig, fig_d, stats_row

# ── VAR ──
@app.callback(
    Output("var-chart", "figure"), Output("var-stats", "children"),
    Input("var-btn", "n_clicks"),
    State("var-S", "value"), State("var-mu", "value"), State("var-sig", "value"),
    State("var-c", "value"), State("var-n", "value"),
    prevent_initial_call=False
)
def update_var(_, S, mu, sigma, c, n):
    S, mu, sigma, c, n = S or 1e6, mu or 0.0005, sigma or 0.015, c or 0.99, int(n or 1)
    var_param = calculate_var(S, c, mu, sigma, n)
    var_mc, sim_prices = mc_var(S, mu, sigma, c, n)
    pnl = sim_prices - S
    cutoff = np.percentile(pnl, (1 - c) * 100)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl, nbinsx=100, marker_color=ACCENT2, opacity=0.7, name="P&L"))
    fig.add_vline(x=cutoff, line_dash="dash", line_color=RED, line_width=2,
                  annotation_text=f"VaR {c*100:.0f}% = ${abs(cutoff):,.0f}",
                  annotation_font_color=RED)
    # shade loss tail
    loss_x = pnl[pnl <= cutoff]
    fig.add_trace(go.Histogram(x=loss_x, nbinsx=30, marker_color=RED, opacity=0.8, name=f"Loss tail"))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420,
                      xaxis_title="P&L ($)", yaxis_title="Count",
                      xaxis_tickformat="$,.0f")

    stats_row = html.Div([
        html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
            stat_box(f"MC VaR ({c*100:.0f}%)", f"${var_mc:,.0f}", RED),
            stat_box(f"PARAM VaR",            f"${var_param:,.0f}", ACCENT3),
            stat_box("HORIZON",               f"{n}d", MUTED),
            stat_box("INVESTMENT",            f"${S:,.0f}", ACCENT),
        ])
    ])
    return fig, stats_row

# ── BONDS ──
@app.callback(
    Output("zcb-chart", "figure"), Output("vas-chart", "figure"),
    Output("zcb-stats", "children"), Output("vas-stats", "children"),
    Input("bond-btn", "n_clicks"),
    State("zcb-par", "value"), State("zcb-mat", "value"), State("zcb-rate", "value"),
    State("vas-par", "value"), State("vas-r0", "value"),  State("vas-kappa", "value"),
    State("vas-theta","value"), State("vas-sigma","value"),State("vas-T",    "value"),
    prevent_initial_call=False
)
def update_bonds(_, par, mat, rate, vpar, r0, kappa, theta, sigma, T):
    par, mat, rate = par or 1000, mat or 5, rate or 0.05
    vpar, r0, kappa = vpar or 1000, r0 or 0.10, kappa or 0.30
    theta, sigma, T = theta or 0.30, sigma or 0.03, T or 1.0

    price = zcb_price(par, mat, rate)
    ytm   = zcb_ytm(par, price, mat)
    mats  = np.arange(1, 31)
    prices_curve = [zcb_price(par, m, rate) for m in mats]

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=mats, y=prices_curve, line=dict(color=ACCENT, width=2), fill="tozeroy",
                                fillcolor="rgba(0,212,170,0.1)", name="ZCB Price"))
    fig_z.add_vline(x=mat, line_dash="dash", line_color=ACCENT3, annotation_text=f"T={mat}")
    fig_z.update_layout(**PLOTLY_TEMPLATE["layout"], height=260,
                        xaxis_title="Maturity (years)", yaxis_title="Price ($)")

    bond_price, sim_data = vasicek_bond_price(vpar, r0, kappa, theta, sigma, T)
    fig_v = go.Figure()
    for i in range(min(50, sim_data.shape[1])):
        fig_v.add_trace(go.Scatter(y=sim_data.iloc[:, i], mode="lines",
                                    line=dict(width=0.5, color=ACCENT2), opacity=0.3, showlegend=False))
    mean_path = sim_data.mean(axis=1)
    fig_v.add_trace(go.Scatter(y=mean_path, line=dict(color=ACCENT, width=2.5), name="Mean Rate"))
    fig_v.add_hline(y=theta, line_dash="dot", line_color=ACCENT3, annotation_text=f"θ={theta}")
    fig_v.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                        xaxis_title="Time Steps", yaxis_title="Interest Rate")

    zcb_stats = html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}, children=[
        stat_box("PRICE",  f"${price:.2f}", ACCENT),
        stat_box("YTM",    f"{ytm*100:.3f}%", GREEN),
        stat_box("DISCOUNT", f"{rate*100:.1f}%", MUTED),
    ])
    vas_stats = html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}, children=[
        stat_box("BOND PRICE", f"${bond_price:.2f}", ACCENT2),
    ])
    return fig_z, fig_v, zcb_stats, vas_stats

# ── GBM / OU ──
@app.callback(
    Output("gbm-chart", "figure"), Output("ou-chart", "figure"),
    Input("gbm-btn", "n_clicks"),
    State("gbm-S0", "value"), State("gbm-mu", "value"), State("gbm-sig", "value"),
    State("gbm-T",  "value"), State("gbm-paths","value"),
    State("ou-r0",  "value"), State("ou-kappa","value"), State("ou-theta","value"),
    State("ou-sigma","value"),State("ou-T",    "value"), State("ou-paths","value"),
    prevent_initial_call=False
)
def update_gbm(_, S0, mu, sig, T, n_gbm, r0, kappa, theta, ou_sig, ou_T, n_ou):
    S0, mu, sig, T, n_gbm = S0 or 100, mu or 0.10, sig or 0.20, T or 2.0, int(n_gbm or 20)
    r0, kappa, theta = r0 or 0.10, kappa or 0.70, theta or 0.10
    ou_sig, ou_T, n_ou = ou_sig or 0.02, ou_T or 5.0, int(n_ou or 15)

    N_gbm = int(T * 252)
    paths = simulate_gbm(S0, mu, sig, T, N_gbm, n_gbm)
    t_gbm = np.linspace(0, T, N_gbm)

    fig_gbm = go.Figure()
    for p in paths:
        fig_gbm.add_trace(go.Scatter(x=t_gbm, y=p, mode="lines",
                                      line=dict(width=0.8, color=ACCENT2), opacity=0.5, showlegend=False))
    mean_path = np.mean(paths, axis=0)
    fig_gbm.add_trace(go.Scatter(x=t_gbm, y=mean_path, line=dict(color=ACCENT, width=2.5), name="Mean"))
    fig_gbm.update_layout(**PLOTLY_TEMPLATE["layout"], height=350,
                          xaxis_title="Time (years)", yaxis_title="Price S(t)")

    # OU process
    N_ou = int(ou_T * 252)
    dt = ou_T / N_ou
    ou_paths = []
    for _ in range(n_ou):
        rates = [r0]
        for _ in range(N_ou):
            dr = kappa * (theta - rates[-1]) * dt + ou_sig * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        ou_paths.append(rates)

    t_ou = np.linspace(0, ou_T, N_ou + 1)
    fig_ou = go.Figure()
    for p in ou_paths:
        fig_ou.add_trace(go.Scatter(x=t_ou, y=p, mode="lines",
                                     line=dict(width=0.8, color=ACCENT3), opacity=0.5, showlegend=False))
    mean_ou = np.mean(ou_paths, axis=0)
    fig_ou.add_trace(go.Scatter(x=t_ou, y=mean_ou, line=dict(color=ACCENT, width=2.5), name="Mean"))
    fig_ou.add_hline(y=theta, line_dash="dot", line_color=RED, annotation_text=f"θ={theta}")
    fig_ou.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                         xaxis_title="Time (years)", yaxis_title="r(t)")

    return fig_gbm, fig_ou

# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ▣  QF Terminal running at  http://127.0.0.1:8050\n")
    app.run(debug=False)
