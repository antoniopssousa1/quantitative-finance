# ─────────────────────────────────────────
#  MODEL: Black-Scholes + Greeks
# ─────────────────────────────────────────

import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2), d1, d2


def put_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -S * norm.cdf(-d1) + K * exp(-r * T) * norm.cdf(-d2), d1, d2


def greeks(S, K, T, r, sigma):
    """Returns dict of all Greeks for a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    delta_call =  norm.cdf(d1)
    delta_put  = -norm.cdf(-d1)
    gamma      =  norm.pdf(d1) / (S * sigma * sqrt(T))
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T))
                  - r * K * exp(-r * T) * norm.cdf(d2)) / 365
    theta_put  = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T))
                  + r * K * exp(-r * T) * norm.cdf(-d2)) / 365
    vega       =  S * norm.pdf(d1) * sqrt(T) / 100
    rho_call   =  K * T * exp(-r * T) * norm.cdf(d2)  / 100
    rho_put    = -K * T * exp(-r * T) * norm.cdf(-d2) / 100
    return dict(
        delta_call=delta_call, delta_put=delta_put,
        gamma=gamma,
        theta_call=theta_call, theta_put=theta_put,
        vega=vega,
        rho_call=rho_call, rho_put=rho_put,
    )


def implied_volatility(market_price, S, K, T, r, option_type="call", tol=1e-6, max_iter=200):
    """Newton-Raphson implied volatility solver."""
    sigma = 0.2
    for _ in range(max_iter):
        if option_type == "call":
            price, _, _ = call_price(S, K, T, r, sigma)
        else:
            price, _, _ = put_price(S, K, T, r, sigma)
        g = greeks(S, K, T, r, sigma)
        vega_val = g["vega"] * 100  # un-scale
        if abs(vega_val) < 1e-10:
            break
        diff = price - market_price
        sigma -= diff / vega_val
        sigma = max(1e-6, sigma)
        if abs(diff) < tol:
            break
    return sigma


def mc_option_price(S, K, T, r, sigma, iterations=100_000):
    """Monte-Carlo option pricing (European call & put)."""
    rand = np.random.normal(0, 1, iterations)
    ST   = S * np.exp(T * (r - 0.5 * sigma ** 2) + sigma * sqrt(T) * rand)
    call = exp(-r * T) * np.mean(np.maximum(ST - K, 0))
    put  = exp(-r * T) * np.mean(np.maximum(K - ST, 0))
    return call, put
