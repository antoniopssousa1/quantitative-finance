# ─────────────────────────────────────────
#  MODEL: Black-Scholes + Greeks
# ─────────────────────────────────────────
"""Closed-form Black-Scholes pricing, Greeks, IV solver, and Monte-Carlo pricer."""

from __future__ import annotations

import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import norm

__all__ = [
    "call_price", "put_price", "greeks",
    "implied_volatility", "mc_option_price",
]


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d₁ and d₂ for the Black-Scholes formula."""
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price: C = S·N(d₁) − K·e^{-rT}·N(d₂)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price: P = K·e^{-rT}·N(-d₂) − S·N(-d₁)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return float(-S * norm.cdf(-d1) + K * exp(-r * T) * norm.cdf(-d2))


def greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict[str, float]:
    """Return all first-order Greeks plus gamma for a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    call_delta =  norm.cdf(d1)
    put_delta  = -norm.cdf(-d1)
    gamma      =  norm.pdf(d1) / (S * sigma * sqrt(T))
    call_theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T))
                  - r * K * exp(-r * T) * norm.cdf(d2)) / 365
    put_theta  = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T))
                  + r * K * exp(-r * T) * norm.cdf(-d2)) / 365
    vega       =  S * norm.pdf(d1) * sqrt(T) / 100
    rho_call   =  K * T * exp(-r * T) * norm.cdf(d2)  / 100
    rho_put    = -K * T * exp(-r * T) * norm.cdf(-d2) / 100
    return dict(
        call_delta=call_delta, put_delta=put_delta,
        gamma=gamma,
        call_theta=call_theta, put_theta=put_theta,
        vega=vega,
        rho_call=rho_call, rho_put=rho_put,
    )


def implied_volatility(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Newton-Raphson implied-volatility solver."""
    sigma = 0.2
    for _ in range(max_iter):
        price    = call_price(S, K, T, r, sigma) if option_type == "call" else put_price(S, K, T, r, sigma)
        g        = greeks(S, K, T, r, sigma)
        vega_val = g["vega"] * 100   # un-scale
        if abs(vega_val) < 1e-10:
            break
        diff  = price - market_price
        sigma -= diff / vega_val
        sigma  = max(1e-6, sigma)
        if abs(diff) < tol:
            break
    return sigma


def mc_option_price(
    S: float, K: float, T: float, r: float, sigma: float,
    iterations: int = 100_000,
) -> tuple[float, float]:
    """Monte-Carlo European option pricing. Returns ``(call, put)``."""
    rand = np.random.normal(0, 1, iterations)
    ST   = S * np.exp(T * (r - 0.5 * sigma ** 2) + sigma * sqrt(T) * rand)
    call = exp(-r * T) * np.mean(np.maximum(ST - K, 0))
    put  = exp(-r * T) * np.mean(np.maximum(K - ST, 0))
    return call, put
