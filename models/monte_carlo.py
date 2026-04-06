# ─────────────────────────────────────────
#  MODEL: Monte Carlo
# ─────────────────────────────────────────
"""GBM stock-price simulation and Monte-Carlo option pricing."""

from __future__ import annotations

import numpy as np
from numpy import sqrt, exp

__all__ = ["stock_price_mc", "bs_mc_price"]


def stock_price_mc(
    S0: float, mu: float, sigma: float, N: int = 252, n_sims: int = 500,
) -> np.ndarray:
    """
    Simulate GBM stock-price paths (vectorised).

    Returns
    -------
    ndarray, shape ``(N + 1, n_sims)``
        Row 0 = S₀, row N = terminal prices.
    """
    dt      = 1.0 / N
    rand    = np.random.normal(0, 1, (N, n_sims))
    log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand
    paths   = np.vstack([np.full(n_sims, S0), S0 * np.exp(np.cumsum(log_ret, axis=0))])
    return paths   # shape (N+1, n_sims)


def bs_mc_price(
    S: float, K: float, T: float, r: float, sigma: float,
    iterations: int = 100_000,
) -> tuple[float, float]:
    """Black-Scholes price via Monte Carlo (vectorised). Returns ``(call, put)``."""
    rand  = np.random.normal(0, 1, iterations)
    ST    = S * exp(T * (r - 0.5 * sigma ** 2) + sigma * sqrt(T) * rand)
    call  = float(exp(-r * T) * np.mean(np.maximum(ST - K, 0)))
    put   = float(exp(-r * T) * np.mean(np.maximum(K - ST, 0)))
    return call, put
