# ─────────────────────────────────────────
#  MODEL: Value-at-Risk
# ─────────────────────────────────────────
"""Parametric, historical, and Monte-Carlo VaR / CVaR (Expected Shortfall)."""

from __future__ import annotations

import numpy as np
from numpy import sqrt
from scipy.stats import norm

__all__ = [
    "var_parametric", "var_historical", "var_monte_carlo",
    "cvar_parametric", "cvar_monte_carlo",
]


def var_parametric(mu: float, sigma: float, c: float, n: int = 1) -> float:
    """
    Parametric (analytical) VaR as a **positive** return-fraction loss.

    Parameters
    ----------
    mu    : Mean daily return.
    sigma : Daily standard deviation.
    c     : Confidence level (e.g. 0.95).
    n     : Holding period in days.
    """
    return -(mu * n - sigma * sqrt(n) * norm.ppf(1 - c))


def var_historical(returns: np.ndarray, c: float, n: int = 1) -> float:
    """Historical-simulation VaR as a positive loss fraction."""
    sorted_r = np.sort(returns)
    idx      = int((1 - c) * len(sorted_r))
    return float(-sorted_r[max(idx, 0)] * sqrt(n))


def var_monte_carlo(
    mu: float, sigma: float, c: float, n: int = 1, iterations: int = 100_000,
) -> float:
    """Monte-Carlo VaR — returns scalar positive loss fraction."""
    rand = np.random.normal(0, 1, iterations)
    pnl  = mu * n + sigma * sqrt(n) * rand   # log-return distribution
    return float(-np.percentile(pnl, (1 - c) * 100))


def cvar_parametric(mu: float, sigma: float, c: float, n: int = 1) -> float:
    """Conditional VaR (Expected Shortfall) — parametric, positive loss fraction."""
    alpha = 1 - c
    return float(-(mu * n) + sigma * sqrt(n) * norm.pdf(norm.ppf(alpha)) / alpha)


def cvar_monte_carlo(
    mu: float, sigma: float, c: float, n: int = 1, iterations: int = 100_000,
) -> float:
    """CVaR (Expected Shortfall) via Monte Carlo — positive loss fraction."""
    rand   = np.random.normal(0, 1, iterations)
    pnl    = mu * n + sigma * sqrt(n) * rand
    cutoff = np.percentile(pnl, (1 - c) * 100)
    tail   = pnl[pnl <= cutoff]
    return float(-tail.mean()) if len(tail) > 0 else float(-cutoff)
