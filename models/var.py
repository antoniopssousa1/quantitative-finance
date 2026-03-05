# ─────────────────────────────────────────
#  MODEL: Value-at-Risk
# ─────────────────────────────────────────

import numpy as np
from numpy import sqrt
from scipy.stats import norm


def var_parametric(mu, sigma, c, n=1):
    """
    Parametric (analytical) VaR as a return fraction.
    Returns positive number representing the loss fraction.
    """
    return -(mu * n - sigma * sqrt(n) * norm.ppf(1 - c))


def var_historical(returns: np.ndarray, c: float, n: int = 1):
    """Historical simulation VaR as a positive loss fraction."""
    sorted_r = np.sort(returns)
    idx      = int((1 - c) * len(sorted_r))
    return float(-sorted_r[max(idx, 0)] * sqrt(n))


def var_monte_carlo(mu, sigma, c, n=1, iterations=100_000):
    """Monte-Carlo VaR — returns scalar positive loss fraction."""
    rand = np.random.normal(0, 1, iterations)
    pnl  = mu * n + sigma * sqrt(n) * rand   # log-return distribution
    return float(-np.percentile(pnl, (1 - c) * 100))


def cvar_parametric(mu, sigma, c, n=1):
    """Conditional VaR (Expected Shortfall) — parametric, positive loss fraction."""
    alpha = 1 - c
    return float(-(mu * n) + sigma * sqrt(n) * norm.pdf(norm.ppf(alpha)) / alpha)


def cvar_monte_carlo(mu, sigma, c, n=1, iterations=100_000):
    """CVaR via Monte Carlo — positive loss fraction."""
    rand   = np.random.normal(0, 1, iterations)
    pnl    = mu * n + sigma * sqrt(n) * rand
    cutoff = np.percentile(pnl, (1 - c) * 100)
    tail   = pnl[pnl <= cutoff]
    return float(-tail.mean()) if len(tail) > 0 else float(-cutoff)
