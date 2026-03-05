# ─────────────────────────────────────────
#  MODEL: Value-at-Risk
# ─────────────────────────────────────────

import numpy as np
from numpy import sqrt
from scipy.stats import norm


def var_parametric(position, c, mu, sigma, n=1):
    """Analytical (parametric) VaR."""
    return position * (mu * n - sigma * sqrt(n) * norm.ppf(1 - c))


def var_historical(returns: np.ndarray, position: float, c: float):
    """Historical simulation VaR."""
    sorted_r = np.sort(returns)
    idx      = int((1 - c) * len(sorted_r))
    return -position * sorted_r[idx]


def var_monte_carlo(position, mu, sigma, c, n=1, iterations=100_000):
    """Monte-Carlo VaR — returns (VaR, array of simulated P&L)."""
    rand      = np.random.normal(0, 1, iterations)
    ST        = position * np.exp(n * (mu - 0.5 * sigma ** 2) + sigma * sqrt(n) * rand)
    pnl       = ST - position
    pnl_sorted = np.sort(pnl)
    var        = -np.percentile(pnl_sorted, (1 - c) * 100)
    return var, pnl_sorted


def cvar_parametric(position, c, mu, sigma, n=1):
    """Conditional VaR (Expected Shortfall) — parametric."""
    alpha = 1 - c
    return position * (-mu * n + sigma * sqrt(n) * norm.pdf(norm.ppf(alpha)) / alpha)


def cvar_monte_carlo(pnl: np.ndarray, c: float):
    """CVaR from a simulated P&L array."""
    cutoff = np.percentile(pnl, (1 - c) * 100)
    tail   = pnl[pnl <= cutoff]
    return float(-tail.mean()) if len(tail) > 0 else np.nan
