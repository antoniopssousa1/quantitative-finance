# ─────────────────────────────────────────
#  MODEL: Markowitz Portfolio Optimisation
# ─────────────────────────────────────────
"""Mean-variance optimisation, efficient frontier, max-Sharpe & min-vol portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.optimize as opt

__all__ = [
    "portfolio_performance",
    "generate_random_portfolios",
    "max_sharpe_portfolio",
    "min_volatility_portfolio",
    "efficient_frontier_curve",
]


def portfolio_performance(
    weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> tuple[float, float]:
    """Annualised ``(return, volatility)`` of a weighted portfolio."""
    ret = np.sum(mean_returns * weights) * trading_days
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * trading_days, weights)))
    return ret, vol


def generate_random_portfolios(
    mean_returns: np.ndarray, cov_matrix: np.ndarray,
    n: int = 3000, trading_days: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(returns, vols, sharpes, weights)`` for *n* random portfolios."""
    n_assets = len(mean_returns)
    results  = np.zeros((3, n))
    all_weights = []
    for i in range(n):
        w = np.random.random(n_assets)
        w /= w.sum()
        all_weights.append(w)
        r, v  = portfolio_performance(w, mean_returns, cov_matrix, trading_days)
        results[0, i] = r
        results[1, i] = v
        results[2, i] = r / v   # crude Sharpe (rf=0)
    return results[0], results[1], results[2], np.array(all_weights)


def max_sharpe_portfolio(
    mean_returns: np.ndarray, cov_matrix: np.ndarray,
    rf: float = 0.05, trading_days: int = 252,
) -> opt.OptimizeResult:
    """Maximise the Sharpe ratio via SLSQP (long-only, fully-invested)."""
    n = len(mean_returns)

    def neg_sharpe(w):
        r, v = portfolio_performance(w, mean_returns, cov_matrix, trading_days)
        return -(r - rf) / v

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds      = tuple((0, 1) for _ in range(n))
    result      = opt.minimize(neg_sharpe, np.ones(n) / n,
                               method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def min_volatility_portfolio(
    mean_returns: np.ndarray, cov_matrix: np.ndarray, trading_days: int = 252,
) -> opt.OptimizeResult:
    """Minimise portfolio volatility via SLSQP (long-only, fully-invested)."""
    n = len(mean_returns)

    def portfolio_vol(w):
        return portfolio_performance(w, mean_returns, cov_matrix, trading_days)[1]

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds      = tuple((0, 1) for _ in range(n))
    result      = opt.minimize(portfolio_vol, np.ones(n) / n,
                               method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def efficient_frontier_curve(
    mean_returns: np.ndarray, cov_matrix: np.ndarray,
    n_points: int = 80, trading_days: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vols, rets)`` along the mean-variance efficient frontier."""
    n       = len(mean_returns)
    min_ret = float(mean_returns.min()) * trading_days
    max_ret = float(mean_returns.max()) * trading_days
    targets = np.linspace(min_ret, max_ret, n_points)
    vols, rets = [], []
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda x:    np.sum(x) - 1},
            {"type": "eq", "fun": lambda x, t=target:
             portfolio_performance(x, mean_returns, cov_matrix, trading_days)[0] - t},
        ]
        bounds = tuple((0, 1) for _ in range(n))
        res = opt.minimize(
            lambda w: portfolio_performance(w, mean_returns, cov_matrix, trading_days)[1],
            np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=constraints
        )
        if res.success:
            vols.append(res.fun)
            rets.append(target)
    return np.array(vols), np.array(rets)
