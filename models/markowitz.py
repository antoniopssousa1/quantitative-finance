# ─────────────────────────────────────────
#  MODEL: Markowitz Portfolio Optimisation
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import scipy.optimize as opt


def portfolio_performance(weights, mean_returns, cov_matrix, trading_days=252):
    ret = np.sum(mean_returns * weights) * trading_days
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * trading_days, weights)))
    return ret, vol


def generate_random_portfolios(mean_returns, cov_matrix, n=3000, trading_days=252):
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


def max_sharpe_portfolio(mean_returns, cov_matrix, rf=0.05, trading_days=252):
    n = len(mean_returns)

    def neg_sharpe(w):
        r, v = portfolio_performance(w, mean_returns, cov_matrix, trading_days)
        return -(r - rf) / v

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds      = tuple((0, 1) for _ in range(n))
    result      = opt.minimize(neg_sharpe, np.ones(n) / n,
                               method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def min_volatility_portfolio(mean_returns, cov_matrix, trading_days=252):
    n = len(mean_returns)

    def portfolio_vol(w):
        return portfolio_performance(w, mean_returns, cov_matrix, trading_days)[1]

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds      = tuple((0, 1) for _ in range(n))
    result      = opt.minimize(portfolio_vol, np.ones(n) / n,
                               method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def efficient_frontier_curve(mean_returns, cov_matrix, n_points=80, trading_days=252):
    """Return (vols, rets) along the efficient frontier."""
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
