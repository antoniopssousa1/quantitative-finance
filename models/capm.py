# ─────────────────────────────────────────
#  MODEL: CAPM
# ─────────────────────────────────────────

import numpy as np
import pandas as pd


def compute_beta_alpha(stock_returns: pd.Series, market_returns: pd.Series):
    """OLS regression: stock = alpha + beta * market."""
    beta, alpha = np.polyfit(market_returns, stock_returns, deg=1)
    return beta, alpha


def expected_return_capm(beta, rf, market_annual_return):
    """E[R] = rf + beta * (E[Rm] - rf)"""
    return rf + beta * (market_annual_return - rf)


def r_squared(stock_returns: pd.Series, market_returns: pd.Series):
    return float(np.corrcoef(market_returns, stock_returns)[0, 1] ** 2)


def sharpe_ratio(annual_return: float, rf: float, annual_vol: float):
    """Sharpe = (Rp - Rf) / σ"""
    return (annual_return - rf) / annual_vol if annual_vol != 0 else np.nan


def treynor_ratio(annual_return: float, rf: float, beta: float):
    """Treynor = (Rp - Rf) / beta"""
    return (annual_return - rf) / beta if beta != 0 else np.nan


def information_ratio(stock_returns, benchmark_returns):
    active = np.asarray(stock_returns) - np.asarray(benchmark_returns)
    std    = active.std()
    return float(active.mean() / std * np.sqrt(252)) if std != 0 else np.nan


def rolling_beta(stock_returns, market_returns, window: int = 36):
    """Rolling beta over a given window. Accepts numpy arrays or Series."""
    s = np.asarray(stock_returns)
    m = np.asarray(market_returns)
    betas = []
    for i in range(window, len(s) + 1):
        b, _ = np.polyfit(m[i - window:i], s[i - window:i], 1)
        betas.append(b)
    return np.array(betas)
