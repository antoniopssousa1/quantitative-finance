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


def sharpe_ratio(returns: pd.Series, rf_daily: float = 0.0):
    excess = returns - rf_daily
    return float(excess.mean() / excess.std() * np.sqrt(252))


def treynor_ratio(annual_return: float, beta: float, rf: float):
    """Treynor = (Rp - Rf) / beta"""
    return (annual_return - rf) / beta if beta != 0 else np.nan


def information_ratio(stock_returns: pd.Series, benchmark_returns: pd.Series):
    active = stock_returns - benchmark_returns
    return float(active.mean() / active.std() * np.sqrt(252)) if active.std() != 0 else np.nan


def rolling_beta(stock_returns: pd.Series, market_returns: pd.Series, window: int = 36):
    """Rolling beta over a given window (months)."""
    betas = []
    for i in range(window, len(stock_returns) + 1):
        s = stock_returns.iloc[i - window:i]
        m = market_returns.iloc[i - window:i]
        b, _ = np.polyfit(m, s, 1)
        betas.append(b)
    idx = stock_returns.index[window - 1:]
    return pd.Series(betas, index=idx)
