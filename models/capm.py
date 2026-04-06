# ─────────────────────────────────────────
#  MODEL: CAPM
# ─────────────────────────────────────────
"""Capital Asset Pricing Model — beta, alpha, performance ratios, rolling beta."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "compute_beta_alpha", "expected_return_capm", "r_squared",
    "sharpe_ratio", "treynor_ratio", "information_ratio", "rolling_beta",
]


def compute_beta_alpha(
    stock_returns: np.ndarray | pd.Series,
    market_returns: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """OLS regression: Rₛ = α + β·Rₘ. Returns ``(beta, alpha)``."""
    beta, alpha = np.polyfit(market_returns, stock_returns, deg=1)
    return beta, alpha


def expected_return_capm(beta: float, rf: float, market_annual_return: float) -> float:
    """CAPM expected return: E[R] = rᶠ + β·(E[Rₘ] − rᶠ)."""
    return rf + beta * (market_annual_return - rf)


def r_squared(
    stock_returns: np.ndarray | pd.Series,
    market_returns: np.ndarray | pd.Series,
) -> float:
    """Coefficient of determination (R²) between stock and market returns."""
    return float(np.corrcoef(market_returns, stock_returns)[0, 1] ** 2)


def sharpe_ratio(annual_return: float, rf: float, annual_vol: float) -> float:
    """Sharpe ratio: (Rₚ − Rᶠ) / σₚ."""
    return (annual_return - rf) / annual_vol if annual_vol != 0 else np.nan


def treynor_ratio(annual_return: float, rf: float, beta: float) -> float:
    """Treynor ratio: (Rₚ − Rᶠ) / β."""
    return (annual_return - rf) / beta if beta != 0 else np.nan


def information_ratio(
    stock_returns: np.ndarray, benchmark_returns: np.ndarray,
) -> float:
    """Annualised information ratio of active returns."""
    active = np.asarray(stock_returns) - np.asarray(benchmark_returns)
    std    = active.std()
    return float(active.mean() / std * np.sqrt(252)) if std != 0 else np.nan


def rolling_beta(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
    window: int = 36,
) -> np.ndarray:
    """Rolling OLS beta over a sliding window. Returns 1-D numpy array."""
    s = np.asarray(stock_returns)
    m = np.asarray(market_returns)
    betas = []
    for i in range(window, len(s) + 1):
        b, _ = np.polyfit(m[i - window:i], s[i - window:i], 1)
        betas.append(b)
    return np.array(betas)
