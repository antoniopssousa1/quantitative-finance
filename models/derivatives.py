# ─────────────────────────────────────────
#  MODEL: Derivatives
# ─────────────────────────────────────────
"""Forward contracts, interest-rate swaps, CDS, and put-call parity."""

from __future__ import annotations

import numpy as np
from numpy import exp, sqrt, log
from scipy.stats import norm

__all__ = [
    "forward_price", "forward_value", "futures_pnl",
    "irs_fixed_rate", "irs_value",
    "cds_spread", "put_call_parity_check",
]


# ── Forward / Futures ─────────────────────

def forward_price(
    S: float, r: float, T: float, q: float = 0.0, storage: float = 0.0,
) -> float:
    """
    Theoretical forward price: F = S·e^{(r − q + u)·T}.

    Parameters
    ----------
    S       : Spot price.
    r       : Risk-free rate (continuous compounding).
    T       : Time to maturity (years).
    q       : Continuous dividend yield.
    storage : Storage cost rate (for commodities).
    """
    return S * exp((r - q + storage) * T)


def forward_value(S: float, K: float, r: float, T: float, position: str = "long") -> float:
    """Mark-to-market value of an existing forward contract."""
    V = S - K * exp(-r * T)
    return V if position == "long" else -V


def futures_pnl(
    entry_price: float, current_price: float, contract_size: int, position: str = "long",
) -> float:
    """Simple P&L of a futures position."""
    diff = current_price - entry_price
    pnl  = diff * contract_size
    return pnl if position == "long" else -pnl


# ── Interest Rate Swap ────────────────────

def irs_fixed_rate(r: float, n_periods: int, freq: int = 1) -> float:
    """
    Par fixed rate of a plain-vanilla IRS under a flat discount curve.

    Parameters
    ----------
    r         : Risk-free rate (annual).
    n_periods : Number of payment periods.
    freq      : Payments per year (default 1 = annual).
    """
    r_p = r / freq
    discount_factors = [1 / (1 + r_p) ** t for t in range(1, n_periods + 1)]
    numerator   = 1 - discount_factors[-1]
    denominator = sum(discount_factors) / freq
    return numerator / denominator if denominator != 0 else np.nan


def irs_value(
    notional: float, fixed_rate: float, market_rate: float,
    maturity: float, r: float, position: str = "pay_fixed",
) -> float:
    """
    Approximate mark-to-market value of an IRS.
    Pay-fixed: value = PV(float leg) − PV(fixed leg).
    """
    n_periods = int(maturity)
    annuity   = sum(1 / (1 + r) ** t for t in range(1, n_periods + 1))
    fixed_pv  = notional * fixed_rate * annuity + notional / (1 + r) ** n_periods
    float_pv  = notional  # floating leg ≈ par at reset
    value     = float_pv - fixed_pv
    return float(value) if position == "pay_fixed" else float(-value)


# ── Credit Default Swap (CDS) ─────────────

def cds_spread(
    hazard_rate: float, recovery_rate: float, maturity: float,
    r: float, dt: float = 0.25,
) -> float:
    """
    CDS par spread under a flat hazard-rate model.

    Parameters
    ----------
    hazard_rate   : λ (constant).
    recovery_rate : R (recovery fraction).
    maturity      : Years.
    r             : Risk-free rate (continuous).
    dt            : Payment frequency in years (0.25 = quarterly).
    """
    times = np.arange(dt, maturity + dt, dt)
    survival  = np.exp(-hazard_rate * times)
    discounts = np.exp(-r * times)

    # Protection leg PV
    default_probs = np.diff(np.concatenate([[1.0], survival]))
    protection_pv = np.sum((1 - recovery_rate) * np.abs(default_probs) * discounts)

    # Premium leg annuity factor
    premium_annuity = np.sum(dt * survival * discounts)

    return float(protection_pv / premium_annuity) if premium_annuity != 0 else np.nan


# ── Put-Call Parity ───────────────────────

def put_call_parity_check(
    call: float, put: float, S: float, K: float, r: float, T: float,
) -> float:
    """
    Check put-call parity: C − P = S − K·e^{−rT}.
    Returns the arbitrage residual (≈ 0 for fair pricing).
    """
    return (call - put) - (S - K * exp(-r * T))
