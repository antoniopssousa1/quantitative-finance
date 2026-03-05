# ─────────────────────────────────────────
#  MODEL: Derivatives
# ─────────────────────────────────────────

import numpy as np
from numpy import exp, sqrt, log
from scipy.stats import norm


# ── Forward / Futures ─────────────────────

def forward_price(S, r, T, q=0.0, storage=0.0):
    """
    S       : spot price
    r       : risk-free rate (continuous)
    T       : time to maturity (years)
    q       : dividend yield (continuous)
    storage : storage cost rate (for commodities)
    """
    return S * exp((r - q + storage) * T)


def forward_value(S, K, r, T, position="long"):
    """Mark-to-market value of an existing forward contract."""
    V = S - K * exp(-r * T)
    return V if position == "long" else -V


def futures_pnl(entry_price, current_price, contract_size, position="long"):
    diff = current_price - entry_price
    pnl  = diff * contract_size
    return pnl if position == "long" else -pnl


# ── Interest Rate Swap ────────────────────

def irs_fixed_rate(r, n_periods, freq=1):
    """
    Par fixed rate of a plain-vanilla IRS using flat discount curve.
    r         : risk-free rate per period
    n_periods : number of payment periods
    freq      : payments per year (default 1 = annual)
    """
    r_p = r / freq
    discount_factors = [1 / (1 + r_p) ** t for t in range(1, n_periods + 1)]
    numerator   = 1 - discount_factors[-1]
    denominator = sum(discount_factors) / freq
    return numerator / denominator if denominator != 0 else np.nan


def irs_value(notional, fixed_rate, market_rate, maturity, r, position="pay_fixed"):
    """
    Approximate mark-to-market value of an IRS.
    Pay-fixed: value = PV(float leg) - PV(fixed leg).
    """
    n_periods = int(maturity)
    annuity   = sum(1 / (1 + r) ** t for t in range(1, n_periods + 1))
    fixed_pv  = notional * fixed_rate * annuity + notional / (1 + r) ** n_periods
    float_pv  = notional  # floating leg ≈ par at reset
    value     = float_pv - fixed_pv
    return float(value) if position == "pay_fixed" else float(-value)


# ── Credit Default Swap (CDS) ─────────────

def cds_spread(hazard_rate, recovery_rate, maturity, r, dt=0.25):
    """
    CDS par spread using a flat hazard rate model.
    hazard_rate   : λ (constant)
    recovery_rate : R
    maturity      : years
    r             : risk-free rate (continuous)
    dt            : payment frequency (0.25 = quarterly)
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

def put_call_parity_check(call, put, S, K, r, T):
    """
    C - P = S - K·e^{-rT}
    Returns the arbitrage difference (should be ~0 for fair pricing).
    """
    return (call - put) - (S - K * exp(-r * T))
