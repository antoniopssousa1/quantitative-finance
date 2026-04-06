# ─────────────────────────────────────────
#  MODEL: Bonds
# ─────────────────────────────────────────
"""Zero-coupon bonds, coupon bonds, duration, convexity, and Vasicek MC pricing."""

from __future__ import annotations

import numpy as np

__all__ = [
    "zcb_price", "zcb_ytm", "zcb_duration",
    "coupon_bond_price", "macaulay_duration", "modified_duration", "bond_convexity",
    "yield_curve", "vasicek_bond_mc",
]


# ── Zero-Coupon Bond ──────────────────────

def zcb_price(face: float, maturity: float, rate: float) -> float:
    """Present value of a zero-coupon bond: PV = F / (1+r)^T."""
    return float(face / (1 + rate) ** maturity)


def zcb_ytm(price: float, face: float, maturity: float) -> float:
    """Yield-to-maturity of a ZCB: YTM = (F/P)^{1/T} − 1."""
    return float((face / price) ** (1 / maturity) - 1)


def zcb_duration(maturity: float) -> float:
    """Macaulay duration of a ZCB equals its maturity."""
    return float(maturity)


# ── Coupon Bond ───────────────────────────

def coupon_bond_price(
    face: float, coupon_rate: float, freq: int, maturity: float, ytm: float,
) -> float:
    """
    Price of a coupon-bearing bond.

    Parameters
    ----------
    face        : Face (par) value.
    coupon_rate : Annual coupon rate (e.g. 0.06 for 6 %).
    freq        : Coupon payments per year (2 = semi-annual).
    maturity    : Years to maturity.
    ytm         : Annual yield-to-maturity (decimal).
    """
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    r_per  = ytm / freq
    price  = sum(coupon / (1 + r_per) ** t for t in range(1, n + 1))
    price += face / (1 + r_per) ** n
    return float(price)


def macaulay_duration(
    face: float, coupon_rate: float, freq: int, maturity: float, ytm: float,
) -> float:
    """Macaulay duration (years) of a coupon bond."""
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    r_per  = ytm / freq
    price  = coupon_bond_price(face, coupon_rate, freq, maturity, ytm)
    weighted = sum((t / freq) * (coupon / (1 + r_per) ** t) for t in range(1, n + 1))
    weighted += maturity * (face / (1 + r_per) ** n)
    return float(weighted / price)


def modified_duration(
    face: float, coupon_rate: float, freq: int, maturity: float, ytm: float,
) -> float:
    """Modified duration = Macaulay duration / (1 + ytm/freq)."""
    return macaulay_duration(face, coupon_rate, freq, maturity, ytm) / (1 + ytm / freq)


def bond_convexity(
    face: float, coupon_rate: float, freq: int, maturity: float, ytm: float,
) -> float:
    """Dollar convexity of a coupon bond."""
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    r_per  = ytm / freq
    price  = coupon_bond_price(face, coupon_rate, freq, maturity, ytm)
    conv   = sum(t * (t + 1) * (coupon / (1 + r_per) ** (t + 2)) for t in range(1, n + 1))
    conv  += n * (n + 1) * (face / (1 + r_per) ** (n + 2))
    return float(conv / (price * freq ** 2))


def yield_curve(
    face: float, coupon_rate: float, freq: int, maturity: float, price_target: float,
) -> float:
    """Solve for YTM numerically (Brent's method) given a market price."""
    from scipy.optimize import brentq
    f = lambda y: coupon_bond_price(face, coupon_rate, freq, maturity, y) - price_target
    try:
        return brentq(f, 0.0001, 0.9999)
    except ValueError:
        return np.nan


# ── Vasicek Bond Pricing (Monte-Carlo) ────

def vasicek_bond_mc(
    r0: float, kappa: float, theta: float, sigma: float,
    T: float = 1.0, n_steps: int = 200, n_paths: int = 200, face: float = 1000,
) -> tuple[np.ndarray, float]:
    """
    Price a zero-coupon bond under the Vasicek short-rate model via Monte Carlo.

    Returns
    -------
    paths : ndarray, shape ``(n_paths, n_steps + 1)``
        Simulated short-rate paths.
    price : float
        Estimated bond price = face × E[exp(−∫r dt)].
    """
    dt   = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0
    for j in range(1, n_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[:, j] = (paths[:, j - 1]
                       + kappa * (theta - paths[:, j - 1]) * dt
                       + sigma * dW)
    # Bond price = face * E[exp(-integral r dt)]
    integral   = paths[:, :-1].sum(axis=1) * dt
    bond_price = face * float(np.mean(np.exp(-integral)))
    return paths, bond_price
