# ─────────────────────────────────────────
#  MODEL: Bonds
# ─────────────────────────────────────────

import numpy as np


# ── Zero-Coupon Bond ──────────────────────

def zcb_price(face, maturity, rate):
    return float(face / (1 + rate) ** maturity)


def zcb_ytm(price, face, maturity):
    """YTM given market price. Args: price, face, maturity."""
    return float((face / price) ** (1 / maturity) - 1)


def zcb_duration(maturity):
    """Macaulay duration of a ZCB equals its maturity."""
    return float(maturity)


# ── Coupon Bond ───────────────────────────

def coupon_bond_price(face, coupon_rate, freq, maturity, ytm):
    """
    face        : face value
    coupon_rate : annual coupon rate (decimal, e.g. 0.05)
    freq        : coupon payments per year (e.g. 2 for semi-annual)
    maturity    : years to maturity
    ytm         : annual yield to maturity (decimal)
    """
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    r_per  = ytm / freq
    price  = sum(coupon / (1 + r_per) ** t for t in range(1, n + 1))
    price += face / (1 + r_per) ** n
    return float(price)


def macaulay_duration(face, coupon_rate, freq, maturity, ytm):
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    r_per  = ytm / freq
    price  = coupon_bond_price(face, coupon_rate, freq, maturity, ytm)
    weighted = sum((t / freq) * (coupon / (1 + r_per) ** t) for t in range(1, n + 1))
    weighted += maturity * (face / (1 + r_per) ** n)
    return float(weighted / price)


def modified_duration(face, coupon_rate, freq, maturity, ytm):
    return macaulay_duration(face, coupon_rate, freq, maturity, ytm) / (1 + ytm / freq)


def bond_convexity(face, coupon_rate, freq, maturity, ytm):
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    r_per  = ytm / freq
    price  = coupon_bond_price(face, coupon_rate, freq, maturity, ytm)
    conv   = sum(t * (t + 1) * (coupon / (1 + r_per) ** (t + 2)) for t in range(1, n + 1))
    conv  += n * (n + 1) * (face / (1 + r_per) ** (n + 2))
    return float(conv / (price * freq ** 2))


def yield_curve(face, coupon_rate, maturity, price_target):
    """Solve for YTM numerically given a market price."""
    from scipy.optimize import brentq
    f = lambda y: coupon_bond_price(face, coupon_rate, maturity, y) - price_target
    try:
        return brentq(f, 0.0001, 0.9999)
    except ValueError:
        return np.nan


# ── Vasicek Bond Pricing (Monte-Carlo) ────

def vasicek_bond_mc(r0, kappa, theta, sigma, T=1.0, n_steps=200, n_paths=200, face=1000):
    """
    Simulate Vasicek short-rate paths and price a ZCB.
    Returns (paths_array, bond_price) where paths_array has shape (n_paths, n_steps+1).
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
