# ─────────────────────────────────────────
#  MODEL: Bonds
# ─────────────────────────────────────────

import numpy as np


# ── Zero-Coupon Bond ──────────────────────

def zcb_price(face, maturity, rate):
    return face / (1 + rate) ** maturity


def zcb_ytm(face, price, maturity):
    return (face / price) ** (1 / maturity) - 1


def zcb_duration(maturity):
    """Macaulay duration of a ZCB equals its maturity."""
    return float(maturity)


# ── Coupon Bond ───────────────────────────

def coupon_bond_price(face, coupon_rate, maturity, ytm):
    """
    face        : face value
    coupon_rate : annual coupon rate (decimal, e.g. 0.05)
    maturity    : years to maturity (int)
    ytm         : yield to maturity (decimal)
    """
    coupon = face * coupon_rate
    price  = sum(coupon / (1 + ytm) ** t for t in range(1, maturity + 1))
    price += face / (1 + ytm) ** maturity
    return price


def macaulay_duration(face, coupon_rate, maturity, ytm):
    coupon = face * coupon_rate
    price  = coupon_bond_price(face, coupon_rate, maturity, ytm)
    weighted = sum(t * (coupon / (1 + ytm) ** t) for t in range(1, maturity + 1))
    weighted += maturity * (face / (1 + ytm) ** maturity)
    return weighted / price


def modified_duration(face, coupon_rate, maturity, ytm):
    return macaulay_duration(face, coupon_rate, maturity, ytm) / (1 + ytm)


def bond_convexity(face, coupon_rate, maturity, ytm):
    coupon = face * coupon_rate
    price  = coupon_bond_price(face, coupon_rate, maturity, ytm)
    conv   = sum(t * (t + 1) * (coupon / (1 + ytm) ** (t + 2))
                 for t in range(1, maturity + 1))
    conv  += maturity * (maturity + 1) * (face / (1 + ytm) ** (maturity + 2))
    return conv / price


def yield_curve(face, coupon_rate, maturity, price_target):
    """Solve for YTM numerically given a market price."""
    from scipy.optimize import brentq
    f = lambda y: coupon_bond_price(face, coupon_rate, maturity, y) - price_target
    try:
        return brentq(f, 0.0001, 0.9999)
    except ValueError:
        return np.nan


# ── Vasicek Bond Pricing (Monte-Carlo) ────

def vasicek_bond_mc(face, r0, kappa, theta, sigma, T=1.0, n_sims=1000, n_steps=200):
    dt = T / n_steps
    paths = []
    for _ in range(n_sims):
        r = r0
        rates = [r]
        for _ in range(n_steps):
            dr = kappa * (theta - r) * dt + sigma * np.sqrt(dt) * np.random.normal()
            r += dr
            rates.append(r)
        paths.append(rates)
    sim = np.array(paths)                  # (n_sims, n_steps+1)
    integral = sim[:, :-1].sum(axis=1) * dt
    bond_price = face * np.mean(np.exp(-integral))
    return bond_price, sim
