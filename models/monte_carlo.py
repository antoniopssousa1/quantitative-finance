# ─────────────────────────────────────────
#  MODEL: Monte Carlo
# ─────────────────────────────────────────

import numpy as np
from numpy import sqrt, exp


def stock_price_mc(S0, mu, sigma, N=252, n_sims=500):
    """Simulate stock price paths and return DataFrame-like arrays."""
    import pandas as pd
    results = []
    for _ in range(n_sims):
        prices = [S0]
        for _ in range(N):
            prices.append(prices[-1] * np.exp(
                (mu - 0.5 * sigma ** 2) + sigma * np.random.normal()
            ))
        results.append(prices)
    df       = pd.DataFrame(results).T
    df["mean"] = df.mean(axis=1)
    return df


def bs_mc_price(S, K, T, r, sigma, iterations=100_000):
    """Black-Scholes price via Monte-Carlo (vectorised)."""
    rand  = np.random.normal(0, 1, iterations)
    ST    = S * exp(T * (r - 0.5 * sigma ** 2) + sigma * sqrt(T) * rand)
    call  = exp(-r * T) * np.mean(np.maximum(ST - K, 0))
    put   = exp(-r * T) * np.mean(np.maximum(K - ST, 0))
    return call, put, ST
