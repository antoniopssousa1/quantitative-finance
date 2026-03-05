# ─────────────────────────────────────────
#  MODEL: Stochastic Processes
#         Wiener · GBM · OU · Vasicek
# ─────────────────────────────────────────

import numpy as np


# ── Wiener Process ────────────────────────

def wiener_process(T=10.0, n_steps=1000, n_paths=1, dt=None):
    dt     = dt or T / n_steps
    t      = np.linspace(0, T, n_steps + 1)
    paths  = np.zeros((n_paths, n_steps + 1))
    for i in range(n_paths):
        increments       = np.random.normal(0, np.sqrt(dt), n_steps)
        paths[i, 1:]     = np.cumsum(increments)
    return t, paths


# ── Geometric Brownian Motion ─────────────

def gbm(S0, mu, sigma, T=1.0, n_steps=252, n_paths=10):
    dt   = T / n_steps
    t    = np.linspace(0, T, n_steps)
    paths = []
    for _ in range(n_paths):
        W = np.cumsum(np.random.standard_normal(n_steps)) * np.sqrt(dt)
        X = (mu - 0.5 * sigma ** 2) * t + sigma * W
        paths.append(S0 * np.exp(X))
    return t, np.array(paths)


# ── Ornstein-Uhlenbeck Process ────────────

def ornstein_uhlenbeck(x0, kappa, theta, sigma, T=5.0, n_steps=1000, n_paths=10):
    dt    = T / n_steps
    t     = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for i in range(n_paths):
        for j in range(1, n_steps + 1):
            dx = kappa * (theta - paths[i, j - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            paths[i, j] = paths[i, j - 1] + dx
    return t, paths


# ── Vasicek Interest Rate Model ───────────

def vasicek(r0, kappa, theta, sigma, T=1.0, n_steps=252, n_paths=50):
    dt    = T / n_steps
    t     = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0
    for i in range(n_paths):
        for j in range(1, n_steps + 1):
            dr = kappa * (theta - paths[i, j - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            paths[i, j] = paths[i, j - 1] + dr
    return t, paths


# ── Heston Stochastic Volatility (NEW) ────

def heston(S0, V0, mu, kappa, theta, sigma_v, rho, T=1.0, n_steps=252, n_paths=50):
    """
    Heston model: stochastic volatility correlated with price.
    V0      : initial variance
    kappa   : mean reversion speed of variance
    theta   : long-run variance
    sigma_v : vol-of-vol
    rho     : correlation between price and variance Brownian motions
    """
    dt     = T / n_steps
    t      = np.linspace(0, T, n_steps + 1)
    S_paths = np.zeros((n_paths, n_steps + 1))
    V_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    V_paths[:, 0] = V0

    for i in range(n_paths):
        for j in range(1, n_steps + 1):
            z1 = np.random.normal()
            z2 = np.random.normal()
            zv = z1
            zs = rho * z1 + np.sqrt(1 - rho ** 2) * z2
            V  = max(V_paths[i, j - 1], 0)
            V_paths[i, j] = (V + kappa * (theta - V) * dt
                              + sigma_v * np.sqrt(V * dt) * zv)
            V_paths[i, j] = max(V_paths[i, j], 0)
            S_paths[i, j] = S_paths[i, j - 1] * np.exp(
                (mu - 0.5 * V) * dt + np.sqrt(V * dt) * zs
            )
    return t, S_paths, V_paths
