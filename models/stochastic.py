# ─────────────────────────────────────────
#  MODEL: Stochastic Processes
#         Wiener · GBM · OU · Vasicek
# ─────────────────────────────────────────

import numpy as np


# ── Wiener Process ────────────────────────

def wiener_process(n_steps=1000, T=10.0, n_paths=1):
    """Returns paths array of shape (n_paths, n_steps+1)."""
    dt    = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    for i in range(n_paths):
        paths[i, 1:] = np.cumsum(np.random.normal(0, np.sqrt(dt), n_steps))
    return paths


# ── Geometric Brownian Motion ─────────────

def gbm(S0, mu, sigma, n_steps=252, n_paths=10, T=1.0):
    """
    Returns price paths array of shape (n_steps, n_paths).
    Tab usage: gbm(S0, mu, sig, N_gbm, paths) — positional.
    """
    dt    = T / n_steps
    t     = np.linspace(0, T, n_steps)
    paths = np.zeros((n_steps, n_paths))
    for i in range(n_paths):
        W          = np.cumsum(np.random.standard_normal(n_steps)) * np.sqrt(dt)
        paths[:, i] = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
    return paths   # shape (n_steps, n_paths)


# ── Ornstein-Uhlenbeck Process ────────────

def ornstein_uhlenbeck(x0, kappa, theta, sigma, n_steps=1000, n_paths=10, T=5.0):
    """Returns paths array of shape (n_steps+1, n_paths)."""
    dt    = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = x0
    for j in range(1, n_steps + 1):
        dW          = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[j, :] = (paths[j - 1, :]
                       + kappa * (theta - paths[j - 1, :]) * dt
                       + sigma * dW)
    return paths   # shape (n_steps+1, n_paths)


# ── Vasicek Interest Rate Model ───────────

def vasicek(r0, kappa, theta, sigma, n_steps=252, n_paths=50, T=1.0):
    """Returns paths array of shape (n_steps+1, n_paths)."""
    dt    = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = r0
    for j in range(1, n_steps + 1):
        dW          = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[j, :] = (paths[j - 1, :]
                       + kappa * (theta - paths[j - 1, :]) * dt
                       + sigma * dW)
    return paths   # shape (n_steps+1, n_paths)


# ── Heston Stochastic Volatility ──────────

def heston(S0, mu, v0, kappa_v, theta_v, sigma_v, rho, n_steps=252, n_paths=50, T=1.0):
    """
    Heston stochastic volatility model.
    Args match tab call: heston(S0, mu, v0, kv, tv, sv, rho, N, paths)
    Returns (S_paths, V_paths) each of shape (n_steps+1, n_paths).
    """
    dt      = T / n_steps
    S_paths = np.zeros((n_steps + 1, n_paths))
    V_paths = np.zeros((n_steps + 1, n_paths))
    S_paths[0, :] = S0
    V_paths[0, :] = v0

    for j in range(1, n_steps + 1):
        z1 = np.random.normal(0, 1, n_paths)
        z2 = np.random.normal(0, 1, n_paths)
        zv = z1
        zs = rho * z1 + np.sqrt(max(1 - rho ** 2, 0)) * z2
        V  = np.maximum(V_paths[j - 1, :], 0)
        V_paths[j, :] = np.maximum(
            V + kappa_v * (theta_v - V) * dt + sigma_v * np.sqrt(V * dt) * zv, 0
        )
        S_paths[j, :] = S_paths[j - 1, :] * np.exp(
            (mu - 0.5 * V) * dt + np.sqrt(V * dt) * zs
        )
    return S_paths, V_paths   # each (n_steps+1, n_paths)
