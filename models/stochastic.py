# ─────────────────────────────────────────
#  MODEL: Stochastic Processes
#         Wiener · GBM · OU · Vasicek · Heston
# ─────────────────────────────────────────
"""Standard stochastic processes used in quantitative finance."""

from __future__ import annotations

import numpy as np

__all__ = [
    "wiener_process", "gbm", "ornstein_uhlenbeck", "vasicek", "heston",
]


# ── Wiener Process ────────────────────────

def wiener_process(n_steps: int = 1000, T: float = 10.0, n_paths: int = 1) -> np.ndarray:
    """
    Standard Brownian motion (Wiener process).

    Returns ndarray of shape ``(n_paths, n_steps + 1)``.
    """
    dt    = T / n_steps
    dW    = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 1:] = np.cumsum(dW, axis=1)
    return paths


# ── Geometric Brownian Motion ─────────────

def gbm(
    S0: float, mu: float, sigma: float,
    n_steps: int = 252, n_paths: int = 10, T: float = 1.0,
) -> np.ndarray:
    """
    Geometric Brownian Motion (vectorised, no per-path loop).

    Returns ndarray of shape ``(n_steps, n_paths)``.
    """
    dt  = T / n_steps
    t   = np.linspace(0, T, n_steps)
    dW  = np.random.standard_normal((n_steps, n_paths))
    W   = np.cumsum(dW, axis=0) * np.sqrt(dt)
    return S0 * np.exp((mu - 0.5 * sigma ** 2) * t[:, None] + sigma * W)


# ── Ornstein-Uhlenbeck Process ────────────

def ornstein_uhlenbeck(
    x0: float, kappa: float, theta: float, sigma: float,
    n_steps: int = 1000, n_paths: int = 10, T: float = 5.0,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck mean-reverting process (vectorised).

    Returns ndarray of shape ``(n_steps + 1, n_paths)``.
    """
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

def vasicek(
    r0: float, kappa: float, theta: float, sigma: float,
    n_steps: int = 252, n_paths: int = 50, T: float = 1.0,
) -> np.ndarray:
    """
    Vasicek short-rate model (vectorised).

    Returns ndarray of shape ``(n_steps + 1, n_paths)``.
    """
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

def heston(
    S0: float, mu: float, v0: float,
    kappa_v: float, theta_v: float, sigma_v: float, rho: float,
    n_steps: int = 252, n_paths: int = 50, T: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heston stochastic-volatility model.

    Returns
    -------
    S_paths : ndarray, shape ``(n_steps + 1, n_paths)``
    V_paths : ndarray, shape ``(n_steps + 1, n_paths)``
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
