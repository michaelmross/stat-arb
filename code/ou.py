"""Ornstein-Uhlenbeck estimation for the spread.

dS = kappa*(mu - S) dt + sigma dW has exact discretization
    S_{t+1} = mu + e^{-kappa dt}(S_t - mu) + eps_t,
    eps_t ~ N(0, sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa)),
i.e. a Gaussian AR(1). Conditional MLE == OLS of S_{t+1} on S_t.

Half-life of mean reversion: ln(2)/kappa (in the same time units as dt).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class OUFit:
    kappa: float
    mu: float
    sigma: float
    half_life: float        # in dt units * (1/dt) -> reported in steps
    ar_coef: float
    stationary_std: float


def fit_ou(spread: np.ndarray, dt: float = 1 / 252) -> OUFit:
    s0, s1 = spread[:-1], spread[1:]
    X = np.column_stack([np.ones_like(s0), s0])
    coef, *_ = np.linalg.lstsq(X, s1, rcond=None)
    a, b = coef
    b = min(max(b, 1e-6), 1 - 1e-6)  # clamp to stationary region
    kappa = -np.log(b) / dt
    mu = a / (1.0 - b)
    resid = s1 - (a + b * s0)
    v = resid.var(ddof=2)
    sigma2 = v * 2.0 * kappa / (1.0 - b**2)
    sigma = np.sqrt(sigma2)
    half_life_steps = np.log(2.0) / kappa / dt
    stat_std = sigma / np.sqrt(2.0 * kappa)
    return OUFit(kappa=kappa, mu=mu, sigma=sigma,
                 half_life=half_life_steps, ar_coef=b,
                 stationary_std=stat_std)
