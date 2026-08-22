"""Synthetic price-pair generators with known ground truth.

Four regimes:
  1. cointegrated:   common stochastic trend + OU spread (H1: tradable)
  2. random_walks:   independent random walks (H0: no relation)
  3. correlated_rw:  correlated increments but NO cointegration (the trap:
                     high correlation, spurious-regression bait)
  4. noisy_rw:       a slowly wandering (unit-root) spread buried under
                     white noise -- what real same-index ETF pairs look
                     like, and the null that breaks ADF lag selection

Regime 4 was added after the first real-data run (2026-08-21). The real
SPY/IVV log spread has a FLAT autocorrelation function -- 0.37 at lag 1 and
still 0.31 at lag 40, where an AR(1) at 0.37 would be 0.007 by lag 5 -- and
its yearly mean wanders -5.9bp -> +3.3bp -> -3.7bp. That is a unit root
under noise, i.e. NOT cointegrated, and it is the classic Schwert (1989)
MA size-distortion case. Regimes 1-3 all emit clean innovations, so none of
them exercises it, and the original validation suite passed while the test
was in fact rejecting this null a third of the time. See the lag-policy
cell in validate.py.

All prices are log prices. Ground-truth parameters are returned so the
estimation pipeline can be scored against them.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PairTruth:
    kind: str                 # 'cointegrated'|'random_walks'|'correlated_rw'|'noisy_rw'
    beta: float | None = None # true hedge ratio (cointegrated only)
    kappa: float | None = None
    mu: float | None = None
    sigma: float | None = None


def simulate_ou(n: int, kappa: float, mu: float, sigma: float,
                dt: float, rng: np.random.Generator, s0: float | None = None):
    """Exact discretization of dS = kappa*(mu - S)dt + sigma dW."""
    s = np.empty(n)
    a = np.exp(-kappa * dt)
    # stationary variance sigma^2 / (2 kappa); innovation variance:
    v = sigma**2 * (1.0 - a**2) / (2.0 * kappa)
    s[0] = mu + rng.normal(0, np.sqrt(sigma**2 / (2 * kappa))) if s0 is None else s0
    eps = rng.normal(0.0, np.sqrt(v), size=n - 1)
    for t in range(1, n):
        s[t] = mu + a * (s[t - 1] - mu) + eps[t - 1]
    return s


def add_micro_noise(logp, logq, bp: float, rng: np.random.Generator):
    """Independent iid observation noise on each leg, in basis points.

    Bid-ask bounce plus non-synchronous closing prints. This is what makes
    an observed ETF spread ARMA rather than pure OU, and the MA component
    is exactly what drives ADF lag selection. Real same-index pairs carry
    roughly 3bp per leg.
    """
    sd = bp / 1e4
    return (logp + rng.normal(0, sd, len(logp)),
            logq + rng.normal(0, sd, len(logq)))


def make_pair(kind: str, n: int = 1500, dt: float = 1 / 252,
              rng: np.random.Generator | None = None,
              trend_vol: float = 0.18,      # annualized vol of common trend
              beta: float = 1.0,
              kappa: float = 25.0,          # half-life ~ 7 trading days
              spread_sigma: float = 0.05,   # annualized OU vol
              idio_vol: float = 0.02,       # small idiosyncratic obs noise
              rw_corr: float = 0.95,
              spread_sd_bp: float = 5.28,   # noisy_rw: total spread width
              persist_share: float = 0.35): # noisy_rw: unit-root var share
    """Return (logP, logQ, PairTruth)."""
    rng = rng or np.random.default_rng()
    sd = np.sqrt(dt)

    if kind == "cointegrated":
        trend = np.cumsum(rng.normal(0, trend_vol * sd, n)) + np.log(100.0)
        spread = simulate_ou(n, kappa, mu=0.0, sigma=spread_sigma, dt=dt, rng=rng)
        q = trend + rng.normal(0, idio_vol * sd, n).cumsum() * 0.0  # Q carries the trend
        p = beta * q + spread + 0.1  # constant offset absorbed by intercept
        return p, q, PairTruth(kind, beta=beta, kappa=kappa, mu=0.0,
                               sigma=spread_sigma)

    if kind == "random_walks":
        p = np.cumsum(rng.normal(0, trend_vol * sd, n)) + np.log(100.0)
        q = np.cumsum(rng.normal(0, trend_vol * sd, n)) + np.log(100.0)
        return p, q, PairTruth(kind)

    if kind == "noisy_rw":
        # Spread = persistent unit-root component + white noise, calibrated
        # to a real same-index pair (default: SPY/IVV, 5.28bp wide, ACF
        # flat near 0.35). NOT cointegrated -- the wander never reverts.
        V = (spread_sd_bp / 1e4) ** 2
        var_rw, var_noise = persist_share * V, (1.0 - persist_share) * V
        q = np.cumsum(rng.normal(0, trend_vol * sd, n)) + np.log(100.0)
        wander = np.cumsum(rng.normal(0.0, 1.0, n))
        # Residualise the walk against q BEFORE scaling: two independent
        # random walks regress spuriously on one another, so the EG step's
        # OLS would otherwise absorb most of the persistent component and
        # leave a near-white residual (ACF 0.08 instead of the target 0.35).
        X = np.column_stack([np.ones(n), q])
        coef, *_ = np.linalg.lstsq(X, wander, rcond=None)
        wander = wander - X @ coef
        sd_w = wander.std()
        if sd_w > 0:
            wander = wander * (np.sqrt(var_rw) / sd_w)
        p = q + wander + rng.normal(0, np.sqrt(var_noise), n)
        return p, q, PairTruth(kind)

    if kind == "correlated_rw":
        z1 = rng.normal(0, 1, n)
        z2 = rw_corr * z1 + np.sqrt(1 - rw_corr**2) * rng.normal(0, 1, n)
        p = np.cumsum(trend_vol * sd * z1) + np.log(100.0)
        q = np.cumsum(trend_vol * sd * z2) + np.log(100.0)
        return p, q, PairTruth(kind)

    raise ValueError(f"unknown kind: {kind}")
