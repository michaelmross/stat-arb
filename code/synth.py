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


def make_margin(n: int, rng: np.random.Generator, kind: str = "ou",
                half_life: float = 20.0, level_sd: float = 10.4,
                roll_every: int = 21, roll_offset_sd: float = 2.14,
                roll_offset_mean: float = -0.23, dt: float = 1 / 252):
    """Roll-gap null: a production margin observed through stitched fronts.

    This is the futures universe's `noisy_rw` -- the structure real series
    have that the clean generators lack. Returns (observed, roll_mask,
    truth_dict).

      kind='ou'  mean-reverting margin (H1: a real physical tether)
      kind='rw'  random-walk margin    (H0: no tether)

    Contract offsets are PIECEWISE CONSTANT, not cumulative. Modelling
    each roll as an increment to a running sum would make the observed
    series contain a genuine random walk: ~200 rolls at the crack's
    observed step sd of 2.14 gives a cumulative component of sd ~30
    against a margin sd of 10.4, which would swamp everything. Real crack
    margins plainly do not wander like that, because the roll step is the
    calendar spread -- itself mean-reverting and seasonal. So each
    contract carries an iid offset that RESETS at the next roll:

        observed_t = margin_t + offset_{contract(t)}

    Defaults are calibrated to the real 3-2-1 crack (level sd 10.4 $/bbl,
    roll-step sd 2.14, mean -0.23 reflecting average contango).
    """
    kappa = np.log(2.0) / (half_life * dt)
    if kind == "ou":
        sigma = level_sd * np.sqrt(2.0 * kappa)
        margin = simulate_ou(n, kappa, mu=0.0, sigma=sigma, dt=dt, rng=rng)
    elif kind == "rw":
        # scaled so the sample sd is comparable to the OU case
        step = level_sd / np.sqrt(n / 6.0)
        margin = np.cumsum(rng.normal(0.0, step, n))
        margin -= margin.mean()
    else:
        raise ValueError(f"unknown kind: {kind}")

    roll_idx = np.arange(roll_every, n, roll_every)
    mask = np.zeros(n, dtype=bool)
    mask[roll_idx] = True
    offsets = np.zeros(n)
    bounds = np.concatenate(([0], roll_idx, [n]))
    for i in range(len(bounds) - 1):
        offsets[bounds[i]:bounds[i + 1]] = rng.normal(roll_offset_mean,
                                                      roll_offset_sd)
    return margin + offsets, mask, dict(kind=kind, half_life=half_life,
                                        level_sd=level_sd, n_rolls=len(roll_idx))


def dejump(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove roll-day increments: diff, zero the masked steps, re-cumulate.

    For TESTING only. This is not back-adjustment of a traded series --
    the level (the anchor) is untouched in `futures.build_margin`. It just
    stops contract-offset steps from being read as margin dynamics.
    """
    d = np.diff(x)
    d[mask[1:]] = 0.0
    return np.concatenate(([x[0]], x[0] + np.cumsum(d)))


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
