"""Statistics for the census: autocorrelation, OU half-life, block bootstrap,
Newey-West t-stats, and the Hayashi-Yoshida asynchronous covariance.

Everything here assumes OVERLAPPING, AUTOCORRELATED observations, because that
is what a tick-sampled residual gives you. Plain i.i.d. standard errors on this
data are wrong by a large factor, always in the direction of overstating
significance -- which is exactly how a null result gets mistaken for an edge.
"""
from __future__ import annotations

import math

import numpy as np


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation, lags 0..nlags."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < nlags + 2:
        return np.full(nlags + 1, np.nan)
    xd = x - x.mean()
    denom = float(np.dot(xd, xd))
    if denom == 0:
        return np.full(nlags + 1, np.nan)
    return np.array([float(np.dot(xd[: n - k], xd[k:])) / denom
                     for k in range(nlags + 1)])


def ar1_halflife(x: np.ndarray, dt_s: float):
    """Fit x_t = c + phi*x_{t-1} + e on a REGULARLY sampled series.

    Returns (phi, halflife_seconds). Half-life is None when phi is not in (0,1),
    i.e. when the series is not mean-reverting at this sampling frequency --
    which is a real answer, not a failure.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        return float("nan"), None
    y, lag = x[1:], x[:-1]
    A = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    phi = float(coef[1])
    if not (0.0 < phi < 1.0):
        return phi, None
    return phi, -math.log(2.0) * dt_s / math.log(phi)


def newey_west_se(x: np.ndarray, lags: int | None = None) -> float:
    """HAC standard error of the sample mean (Bartlett kernel)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    if lags is None:
        lags = max(1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))
    lags = min(lags, n - 2)
    xd = x - x.mean()
    gamma0 = float(np.dot(xd, xd)) / n
    s = gamma0
    for k in range(1, lags + 1):
        gk = float(np.dot(xd[: n - k], xd[k:])) / n
        s += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    s = max(s, 0.0)
    return math.sqrt(s / n)


def nw_tstat(x: np.ndarray, lags: int | None = None):
    """(mean, HAC se, t) of a possibly overlapping sample."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    se = newey_west_se(x, lags)
    return m, se, (m / se if se and np.isfinite(se) and se > 0 else float("nan"))


def stationary_bootstrap_ci(x: np.ndarray, iters: int = 1000,
                            mean_block: float = 50.0, alpha: float = 0.05,
                            seed: int = 0):
    """Politis-Romano stationary bootstrap CI for the mean.

    Geometric block lengths preserve short-range dependence, which the reversion
    curve has in abundance (consecutive ticks are nearly the same observation).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_block, 1.0)
    means = np.empty(iters)
    for it in range(iters):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(0, n) if rng.random() < p else (i + 1) % n
        means[it] = x[idx].mean()
    return (float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)))


def circular_block_bootstrap_ci(x: np.ndarray, iters: int = 1000,
                                block: int = 200, alpha: float = 0.05,
                                seed: int = 0, max_n: int = 40000):
    """Vectorized circular block bootstrap CI for the mean.

    Used as the cross-check on the Newey-West interval for the bins that decide
    the gate. Samples larger than max_n are systematically thinned (every m-th
    observation) rather than randomly subsampled, so the dependence structure
    the blocks are meant to preserve survives the thinning.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 30:
        return float("nan"), float("nan")
    # Small bins still deserve an interval; shrink the block rather than refuse,
    # and never let the block exceed a tenth of the sample.
    block = max(2, min(block, n // 10))
    if n > max_n:
        m = int(np.ceil(n / max_n))
        x = x[::m]
        block = max(2, block // m)
        n = x.size
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    offs = np.arange(block)
    means = np.empty(iters)
    chunk = max(1, int(4_000_000 / max(nb * block, 1)))
    done = 0
    while done < iters:
        k = min(chunk, iters - done)
        starts = rng.integers(0, n, size=(k, nb))
        idx = (starts[:, :, None] + offs[None, None, :]) % n
        means[done:done + k] = x[idx.reshape(k, -1)[:, :n]].mean(axis=1)
        done += k
    return (float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)))


def hayashi_yoshida(t1: np.ndarray, p1: np.ndarray,
                    t2: np.ndarray, p2: np.ndarray, lag_s: float = 0.0,
                    seg1=None, seg2=None):
    """Hayashi-Yoshida covariance/correlation of two ASYNCHRONOUS price series.

    t1/t2 are seconds, p1/p2 are log prices. Series 2's timestamps become
    t2 + lag_s. If series 1 LEADS series 2 by L seconds, correlation peaks at
    lag_s = -L: series 2 must be moved EARLIER to line up with its leader. So a
    NEGATIVE best lag means series 1 leads (selftest checks exactly this case).
    An earlier version of this docstring stated the sign backwards.

    No interpolation and no common grid -- that is the whole point of the
    estimator, and the reason it is the right cross-check for lead-lag here.

    seg1/seg2 label collector runs. A return spanning two runs is an overnight
    move, not a tick return; it is zeroed so the two series' overnight moves do
    not get multiplied together through their (huge, overlapping) intervals.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float) + lag_s
    r1 = np.diff(np.asarray(p1, dtype=float))
    r2 = np.diff(np.asarray(p2, dtype=float))
    if seg1 is not None:
        seg1 = np.asarray(seg1)
        r1 = np.where(seg1[1:] != seg1[:-1], 0.0, r1)
    if seg2 is not None:
        seg2 = np.asarray(seg2)
        r2 = np.where(seg2[1:] != seg2[:-1], 0.0, r2)
    if r1.size < 2 or r2.size < 2:
        return float("nan"), float("nan")
    a0, a1 = t1[:-1], t1[1:]
    b0, b1 = t2[:-1], t2[1:]

    cov = 0.0
    j_start = 0
    m = r2.size
    for i in range(r1.size):
        while j_start < m and b1[j_start] <= a0[i]:
            j_start += 1
        j = j_start
        while j < m and b0[j] < a1[i]:
            if b1[j] > a0[i]:            # intervals overlap
                cov += r1[i] * r2[j]
            j += 1
    v1 = float(np.dot(r1, r1))
    v2 = float(np.dot(r2, r2))
    if v1 <= 0 or v2 <= 0:
        return cov, float("nan")
    return cov, cov / math.sqrt(v1 * v2)


def resample_last(ts: np.ndarray, x: np.ndarray, step_s: float, seg=None):
    """Last-observation-carried-forward onto a regular grid.

    Returns (grid_times, values). Grid points before the first observation are
    dropped rather than back-filled -- back-filling would invent data.

    seg, if given, labels contiguous segments (one per collector run). Each is
    gridded on its own: carrying a value forward across the gap between two runs
    would fill an overnight hole with ~10^6 copies of the last quote, dragging
    AR(1) phi toward 1 and inventing a long half-life.
    """
    ts = np.asarray(ts, dtype=float)
    x = np.asarray(x, dtype=float)
    if ts.size == 0:
        return np.array([]), np.array([])
    if seg is not None:
        seg = np.asarray(seg)
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        parts = [resample_last(t_, x_, step_s)
                 for t_, x_ in zip(np.split(ts, cut), np.split(x, cut))]
        return (np.concatenate([p[0] for p in parts]),
                np.concatenate([p[1] for p in parts]))
    grid = np.arange(ts[0], ts[-1] + step_s, step_s)
    idx = np.searchsorted(ts, grid, side="right") - 1
    keep = idx >= 0
    return grid[keep], x[idx[keep]]


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
        "skew": float(_moment(x, 3)),
        "excess_kurtosis": float(_moment(x, 4) - 3.0),
    }


def _moment(x: np.ndarray, k: int) -> float:
    sd = x.std()
    if sd == 0:
        return float("nan")
    return float((((x - x.mean()) / sd) ** k).mean())
