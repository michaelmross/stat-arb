"""Walk-forward z-score backtest with transaction costs.

Protocol (no look-ahead):
  - In-sample window: fit hedge ratio (Engle-Granger OLS) and OU params.
  - Out-of-sample window: compute spread with FROZEN in-sample parameters,
    trade z-score thresholds, roll forward, refit.
  - Tradability gate re-checked at every refit: EG p-value and half-life
    bounds. Cointegration can die; the gate lets the strategy stand down.

PnL accounting (deliberately explicit):
  Position sign x in {-1, 0, +1} refers to the SPREAD: x=+1 is long P,
  short beta*Q. Leg weights are w_P = x/(1+|beta|), w_Q = -x*beta/(1+|beta|)
  so gross notional = |w_P| + |w_Q| = |x| (unit capital when in a trade).
  Daily return on capital: r_t = w_P*dp_t + w_Q*dq_t with dp, dq log
  returns, using the position held ENTERING day t. Trades execute at the
  close of day t (position decided from z_t applies to t+1's move).
  One-way cost `cost_bps` charged on traded notional (sum over both legs).
"""

from dataclasses import dataclass, field
import numpy as np
from coint import engle_granger
from ou import fit_ou


@dataclass
class BTResult:
    daily_ret: np.ndarray
    positions: np.ndarray
    zscores: np.ndarray
    n_round_trips: int
    sharpe: float
    ann_ret: float
    ann_vol: float
    max_dd: float
    time_in_market: float
    windows: list = field(default_factory=list)


def _perf(r: np.ndarray):
    if len(r) == 0 or r.std(ddof=1) == 0:
        return 0.0, 0.0, 0.0, 0.0
    mu = r.mean() * 252
    sd = r.std(ddof=1) * np.sqrt(252)
    eq = np.cumprod(1 + r)
    dd = 1 - eq / np.maximum.accumulate(eq)
    return mu / sd, mu, sd, dd.max()


def zscore_backtest(logp: np.ndarray, logq: np.ndarray,
                    train: int = 504, trade: int = 126,
                    z_entry: float = 2.0, z_exit: float = 0.5,
                    z_stop: float = 4.0,
                    cost_bps: float = 2.0,
                    max_half_life: float = 60.0,
                    min_half_life: float = 1.0,
                    pval_gate: float = 0.05) -> BTResult:
    n = len(logp)
    pos = np.zeros(n)          # position held ENTERING day t
    z_all = np.full(n, np.nan)
    rets = np.zeros(n)
    cost = cost_bps / 1e4
    round_trips = 0
    windows = []

    start = 0
    while start + train + 2 <= n:
        i1 = start + train
        j1 = min(i1 + trade, n)
        eg = engle_granger(logp[start:i1], logq[start:i1])
        ou = fit_ou(eg.spread)
        tradable = (eg.pvalue < pval_gate and
                    min_half_life < ou.half_life < max_half_life)
        windows.append(dict(start=start, pvalue=float(eg.pvalue),
                            half_life=float(ou.half_life),
                            tradable=bool(tradable)))
        if tradable:
            alpha, beta = eg.alpha, eg.beta
            mu, sd = ou.mu, max(ou.stationary_std, 1e-12)
            wnorm = 1.0 + abs(beta)
            state = 0.0
            for t in range(i1, j1):
                # 1) accrue pnl on position held entering t
                if state != 0.0:
                    dp = logp[t] - logp[t - 1]
                    dq = logq[t] - logq[t - 1]
                    rets[t] += state * (dp - beta * dq) / wnorm
                pos[t] = state
                # 2) observe today's close, decide position for t+1
                s = logp[t] - alpha - beta * logq[t]
                z = (s - mu) / sd
                z_all[t] = z
                new = state
                if state == 0.0:
                    if z > z_entry:
                        new = -1.0
                    elif z < -z_entry:
                        new = +1.0
                elif abs(z) < z_exit or abs(z) > z_stop:
                    new = 0.0
                if new != state:
                    rets[t] -= abs(new - state) * cost  # both legs sum to |Δx| notional
                    if new == 0.0:
                        round_trips += 1
                    state = new
        start += trade

    oos = rets[train:]
    sharpe, ann_ret, ann_vol, max_dd = _perf(oos)
    tim = float((pos[train:] != 0).mean())
    return BTResult(daily_ret=rets, positions=pos, zscores=z_all,
                    n_round_trips=round_trips, sharpe=sharpe,
                    ann_ret=ann_ret, ann_vol=ann_vol, max_dd=max_dd,
                    time_in_market=tim, windows=windows)
