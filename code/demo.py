"""End-to-end demo on one synthetic cointegrated pair: EG fit, OU fit,
Kalman beta tracking, walk-forward backtest, diagnostic figure."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth import make_pair
from coint import engle_granger
from ou import fit_ou
from kalman import kalman_hedge
from backtest import zscore_backtest

rng = np.random.default_rng(7)
logp, logq, truth = make_pair("cointegrated", n=2000, rng=rng)

eg = engle_granger(logp, logq)
ou = fit_ou(eg.spread)
kf = kalman_hedge(logp, logq)
bt = zscore_backtest(logp, logq)

print(f"EG p-value      : {eg.pvalue:.2e}")
print(f"OLS beta        : {eg.beta:.4f}   (true {truth.beta})")
print(f"Kalman beta[-1] : {kf.beta[-1]:.4f}")
print(f"OU half-life    : {ou.half_life:.1f} days "
      f"(true {np.log(2)/truth.kappa*252:.1f})")
print(f"OOS Sharpe      : {bt.sharpe:.2f}   ann ret {bt.ann_ret:.1%}  "
      f"vol {bt.ann_vol:.1%}  maxDD {bt.max_dd:.1%}")
print(f"Round trips     : {bt.n_round_trips}   "
      f"time in market {bt.time_in_market:.0%}")

fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
t = np.arange(len(logp))
ax[0].plot(t, np.exp(logp), lw=.8, label="P")
ax[0].plot(t, np.exp(logq), lw=.8, label="Q")
ax[0].set_title("Synthetic cointegrated pair (price)")
ax[0].legend()

z = (eg.spread - ou.mu) / ou.stationary_std
ax[1].plot(t, z, lw=.7, color="k")
for lvl, c in [(2, "r"), (-2, "r"), (0.5, "g"), (-0.5, "g")]:
    ax[1].axhline(lvl, ls="--", lw=.7, color=c)
ax[1].set_title(f"Spread z-score (half-life {ou.half_life:.1f}d)")

ax[2].plot(t, kf.beta, lw=.9, label="Kalman beta")
ax[2].axhline(truth.beta, color="r", ls="--", lw=.8, label="true beta")
ax[2].axhline(eg.beta, color="g", ls=":", lw=.8, label="OLS beta")
ax[2].set_ylim(truth.beta - .1, truth.beta + .1)
ax[2].set_title("Hedge ratio tracking")
ax[2].legend()

eq = np.cumprod(1 + bt.daily_ret)
ax[3].plot(t, eq, lw=1.0, color="navy")
ax[3].set_title(f"Walk-forward equity (OOS Sharpe {bt.sharpe:.2f}, "
                f"costs 2bp/leg)")
ax[3].set_xlabel("trading day")
fig.tight_layout()
fig.savefig("diagnostics.png", dpi=110)
print("wrote diagnostics.png")
