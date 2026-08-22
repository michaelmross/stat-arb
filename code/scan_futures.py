"""Test the production margins, with the roll treatment validated first.

Same protocol as the ETF study: parameters and the tradability gate are
fitted in-sample and re-checked every window, and performance is scored
only after the discovery split (2018-12-31), so the out-of-sample period
is out-of-sample for the choice of spread as well as the parameters.

Only two spreads are confirmatory. The cattle crush is reported but
excluded from the headline: the true feedlot margin is time-offset (buy
feeders and corn now, sell live cattle five to six months deferred) and
front-month-only data cannot represent the deferred leg.

Registered in advance (conversation9.md): the crack and crush margins
should show genuinely decaying ACFs and d_hat well below the ETF
discoveries' 0.51 -- the tether is physical -- with the open question
being whether half-lives land in the tradable window once roll costs are
charged.

d_hat MUST be read against a matched-persistence anchor, never against
0. validate_roll.py shows exact local Whittle reads +0.605 on a
synthetic OU margin that is I(0) by construction, because ELW is biased
upward on strongly autoregressive series. The anchor below is that same
bias measured at each margin's own fitted half-life and level sd.
"""

from __future__ import annotations

import paths

import json

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf

from data import load_panel
from futures import (SPREADS, build_margin, backtest_margin,
                             fit_ou_masked, cost_per_side)
from synth import make_margin, dejump
from fractional_census import elw

SPLIT = "2018-12-31"


def d_anchor(half_life, level_sd, n, rng, n_sims=25):
    """What ELW reads on a margin that is I(0) BY CONSTRUCTION, at this
    persistence. Anything at or below this is consistent with I(0)."""
    est = []
    for _ in range(n_sims):
        x, _, _ = make_margin(n, rng, kind="ou", half_life=half_life,
                              level_sd=level_sd, roll_offset_sd=0.0,
                              roll_offset_mean=0.0)
        est.append(elw(x)[0])
    return float(np.mean(est)), float(np.std(est))


def main():
    rng = np.random.default_rng(20260822)
    paths.require_prices('futures', min_files=3)
    panel = load_panel(str(paths.FUTURES), prefer_adjusted=False, log_transform=False)
    out = []

    for name, sp in SPREADS.items():
        m = build_margin(panel.frame, sp)
        x = m.value.to_numpy(float)
        idx = m.value.index
        disc = np.asarray(idx <= pd.Timestamp(SPLIT))

        ou = fit_ou_masked(x[disc], m.roll[disc])
        pv_raw = adfuller(x[disc], autolag="aic")[1]
        pv_msk = adfuller(dejump(x[disc], m.roll[disc]), autolag="aic")[1]
        a = acf(x[disc], nlags=40, fft=True)
        dh, _ = elw(x[disc])
        anc, anc_sd = d_anchor(ou["half_life"], ou["stationary_std"],
                               int(disc.sum()), rng)

        bt = backtest_margin(m, score_from=int(np.argmax(~disc)))

        tag = "CONFIRMATORY" if sp.confirmatory else "exploratory"
        print(f"=== {name}  [{tag}]  {sp.unit} ===")
        print(f"  {sp.note}")
        print(f"  n={len(x)}  roll days {m.meta['roll_frac']:.1%}  "
              f"level mean {x.mean():.3f} sd {x.std():.3f}")
        print(f"  discovery window (<= {SPLIT}, n={int(disc.sum())}):")
        print(f"    ADF p  raw {pv_raw:.2e}   roll-masked {pv_msk:.2e}"
              f"   <- masked is the honest one")
        print(f"    half-life {ou['half_life']:.1f}d   "
              f"stationary sd {ou['stationary_std']:.3f}")
        print(f"    ACF 1/10/40: {a[1]:.2f} / {a[10]:.2f} / {a[40]:.2f}"
              f"   {'DECAYS' if a[40] < 0.5 * a[1] else 'FLAT'}")
        print(f"    d_hat {dh:+.3f}  vs matched-I(0) anchor "
              f"{anc:+.3f} +/- {anc_sd:.3f}"
              f"   -> {'consistent with I(0)' if dh <= anc + 2 * anc_sd else 'ABOVE anchor'}")
        print(f"  out-of-sample (> {SPLIT}, n={bt['n_scored']}):")
        print(f"    Sharpe {bt['sharpe']:+.2f}   ann P&L "
              f"{bt['ann_pnl']:+.3f} {sp.unit.split()[0]}/unit   "
              f"t={bt['sharpe'] * np.sqrt(bt['n_scored'] / 252):+.2f}")
        print(f"    notional Sharpe {bt['notional_sharpe']:+.2f}  "
              f"(ETF-comparable)   ann ret {bt['ann_notional_ret']:+.2%}")
        print(f"    time in market {bt['time_in_market']:.1%}   "
              f"round trips {bt['round_trips']}   "
              f"windows tradable {bt['windows_tradable']:.0%}")
        print(f"    cost/side {bt['cost_per_side']:.5f} = "
              f"{2 * bt['cost_per_side'] / x.std() * 100:.2f}% of margin sd "
              f"per round trip")
        print()

        out.append(dict(spread=name, confirmatory=sp.confirmatory,
                        n=len(x), roll_frac=m.meta["roll_frac"],
                        adf_p_raw=float(pv_raw), adf_p_masked=float(pv_msk),
                        half_life=ou["half_life"],
                        stationary_std=ou["stationary_std"],
                        acf1=float(a[1]), acf10=float(a[10]),
                        acf40=float(a[40]), d_hat=float(dh),
                        d_anchor=anc, d_anchor_sd=anc_sd,
                        sharpe=bt["sharpe"],
                        t=float(bt["sharpe"] * np.sqrt(bt["n_scored"] / 252)),
                        notional_sharpe=bt["notional_sharpe"],
                        ann_notional_ret=bt["ann_notional_ret"],
                        time_in_market=bt["time_in_market"],
                        round_trips=bt["round_trips"],
                        windows_tradable=bt["windows_tradable"],
                        cost_per_side=bt["cost_per_side"]))

    with open(str(paths.data("futures_results.json")), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote futures_results.json")


if __name__ == "__main__":
    main()
