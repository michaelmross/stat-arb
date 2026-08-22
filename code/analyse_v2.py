"""Scan v2 follow-ups: significance, cohort split, and the March 2020 test.

The March 2020 episode is the point of extending into fixed income. Bond
ETFs traded at large, persistent discounts to NAV for weeks -- creation-
redemption, the very tether that motivated this cohort, was visibly
impaired. So either the spreads widened and reverted (the hypothesis
holds, and this is where the strategy should earn its keep), or they
broke (the tether is conditional on exactly the calm it is supposed to
survive). This script asks which.
"""

from __future__ import annotations

import paths

import json
import numpy as np
import pandas as pd

from data import load_panel
from coint import engle_granger
from ou import fit_ou
from backtest import zscore_backtest, _perf

CRISIS = ("2020-02-15", "2020-04-30")
SPLIT = "2018-12-31"


def main():
    res = json.load(open(str(paths.data("scan_v2.json"))))
    ev = pd.DataFrame(res["evaluation"])
    disc = pd.DataFrame(res["discovery"])
    meta = disc.set_index(disc.a + "/" + disc.b)

    paths.require_prices('etf', min_files=2)
    panel = load_panel(str(paths.PRICES))

    print("=== significance of realised OOS Sharpes ===")
    print(f"{'pair':12s} {'cohort':17s} {'Sharpe':>7s} {'t':>6s} "
          f"{'tim':>6s}  verdict")
    rows = []
    for r in ev.itertuples():
        key = f"{r.a}/{r.b}"
        coh = meta.loc[key, "cohort"] if key in meta.index else "?"
        yrs = r.n_days_oos / 252.0
        t = r.sharpe * np.sqrt(yrs)
        if r.time_in_market == 0.0:
            v = "NEVER TRADED OOS (gate stood down)"
        elif abs(t) < 1.96:
            v = "indistinguishable from zero"
        else:
            v = "SIGNIFICANT"
        print(f"{key:12s} {coh:17s} {r.sharpe:7.2f} {t:6.2f} "
              f"{r.time_in_market:6.1%}  {v}")
        rows.append(dict(pair=key, cohort=coh, sharpe=r.sharpe, t=t,
                         tim=r.time_in_market))
    ev2 = pd.DataFrame(rows)
    traded = ev2[ev2.tim > 0]
    print(f"\n{len(traded)} of {len(ev2)} discoveries actually traded OOS; "
          f"of those, {(traded.t.abs() > 1.96).sum()} are significant "
          f"(all negative: {(traded[traded.t.abs()>1.96].t < 0).all()})")
    for coh, g in ev2.groupby("cohort"):
        gt = g[g.tim > 0]
        print(f"  {coh:17s} n={len(g):2d}  traded={len(gt):2d}  "
              f"median Sharpe {g.sharpe.median():+.2f}")

    # ---------------- March 2020 ----------------------------------------
    print(f"\n=== March 2020 stress ({CRISIS[0]} -> {CRISIS[1]}) ===")
    print("Spread z computed on parameters frozen from data BEFORE the "
          "crisis window,\nso this is what a live book would have seen.\n")
    print(f"{'pair':12s} {'cohort':17s} {'maxZ':>6s} {'endZ':>6s} "
          f"{'crisis ret':>11s}  outcome")
    out = []
    for r in ev.itertuples():
        key = f"{r.a}/{r.b}"
        coh = meta.loc[key, "cohort"] if key in meta.index else "?"
        sub = panel.log_prices[[r.a, r.b]].dropna()
        # freeze on the two years ending just before the crisis window
        pre = sub[sub.index < CRISIS[0]].tail(504)
        eg = engle_granger(pre[r.a].to_numpy(), pre[r.b].to_numpy())
        ou = fit_ou(eg.spread)
        sd = max(ou.stationary_std, 1e-12)
        win = sub[(sub.index >= CRISIS[0]) & (sub.index <= CRISIS[1])]
        s = (win[r.a].to_numpy() - eg.alpha - eg.beta * win[r.b].to_numpy())
        z = (s - ou.mu) / sd
        # did it revert? compare |z| at the end vs its peak
        peak, end = float(np.abs(z).max()), float(z[-1])
        # realised strategy return inside the window
        bt = zscore_backtest(sub[r.a].to_numpy(), sub[r.b].to_numpy())
        mask = np.asarray((sub.index >= CRISIS[0]) & (sub.index <= CRISIS[1]))
        cret = float(np.prod(1 + bt.daily_ret[mask]) - 1)
        if peak > 4.0:
            outcome = "BROKE (blew through 4-sigma stop)"
        elif peak > 2.0 and abs(end) < 1.0:
            outcome = "widened and reverted"
        elif peak > 2.0:
            outcome = "widened, unresolved in window"
        else:
            outcome = "no dislocation"
        print(f"{key:12s} {coh:17s} {peak:6.1f} {end:6.1f} "
              f"{cret:10.2%}  {outcome}")
        out.append(dict(pair=key, cohort=coh, peak_z=peak, end_z=end,
                        crisis_ret=cret, outcome=outcome))
    o = pd.DataFrame(out)
    print(f"\nblew through the 4-sigma stop: {(o.peak_z > 4).sum()}/{len(o)}")
    print(f"total crisis-window return, equal-weighted: "
          f"{o.crisis_ret.mean():+.2%}")
    o.to_json(str(paths.data("crisis_2020.json")), orient="records", indent=2)
    print("\nwrote crisis_2020.json")


if __name__ == "__main__":
    main()
