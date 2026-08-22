"""BH-corrected cointegration scan over a structurally related ETF universe.

Protocol (the point of the split):
  1. DISCOVERY  -- on dates <= --discovery-end only, run Engle-Granger on
     every candidate pair, apply Benjamini-Hochberg at level q.
  2. EVALUATION -- walk-forward backtest the discoveries, scoring ONLY the
     dates after --discovery-end.

Selecting pairs on the whole sample and then backtesting that same sample
is look-ahead through the back door: the backtester is honest window by
window, but the choice of WHICH pairs to run already used the future.
Splitting makes the reported OOS number out-of-sample for selection too.

Also reports an explicit cost-viability ratio, because README caveat 1 is
the thing that actually kills real ETF pairs:

    expected capture per round trip   1.5 * stationary_std / (1 + |beta|)
    ------------------------------- = ----------------------------------
    round-trip cost                            2 * cost_bps/1e4

entering at |z|=2 and exiting at |z|=0.5. Below 1.0 the spread cannot pay
for its own execution no matter how clean the cointegration is.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from statarb.data import load_panel
from statarb.universe import candidate_pairs, all_tickers
from statarb.coint import engle_granger, benjamini_hochberg
from statarb.ou import fit_ou
from statarb.backtest import zscore_backtest, _perf
from statsmodels.tsa.stattools import acf


@dataclass
class PairScan:
    a: str
    b: str
    relation: str
    family: str
    cohort: str
    n_obs_disc: int
    pvalue: float
    beta: float
    half_life: float
    stationary_std: float
    ret_corr: float
    edge_ratio: float
    gate_pass: bool           # half-life within tradable bounds
    acf1: float
    acf40: float
    noise_dominated: bool     # flat ACF -> EG p-value is not trustworthy


def scan_discovery(panel, cands, discovery_end, min_obs=756,
                   cost_bps=2.0, min_half_life=1.0, max_half_life=60.0):
    """Engle-Granger every candidate pair on discovery-window data only."""
    rows, skipped = [], []
    for c in cands:
        cols = panel.log_prices.columns
        if c.a not in cols or c.b not in cols:
            skipped.append((c.a, c.b, "missing data"))
            continue
        sub = panel.log_prices[[c.a, c.b]].dropna()
        sub = sub[sub.index <= pd.Timestamp(discovery_end)]
        if len(sub) < min_obs:
            skipped.append((c.a, c.b, f"only {len(sub)} obs in discovery window"))
            continue
        lp, lq = sub[c.a].to_numpy(), sub[c.b].to_numpy()
        eg = engle_granger(lp, lq)
        ou = fit_ou(eg.spread)
        capture = 1.5 * ou.stationary_std / (1.0 + abs(eg.beta))
        edge = capture / (2.0 * cost_bps / 1e4)

        # Spread ACF shape. A genuinely mean-reverting spread DECAYS; a
        # spread that is a persistent component buried under white noise
        # has a FLAT ACF (real SPY/IVV: 0.37 at lag 1, still 0.31 at lag
        # 40). validate_lag.py shows EG rejects such nulls 33% of the time
        # at nominal 5% even under the most conservative lag policy, so a
        # small p-value here means much less than it appears to.
        a = acf(eg.spread, nlags=40, fft=True)
        flat = bool(a[1] > 0.02 and a[40] / a[1] > 0.5)
        rows.append(PairScan(
            a=c.a, b=c.b, relation=c.relation, family=c.family,
            cohort=c.cohort,
            n_obs_disc=len(sub), pvalue=float(eg.pvalue), beta=float(eg.beta),
            half_life=float(ou.half_life),
            stationary_std=float(ou.stationary_std),
            ret_corr=float(np.corrcoef(np.diff(lp), np.diff(lq))[0, 1]),
            edge_ratio=float(edge),
            gate_pass=bool(min_half_life < ou.half_life < max_half_life),
            acf1=float(a[1]), acf40=float(a[40]), noise_dominated=flat,
        ))
    return rows, skipped


def evaluate(panel, a, b, relation, discovery_end, cost_bps=2.0,
             train=504, trade=126):
    """Walk-forward backtest; score only post-discovery dates."""
    sub = panel.log_prices[[a, b]].dropna()
    lp, lq = sub[a].to_numpy(), sub[b].to_numpy()
    bt = zscore_backtest(lp, lq, cost_bps=cost_bps, train=train, trade=trade)
    post = np.asarray(sub.index > pd.Timestamp(discovery_end))
    post[:train] = False          # never score inside the first training block
    r = bt.daily_ret[post]
    sharpe, ann_ret, ann_vol, max_dd = _perf(r)
    return dict(a=a, b=b, relation=relation,
                n_days_oos=int(post.sum()),
                sharpe=float(sharpe), ann_ret=float(ann_ret),
                ann_vol=float(ann_vol), max_dd=float(max_dd),
                time_in_market=float((bt.positions[post] != 0).mean()
                                     if post.any() else 0.0),
                round_trips=int(bt.n_round_trips),
                windows_tradable=float(np.mean([w["tradable"] for w in bt.windows])
                                       if bt.windows else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--discovery-end", required=True,
                    help="last date usable for pair SELECTION, YYYY-MM-DD")
    ap.add_argument("--q", type=float, default=0.05, help="BH FDR level")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--min-obs", type=int, default=756)
    ap.add_argument("--train", type=int, default=504)
    ap.add_argument("--trade", type=int, default=126)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--out", default="scan_results.json")
    args = ap.parse_args()

    cands = candidate_pairs(include_controls=not args.no_controls)
    panel = load_panel(args.data_dir, tickers=all_tickers())
    have = set(panel.log_prices.columns)
    print(f"Loaded {len(have)}/{len(all_tickers())} tickers  "
          f"{panel.log_prices.index[0].date()} -> "
          f"{panel.log_prices.index[-1].date()}")
    if panel.any_dividend_unadjusted:
        print("  WARNING: at least one series is not dividend-adjusted; "
              "ex-div sawtooth will contaminate spreads.")
    for t, i in sorted(panel.info.items()):
        for w in i.warnings:
            print(f"  [{t}] {w}")

    rows, skipped = scan_discovery(panel, cands, args.discovery_end,
                                   min_obs=args.min_obs,
                                   cost_bps=args.cost_bps)
    if skipped:
        print(f"\nSkipped {len(skipped)} pairs:")
        for a, b, why in skipped[:25]:
            print(f"  {a}/{b}: {why}")

    if not rows:
        print("\nNo pairs had enough data to test.")
        return

    pvals = np.array([r.pvalue for r in rows])
    bh = benjamini_hochberg(pvals, q=args.q)
    raw = pvals < args.q

    print(f"\n=== DISCOVERY (<= {args.discovery_end}, "
          f"{len(rows)} pairs tested) ===")
    print(f"raw p<{args.q}: {int(raw.sum())}   BH q={args.q}: {int(bh.sum())}")

    df = pd.DataFrame([asdict(r) for r in rows])
    df["bh"] = bh
    df["raw"] = raw
    df = df.sort_values("pvalue").reset_index(drop=True)

    print("\nBy relation (raw / BH discoveries out of tested):")
    for rel, g in df.groupby("relation"):
        print(f"  {rel:14s} {int(g['raw'].sum()):3d} / "
              f"{int(g['bh'].sum()):3d} of {len(g):3d}")

    fmt = lambda v: f"{v:,.4g}"
    show = df[df["bh"]].copy()
    print(f"\n--- BH discoveries ({len(show)}) ---")
    if len(show):
        print(show[["a", "b", "cohort", "family", "pvalue", "beta", "half_life",
                    "stationary_std", "edge_ratio", "acf1", "acf40",
                    "noise_dominated", "gate_pass"]]
              .to_string(index=False, float_format=fmt))

    tradeable = show[show["gate_pass"]]
    print(f"\n=== EVALUATION (> {args.discovery_end}) ===")
    print(f"{len(tradeable)} of {len(show)} discoveries pass the half-life gate")
    results = []
    for _, r in tradeable.iterrows():
        try:
            results.append(evaluate(panel, r["a"], r["b"], r["relation"],
                                    args.discovery_end,
                                    cost_bps=args.cost_bps,
                                    train=args.train, trade=args.trade))
        except Exception as e:
            print(f"  {r['a']}/{r['b']}: evaluation failed: {e}")
    if results:
        res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        print(res.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        print(f"\nmedian OOS Sharpe {res['sharpe'].median():.2f}   "
              f"mean {res['sharpe'].mean():.2f}   "
              f"positive {int((res['sharpe'] > 0).sum())}/{len(res)}")
    else:
        print("  nothing to evaluate")

    with open(args.out, "w") as f:
        json.dump({"discovery": df.to_dict("records"),
                   "evaluation": results,
                   "skipped": skipped,
                   "params": vars(args)}, f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
