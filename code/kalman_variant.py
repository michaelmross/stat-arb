"""Roadmap item 2: Kalman-innovation trading variant, evaluated.

Fully online: state (alpha_t, beta_t) evolves as a random walk, observed
through logP_t = alpha_t + beta_t logQ_t + eps. The trading signal is the
standardized one-step innovation z_t = e_t / sqrt(S_t). No frozen
windows, no refits, no tradability gate -- the filter adapts
continuously, which is exactly the design difference being tested
against the walk-forward baseline.

Honesty constraints:
  * The state-noise scale `delta` is chosen ON SYNTHETIC ground truth
    (calibrate() below), never on real-data outcomes.
  * Per-pair observation variance r is set from a 252-day burn-in OLS,
    using only data before trading starts.
  * Costs (2bp/leg one-way) are charged on ALL turnover, including the
    continuous rebalancing induced by beta_t drift while in a position.
  * Same entry/exit/stop (2 / 0.5 / 4) on standardized innovations, and
    scoring only on dates after the discovery split, as the baseline.
"""

from __future__ import annotations
import numpy as np


def kalman_trade(logp, logq, delta=1e-5, burn=252,
                 z_entry=2.0, z_exit=0.5, z_stop=4.0,
                 cost_bps=2.0, score_from=None):
    """Run the online variant. Returns daily returns, positions, z."""
    n = len(logp)
    cost = cost_bps / 1e4
    # burn-in OLS for r and state init
    X = np.column_stack([np.ones(burn), logq[:burn]])
    coef, *_ = np.linalg.lstsq(X, logp[:burn], rcond=None)
    resid = logp[:burn] - X @ coef
    r = max(resid.var(ddof=2), 1e-10)
    x = coef.copy()                       # (alpha, beta)
    P = np.eye(2) * r
    Q = np.eye(2) * (delta / (1.0 - delta)) * r

    rets = np.zeros(n)
    pos = np.zeros(n)
    zs = np.full(n, np.nan)
    state = 0.0
    w_p_prev = w_q_prev = 0.0
    for t in range(burn, n):
        beta_held = x[1]
        # accrue pnl on position held entering t (weights fixed overnight)
        if state != 0.0 and t > burn:
            rets[t] += w_p_prev * (logp[t] - logp[t - 1]) \
                     + w_q_prev * (logq[t] - logq[t - 1])
        pos[t] = state
        # filter update at close t
        H = np.array([1.0, logq[t]])
        P = P + Q
        e = logp[t] - H @ x
        S = H @ P @ H + r
        K = P @ H / S
        x = x + K * e
        P = P - np.outer(K, H @ P)
        z = e / np.sqrt(S)
        zs[t] = z
        # decide position for t+1
        new = state
        if state == 0.0:
            if z > z_entry:
                new = -1.0
            elif z < -z_entry:
                new = +1.0
        elif abs(z) < z_exit or abs(z) > z_stop:
            new = 0.0
        # weights for t+1 at current beta estimate
        b = x[1]
        wn = 1.0 + abs(b)
        w_p, w_q = new / wn, -new * b / wn
        turnover = abs(w_p - w_p_prev) + abs(w_q - w_q_prev)
        if turnover > 1e-12:
            rets[t] -= turnover * cost
        w_p_prev, w_q_prev, state = w_p, w_q, new

    if score_from is not None:
        sl = np.zeros(n, bool)
        sl[score_from:] = True
    else:
        sl = np.ones(n, bool)
        sl[:burn] = False
    r_ = rets[sl]
    sd = r_.std(ddof=1)
    sharpe = r_.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    return dict(sharpe=float(sharpe),
                ann_ret=float(r_.mean() * 252),
                time_in_market=float((pos[sl] != 0).mean()),
                daily_ret=rets, positions=pos, z=zs)


def calibrate(deltas=(1e-4, 1e-5, 1e-6), widths_bp=(12, 16, 44),
              n_sims=25, n_obs=2000, kappa=25.0, seed=777):
    """Pick delta on synthetic cointegrated pairs (ground truth only)."""
    from statarb.synth import make_pair
    rng = np.random.default_rng(seed)
    table = {}
    for d in deltas:
        sh = []
        for bp in widths_bp:
            ss = bp / 1e4 * np.sqrt(2 * kappa)
            for _ in range(n_sims):
                p, q, _ = make_pair("cointegrated", n=n_obs, rng=rng,
                                    kappa=kappa, spread_sigma=ss)
                sh.append(kalman_trade(p, q, delta=d)["sharpe"])
        table[d] = float(np.median(sh))
    best = max(table, key=table.get)
    return best, table


def run_real(scan_path="scan_v2.json", data_dir="data", delta=1e-6,
             discovery_end="2018-12-31", out="kalman_results.json"):
    """Run the online variant on the pairs the walk-forward scan evaluated.

    Same pairs, same out-of-sample window, same costs -- so the only
    difference from the baseline is the design being tested (continuous
    adaptation vs frozen windows plus a tradability gate). `delta` comes
    from calibrate(), i.e. from synthetic ground truth, never from these
    outcomes.
    """
    import json
    import pandas as pd
    from statarb.data import load_panel

    scan = json.load(open(scan_path))
    panel = load_panel(data_dir)
    cut = pd.Timestamp(discovery_end)

    rows = []
    for e in scan["evaluation"]:
        a, b = e["a"], e["b"]
        sub = panel.log_prices[[a, b]].dropna()
        score_from = int(np.searchsorted(sub.index.values,
                                         np.datetime64(cut), side="right"))
        res = kalman_trade(sub[a].to_numpy(), sub[b].to_numpy(),
                           delta=delta, score_from=score_from)
        yrs = (len(sub) - score_from) / 252.0
        rows.append(dict(pair=f"{a}/{b}", wf_sharpe=e["sharpe"],
                         kalman_sharpe=res["sharpe"],
                         t=res["sharpe"] * np.sqrt(yrs),
                         tim=res["time_in_market"]))
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)
    return rows


def cost_attribution(scan_path="scan_v2.json", data_dir="data", delta=1e-6,
                     discovery_end="2018-12-31",
                     out="kalman_cost_attribution.json"):
    """Re-run each pair at zero cost to split churn from adverse level.

    A large negative online Sharpe has two possible causes. If it comes
    back to roughly zero at zero cost, the strategy was trading
    closing-print bounce that nets nothing and paying for every pass --
    churn. If it stays negative, the level itself moved against the
    position, which no cost model can rescue.

    Written to its own file: `kalman_results.json` is the note's archived
    artifact and is left alone.
    """
    import json
    import pandas as pd
    from statarb.data import load_panel

    scan = json.load(open(scan_path))
    panel = load_panel(data_dir)
    cut = pd.Timestamp(discovery_end)

    rows = []
    for e in scan["evaluation"]:
        a, b = e["a"], e["b"]
        sub = panel.log_prices[[a, b]].dropna()
        sf = int(np.searchsorted(sub.index.values,
                                 np.datetime64(cut), side="right"))
        lp, lq = sub[a].to_numpy(), sub[b].to_numpy()
        full = kalman_trade(lp, lq, delta=delta, score_from=sf)["sharpe"]
        free = kalman_trade(lp, lq, delta=delta, cost_bps=0.0,
                            score_from=sf)["sharpe"]
        rows.append(dict(pair=f"{a}/{b}", sharpe_2bp=full, sharpe_0bp=free,
                         cost_drag=full - free,
                         verdict="churn" if free > -0.5 else "adverse level"))
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)
    print(f"{'pair':12s} {'2bp/leg':>8s} {'zero cost':>10s} "
          f"{'drag':>7s}  verdict")
    for r in sorted(rows, key=lambda x: x["sharpe_2bp"]):
        print(f"{r['pair']:12s} {r['sharpe_2bp']:8.2f} {r['sharpe_0bp']:10.2f} "
              f"{r['cost_drag']:7.2f}  {r['verdict']}")
    return rows


def summarise(rows):
    med_wf = float(np.median([r["wf_sharpe"] for r in rows]))
    med_kf = float(np.median([r["kalman_sharpe"] for r in rows]))
    sig = [r for r in rows if abs(r["t"]) > 1.96]
    print(f"{'pair':12s} {'walk-fwd':>9s} {'kalman':>8s} {'t':>7s} {'tim':>7s}")
    for r in sorted(rows, key=lambda x: -x["kalman_sharpe"]):
        print(f"{r['pair']:12s} {r['wf_sharpe']:9.2f} "
              f"{r['kalman_sharpe']:8.2f} {r['t']:7.2f} {r['tim']:7.1%}")
    print(f"\nmedian: walk-forward {med_wf:+.2f} -> kalman {med_kf:+.2f}"
          f"   significant at |t|>1.96: {len(sig)}/{len(rows)}")
    print(f"mean time in market: "
          f"{np.mean([r['tim'] for r in rows]):.1%} "
          f"(baseline gate stands down; the filter never does)")


if __name__ == "__main__":
    import sys
    if "--real" in sys.argv:
        rows = run_real()
        summarise(rows)
    elif "--costs" in sys.argv:
        cost_attribution()
    else:
        best, table = calibrate()
        print("synthetic median Sharpe by delta:", table, "-> chosen:", best)
