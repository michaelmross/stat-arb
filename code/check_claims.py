"""Adversarial checks on the three inferential claims in the write-up.

C1. "their defining statistics are significantly anti-correlated ...
     competitive equilibrium made visible in scan statistics"
    -> Both edge_ratio and the EG p-value are monotone functions of the
       SAME estimated mean-reversion speed: stationary_std = sigma /
       sqrt(2*kappa) blows up as kappa -> 0, and the p-value -> 1 as
       kappa -> 0. So a positive rank correlation is expected with no
       competitive mechanism anywhere. Test: does synthetic data, which
       contains no arbitrageurs, reproduce the sign and magnitude?

C2. "detects synthetic edge of realistic shape at Sharpe 1.9"
    -> The Sharpe-1.9 cell uses spread_sigma=0.05, i.e. a stationary
       spread sd of ~70bp. Real same-index spreads measured here are
       5-44bp. Test: sweep spread amplitude and find where the machinery
       actually retains power at 2bp/leg costs.

C3. "out-of-sample performance ... statistically indistinguishable from
     zero"
    -> Never actually tested. Compute t-stats on the realised Sharpes.

Run:  python check_claims.py
"""

from __future__ import annotations

import paths

import json
import numpy as np
from scipy import stats

from synth import make_pair
from coint import engle_granger
from ou import fit_ou
from backtest import zscore_backtest


def hl_to_kappa(half_life_days):
    return np.log(2.0) * 252.0 / half_life_days


def c1_mechanical_confound(n_sims=140, n_obs=2264, seed=20260821):
    """Do p-value and edge_ratio correlate in data with NO market in it?"""
    rng = np.random.default_rng(seed)
    pvals, edges = [], []
    # sweep persistence across the range real pairs span, plus true nulls
    half_lives = np.geomspace(1.5, 400.0, n_sims)
    for hl in half_lives:
        kappa = hl_to_kappa(hl)
        # hold the OU innovation vol fixed; stationary_std then varies
        # with kappa exactly as it does across real pairs
        p, q, _ = make_pair("cointegrated", n=n_obs, rng=rng, kappa=kappa,
                            spread_sigma=0.05)
        eg = engle_granger(p, q)
        ou = fit_ou(eg.spread)
        edge = (1.5 * ou.stationary_std / (1.0 + abs(eg.beta))) / (2 * 2e-4)
        pvals.append(eg.pvalue)
        edges.append(edge)
    rho, pv = stats.spearmanr(pvals, edges)
    print("C1  synthetic control (no competition, only estimation)")
    print(f"    Spearman(p-value, edge_ratio) = {rho:+.3f}  p={pv:.2g}"
          f"   n={n_sims}")
    print(f"    real universe                 = +0.446  p=4.2e-06   n=98")
    return rho, pv


def c2_power_vs_amplitude(n_sims=25, n_obs=4000, half_life=5.0,
                          seed=20260821):
    """At what spread width does the backtest still find edge at 2bp/leg?"""
    rng = np.random.default_rng(seed)
    kappa = hl_to_kappa(half_life)
    print(f"\nC2  OOS Sharpe vs spread amplitude "
          f"(half-life {half_life:.0f}d, 2bp/leg, n={n_obs})")
    print(f"    {'spread sd':>10s} {'median Sharpe':>14s} "
          f"{'IQR':>18s} {'>0':>6s}")
    out = {}
    for sd_bp in [5, 10, 20, 40, 70, 140]:
        sigma = (sd_bp / 1e4) * np.sqrt(2.0 * kappa)
        sh = []
        for _ in range(n_sims):
            p, q, _ = make_pair("cointegrated", n=n_obs, rng=rng,
                                kappa=kappa, spread_sigma=sigma)
            sh.append(zscore_backtest(p, q, cost_bps=2.0).sharpe)
        sh = np.array(sh)
        q1, q3 = np.quantile(sh, [.25, .75])
        print(f"    {sd_bp:8d}bp {np.median(sh):14.2f} "
              f"{f'[{q1:.2f}, {q3:.2f}]':>18s} {(sh>0).mean():6.0%}")
        out[sd_bp] = float(np.median(sh))
    return out


def c3_sharpe_tstats(path=str(paths.data("scan_full.json"))):
    """Is 'indistinguishable from zero' actually true for each pair?"""
    res = json.load(open(path))["evaluation"]
    print("\nC3  t-stats on realised OOS Sharpes  (t ~ SR * sqrt(years))")
    print(f"    {'pair':12s} {'Sharpe':>8s} {'years':>7s} {'t':>7s}   verdict")
    for e in sorted(res, key=lambda r: -r["sharpe"]):
        yrs = e["n_days_oos"] / 252.0
        t = e["sharpe"] * np.sqrt(yrs)
        v = "indistinguishable" if abs(t) < 1.96 else "SIGNIFICANT"
        print(f"    {e['a']+'/'+e['b']:12s} {e['sharpe']:8.2f} {yrs:7.1f} "
              f"{t:7.2f}   {v}")


if __name__ == "__main__":
    c1_mechanical_confound()
    c2_power_vs_amplitude()
    c3_sharpe_tstats()
