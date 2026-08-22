"""Checks behind the v2 revision (adversarial review, 2026-08-21).

1. CONTROLS: Spearman(EG p-value, edge ratio) on no-market universes of
   genuinely cointegrated pairs, half-life swept over the range real
   pairs span (2-250 days). Three designs differing only in how spread
   width covaries with persistence:
     (a) diffusion sigma fixed  -> width inherits persistence  -> +0.997
     (b) stationary width fixed                                -> -0.294
     (c) width and persistence swept independently             -> -0.068
   The real-data value (+0.446) lies inside a range spanned by
   no-economics generators, so it supports no economic inference in
   either direction. It is reported descriptively only.

2. POWER: median walk-forward OOS Sharpe on synthetic cointegrated
   pairs vs stationary spread width, at the study's costs (2bp/leg).
   Well-powered at bond-pair amplitudes (12-44bp), underpowered at
   tight-tracker amplitudes (~5bp).

3. T-STATS: annualized Sharpe times sqrt(7.62 years) for the five
   evaluated pairs. Four indistinguishable from zero; LQD/USIG is
   significantly negative (t=-3.29) and was the one pair flagged
   noise-dominated in advance.

Run:  python controls_and_power.py
"""

import numpy as np
from scipy.stats import spearmanr
from synth import make_pair
from coint import engle_granger
from ou import fit_ou
from backtest import zscore_backtest

COST_RT = 4e-4  # round-trip cost on capital (2bp/leg one-way)


def edge(eg, ou):
    return 1.5 * ou.stationary_std / (1 + abs(eg.beta)) / COST_RT


def control(design: str, n_draws=200, n_obs=2264, seed=0):
    rng = np.random.default_rng(seed)
    pvs, edges = [], []
    for _ in range(n_draws):
        hl = np.exp(rng.uniform(np.log(2), np.log(250)))
        kappa = np.log(2) * 252 / hl
        if design == "diffusion_fixed":
            spread_sigma = 0.02
        elif design == "width_fixed":
            spread_sigma = 0.0015 * np.sqrt(2 * kappa)
        elif design == "independent":
            w = np.exp(rng.uniform(np.log(4e-4), np.log(45e-4)))
            spread_sigma = w * np.sqrt(2 * kappa)
        else:
            raise ValueError(design)
        p, q, _ = make_pair("cointegrated", n=n_obs, rng=rng,
                            kappa=kappa, spread_sigma=spread_sigma)
        eg = engle_granger(p, q)
        pvs.append(eg.pvalue)
        edges.append(edge(eg, fit_ou(eg.spread)))
    return spearmanr(pvs, edges)


def power_sweep(widths_bp=(5, 8, 12, 16, 20, 44, 70), n_sims=30,
                n_obs=2000, kappa=25.0, seed=2718):
    rng = np.random.default_rng(seed)
    out = {}
    for bp in widths_bp:
        spread_sigma = bp / 1e4 * np.sqrt(2 * kappa)
        sh = [zscore_backtest(*make_pair("cointegrated", n=n_obs, rng=rng,
                                         kappa=kappa,
                                         spread_sigma=spread_sigma)[:2]).sharpe
              for _ in range(n_sims)]
        out[bp] = (float(np.median(sh)),
                   float(np.quantile(sh, .25)), float(np.quantile(sh, .75)))
    return out


if __name__ == "__main__":
    for d, s in [("diffusion_fixed", 161803), ("width_fixed", 31415),
                 ("independent", 27182)]:
        rho, p = control(d, seed=s)
        print(f"control {d:16s} Spearman {rho:+.3f}  (p={p:.2e})")
    print()
    for bp, (med, lo, hi) in power_sweep().items():
        print(f"width {bp:3d}bp  median OOS Sharpe {med:+.2f}  [{lo:+.2f},{hi:+.2f}]")
    print()
    yrs = 1920 / 252
    for pair, s in [("VTI/SCHB", .2509), ("BND/SPAB", .1494),
                    ("AGG/SPAB", -.0793), ("AGG/BND", -.3679),
                    ("LQD/USIG", -1.1933)]:
        print(f"{pair:10s} Sharpe {s:+.2f}  t = {s*np.sqrt(yrs):+.2f}")
