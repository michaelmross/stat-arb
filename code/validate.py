"""Validation on synthetic ground truth (the 'mutant census' step).

1. SIZE: on non-cointegrated pairs, does the EG test reject at ~5%?
   Includes correlated random walks -- the spurious-regression trap --
   and shows the naive ADF-on-residuals test is anti-conservative.
2. POWER: on genuinely cointegrated pairs, how often do we detect?
3. RECOVERY: are beta, kappa, half-life estimates consistent?
4. MULTIPLE TESTING: BH-FDR discovery counts on a mixed universe.
5. BACKTEST SANITY: positive OOS Sharpe on cointegrated pairs, ~zero
   (cost-bleeding) on impostors -- and the gate should keep us OUT of
   impostor trades almost always.
6. LAG POLICY: on the noisy_rw null -- a unit-root spread buried under
   white noise, which is what real same-index ETF pairs actually look
   like -- which ADF augmentation policy holds size?

Cell 6 exists because cells 1-5 all passed while the test was rejecting
the noisy_rw null a third of the time. Regimes 1-3 emit clean
innovations, so nothing here exercised MA-induced size distortion, and
the suite reported a clean bill of health on a case it never ran. The
lesson generalises: a validation suite certifies the generator's
coverage, not the estimator.
"""

import numpy as np
from synth import make_pair
from coint import engle_granger, adf_naive, benjamini_hochberg
from ou import fit_ou
from backtest import zscore_backtest


def run(n_sims: int = 200, n_obs: int = 1500, seed: int = 20260821,
        verbose: bool = True):
    rng = np.random.default_rng(seed)
    out = {}

    # ---- 1 & 2: size and power -------------------------------------------
    for kind in ("random_walks", "correlated_rw", "noisy_rw", "cointegrated"):
        pv_eg, pv_naive = [], []
        for _ in range(n_sims):
            p, q, _ = make_pair(kind, n=n_obs, rng=rng)
            eg = engle_granger(p, q)
            pv_eg.append(eg.pvalue)
            pv_naive.append(adf_naive(eg.spread))
        pv_eg, pv_naive = np.array(pv_eg), np.array(pv_naive)
        out[kind] = dict(eg_reject_05=float((pv_eg < .05).mean()),
                         naive_reject_05=float((pv_naive < .05).mean()))

    # ---- 3: parameter recovery on cointegrated pairs ---------------------
    betas, hls = [], []
    true_hl = None
    for _ in range(n_sims):
        p, q, truth = make_pair("cointegrated", n=n_obs, rng=rng)
        eg = engle_granger(p, q)
        ou = fit_ou(eg.spread)
        betas.append(eg.beta)
        hls.append(ou.half_life)
        true_hl = np.log(2) / truth.kappa * 252
    out["recovery"] = dict(beta_mean=float(np.mean(betas)),
                           beta_sd=float(np.std(betas)),
                           true_beta=1.0,
                           hl_median=float(np.median(hls)),
                           true_half_life_days=float(true_hl))

    # ---- 4: BH-FDR on a mixed universe -----------------------------------
    kinds = (["cointegrated"] * 50 + ["random_walks"] * 75
             + ["correlated_rw"] * 75)
    pvs, is_true = [], []
    for k in kinds:
        p, q, _ = make_pair(k, n=n_obs, rng=rng)
        pvs.append(engle_granger(p, q).pvalue)
        is_true.append(k == "cointegrated")
    pvs, is_true = np.array(pvs), np.array(is_true)
    disc_raw = pvs < 0.05
    disc_bh = benjamini_hochberg(pvs, q=0.05)
    def _fdr(d):
        return float((d & ~is_true).sum() / max(d.sum(), 1))
    out["scan"] = dict(raw_discoveries=int(disc_raw.sum()),
                       raw_false=int((disc_raw & ~is_true).sum()),
                       raw_fdr=_fdr(disc_raw),
                       bh_discoveries=int(disc_bh.sum()),
                       bh_false=int((disc_bh & ~is_true).sum()),
                       bh_fdr=_fdr(disc_bh),
                       n_true=int(is_true.sum()))

    # ---- 5: backtest sanity ----------------------------------------------
    for kind in ("cointegrated", "correlated_rw"):
        sharpes, tims = [], []
        for _ in range(40):
            p, q, _ = make_pair(kind, n=2000, rng=rng)
            bt = zscore_backtest(p, q)
            sharpes.append(bt.sharpe)
            tims.append(bt.time_in_market)
        out[f"bt_{kind}"] = dict(sharpe_median=float(np.median(sharpes)),
                                 sharpe_iqr=[float(np.quantile(sharpes, .25)),
                                             float(np.quantile(sharpes, .75))],
                                 time_in_mkt=float(np.mean(tims)))

    # ---- 6: ADF lag policy on the realistic noisy_rw null ----------------
    out["lag_policy"] = lag_policy(n_sims=n_sims, n_obs=n_obs, rng=rng)

    if verbose:
        import json
        print(json.dumps(out, indent=2))
    return out


# lag policies compared, as kwargs to statsmodels.tsa.stattools.coint
LAG_POLICIES = {
    "aic_auto_CURRENT": dict(autolag="aic", maxlag=None),
    "bic_auto":         dict(autolag="bic", maxlag=None),
    "aic_cap8":         dict(autolag="aic", maxlag=8),
    "fixed_lag_1":      dict(autolag=None, maxlag=1),
    "fixed_lag_5":      dict(autolag=None, maxlag=5),
}


def lag_policy(n_sims: int = 200, n_obs: int = 1500,
               rng: np.random.Generator | None = None):
    """Rejection rate per lag policy on noisy_rw (H0) and cointegrated (H1).

    Reported as a size/power pair per policy. The right choice is the one
    that HOLDS SIZE on noisy_rw, not the one that maximises discoveries --
    picking on discovery count is p-hacking. Note that no policy fully
    controls size here; the least-bad is the incumbent, and the residual
    over-rejection is why scan.py carries a noise_dominated flag instead
    of trusting a small p-value on a flat-ACF spread.
    """
    from statsmodels.tsa.stattools import coint
    rng = rng or np.random.default_rng(20260821)
    res = {k: {"size_noisy_rw": 0, "power": 0} for k in LAG_POLICIES}

    for kind, key in (("noisy_rw", "size_noisy_rw"),
                      ("cointegrated", "power")):
        for _ in range(n_sims):
            p, q, _ = make_pair(kind, n=n_obs, rng=rng)
            for name, kw in LAG_POLICIES.items():
                try:
                    _, pv, _ = coint(p, q, trend="c", **kw)
                except Exception:
                    pv = 1.0
                if pv < 0.05:
                    res[name][key] += 1

    return {name: dict(size_noisy_rw=v["size_noisy_rw"] / n_sims,
                       power=v["power"] / n_sims)
            for name, v in res.items()}


if __name__ == "__main__":
    run()
