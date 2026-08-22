"""Standalone deep-dive on ADF lag policy (calibration + size/power).

The headline cell now lives in statarb/validate.py as cell 6; this
script keeps the calibration diagnostics that derived it -- it shows
HOW the noisy_rw null was matched to real SPY/IVV, which validate.py
only consumes. The generator itself is synth.make_pair.

Original note follows.

Which ADF lag policy should coint.py use? Decide it on ground truth.

WHY THIS EXISTS
On real ETF pairs the default `autolag='aic'` with `maxlag=None` selects
~27 augmentation lags and drives the SPY/IVV ADF t-stat from -32 to -2.3
(p 0.38). That LOOKS like the well-known ADF power loss from over-long
augmentation, and the first reading of it here was exactly that: a bug
under-detecting cointegration.

That reading was wrong, and this script is what proved it wrong.

The real SPY/IVV spread is not an AR(1). Its ACF is flat at ~0.31-0.39
out to lag 40 (an AR(1) at 0.37 would be 0.007 by lag 5) and its yearly
mean wanders -5.9bp -> +3.3bp -> -3.7bp. That is a unit root buried under
white noise -- not cointegration -- and it is the classic Schwert (1989)
MA size-distortion case, where too FEW lags make ADF over-reject
catastrophically. The 27 lags were absorbing noise, not signal. The -32
t-stat at maxlag=0 was the spurious number.

Every candidate "fix" turns out anti-conservative on that null: bic/auto
rejects 95% of the time, and every fixed or capped policy rejects 100%,
at nominal 5%. The incumbent aic/auto is the least-bad at ~33%.

The residual 33% is itself the important result: for noise-dominated
spreads the EG p-value cannot be trusted at all, whatever the lag policy.
That is why scan.py carries a noise_dominated flag rather than reading a
small p-value at face value.

Method note: pick on size, never on discovery count. The policy that
maximises discoveries is p-hacking; the target is nominal size with the
best power available at that size.

Run:  python validate_lag.py
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from synth import make_pair, add_micro_noise

# lag policies to compare: (label, kwargs to statsmodels.coint)
POLICIES = [
    ("aic/auto  (current)", dict(autolag="aic", maxlag=None)),
    ("bic/auto",            dict(autolag="bic", maxlag=None)),
    ("aic/cap8",            dict(autolag="aic", maxlag=8)),
    ("fixed lag 1",         dict(autolag=None, maxlag=1)),
    ("fixed lag 5",         dict(autolag=None, maxlag=5)),
]


def observed_ar1(logp, logq):
    """AR(1) of the OLS residual -- the thing we match to real data."""
    X = sm.add_constant(logq)
    resid = sm.OLS(logp, X).fit().resid
    return float(np.corrcoef(resid[:-1], resid[1:])[0, 1])


def calibrate(target_sd_bp, target_ar1, kappa, dt=1 / 252):
    """Solve for (OU vol, per-leg noise) reproducing a REAL pair's spread.

    observed = OU + iid noise on each leg, so with V the observed variance
    and rho the OU autocorrelation:
        observed_AR1 = rho * var_ou / V      -> var_ou = AR1 * V / rho
        V = var_ou + 2 * var_noise           -> var_noise = (V - var_ou)/2
    Returns (spread_sigma, noise_bp) for make_pair/add_micro_noise.

    This matters more than it looks: the default synthetic spread is ~70bp
    wide against 4bp of noise -- trivially detectable. Real same-index ETF
    spreads are ~5bp wide against ~3bp of noise. Lag selection behaves
    completely differently in those two regimes.
    """
    V = (target_sd_bp / 1e4) ** 2
    rho = np.exp(-kappa * dt)
    var_ou = target_ar1 * V / rho
    if var_ou > V:
        raise ValueError("target AR(1) too high for this kappa")
    var_noise = (V - var_ou) / 2.0
    spread_sigma = np.sqrt(var_ou) * np.sqrt(2.0 * kappa)
    return spread_sigma, np.sqrt(var_noise) * 1e4


def run(n_sims=200, n_obs=2264, seed=20260821,
        target_sd_bp=5.28, target_ar1=0.370, kappa=25.0):
    rng = np.random.default_rng(seed)
    spread_sigma, noise_bp = calibrate(target_sd_bp, target_ar1, kappa)

    # --- calibration check: does this reproduce the real pair? ---
    p, q, _ = make_pair("cointegrated", n=n_obs, rng=rng,
                        kappa=kappa, spread_sigma=spread_sigma)
    pn, qn = add_micro_noise(p, q, noise_bp, rng)
    X = sm.add_constant(qn)
    resid = sm.OLS(pn, X).fit().resid
    print(f"calibrated to SPY/IVV: OU vol {spread_sigma:.5f}, "
          f"noise {noise_bp:.2f}bp/leg, true half-life "
          f"{np.log(2)/kappa*252:.1f}d")
    print(f"  synthetic spread: sd {resid.std()*1e4:.2f}bp  "
          f"AR(1) {observed_ar1(pn, qn):.3f}")
    print(f"  real SPY/IVV    : sd {target_sd_bp:.2f}bp  "
          f"AR(1) {target_ar1:.3f}\n")

    # sanity-check the new null reproduces the real flat ACF
    from statsmodels.tsa.stattools import acf as _acf
    tp, tq, _ = make_pair('noisy_rw', n=n_obs, rng=rng)
    _r = sm.OLS(tp, sm.add_constant(tq)).fit().resid
    _a = _acf(_r, nlags=40, fft=True)
    print(f"noisy-RW null check: spread sd {_r.std()*1e4:.2f}bp  "
          f"ACF lag1 {_a[1]:.2f} lag10 {_a[10]:.2f} lag40 {_a[40]:.2f}")
    print(f"  real SPY/IVV     : spread sd 5.28bp  "
          f"ACF lag1 0.37 lag10 0.35 lag40 0.31\n")

    results = {lab: {"size_rw": 0, "size_crw": 0, "size_noisy": 0, "power": 0}
               for lab, _ in POLICIES}

    for kind, key in [("random_walks", "size_rw"),
                      ("correlated_rw", "size_crw"),
                      ("noisy_rw", "size_noisy"),
                      ("cointegrated", "power")]:
        for _ in range(n_sims):
            if kind == "noisy_rw":
                pn, qn, _ = make_pair('noisy_rw', n=n_obs, rng=rng)
                for lab, kw in POLICIES:
                    try:
                        _, pv, _ = coint(pn, qn, trend="c", **kw)
                    except Exception:
                        pv = 1.0
                    if pv < 0.05:
                        results[lab][key] += 1
                continue
            kw_synth = (dict(kappa=kappa, spread_sigma=spread_sigma)
                        if kind == "cointegrated" else {})
            p, q, _ = make_pair(kind, n=n_obs, rng=rng, **kw_synth)
            pn, qn = add_micro_noise(p, q, noise_bp, rng)
            for lab, kw in POLICIES:
                try:
                    _, pv, _ = coint(pn, qn, trend="c", **kw)
                except Exception:
                    pv = 1.0
                if pv < 0.05:
                    results[lab][key] += 1

    print(f"n_sims={n_sims}  n_obs={n_obs}  noise={noise_bp}bp  "
          f"(nominal size 5%)\n")
    print(f"{'policy':22s} {'size:RW':>8s} {'corrRW':>8s} "
          f"{'noisyRW':>9s} {'POWER':>8s}   verdict")
    print("-" * 74)
    best = []
    for lab, _ in POLICIES:
        r = results[lab]
        s1 = r["size_rw"] / n_sims
        s2 = r["size_crw"] / n_sims
        s3 = r["size_noisy"] / n_sims
        pw = r["power"] / n_sims
        se = np.sqrt(0.05 * 0.95 / n_sims)
        ok = max(s1, s2, s3) <= 0.05 + 2.5 * se
        verdict = "size OK" if ok else "ANTI-CONSERVATIVE"
        if ok:
            best.append((pw, lab))
        print(f"{lab:22s} {s1:7.1%} {s2:8.1%} {s3:9.1%} {pw:8.1%}   {verdict}")

    print()
    if best:
        pw, lab = max(best)
        print(f"-> Best power among size-valid policies: {lab} ({pw:.1%})")
    else:
        print("-> No policy held nominal size; do not trust any of them.")
    return results


if __name__ == "__main__":
    run()
