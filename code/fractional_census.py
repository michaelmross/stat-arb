"""Fractional integration census of the discovery-window spreads.

The noise_dominated flag is a binary two-point ACF condition. The
continuous, theoretically grounded version of the same question is the
memory parameter d: a spread is I(d) with d=0 stationary short-memory,
d=1 a unit root, and d in (0.5,1) nonstationary but mean-reverting at
long horizons. We estimate d for every tested spread by exact local
Whittle (Shimotsu-Phillips), which unlike standard local Whittle is
consistent over the whole range needed here.

Protocol: the estimator is validated on ARFIMA ground truth and on the
project's two calibrated composite nulls BEFORE touching real spreads,
and the bandwidth m = n^0.6 is fixed in advance.

ELW objective:  R(d) = log( mean_j I_{(1-L)^d x}(lam_j) )
                        - (2d/m) sum_j log(lam_j),   j = 1..m
minimized over d. Series are demeaned (adequate for a census; the
two-step mean correction changes nothing material here).
"""

from __future__ import annotations

import paths
import numpy as np


def fracdiff(x, d):
    """(1-L)^d x via FFT convolution of binomial coefficients."""
    n = len(x)
    c = np.empty(n)
    c[0] = 1.0
    for k in range(1, n):
        c[k] = c[k - 1] * (k - 1 - d) / k
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    return np.fft.irfft(np.fft.rfft(c, nf) * np.fft.rfft(x, nf), nf)[:n]


def elw(x, m=None, grid=(-0.4, 1.4), step=0.01):
    """Exact local Whittle estimate of d. Returns (d_hat, approx_se)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if m is None:
        m = int(n ** 0.6)
    j = np.arange(1, m + 1)
    lam = 2 * np.pi * j / n
    slog = np.sum(np.log(lam))
    ds = np.arange(grid[0], grid[1] + 1e-9, step)
    best, bd = np.inf, ds[0]
    for d in ds:
        u = fracdiff(x, d)
        w = np.fft.rfft(u)
        I = (np.abs(w[1:m + 1]) ** 2) / (2 * np.pi * n)
        r = np.log(I.mean()) - 2 * d * slog / m
        if r < best:
            best, bd = r, d
    return float(bd), 1.0 / (2 * np.sqrt(m))


def arfima(n, d, rng):
    """ARFIMA(0,d,0): fractionally integrate white noise."""
    return fracdiff(rng.normal(0, 1, n), -d)


def validate(n=2264, n_sims=40, seed=90210):
    from synth import make_pair, add_micro_noise
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    print(f"ELW validation, n={n}, m={int(n**0.6)}, "
          f"nominal se={1/(2*np.sqrt(int(n**0.6))):.3f}")
    for d0 in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        est = [elw(arfima(n, d0, rng))[0] for _ in range(n_sims)]
        print(f"  ARFIMA d={d0:.1f}: d_hat = {np.mean(est):+.3f} "
              f"+/- {np.std(est):.3f}")
    # composite nulls, calibrated to real pairs, spreads via EG residual
    for kind, label in (("cointegrated", "OU+noise (H1, SPY/IVV-scale)"),
                        ("noisy_rw", "RW+noise (H0, SPY/IVV-calibrated)")):
        est = []
        for _ in range(n_sims):
            if kind == "cointegrated":
                p, q, _ = make_pair(kind, n=n, rng=rng, kappa=25.0,
                                    spread_sigma=0.00316)
                p, q = add_micro_noise(p, q, 3.16, rng)
            else:
                p, q, _ = make_pair(kind, n=n, rng=rng)
            X = sm.add_constant(q)
            resid = p - X @ np.linalg.lstsq(X, p, rcond=None)[0]
            est.append(elw(resid)[0])
        print(f"  {label}: d_hat = {np.mean(est):+.3f} +/- {np.std(est):.3f}")


def anchors(scan_path=str(paths.data("scan_v2.json")), n=2264, n_sims=40, seed_ou=555,
            seed_rw=90210):
    """Recompute the two ground-truth anchor bands drawn on fig_dcensus.

    make_figures.fig_dcensus hardcodes these so a figure build stays fast,
    but a hardcoded constant with no derivation is exactly the kind of
    number an archive cannot regenerate. This is the derivation.

      OU anchor  -- cointegrated spread with half-life AND stationary
                    width matched to the BH discoveries' own medians
                    (read from the scan, not hardcoded), no micro noise.
                    This is 'what d looks like if the discoveries really
                    were the I(0) spreads the EG test says they are'.
      RW anchor  -- the SPY/IVV-calibrated noisy_rw null, seed 90210,
                    identical to the composite null in validate().
    """
    import json
    import numpy as _np
    import statsmodels.api as sm
    from synth import make_pair

    bh = [r for r in json.load(open(scan_path))["discovery"] if r["bh"]]
    hl = float(_np.median([r["half_life"] for r in bh]))
    width = float(_np.median([r["stationary_std"] for r in bh]))
    kappa = _np.log(2) * 252 / hl

    rng = _np.random.default_rng(seed_ou)
    ou = []
    for _ in range(n_sims):
        p, q, _ = make_pair("cointegrated", n=n, rng=rng, kappa=kappa,
                            spread_sigma=width * _np.sqrt(2 * kappa))
        X = sm.add_constant(q)
        ou.append(elw(p - X @ _np.linalg.lstsq(X, p, rcond=None)[0])[0])

    rng = _np.random.default_rng(seed_rw)
    rw = []
    for _ in range(n_sims):
        p, q, _ = make_pair("noisy_rw", n=n, rng=rng)
        X = sm.add_constant(q)
        rw.append(elw(p - X @ _np.linalg.lstsq(X, p, rcond=None)[0])[0])

    out = dict(ou_hl_days=hl, ou_width_bp=width * 1e4,
               ou_anchor=float(_np.mean(ou)), ou_sd=float(_np.std(ou)),
               rw_anchor=float(_np.mean(rw)), rw_sd=float(_np.std(rw)))
    print(f"OU anchor (matched hl {hl:.2f}d, width {width*1e4:.1f}bp, "
          f"seed {seed_ou}): {out['ou_anchor']:+.3f} +/- {out['ou_sd']:.3f}")
    print(f"RW anchor (SPY/IVV-calibrated noisy_rw, seed {seed_rw}): "
          f"{out['rw_anchor']:+.3f} +/- {out['rw_sd']:.3f}")
    print("fig_dcensus hardcodes 0.066 +/- 0.053 and 0.543 +/- 0.042")
    return out


def census(scan_path=str(paths.data("scan_v2.json")), data_dir=str(paths.PRICES),
           discovery_end="2018-12-31", out=str(paths.data("fractional_census_results.json"))):
    """Estimate d for every tested discovery-window spread in a scan.

    Recomputes each spread the same way scan.py did -- Engle-Granger OLS
    on data up to `discovery_end` -- so d_hat describes exactly the
    residual whose p-value and noise_dominated flag the scan reported.
    """
    import json
    import pandas as pd
    from data import load_panel
    paths.require_prices('etf', min_files=2)
    from coint import engle_granger

    scan = json.load(open(scan_path))
    rows = scan["discovery"]
    panel = load_panel(data_dir)
    cut = pd.Timestamp(discovery_end)

    out_rows = []
    for r in rows:
        a, b = r["a"], r["b"]
        sub = panel.log_prices[[a, b]].dropna()
        sub = sub[sub.index <= cut]
        eg = engle_granger(sub[a].to_numpy(), sub[b].to_numpy())
        d_hat, se = elw(eg.spread)
        out_rows.append(dict(pair=f"{a}/{b}", relation=r["relation"],
                             cohort=r.get("cohort", "v1"),
                             pvalue=r["pvalue"], bh=r["bh"],
                             flag=r["noise_dominated"], gate=r["gate_pass"],
                             d_hat=d_hat, se=se))
    if out:
        with open(out, "w") as f:
            json.dump(out_rows, f, indent=1)
    return out_rows


def summarise(rows):
    """Group medians -- the numbers the write-up quotes."""
    import numpy as _np
    def med(sel):
        v = [r["d_hat"] for r in rows if sel(r)]
        return (float(_np.median(v)), len(v)) if v else (float("nan"), 0)

    groups = [
        ("all tested",              lambda r: True),
        ("BH discoveries",          lambda r: r["bh"]),
        ("  unflagged discoveries", lambda r: r["bh"] and not r["flag"]),
        ("  flagged discoveries",   lambda r: r["bh"] and r["flag"]),
        ("non-discoveries",         lambda r: not r["bh"]),
        ("controls",                lambda r: r["relation"] == "control"),
    ]
    print(f"{'group':26s} {'median d':>9s} {'n':>4s}")
    for name, sel in groups:
        m, n = med(sel)
        print(f"{name:26s} {m:9.2f} {n:4d}")

    # agreement between the binary flag and a d-based rule
    agree = sum((r["flag"]) == (r["d_hat"] > 0.43) for r in rows)
    print(f"\nflag vs (d_hat > 0.43) agreement: "
          f"{agree}/{len(rows)} = {agree/len(rows):.0%}")


if __name__ == "__main__":
    import sys
    if "--census" in sys.argv:
        rows = census()
        summarise(rows)
    elif "--anchors" in sys.argv:
        anchors()
    else:
        validate()
