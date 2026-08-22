"""Ground-truth validation of the roll-gap treatment, BEFORE real margins.

Same discipline as everywhere else in this project: establish that the
instrument reads a known answer correctly before pointing it at data
whose answer we do not know.

Two questions, each asked with and without roll masking:

  SIZE   On a random-walk margin (H0: no physical tether) observed
         through stitched contracts, how often does ADF reject at 5%?
         Contract offsets are extra variance with no memory, so the
         untreated series should over-reject -- offsets look like fast
         mean reversion around a shifting level.

  BIAS   On an OU margin (H1: a real tether), what does the memory
         parameter d read, and does masking recover the true value?

Note the test here is ORDINARY ADF, not the MacKinnon cointegration
null. The margin weights are fixed technical ratios, so nothing is
estimated and the cointegration correction does not apply -- one of the
real statistical advantages of production spreads over ETF pairs.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import adfuller

from statarb.synth import make_margin, dejump
from fractional_census import elw

# calibrated to the two confirmatory spreads (see fetch_futures diagnostics)
CALIB = {
    "crack-like": dict(level_sd=10.4, roll_offset_sd=2.14,
                       roll_offset_mean=-0.23),
    "crush-like": dict(level_sd=0.667, roll_offset_sd=0.208,
                       roll_offset_mean=-0.028),
}


def run(n_sims=200, n_obs=4183, half_life=20.0, seed=20260822):
    rng = np.random.default_rng(seed)
    print(f"n_sims={n_sims}  n_obs={n_obs}  true half-life={half_life}d  "
          f"(nominal size 5%)\n")

    for label, kw in CALIB.items():
        print(f"--- {label} "
              f"(level sd {kw['level_sd']}, roll step sd "
              f"{kw['roll_offset_sd']}) ---")

        # ---- SIZE on a random-walk margin --------------------------------
        rej_raw = rej_msk = 0
        for _ in range(n_sims):
            x, mask, _ = make_margin(n_obs, rng, kind="rw",
                                     half_life=half_life, **kw)
            rej_raw += adfuller(x, autolag="aic")[1] < 0.05
            rej_msk += adfuller(dejump(x, mask), autolag="aic")[1] < 0.05
        print(f"  SIZE (H0 random-walk margin, nominal 5%)")
        print(f"    untreated : {rej_raw / n_sims:6.1%}"
              f"   {'ANTI-CONSERVATIVE' if rej_raw / n_sims > 0.09 else 'ok'}")
        print(f"    masked    : {rej_msk / n_sims:6.1%}"
              f"   {'ANTI-CONSERVATIVE' if rej_msk / n_sims > 0.09 else 'ok'}")

        # ---- POWER and d-BIAS on an OU margin ----------------------------
        pw_raw = pw_msk = 0
        d_raw, d_msk = [], []
        for _ in range(max(n_sims // 4, 25)):
            x, mask, _ = make_margin(n_obs, rng, kind="ou",
                                     half_life=half_life, **kw)
            xm = dejump(x, mask)
            pw_raw += adfuller(x, autolag="aic")[1] < 0.05
            pw_msk += adfuller(xm, autolag="aic")[1] < 0.05
            d_raw.append(elw(x)[0])
            d_msk.append(elw(xm)[0])
        k = max(n_sims // 4, 25)
        print(f"  POWER (H1 OU margin, half-life {half_life}d)")
        print(f"    untreated : {pw_raw / k:6.1%}   d_hat "
              f"{np.mean(d_raw):+.3f} +/- {np.std(d_raw):.3f}")
        print(f"    masked    : {pw_msk / k:6.1%}   d_hat "
              f"{np.mean(d_msk):+.3f} +/- {np.std(d_msk):.3f}")
        print()

    print("A true OU margin is I(0), so d_hat should sit near 0.0;")
    print("distance from 0 is the contract-offset contamination.")


if __name__ == "__main__":
    run()
