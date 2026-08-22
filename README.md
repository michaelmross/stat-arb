# statarb — Cointegration-Based ETF Pairs Framework

A validated pipeline for statistical arbitrage research on ETF pairs:
Engle–Granger cointegration testing, Ornstein–Uhlenbeck spread modeling,
Kalman-filtered hedge ratios, and a look-ahead-free walk-forward backtester —
all verified against synthetic ground truth before any real data is touched.

## Modules

| Module | Purpose |
|---|---|
| `statarb/synth.py` | Ground-truth generators: cointegrated pairs (common trend + OU spread), independent random walks, correlated-but-not-cointegrated random walks (the spurious-regression trap), and `noisy_rw` — a unit-root spread buried under white noise, calibrated to a real same-index ETF pair. |
| `statarb/coint.py` | Engle–Granger two-step with MacKinnon cointegration p-values (correct null for an estimated hedge ratio), plus Benjamini–Hochberg FDR control for pair scans. |
| `statarb/ou.py` | Exact AR(1) MLE for OU parameters; half-life and stationary standard deviation. |
| `statarb/kalman.py` | Random-walk state-space model for time-varying (alpha, beta); one-step innovations usable as an online spread. |
| `statarb/backtest.py` | Walk-forward z-score backtest. Parameters frozen in-sample, traded out-of-sample, refit each roll. Tradability gate (EG p-value + half-life bounds) re-checked every window. Explicit dollar-neutral leg weights and per-leg costs. |
| `statarb/validate.py` | The validation suite (below). |
| `statarb/data.py` | Daily OHLC CSV loader for Stooq and Tiingo exports. Format auto-detection, adjusted-close preference, per-pair date alignment, and dirty-data diagnostics (duplicate dates, non-positive prices, stale-price runs). |
| `statarb/universe.py` | Candidate-pair universe of structurally related ETFs: same-index trackers, sector/parent, hedged/unhedged, HOLDRs-style baskets, plus deliberate negative controls. 84 tickers, 104 pairs. |
| `fetch_data.py` | Fetches the universe from Tiingo (token via env var or file, sent as an auth header). Stooq path is verify-only — see the real-data notes. |
| `scan.py` | BH-corrected scan with a discovery/evaluation split, cost-viability ratio, and a noise-dominated reliability flag. |
| `validate_lag.py` | Standalone deep-dive on ADF lag policy: how the `noisy_rw` null was calibrated to a real pair, and the size/power table it produced. |
| `test_data.py` | Ground-truth tests for the real-data path (29 checks): CSV round-trip fidelity, beta recovery through disk, dirty-data detection, universe invariants, end-to-end scan. |
| `fractional_census.py` | Exact local Whittle (Shimotsu–Phillips) memory-parameter estimator. `validate()` on ARFIMA ground truth, `anchors()` for the figure's calibration bands, `--census` for the 130-spread census. |
| `kalman_variant.py` | Roadmap 2: fully online Kalman-innovation strategy. `calibrate()` picks delta on synthetics only; `--real` evaluates the scan's pairs. |
| `controls_and_power.py` | Adversarial checks behind the v2 revision: no-market control universes for the p-vs-edge correlation, power vs spread amplitude, and Sharpe t-stats. |
| `make_figures.py` | Regenerates all five figures from `scan_v2.json` + census + `data/`. |
| `texcheck.py` | Structural integrity check on `etf_pairs_note.tex` — figure targets, `\ref` resolution, citations, environment and brace balance. Not a compiler; catches what would break a build or silently emit `??`. |
| `demo.py` | End-to-end run on one synthetic pair; writes `diagnostics.png`. |

## Validation results (n = 200 sims per cell, seed 20260821)

**Size.** On independent random walks the EG test rejects at 3.5%
(nominal 5%); on correlated random walks, 4.0%. The naive ADF-on-residuals
test rejects at 13–14.5% — anti-conservative, as theory predicts, which is
why `coint.py` uses the MacKinnon cointegration null.

**Power.** 100% detection on genuinely cointegrated pairs at these
parameter settings (n = 1500 obs, half-life ≈ 7 days).

**Recovery.** Hedge ratio: 1.0003 ± 0.006 against a true value of 1.
Half-life: median 6.75 days against a true 6.99.

**Multiple testing.** On a 200-pair universe (50 true, 150 impostors), raw
5% thresholding yields FDR ≈ 21%. BH control brings realized FDR to 5.7%
while keeping 50/53 discoveries true. Scanning without correction is how
spurious "pairs" enter a book.

**Backtest discrimination.** Median out-of-sample Sharpe ≈ 1.9 on true
cointegrated pairs (2bp/leg costs) versus ≈ 0.0 on correlated impostors —
and the tradability gate keeps the strategy out of the market ~97% of the
time on impostors. The machinery finds edge where edge exists and stands
down where it doesn't.

**Lag policy (added 2026-08-21, after the first real-data run).** The
five cells above all passed while the EG test was rejecting a realistic
null a third of the time. Regimes 1–3 emit clean innovations, so nothing
in the suite exercised MA-induced size distortion. Regime 4 (`noisy_rw`)
fixes that: a slowly wandering unit-root spread buried under white noise,
calibrated to the real SPY/IVV spread (5.28bp wide, ACF flat near 0.35
out to lag 40). Rejection rates at nominal 5%, n = 2264:

| ADF lag policy | size on `noisy_rw` | power |
|---|---|---|
| **`aic`/auto (incumbent)** | **33.5%** | 100% |
| `bic`/auto | 95.0% | 100% |
| `aic`, maxlag 8 | 100% | 100% |
| fixed lag 1 | 100% | 100% |
| fixed lag 5 | 100% | 100% |

The incumbent setting is correct and is kept. Every alternative is
catastrophically anti-conservative on the null that real same-index ETF
pairs actually follow — the long augmentation is absorbing noise, not
signal. The residual 33% is the real lesson: on a noise-dominated spread
the EG p-value cannot be trusted whatever the lag policy, which is why
`scan.py` reports a `noise_dominated` flag alongside every p-value.

Method note: the policy is chosen on **size**, never on discovery count.
Picking the setting that yields the most pairs is p-hacking.

## Real-data results (2026-08-21)

84 ETFs from Tiingo (`adjClose`, total-return adjusted), 2010-01-04 to
2026-08-21. 104 structural candidate pairs, 98 with enough history.
Pair *selection* on data through 2018-12-31; walk-forward backtest scored
only on 2019 onward, so the out-of-sample period is out-of-sample for the
choice of pair as well as for the parameters.

**The core tradeoff, measured.** Across all 98 pairs, Spearman
correlation between EG p-value and cost-viability ratio is **+0.446
(p = 4.2e-06)**: the wider a spread is relative to costs, the weaker its
cointegration. Caveat 1 below is no longer a caution, it is a result.

| relation | n | median p | median edge ratio | median half-life | noise-dominated |
|---|---|---|---|---|---|
| same_index | 43 | 0.29 | 9.2 | 11.8d | 38/43 |
| sector_parent | 30 | 0.36 | 100.2 | 115.4d | 30/30 |
| hedged | 11 | 0.52 | 90.3 | 57.1d | 10/11 |
| holdrs | 10 | 0.54 | 152.8 | 141.7d | 10/10 |
| control | 4 | 0.38 | 397.2 | 163.4d | 4/4 |

**Discoveries.** 17 pairs at raw p < 0.05, **8 after BH** — all of them
same-index. Nothing survived in sector/parent, hedged, HOLDRs, or the
controls. BH did visible work: sector/parent produced five raw hits
(`XLF/VFH`, `XLV/VHT`, `XLI/SPY`, `XLU/SPY`, `XBI/XLV`), every one
noise-dominated, none surviving correction. That is the mechanism by
which spurious pairs enter a book, caught on live data.

Three of the eight (`SPY/VOO`, `IVV/VOO`, `IWM/VTWO`) have half-lives of
0.28–0.42 days — sub-daily, noise-dominated, rejected by the tradability
gate. Statistically significant and economically empty.

**Out-of-sample.** Only 5 of 98 pairs clear all three filters
(cost-viable, not noise-dominated, half-life in bounds):

| pair | OOS Sharpe | time in market | |
|---|---|---|---|
| VTI/SCHB | +0.25 | 0.1% | 5 round trips in 7.6 years |
| BND/SPAB | +0.15 | 13.0% | |
| AGG/SPAB | −0.08 | 3.1% | |
| AGG/BND | −0.37 | 16.8% | |
| LQD/USIG | −1.19 | 11.3% | flagged noise-dominated |

Median OOS Sharpe **−0.08**. The synthetic pipeline returns ≈ 1.9 on
genuinely cointegrated pairs; real ETFs return approximately zero. The
machinery is not the problem — the spreads are not there.

**Where the survivors cluster.** Everything with credible cointegration
and a decaying (not flat) spread ACF is a *bond* pair: AGG, BND, SPAB,
LQD, USIG. NAV-anchored creation-redemption is a real economic tether in
a way that shared index membership is not. If this line is worth
continuing, that is the thread to pull.

## Scan v2 — fixed-income cohort (confirmatory follow-up)

Scan v1 ended with a hypothesis: every credible discovery was a bond
pair, so NAV-anchored creation-redemption may be a real economic tether
where shared index membership is not. v2 tests that on **new pairs**
rather than re-cutting v1's data — 30 added tickers across 13
duration-disciplined families (Treasury by maturity bucket, TIPS broad vs
short, IG intermediate vs short, MBS, EM USD sovereign, muni, preferred,
aggregate). Universe goes to 114 tickers / 136 pairs, 130 testable. Same
2010 start, same `--discovery-end 2018-12-31`.

Cohorts are labelled in `universe.py` (`Candidate.cohort`) and reported
separately, so "identified in v1, tested in v2" is a property of the code
rather than a claim in prose. The BH denominator moves 98 → 130, which
re-thresholds every pair, so `scan_v2.json` supersedes `scan_full.json`.

**Discovery.** 28 raw at p < 0.05, **14 after BH** — the same 8 from v1
plus 6 new: `SCHZ/SPAB`, `SCHZ/AGG`, `VGSH/SCHO`, `IGSB/SPSB`,
`VTIP/STIP`, `VCIT/IGIB`. The bond cluster replicates.

**Out-of-sample it does not.** Of 14 discoveries, 10 clear the half-life
gate and only 7 ever take a position. Median OOS Sharpe **0.00**. The
best is `VCIT/IGIB` at +0.53 (t = 1.46, not significant, and flagged
noise-dominated). The only significant result in the whole study remains
`LQD/USIG` at −1.19 (t = −3.29) — significantly *negative*. The
fixed-income cohort's median Sharpe is +0.00 against v1's −0.08: the
hypothesis survives as a *detection* claim and dies as a *return* claim.

**March 2020 — the actual test.** This is why the cohort was worth
running. Spread z-scores computed on parameters frozen before
2020-02-15, i.e. what a live book would have seen:

| | |
|---|---|
| pairs breaching the 4σ stop | **10 of 10** |
| peak \|z\| range | **9.5σ – 72.3σ** |
| worst dislocation (`BND/SPAB`, 2020-03-13) | 476bp, against a 6.6bp pre-crisis spread sd |

On 2020-03-13 BND closed **+4.2%** while SPAB closed **−0.6%** — two
aggregate-bond trackers on nearly the same index, 4.8% apart in one
session. The tether does not merely weaken under stress; it fails
hardest exactly where it was supposed to be strongest, because both legs
decoupled from NAV by *different* amounts. Verified against raw closes,
not a data artifact.

The machinery handled it: the tradability gate had stood 7 of the 10
pairs down before the window, and the 4σ stop capped the three that were
in the market (`AGG/BND` −1.05%, `VCIT/IGIB` +1.56%, `LQD/USIG` −1.65%
over the window). Equal-weighted crisis return −0.11%. That is the
system working as designed — and it is the same reason it earns nothing.
It survives by not participating.

**Conclusion.** The bond-cluster hypothesis is confirmed as a
*statistical* phenomenon and refuted as a *tradable* one. Cointegration
among bond ETFs is real, detectable, and replicates on fresh pairs; its
spreads are still too thin to clear 2bp/leg in calm markets, and in the
one episode wide enough to pay, the relationship broke by up to 72σ.

## Roadmap 2 — Kalman-innovation variant (done, negative)

`kalman_variant.py`. Fully online: state `(alpha_t, beta_t)` is a random
walk, the signal is the standardized one-step innovation, and there are
no frozen windows, no refits, and **no tradability gate**. Costs are
charged on all turnover including the continuous rebalancing that beta
drift induces. The state-noise scale is chosen on synthetic ground truth
only (`calibrate()`): median synthetic Sharpe 1.026 / 1.403 / **1.485**
at delta = 1e-4 / 1e-5 / **1e-6**.

On the same ten pairs, same out-of-sample window, same costs:

| | walk-forward | Kalman |
|---|---|---|
| median OOS Sharpe | +0.00 | **−0.15** |
| significant at \|t\|>1.96 | 1 of 10 | **3 of 10** |
| mean time in market | ~7% | **34.9%** |

All three significant results are negative (`VTIP/STIP` t = −2.53,
`IGSB/SPSB` t = −11.07, `VCIT/IGIB` t = −15.64). `VCIT/IGIB`, the
baseline's best pair at +0.53, becomes −5.67 online.

**The gate-attribution finding.** This is the useful part. The
walk-forward baseline's headline result was "median Sharpe 0.00" — no
edge, but no bleeding either. That number was not produced by the
estimator; it was produced by the **tradability gate standing the
strategy down**. The Kalman variant is the same statistical machinery
with the gate removed and adaptation improved, and it is in the market
five times as often and loses money. What looked like a well-behaved
null result was an abstention result. Continuous adaptation does not
rescue a spread that has no exploitable reversion — it just collects
more of the costs.

## Roadmap 6 — fractional integration census (substantially answers it)

`fractional_census.py`. The `noise_dominated` flag is a binary two-point
ACF condition; the continuous version of the same question is the memory
parameter d, estimated by exact local Whittle (Shimotsu–Phillips), with
bandwidth m = n^0.6 fixed in advance and the estimator validated on
ground truth before touching real spreads.

Validation (n = 2264, nominal se 0.049): ARFIMA d = 0.0 → −0.012,
0.4 → +0.390, 0.8 → +0.820, **1.0 → +1.000**. On the project's two
calibrated composite nulls: OU+noise (H1) → **0.317**, RW+noise (H0) →
**0.543**. The two separate by ~3.5 se — where the EG p-value rejected
the H0 null 33% of the time, d̂ distinguishes them. That is what makes
this a partial answer to roadmap 6.

Census over all 130 tested spreads:

| group | median d̂ | n |
|---|---|---|
| all tested | 0.93 | 130 |
| BH discoveries | **0.51** | 14 |
| non-discoveries | 0.94 | 116 |
| controls | 1.01 | 4 |

**The discoveries are not I(0).** Median d̂ = 0.51 puts them just past
the stationarity boundary — nonstationary, mean-reverting only at long
horizons. Cointegration testing labels them stationary; the memory
parameter says they are marginal. That is a cleaner explanation of the
whole study's out-of-sample record than anything in the scan output: a
spread with d ≈ 0.5 reverts too slowly to pay for 2bp/leg before it
wanders again. Controls at 1.01 and non-discoveries at 0.94 confirm the
estimator is reading true unit roots correctly.

**Caveat on the flag.** d̂ agrees with a `d̂ > 0.43` rule on 92% of
pairs, but that agreement is carried almost entirely by the
non-discovery mass. *Within* the 14 discoveries, flagged and unflagged
have essentially the same memory (0.50 vs 0.51). So d̂ and the binary
flag are not measuring the same thing at the top of the table, and the
flag should not be read as a proxy for d̂ there.

## Production margins — crack and crush (2026-08-22)

A different universe with the opposite structural properties: crack and
crush margins are enforced by physical production, not index membership,
so the tether is real. Nine continuous front-month contracts from Yahoo,
2010–2026. Method decisions and their reasons are in
`statarb/futures.py`; the short version:

- **Price space, not log space.** A margin is a linear combination of
  dollar prices with fixed technical weights. Nothing needs logs, and log
  space cannot represent 2020-04-20, when WTI settled at −$37.63.
  `load_panel(..., log_transform=False)` retains that observation instead
  of deleting it and fabricating a −60% two-day move across the hole.
- **Fixed weights, so nothing is estimated.** Ordinary ADF applies (no
  MacKinnon correction), there is no spurious-regression channel, and the
  d-census runs on the margin directly rather than on a residual.
- **Unadjusted stitching is correct.** The stitched front *is* the true
  prompt margin every day. Back-adjustment would smooth the roll steps at
  the cost of destroying the level, and the level is the anchor. Rolls are
  handled in the return accounting instead.
- **Roll window chosen empirically**, not assumed: −0/+1 business day
  around each leg's expiry. Exact-expiry-only catches 1–3% of large jumps
  while −0/+1 catches 27–40%, which pins the mechanism — the series
  switches contract the day *after* expiry.

**Roll masking is essential, and it changes the answer.**
`validate_roll.py` measures ADF size on a random-walk margin observed
through stitched contracts: **11.5%** (crack-calibrated) and **30.0%**
(crush-calibrated) against a nominal 5%, falling to **2.5%** and **4.0%**
once masked. On the real data this is not academic — the crack's ADF
p-value is 0.013 raw and **0.252** masked. Untreated, we would have
"discovered" the crack as a cointegrated spread.

**The registered prediction was wrong.** It was recorded in advance that
these margins would show decaying ACFs and d̂ well below the ETF
discoveries' 0.51, because the tether is physical. Neither held:

| | crack 3-2-1 | board crush | cattle (exploratory) |
|---|---|---|---|
| ADF p (masked) | 0.252 | 0.921 | 0.0013 |
| half-life | **158.4d** | **160.1d** | 41.6d |
| ACF 1 / 10 / 40 | 0.99 / 0.91 / 0.69 | 0.98 / 0.82 / 0.68 | 0.98 / 0.85 / 0.51 |
| d̂ | 0.880 | 0.650 | 0.890 |
| OOS Sharpe | −0.01 | +0.00 | +0.54 (t=1.49) |
| round trips | 3 | **0** | 5 |
| windows tradable | 27% | 3% | 17% |

The ACFs are flat, not decaying, and d̂ is *above* 0.51, not below.

**The failure mode is the exact opposite of the ETF study.** There,
cointegration was real but spreads were too thin to pay for execution.
Here execution is nearly free — a round trip costs **0.32%** of one
margin sd for the crack and 1.96% for the crush — but the margin does not
revert fast enough to trade. Half-lives near 160 days sit ~3× beyond the
60-day tradability gate, so the gate stands the strategy down and costs
never get a chance to matter. The crush took **zero** round trips in
seven out-of-sample years. Physical arbitrage anchors the *level* of
these margins on a multi-quarter horizon; it does not make them
mean-revert on a horizon a daily strategy can trade.

**A caveat on the d̂ anchor.** ELW is biased upward on strongly
autoregressive I(0) series — `validate_roll.py` shows it reading +0.605
on a synthetic margin that is I(0) by construction. So d̂ is compared
against an anchor computed at each margin's own fitted persistence. But
that comparison loses power precisely when persistence is high: at a
fitted half-life of ~160 days the anchor is ~0.97, so almost any d̂ reads
as "consistent with I(0)". Here that verdict means *cannot distinguish*,
not *is stationary*.

## Archive integrity

`etf_pairs_note.tex` is the write-up. Its numeric claims were checked
against the archived JSONs by independent recomputation on 2026-08-22:
**17 of 17 verified**, the single apparent mismatch being a rounding
convention on an exact tie (LQD/USIG time-in-market is exactly 11.25%;
the note rounds half-up to 11.3%, Python's banker's rounding gives
11.2%). Verified quantities include the 98-pair and 130-pair test
counts, ρ = +0.446, 5-of-98 disjointness, 95-of-98 edge ratios above 1,
28 raw / 14 BH discoveries with the 8/6 cohort split, the crisis peak-|z|
range 9.5–72.3, and the full LQD/USIG row.

```bash
python texcheck.py            # structural check on the .tex
python make_figures.py        # regenerate all five figures
python fractional_census.py --anchors   # derive fig_dcensus anchor bands
```

Two things this repo does **not** establish. No TeX toolchain is
installed here, so the note has been structurally validated but never
compiled — `texcheck.py` is a proxy, not a build. And figure PDFs are
not byte-reproducible across matplotlib and platform builds; the
underlying census data reproduces to 1e-15, the bytes do not.

Scope note: the note is the v2 paper. It covers the initial scan, the
fixed-income cohort, the crisis episode, lag policy, and the
identification controls. It does **not** cover the Kalman variant
(roadmap 2) or the fractional census (roadmap 6) — which is why
`fig_dcensus.pdf` is generated but unreferenced. Those two sections
above are the material for a v3.

## Honest caveats before real capital

1. Synthetic Sharpe ≈ 2 says the *pipeline* works, not that markets offer
   such spreads. Real ETF pairs with strong cointegration (SPY/IVV class)
   have tiny spreads that costs mostly consume; pairs with fat spreads
   have weaker, breakable cointegration. The tradeoff is the whole game.
   **Now measured** — Spearman(p-value, edge ratio) = +0.446 across 98
   real pairs, median out-of-sample Sharpe −0.08. See real-data results.
2. Costs here are 2bp/leg one-way. Add short borrow fees, and note that
   spread z-scores are computed at the close but real fills slip.
3. Cointegration regimes break (2007 quant unwind). The per-window gate
   mitigates but does not eliminate this; add portfolio-level drawdown
   stops before live use.
4. Position sizing is binary ±1 here. OU-optimal band placement
   (Leung–Li) and volatility targeting are natural upgrades.

## Roadmap

1. ~~Real data: loader for daily OHLC CSVs (Stooq/Tiingo exports), a
   candidate-pair universe of structurally related ETFs, BH-corrected
   scan.~~ **Done 2026-08-21 — result was negative.** See real-data
   results above.
2. ~~Kalman-innovation trading variant (fully online, no frozen
   windows).~~ **Done 2026-08-21 — negative, and diagnostic.** Median OOS
   Sharpe −0.15 against the baseline's 0.00, 3 of 10 significantly
   negative. Established that the baseline's null result came from the
   tradability gate abstaining, not from the estimator. See above.
3. Johansen baskets for 3+ ETFs.
4. Cost model refinement: half-spread + borrow; capacity estimates.
5. Paper trading harness (broker API, dry-run mode, kill switch, size
   caps) — only after 1–4 hold up.

**Read 2–5 in light of 1.** Items 2–4 all improve *extraction*, and
extraction is not the binding constraint: cost-viable spreads and
cointegrated spreads are close to disjoint sets in this universe. Item 5
is explicitly gated on 1–4 holding up, and 1 did not. The honest next
move is not a better estimator on this universe but a different universe
— the bond-ETF cluster is the only place the evidence points.

Two smaller follow-ups that are genuinely open:

6. ~~The `noisy_rw` null over-rejects at 33% even under the best lag
   policy. A test with correct size on that null would make the scan
   trustworthy rather than merely flagged.~~ **Substantially answered
   2026-08-21** by the fractional-integration census: exact local
   Whittle separates the two composite nulls (0.317 vs 0.543, ~3.5 se)
   where the EG p-value could not, and prices every spread on a
   continuous scale. Remaining gap: d̂ is a descriptive statistic here,
   not yet a sized hypothesis test with a stated rejection rule.
7. `scan.py` aligns each pair independently and applies BH across
   whatever survived the history filter. Pairs skipped for short history
   (`QQQM`, `GLDM`, `USHY` families) silently shrink the test count and
   therefore loosen the BH threshold for everyone else.
8. **Deferred by choice:** promote d̂ to a first-class `scan.py` output
   column alongside `noise_dominated`. It costs ~15s for the whole
   universe and the census showed d̂ is the more informative statistic.
   Not done yet because `scan_v2.json` is the note's dataset: changing
   the output schema now would mean the archived file no longer matches
   what `scan.py` emits. Do it after tagging, as an explicit
   `schema_version: 3`. Until then d̂ lives in
   `fractional_census_results.json` and joins on pair name.

## Run

```bash
pip install numpy scipy pandas statsmodels matplotlib requests
python -m statarb.validate    # ~15–20 min full suite (incl. lag policy)
python demo.py                # single-pair demo + figure
```

Real data:

```bash
python test_data.py                              # 29 checks, no network
python fetch_data.py --source tiingo --out data --start 2010-01-01
python scan.py --data-dir data --discovery-end 2018-12-31
python validate_lag.py                           # lag-policy deep-dive
```

Follow-up studies (both ground-truth-first; run with no flag to validate,
with the flag to reproduce the real-data JSON):

```bash
python fractional_census.py            # ELW validation on ARFIMA truth
python fractional_census.py --census   # -> fractional_census_results.json
python kalman_variant.py               # delta calibration on synthetics
python kalman_variant.py --real        # -> kalman_results.json
```

`validate()` in `fractional_census.py` and `calibrate()` in
`kalman_variant.py` are the natural fast-mode CI cells — they are the
same ground-truth-before-real-data pattern the rest of the project uses,
and both run in well under a minute.

`fetch_data.py` reads a Tiingo token from `TIINGO_API_KEY` or from
`.tiingo_token` beside the script (gitignored), and sends it as an
`Authorization` header rather than a URL parameter.

**Two real-data gotchas, both load-bearing.**

Tiingo signals its hourly rate cap with **HTTP 200 and
`Content-Type: text/csv`**, putting an error sentence in the body. A
loader that trusts the status code writes 136-byte "price files" and
scans them as data. `fetch_data.py` validates the body, names the cap,
and resumes from the last good ticker on re-run.

Stooq's CSV endpoint is now behind a JavaScript proof-of-work bot check,
so scripted fetching no longer works. Download by hand (per ticker, or
the bulk archive at `stooq.com/db/h/`) and use `--source stooq` to verify
what is already on disk. Note that Stooq closes are split- but **not**
dividend-adjusted: for same-index trackers the differing ex-dividend
dates stamp a sawtooth into the log spread that reads as mean reversion
and is not tradable. Prefer Tiingo `adjClose`.
