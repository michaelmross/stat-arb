# Triangular FX Statistical Arbitrage — Experiment Specification

**Instruments:** EUR/USD (A), GBP/USD (B), EUR/GBP (C). Identity: C = A/B.
**Venue:** OANDA fxTrade practice account, v20 REST/streaming API (free with a practice account; streams independent bid/ask for all three pairs; paper execution mirrors production endpoints).
**Mode:** Paper money, real-time data. Pre-registered hypotheses and decision gates, census-style reporting.

---

## 1. R&D summary and design rationale

### 1.1 The deterministic triangle is dominated at retail costs

Define the log-residual on mids:

    ε_t = ln A_t − ln B_t − ln C_t

The executable (bid/ask) cycle returns for the two directions are:

    D1 (USD→EUR→GBP→USD):  R1 = (1/A_ask) · C_bid · B_bid − 1
    D2 (USD→GBP→EUR→USD):  R2 = (1/B_ask) · (1/C_ask) · A_bid − 1

A deterministic triangular trade is profitable iff R1 > 0 or R2 > 0. This requires |ε| to exceed the **sum of three half-spreads**. Typical OANDA core spreads: EUR/USD ≈ 0.6–1.0 pip (~0.7 bp), GBP/USD ≈ 0.9–1.3 pip (~0.8 bp), EUR/GBP ≈ 0.9–1.5 pip (~1.3 bp). Total cycle cost ≈ **2.5–3.5 bp**.

The interdealer literature (Aiba et al. 2002; Fenn et al. 2009; Foucault, Kozhan & Tham 2017 on EBS data) found cost-exceeding triangular deviations already lasted well under one second by the late 2000s, on feeds with ~0.1–0.5 bp costs and co-located latency. On a retail feed with ~100–300 ms network latency and ~3 bp cycle cost, the expected net opportunity rate is zero. **The deterministic triangle is retained as a measurement arm (census), not a trading arm.**

### 1.2 The optimal statistical version: one-leg cross reversion

The cost structure dictates the algorithm. Instead of trading all three legs (three spreads), trade only the lagging instrument against the rate implied by the other two (one spread):

- Implied cross from majors' mids: **Ĉ_t = A_t / B_t**
- Signal: **z_t = (ln C_t − ln Ĉ_t) / σ_t**, with σ_t a rolling EWMA of the residual's standard deviation.
- If z_t > k_in: sell C (quoted cross rich vs. implied); if z_t < −k_in: buy C. Exit at |z| ≤ k_out or timeout T. Unhedged during the holding period (seconds), which is what makes it statistical rather than deterministic.

Justification for trading C rather than A or B: price discovery in this triangle is concentrated in the dollar majors (deeper liquidity; the cross is largely derivative). This is an assumption to be **verified in Phase 1** via tick-level lead-lag estimation, with a direction filter in the trading arm (only trade when the majors moved last).

Round-trip cost: one full EUR/GBP spread (market in, market out) ≈ **1.0–1.5 pip ≈ 1.2–1.7 bp**.

### 1.3 The decisive unknown (pre-registered)

Retail feeds commonly **construct the cross internally from the majors**. If OANDA's EUR/GBP quote is derived as A/B plus a spread, the mid-residual is identically ~0 and the experiment terminates at Phase 1 with a clean negative result: *no independent cross price exists on this feed, hence no statistical triangle*. This is the first thing Phase 1 measures, and it is the most likely outcome. The residual distribution on this specific feed is the genuine unknown the experiment resolves.

### 1.4 Pre-registered hypotheses

- **H1 (deterministic census):** P(R1 > 0 or R2 > 0 at synchronized executable quotes) ≈ 0 over the sample. Expected: confirmed.
- **H2 (residual structure):** stationary residual σ at 1 s sampling is ≤ 0.3 pip EUR/GBP-equivalent, i.e., well below the one-leg breakeven. Expected: confirmed (possibly σ ≈ 0 per §1.3).
- **H3 (one-leg EV):** E[net per trade] < 0 at all thresholds under the measured cost model. Expected: confirmed.
- Success criterion for a *positive* surprise: E[net] > 0 with t > 2 over ≥ 100 trades, robust across the two calibration/validation weeks.

The deliverable either way is the census: residual distribution, half-life, lead-lag structure, opportunity counts/durations/magnitudes, and the conditional reversion curve. A rigorous negative here is consistent with, and extends, the ETF-pairs and futures-spread conclusions in the stat-arb repo.

---

## 2. Phased protocol

### Phase 0 — Infrastructure (1–2 days)
1. OANDA practice account; generate API token (Account Management Portal → Manage API Access).
2. Stream `/v3/accounts/{id}/pricing/stream?instruments=EUR_USD,GBP_USD,EUR_GBP`. Handle heartbeats (~5 s), reconnect with backoff, backfill gaps flagged (never silently).
3. Log every tick append-only (parquet, daily files): instrument, bid, ask, OANDA timestamp, local receive timestamp (monotonic clock + NTP-synced wall clock).
4. Characterize latency: distribution of (local receive − OANDA timestamp); tick inter-arrival per pair; quote-age structure.

### Phase 1 — Shadow measurement, no trading (2 weeks, London/NY overlap 08:00–17:00 ET emphasized)
Maintain latest quote per pair. Compute the residual **only when all three quote ages < τ = 250 ms** (record ages; report sensitivity at τ ∈ {100, 250, 500, 1000} ms).

Outputs, computed daily and pooled:
1. **Residual distribution:** σ, kurtosis, autocorrelation, AR(1)/OU half-life at 100 ms and 1 s sampling. Explicit test of §1.3: fraction of synchronized observations with |ε| < 0.05 pip.
2. **Deterministic census (H1):** every event where R1 > 0 or R2 > 0 at executable quotes: timestamp, magnitude, duration until closed, which leg moved to close it.
3. **Lead-lag:** which pair updated last before residual formation and which pair's move closes it (event-study around |z| ≥ 2 crossings); Hayashi–Yoshida lagged correlations as cross-check. This validates (or kills) the trade-the-cross assumption.
4. **Conditional reversion curve:** nonparametric estimate of g(x, Δ) = E[ε_{t+Δ} − ε_t | ε_t = x] for Δ ∈ {0.5, 1, 2, 5, 10, 30} s, with confidence bands. This is the object that determines optimal thresholds — no backtest grid search, hence no multiple-testing inflation.

**Gate G1:** proceed to Phase 2 only if σ > 0.1 pip (residual is not internally derived to zero) AND the reversion curve shows E[capture] within 30 s exceeding 25% of one EUR/GBP spread at some attainable x. Otherwise skip to Phase 3 and report.

### Phase 2 — Paper trading arm (2 weeks, only if G1 passes; else run 2–3 days purely as pipeline validation)
- Entry: market order on EUR_GBP when |z| ≥ k_in and majors-led filter passes and quote ages < τ. Exit: |z| ≤ k_out or timeout T. One position at a time; fixed unit size (e.g., 100k notional — size is irrelevant to the measurement, keep it constant).
- Thresholds from Phase 1: k_in = argmax over x of [g(x, ·) capture − measured spread − 2·measured slippage], k_out = 0, T = 5× measured half-life. Freeze parameters for week 1; one revision allowed for week 2 (declared in the log before use).
- Log per trade: signal timestamp, signal price, order timestamp, fill price/timestamp (practice-fill slippage = fill − quote at signal), exit analogues, gross PnL, net PnL under cost model.
- **Cost model caveat (must appear in the report):** practice fills at OANDA's quoted price are optimistic — no queue, no market impact, no last-look rejection. Report gross, net-at-quoted-spread, and net-with-measured-latency-slippage separately.

### Phase 3 — Report
Census-style writeup for the stat-arb repo: residual/latency characterization, H1–H3 verdicts, conditional reversion curves, trade log summary, and the standing conclusion. Include the decomposition: how much of any apparent edge is feed-construction artifact vs. genuine cross-market lag.

---

## 3. Implementation notes for Claude Code

- **Language/stack:** Python 3.11+, `httpx` or `requests` for streaming (chunked), `polars`/`pyarrow` for tick storage, no async complexity needed at three instruments. The `oandapyV20` package is serviceable but a thin hand-rolled client over the two endpoints (pricing stream, orders) is fewer dependencies and easier to audit.
- **Endpoints:** practice host `api-fxpractice.oanda.com` / `stream-fxpractice.oanda.com`. Orders: POST `/v3/accounts/{id}/orders` with MARKET type; fills returned synchronously in the transaction response.
- **Clock discipline:** never compare OANDA timestamps to local wall clock for signal logic; use local receive times on a monotonic clock for ages, OANDA timestamps for cross-pair event ordering.
- **No look-ahead:** the residual at decision time uses only quotes already received; the tick logger and the signal engine must share one in-memory book, with the decision recorded before the order call.
- **Weekend/rollover:** halt 16:55–18:05 ET daily (spread blowout at 5 pm ET rollover), and Friday close → Sunday open. Flag scheduled releases (NFP, CPI, BoE/ECB/Fed) in the tick log for regime-conditioned analysis; do not filter them out of the census.
- **Kill switches:** max 200 paper trades/day, halt on 3 consecutive reconnect failures, halt if position count ≠ {0,1}.
- **Config:** single TOML for {τ, k_in, k_out, T, session windows}; every run logs its config hash so the report can prove parameters were frozen.

## 4. Breakeven arithmetic (reference)

EUR/GBP pip = 1e-4 ≈ 1.16 bp at C ≈ 0.86. One-leg round trip at market ≈ 1.0–1.5 pip. For entry at k·σ with capture fraction θ ≤ 1, breakeven requires k·σ·θ > spread. At σ = 0.2 pip (plausible upper bound if the cross is independently priced at all), k·θ > 5–7.5 — i.e., entries at 5σ+ with full capture, events which at retail latency are dominated by genuine information (adverse selection), not noise. This is why H3 is expected to hold, and why the experiment's value is the measurement, not the PnL.

## 5. References

- Aiba, Hatano, Takayasu, Marumo, Shimizu (2002), "Triangular arbitrage as an interaction among foreign exchange rates," *Physica A*.
- Fenn, Howison, McDonald, Williams, Johnson (2009), "The mirage of triangular arbitrage in the spot foreign exchange market," *IJTAF*.
- Foucault, Kozhan, Tham (2017), "Toxic Arbitrage," *Review of Financial Studies*.
- Hasbrouck (1995) information shares / Hayashi–Yoshida (2005) asynchronous covariance, for the lead-lag layer.
