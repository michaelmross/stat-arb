# Design notes

For anyone changing the code. The user-facing overview is `README.md`; the
protocol is `triangular-fx-statarb-spec.md`; the results are in
`paper/main.tex`.

## Design decisions

**Two residual conventions, both kept.** `eps = ln A − ln B − ln C` (spec §1.1)
and `resid = ln C − ln Ĉ = −eps` (spec §1.2, the signal numerator). `z =
resid/σ`; `z > 0` means the quoted cross is rich, so the trade is to **sell** C.
Confusing the two silently inverts every trade.

**The analysis does not re-implement the signal.** `phase1_analyze.py` replays
the raw tick log through the same `TriangleBook` the live engine uses, with the
recorded monotonic receive stamps. The shadow measurement and the live signal
therefore cannot drift apart.

**Three clocks, never mixed.** Quote ages use the local monotonic clock;
cross-pair event ordering uses OANDA timestamps; sessions and calendars use the
wall clock. The latency figure in Phase 0 is a diagnostic only and never feeds
signal logic.

**The wall clock is measured, not assumed.** The spec says to stamp ticks with
an "NTP-synced wall clock", but a Windows desktop is often hundreds of
milliseconds out, and all of that error lands in the reported latency. Every
collector run takes an SNTP reading at start-up (`fxlib/clocks.py`) and records
it in the run manifest; Phase 0 and Phase 1 report latency both raw and
corrected. On this machine's first run the clock was **556 ms fast**, which
turned a 160 ms median latency into a reported 715 ms — a number that would have
looked like a fatal infrastructure problem. Skip the reading with
`--no-clock-check` if the machine is offline.

**z is demeaned by default.** `[signal].demean = true` uses `(resid − EWMA
mean)/σ`. The spec's literal `resid/σ` is available with `demean = false`, but a
broker whose cross carries a constant offset would pin the literal z permanently
to one side. Both are recorded on every snapshot (`z` and `z_raw`).

**Exit is directional, not `|z| ≤ k_out`.** With the spec's `k_out = 0` an
absolute test can never fire, and it would refuse to exit an overshoot (z going
+3 → −2 has fully reverted). A short exits at `z ≤ k_out`, a long at `z ≥
−k_out`.

**Collection never halts; trading does.** The 5pm ET rollover and the weekend
are trading halts. The rollover spread blowout is itself a measurement, so ticks
keep being logged and tagged `session="rollover"`. Note the Sunday open (17:05
ET) falls inside the rollover window (16:55–18:05 ET), so trading effectively
resumes Sunday at 18:05 — conservative, and deliberate.

**No silent gaps.** Any inter-message interval beyond the heartbeat timeout is
written to the events log with its duration. Reversion-curve pairs that would
straddle a gap are dropped: a "30-second" move measured across a five-minute
hole is not a 30-second move.

**Parquet part files, not one open writer.** A `ParquetWriter` killed before
close leaves an unreadable footer-less file — a crash would silently destroy the
day. Parts are atomic on rename.

**No threshold grid search.** The reversion curve is estimated once,
nonparametrically, on fixed a-priori z bins, and `k_in` falls out of it. Tuning
thresholds against realized PnL would inflate significance exactly the way this
experiment exists to avoid.

**Overlapping observations get HAC errors.** Tick-sampled residuals are heavily
autocorrelated; i.i.d. standard errors on them overstate significance by a large
factor. Bin means use Newey–West, cross-checked with a circular block bootstrap
on the bin the gate actually turns on.

## Guard rails

`phase2_paper.py` will not send an order to an account unless both hold:

1. **Gate G1 passed.** `out/g1_verdict.json` must say `PASS`. Otherwise the arm
   runs only under `--pipeline-validation`, capped at `--validation-trades`
   orders, with every record stamped `validation_only` and excluded from the
   evidential set in Phase 3.
2. **Parameters are frozen.** The first run writes the frozen fields and their
   hash to `out/param_freeze.json`. Changing `k_in`, `k_out`, `T`, `tau`, the
   EWMA half-life, the majors-led filter, the unit size or the rollover window
   afterwards is refused unless `--declare-revision "why"` is passed, which
   appends the declaration *before* the run. Exactly one revision is permitted
   (spec §2, week 2); a second is refused.

Kill switches: `max_trades_per_day`, three consecutive reconnect failures, and
any observed position count outside {0, 1}.

Synthetic and live ticks can never be pooled — `phase1_analyze.py` aborts if a
tick log contains both — and Gate G1 returns `VOID_SYNTHETIC` rather than a
verdict when the input is synthetic.

## Offline validation

`selftest.py` runs the entire pipeline twice against synthetic feeds with known
ground truth, because a census reporting "no edge" is only worth something if it
would have reported an edge had there been one.

- **derived world** — the cross is A/B plus a spread, rounded to the venue grid.
  Truth: no residual beyond rounding. The pipeline must report σ at the floor,
  zero deterministic events, and a failing gate.
- **independent world** — the cross carries a known OU deviation (σ = 0.6 pip,
  half-life 4 s). The pipeline must recover the half-life, find deterministic
  events, produce a monotone reversion curve, and pass the gate.

The derived world also calibrates the Phase 3 variance decomposition: whatever
it reports as "unexplained" there is the method's own floor, since the true
unexplained component is zero by construction. Compare the live number against
it rather than against zero.

## Costs, honestly

Every trade record carries three PnL figures and the report shows all three:

- **gross (mid to mid)** — the signal's capture with no costs at all;
- **net at quoted spread** — you crossed the spread you could see;
- **net at measured fills** — the prices OANDA actually returned.

Practice fills are optimistic: no queue position, no market impact, no last-look
rejection. Even the rightmost column is a lower bound on real-world cost, and
the report says so in the same breath as the number.

## Hazards found in use

Each of these produced plausible output without raising an error. Section 5 of
the paper has the full account.

**Always pass `--dates` to the census.** `phase1_analyze.py` defaults to every
collected session. The first EUR/GBP census silently pooled the 28 Aug Phase 0
calibration run, which the protocol had declared throwaway; that lifted raw
sigma across Gate G1's 0.10 pip bar and turned FAIL into PASS.

**Say which sigma.** Raw residual sigma includes the drift of the wedge's level
between hours and days, which grows with sample length. Compare triangles on
`residual.required_k_demeaned` (the residual the signal trades), with
`required_k_within_hour` and the spread-matched k as checks. On raw sigma, both
thin-cross arms would have "passed" the preregistered test.

**Charge the spread quoted at the time.** Gate G1's capture criterion divides by
the *median* spread; extreme-z observations happen at news, when the cross is
quoted two to six pips wide. Every reversion bin now carries
`net_at_quoted_spread_pips` beside `capture_pips`. Read them together.

**The monotonic clock restarts at reboot.** It orders ticks within a process
only. `replay.load_ticks` orders runs by wall-clock start and ticks within a run
by `recv_mono`; every time difference downstream uses `t_axis`, on which runs are
`RUN_GAP_S` apart. Never sort a multi-run log on `recv_mono` alone (the paper's
own number script did, once).

**Hayashi-Yoshida sign.** A NEGATIVE best lag means the majors lead the cross.
On this feed the profiles are flat to within 0.025 across +/-2 s, so the argmax
carries no information; `hy_best_lag_at_grid_edge` flags boundary maxima.

**Leg inversion swaps as well as reciprocates.** `bid' = 1/ask`, `ask' = 1/bid`.
Reciprocating without swapping inverts the spread and makes every cycle look
profitable by exactly the round trip.

## Running unattended on Windows

The helpers (`run_collector*.bat`, `launch_*.vbs`, `register_phase1b_tasks.ps1`)
encode what a month of unattended collection taught:

- Call python **directly** from the batch file. Wrapping it in `start` detaches
  its output from the log and gives it a console window that kills the run if
  closed; priority is raised in-process instead.
- Launch through `wscript.exe` with window style 0 (`launch_*.vbs`), or register
  the task with S4U logon, so no console exists to close.
- Use `--until 17:00`, not `--hours`: a restarted collector then stops at the
  right time instead of running a fresh full duration.
- Trigger every 15 minutes across the window with `MultipleInstances=IgnoreNew`.
  Task Scheduler's restart-on-failure does not fire when another process (an
  installer's Restart Manager, a reboot) terminates the collector; the
  retrigger does.
- `run_end` is written first on shutdown, and the closing NTP read is bounded,
  because a DNS outage once hung the shutdown path and held the lock for half an
  hour.
- Registering an S4U task needs an elevated prompt.
