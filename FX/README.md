# Triangular FX statistical arbitrage on a retail feed

A preregistered measurement of triangular deviations on OANDA's retail
foreign-exchange feed: EUR/USD · GBP/USD · EUR/GBP, plus two thin-cross
follow-ups (AUD/NZD and EUR/CZK). Paper money only, real-time data, census-style
reporting. The deliverable is the measurement, not a trading strategy, and the
measurement is negative.

This is the code and archived analysis behind the paper
[*Triangular FX Residuals on a Retail Feed Are Real, Measurable, and a Fourteenth
of Their Cost*](paper/main.tex), a companion to
[*Cost Viability and Cointegration Are Anti-Correlated in Liquid US ETF
Pairs*](https://zenodo.org/records/22059837).

**Status:** complete. Collection ran 31 Aug – 25 Sep 2026; the paper-trading arm
(Phase 2) was never run, because the decision gate failed.

## Results at a glance

| | EUR/GBP | AUD/NZD | EUR/CZK |
|---|---|---|---|
| sessions (03:00–17:00 New York) | 10 | 10 | 10 |
| synchronized observations (all quotes < 250 ms old) | 103,650 | 77,502 | 77,608 |
| executable triangular cycles | **0** | **0** | **0** |
| best cycle ever observed | −0.77 bp | −2.87 bp | −1.48 bp |
| residual σ as a share of the cross's round trip | 7.2% | 6.8% | 5.8% |
| k = round trip / σ (lower is better for a trader) | 14.0 | 14.7 | 17.3 |
| best 30 s reversion bin, net of the spread quoted at the time | −1.10 pip | −3.07 pip | −54.4 pip |

- **No arbitrage.** Not once, in 258,760 synchronized observations, could the
  three legs be traded at their quoted prices for a profit.
- **The residual is real but priced.** The quoted cross deviates from the
  majors-implied cross by a small, fast-reverting amount (half-life about five
  seconds) that never exceeds what it costs to trade.
- **Thin crosses are worse, not better.** The preregistered prediction was that
  AUD/NZD or EUR/CZK would come in below EUR/GBP's k = 13.7. Neither did.
- **The analysis nearly said otherwise.** The paper's Section 5 documents three
  default choices that together turned this negative into a PASS. If you reuse
  this code, read that section, or the short version under
  [Pitfalls](#pitfalls) below.

## What is in this directory

```
FX/
├── README.md                      this file
├── DESIGN.md                      design decisions, guard rails, hazards, unattended operation
├── triangular-fx-statarb-spec.md  the preregistered protocol (hypotheses H1–H3, Gate G1)
├── requirements.txt
├── .gitignore
├── .gitattributes                 keeps the Windows helpers CRLF on any checkout
├── credentials.example.toml       template for credentials.toml (which is never committed)
├── config.toml                    EUR/GBP triangle: every runtime parameter
├── config_audnzd.toml             AUD/NZD arm, same signal parameters, preregistered prediction in header
├── config_eurczk.toml             EUR/CZK arm, USD/CZK leg inverted, prediction in header
├── events.toml                    scheduled-release calendar (7 events; incomplete, enters no verdict)
├── collect_ticks.py               Phase 0/1 collector: streams quotes, logs every tick and gap
├── phase0_latency.py              daily health check: coverage, latency, quote ages, τ sensitivity
├── phase1_analyze.py              the census: H1–H3, reversion curves, lead–lag, Gate G1
├── phase2_paper.py                paper-trading arm (refuses to trade unless Gate G1 passed)
├── phase3_report.py               report: verdicts, cost and variance decomposition
├── scout_triangles.py             ranks every triangle on the venue by cost (read-only)
├── selftest.py                    65 ground-truth checks; no credentials, no network
├── run_collector.bat              ┐ Windows helpers for unattended daily collection
├── run_collector_audnzd.bat       │ (Task Scheduler). Specific to the machine they ran on;
├── run_collector_eurczk.bat       │ kept because they encode what the collection
├── launch_collector.vbs           │ required. See DESIGN.md, "Running unattended".
├── launch_arm.vbs                 │
├── register_phase1b_tasks.ps1     ┘
│
├── fxlib/
│   ├── book.py                    the single quote book and all triangle algebra
│   ├── replay.py                  replays stored ticks through that same book
│   ├── phase1.py                  census statistics and Gate G1
│   ├── stats.py                   HAC t-stats, block bootstrap, Hayashi–Yoshida, OU fit
│   ├── clocks.py                  SNTP client and per-tick clock-error reconstruction
│   ├── oanda.py                   minimal v20 client (pricing stream, orders)
│   ├── execution.py               live and simulated order execution
│   ├── sessions.py                session labels, rollover and weekend halts, event windows
│   ├── storage.py                 append-only parquet parts, event log, single-writer lock
│   ├── synthetic.py               synthetic feeds with known truth
│   ├── config.py                  config loading (refuses anything but a practice account)
│   └── __init__.py
│
├── data/excluded_runs.json        quarantined runs (a duplicate collector), with reasons
├── data_audnzd/excluded_runs.json quarantined runs (a 90 s plumbing test), with reasons
├── data_eurczk/excluded_runs.json quarantined runs (a 90 s plumbing test), with reasons
│
└── paper/
    ├── README.md                  how to regenerate every number and figure
    ├── main.tex                   the paper (LaTeX)
    ├── fig_gate.pdf               reversion vs the gate's bar vs the spread actually paid
    ├── fig_kbasis.pdf             k by σ basis and by session
    ├── fig_hourly.pdf             hourly σ against hourly spread
    ├── fig_reboot.pdf             monotonic-clock ranges across a reboot
    ├── paper_numbers.py           computes everything the paper cites …
    ├── paper_numbers.json         … and its output
    ├── make_figures.py            builds the figures
    ├── make_manifest.py           fingerprints the raw tick files …
    ├── data_manifest.json         … SHA-256 and row count per file (5.6 MB)
    ├── lint_tex.py                structural checks on main.tex
    ├── fig_data/hourly_{eurgbp,audnzd,eurczk}.csv
    └── runs/                      archived census outputs the paper rests on
        ├── eurgbp_main/           EUR/GBP, declared window 31 Aug – 11 Sep (primary)
        ├── eurgbp_all/            EUR/GBP pooled with calibration runs (the documented error)
        ├── audnzd/                AUD/NZD, 14–25 Sep
        ├── eurczk/                EUR/CZK, 14–25 Sep
        └── p3/                    variance decomposition for the declared EUR/GBP window
```

**Not committed** (see `.gitignore`):

- `credentials.toml`: your API token.
- `data*/` other than the quarantine records: the raw ticks, about 74 MB for
  EUR/GBP alone. OANDA's price data is not ours to redistribute;
  `paper/data_manifest.json` fingerprints every file instead. **[TODO: confirm
  OANDA's terms.]**
- `out/`: working outputs, regenerated by the phase scripts.
- `*.log`: they contain account ids and local paths.
- `paper/preview/` and `__pycache__/`.

## Quick start: verify the pipeline (no account needed)

```bash
cd fx_statarb
python -m pip install -r requirements.txt     # Python 3.11+
python selftest.py                            # ~2 min, expect "65/65 checks passed"
```

The self-test runs the whole pipeline against two synthetic worlds whose truth is
known. In one the cross is derived from the majors, so there is nothing to find.
In the other the cross carries a known mean-reverting deviation. The pipeline
must find nothing in the first and recover the deviation in the second.

## Reproducing the paper

**From this repository alone**, without the raw data, you can:

```bash
python paper/make_figures.py     # rebuilds three of the four figures from committed outputs
python paper/lint_tex.py         # checks main.tex (compile it with any LaTeX, e.g. Overleaf)
```

Every number in the paper is in `paper/paper_numbers.json` or in the census
outputs under `paper/runs/`, which also carry the config hash each run used.

**With the original tick files**, you can regenerate everything. Follow
[`paper/README.md`](paper/README.md), then confirm you have the same bytes:

```bash
python paper/make_manifest.py && git diff --stat paper/data_manifest.json   # no diff = identical data
```

**With your own collection**, the numbers will differ, since it is a different
month on a live venue, but every step runs the same way (next section).

## Collecting your own data

1. **Open an OANDA practice account** and generate an API token (Account
   Management Portal → Manage API Access).
2. **Add credentials:**
   ```bash
   cp credentials.example.toml credentials.toml    # then fill in api_token and account_id
   ```
   The code refuses to run against anything but a practice account.
3. **For the thin-cross arms**, `config_audnzd.toml` and `config_eurczk.toml`
   pin `[oanda].account_id` to the practice sub-accounts used in the study, so
   each triangle streams on its own account. Replace those ids with your own
   sub-accounts, or delete the line to use the account in `credentials.toml`
   (then run one triangle at a time). Editing a config changes its config hash,
   as it should: the hash identifies the parameters a run used.
4. **Collect**, one process per triangle, ideally every weekday for two weeks:
   ```bash
   python collect_ticks.py --source oanda --until 17:00                          # EUR/GBP
   python collect_ticks.py --source oanda --until 17:00 --config config_audnzd.toml
   ```
5. **Check each day** after collection stops:
   ```bash
   python phase0_latency.py --dates 2026-09-14
   ```
6. **Run the census on an explicit list of sessions**, never the default:
   ```bash
   python phase1_analyze.py --dates 2026-08-31 2026-09-01 ...   # also writes out/g1_verdict.json
   python phase1_analyze.py --no-gate --config config_audnzd.toml --dates ...
   python phase3_report.py
   ```
   Use `--no-gate` for anything other than the EUR/GBP census; otherwise it
   overwrites the gate verdict.
7. **Optional:** `python scout_triangles.py` ranks every triangle on the venue
   by cost, using read-only calls. It refuses to rank outside the London–New
   York overlap unless you pass `--force`, because off-hours spreads bias the
   ranking.

For unattended daily collection on Windows, see DESIGN.md, "Running
unattended". The core code is portable Python, but it has only been run on
Windows 10.

## Safety

- **Practice only.** `config.toml` must say `environment = "practice"`; the code
  exits otherwise.
- **No accidental trading.** `phase2_paper.py` places orders on the practice
  account only if `out/g1_verdict.json` says PASS **and** the frozen parameters
  in `out/param_freeze.json` are unchanged. One declared revision is allowed; a
  second is refused. Kill switches cap trades per day, reconnect failures and
  open positions. `phase2_paper.py --dry-run` uses a synthetic feed and touches
  no account.
- **No pooling of synthetic and live data.** The census aborts if a tick log
  contains both.

## Pitfalls

Each of these produced clean-looking output and no error. Details are in
DESIGN.md and Section 5 of the paper.

1. **The census defaults to every collected session.** Pass `--dates`. Pooling a
   calibration run declared throwaway turned Gate G1 from FAIL to PASS.
2. **Raw σ is not the residual you trade.** It includes the residual's level
   drifting over hours and days, and it grows with sample length. Compare
   triangles on `required_k_demeaned`. On raw σ, both thin crosses "passed".
3. **Charge the spread quoted at the time, not the median.** The largest
   dislocations happen at news releases, when the cross is quoted two to six
   pips wide. Each reversion bin reports `net_at_quoted_spread_pips` beside
   `capture_pips`.
4. **A monotonic clock restarts at reboot.** Never sort a multi-run tick log on
   `recv_mono` alone; the replay orders runs by wall-clock start.
5. **Lead–lag is not identified on this feed.** The Hayashi–Yoshida profiles are
   flat. If you use the estimator elsewhere, note that a *negative* best lag
   means the majors lead.

## Citation

```bibtex
@misc{ross2026fx,
  author = {Ross, Michael M.},
  title  = {Triangular {FX} Residuals on a Retail Feed Are Real, Measurable,
            and a Fourteenth of Their Cost},
  year   = {2026},
  note   = {Companion to "Cost Viability and Cointegration Are Anti-Correlated
            in Liquid US ETF Pairs"},
  doi    = {[TODO]}
}
```

## License and acknowledgments

[Repository License](https://github.com/michaelmross/stat-arb/blob/main/LICENSE.md)

## Not investment advice

This is a research repository reporting negative results. Nothing here
is a recommendation to trade, and the strategies it evaluates lost money
or failed to trade at all under realistic costs.
