"""End-to-end self-test. No credentials, no network, no account.

    python selftest.py            # ~1 minute
    python selftest.py --keep     # leave the scratch directories behind

The point of this file is falsifiability of the *instrument*, not of the
strategy. A census that reports "no edge" is only worth anything if it would
have reported an edge had one been there. So the test runs the whole pipeline
twice over synthetic feeds whose ground truth is known:

  derived      the cross is A/B plus a spread, rounded to the venue grid.
               Truth: no residual beyond rounding, no arbitrage, no capture.
               The pipeline must report sigma near the rounding floor, zero
               deterministic events, and a FAILING gate.

  independent  the cross carries a known OU deviation with a known half-life.
               Truth: a real, tradeable residual.
               The pipeline must recover the half-life, find deterministic
               events, produce a monotone reversion curve, and PASS the gate.

Plus unit-level checks on the triangle algebra, the session calendar, the
parameter freeze and the gate refusal.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fxlib.book import Quote, TriangleBook          # noqa: E402
from fxlib.config import load_config                # noqa: E402
from fxlib.sessions import EventCalendar, SessionClock  # noqa: E402
from fxlib.stats import ar1_halflife, hayashi_yoshida, resample_last  # noqa: E402

# A fixed, known-tradeable instant: Wednesday 2026-09-02 14:00 UTC = 10:00 ET,
# inside the London/NY overlap. Without pinning this the Phase 2 dry run
# inherits the real clock, and the arm correctly refuses to trade during the
# rollover halt or at a weekend -- so the test recorded zero trades every
# evening after 16:55 ET and all weekend.
SIM_START = "2026-09-02T14:00:00+00:00"

PASS, FAIL = "  ok  ", " FAIL "
_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)
    return bool(ok)


def run(args: list, cwd: Path = HERE) -> subprocess.CompletedProcess:
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------
# unit-level
# ---------------------------------------------------------------------------
def test_triangle_algebra(cfg):
    cfg = json.loads(json.dumps({k: v for k, v in cfg.items()
                                 if not k.startswith("_")}))
    cfg["signal"]["sigma_warmup_n"] = 1
    bk = TriangleBook(cfg, "EUR_USD", "GBP_USD", "EUR_GBP")
    now, t = dt.datetime.now(dt.timezone.utc), 1000.0
    A, B = 1.17, 1.36
    C = A / B
    for name, mid, sp, ms in [("EUR_USD", A, 8e-5, 0), ("GBP_USD", B, 11e-5, 1),
                              ("EUR_GBP", C, 13e-5, 2)]:
        bk.update(name, Quote(mid - sp / 2, mid + sp / 2,
                              now + dt.timedelta(milliseconds=ms), t, now))
    s = bk.snapshot(t + 0.05, now)
    check("identity gives eps == 0", abs(s.eps) < 1e-12, f"eps={s.eps:.2e}")
    check("no free lunch at the identity", s.r1 < 0 and s.r2 < 0,
          f"R1={s.r1*1e4:.3f}bp R2={s.r2*1e4:.3f}bp")
    check("cycle cost equals the loss at parity",
          abs(-s.r1 * 1e4 - s.cycle_cost_bp) < 0.02,
          f"cost={s.cycle_cost_bp:.3f}bp")

    C2 = C + 2e-4
    bk.update("EUR_GBP", Quote(C2 - 6.5e-5, C2 + 6.5e-5,
                               now + dt.timedelta(milliseconds=3), t + 0.01, now))
    s2 = bk.snapshot(t + 0.06, now)
    check("a +2 pip dislocation reads as +2 pips",
          abs(s2.resid_pips - 2.0) < 0.02, f"{s2.resid_pips:+.4f}")

    C3 = C * 1.001
    bk.update("EUR_GBP", Quote(C3 - 6.5e-5, C3 + 6.5e-5,
                               now + dt.timedelta(milliseconds=4), t + 0.02, now))
    s3 = bk.snapshot(t + 0.07, now)
    check("a rich cross opens D1 only", s3.r1 > 0 > s3.r2,
          f"R1={s3.r1*1e4:+.2f}bp R2={s3.r2*1e4:+.2f}bp")

    s4 = bk.snapshot(t + 5.0, now)
    check("stale quotes are not synchronized", not s4.synchronized,
          f"max_age={s4.max_age_ms:.0f}ms")

    old = Quote(1.0, 1.1, now - dt.timedelta(seconds=10), t, now)
    bk.update("EUR_USD", old)
    check("the book never rewinds on a late duplicate",
          bk.q["EUR_USD"].bid != 1.0)


def test_leg_inversion(cfg):
    """A leg quoted the wrong way round must be mathematically invisible.

    EUR/CZK needs CZK_USD but the venue quotes USD_CZK. If invert_quote got the
    bid/ask swap wrong it would flip the spread's sign, and every cycle would
    look profitable by exactly the round trip -- a fake arbitrage that would
    sail through the census as a real one.
    """
    import copy
    from fxlib.book import invert_quote

    base = json.loads(json.dumps({k: v for k, v in cfg.items()
                                  if not k.startswith("_")}))
    base["signal"]["sigma_warmup_n"] = 1
    now, t = dt.datetime.now(dt.timezone.utc), 1000.0
    A, B = 1.17, 1.36
    C = A / B * 1.00002
    q = {"EUR_USD": (A - 4e-5, A + 4e-5), "GBP_USD": (B - 5e-5, B + 5e-5),
         "EUR_GBP": (C - 6e-5, C + 6e-5)}

    orig = Quote(q["GBP_USD"][0], q["GBP_USD"][1], now, t, now)
    inv = invert_quote(orig)
    back = invert_quote(inv)
    check("inversion is its own inverse",
          abs(back.bid - orig.bid) < 1e-15 and abs(back.ask - orig.ask) < 1e-15)
    check("inversion keeps bid below ask", inv.bid < inv.ask)
    rel = lambda x: (x.ask - x.bid) / x.mid            # noqa: E731
    check("inversion preserves the relative spread",
          abs(rel(orig) - rel(inv)) / rel(orig) < 1e-6,
          "%.3e vs %.3e" % (rel(orig), rel(inv)))

    def snap_of(c, feed):
        bk = TriangleBook(c, c["oanda"]["instrument_a"],
                          c["oanda"]["instrument_b"], c["oanda"]["instrument_c"])
        for i, (name, (bid, ask)) in enumerate(feed):
            bk.update(name, Quote(bid, ask,
                                  now + dt.timedelta(milliseconds=i), t, now))
        return bk.snapshot(t + 0.05, now)

    n_cfg = copy.deepcopy(base)
    n_cfg["oanda"].update(instrument_a="EUR_USD", instrument_b="GBP_USD",
                          instrument_c="EUR_GBP")
    s1 = snap_of(n_cfg, [("EUR_USD", q["EUR_USD"]), ("GBP_USD", q["GBP_USD"]),
                         ("EUR_GBP", q["EUR_GBP"])])
    i_cfg = copy.deepcopy(base)
    i_cfg["oanda"].update(instrument_a="EUR_USD", instrument_b="USD_GBP",
                          instrument_c="EUR_GBP", inverted_legs=["USD_GBP"])
    s2 = snap_of(i_cfg, [("EUR_USD", q["EUR_USD"]),
                         ("USD_GBP", (1.0 / q["GBP_USD"][1], 1.0 / q["GBP_USD"][0])),
                         ("EUR_GBP", q["EUR_GBP"])])
    worst, field = 0.0, ""
    for f in ("eps", "resid", "resid_pips", "r1", "r2", "cycle_cost_bp",
              "c_spread_pips"):
        v1, v2 = getattr(s1, f), getattr(s2, f)
        d = abs(v1 - v2) / max(1.0, abs(v1))
        if d > worst:
            worst, field = d, f
    check("an inverted leg reproduces the normal triangle exactly",
          worst < 1e-12, "worst field %s, rel diff %.1e" % (field, worst))

    cz = copy.deepcopy(base)
    cz["oanda"]["pip_size"] = 1e-2
    bk = TriangleBook(cz, "EUR_USD", "GBP_USD", "EUR_GBP")
    check("pip size is configurable per triangle", bk.pip == 1e-2)


def test_phase1b_configs():
    """The Phase 1b arms must be comparable to the primary census."""
    from fxlib.config import apply_account_override
    prim = load_config(HERE / "config.toml")
    frozen = ("tau_ms", "k_in", "k_out", "timeout_s", "ewma_halflife_s",
              "demean", "sigma_warmup_n", "majors_led_filter")
    seen_roots, seen_accts = {prim["storage"]["root"]}, set()
    for fn in ("config_audnzd.toml", "config_eurczk.toml"):
        path = HERE / fn
        if not path.exists():
            check("%s exists" % fn, False)
            continue
        c = load_config(path)
        same = all(c["signal"][k] == prim["signal"][k] for k in frozen)
        check("%s shares the primary's signal parameters" % fn, same)
        root = c["storage"]["root"]
        check("%s writes to its own data root" % fn, root not in seen_roots, root)
        seen_roots.add(root)
        acct = apply_account_override(c, {"account_id": "X"})["account_id"]
        check("%s pins its own account" % fn, acct not in seen_accts, acct)
        seen_accts.add(acct)
        a, b, x = (c["oanda"]["instrument_a"], c["oanda"]["instrument_b"],
                   c["oanda"]["instrument_c"])
        bk = TriangleBook(c, a, b, x)
        check("%s inverted legs are actually legs of the triangle" % fn,
              bk.inverted <= {a, b, x},
              "inverted=%s legs=%s" % (sorted(bk.inverted), [a, b, x]))


def test_sessions(cfg):
    sc, ec = SessionClock(cfg), EventCalendar(cfg)
    cases = [
        ("2026-08-24T14:00:00+00:00", True, "overlap"),    # Mon 10:00 ET
        ("2026-08-24T21:00:00+00:00", False, "rollover"),  # Mon 17:00 ET
        ("2026-08-21T21:30:00+00:00", False, "weekend"),   # Fri 17:30 ET
        ("2026-08-22T12:00:00+00:00", False, "weekend"),   # Sat
        ("2026-08-23T12:00:00+00:00", False, "weekend"),   # Sun 08:00 ET
    ]
    ok = True
    for iso, tradeable, label in cases:
        t = dt.datetime.fromisoformat(iso)
        ok &= (sc.tradeable(t) == tradeable) and (sc.session_label(t) == label)
    check("session halts: rollover, Friday close, weekend", ok)
    nfp = dt.datetime.fromisoformat("2026-08-07T12:30:00+00:00")   # 1st Fri 08:30 ET
    notnfp = dt.datetime.fromisoformat("2026-08-14T12:30:00+00:00")  # 2nd Friday
    check("NFP rule flags the first Friday only",
          ec.flag(nfp) == "NFP" and ec.flag(notnfp) == "")


def test_stats():
    rng = np.random.default_rng(3)
    theta, sd, n, step = np.log(2) / 5.0, 0.4, 200_000, 0.1
    x = np.empty(n)
    x[0] = 0.0
    decay = np.exp(-theta * step)
    noise = sd * np.sqrt(1 - decay ** 2)
    for i in range(1, n):
        x[i] = decay * x[i - 1] + rng.normal(0, noise)
    phi, hl = ar1_halflife(x, step)
    check("AR(1) recovers a 5 s OU half-life", hl is not None and abs(hl - 5.0) < 0.4,
          f"{hl:.3f}s" if hl else "none")

    t = np.arange(0, 2000, 0.5)
    base = np.cumsum(rng.normal(0, 1e-4, t.size))
    # series 2 is series 1 delayed by 2.0 s
    t2 = t + 2.0
    _, c_at_lag0 = hayashi_yoshida(t, base, t2, base, lag_s=0.0)
    _, c_at_lag2 = hayashi_yoshida(t, base, t2, base, lag_s=-2.0)
    check("Hayashi-Yoshida finds a known 2 s lead",
          c_at_lag2 > c_at_lag0, f"corr {c_at_lag0:.3f} -> {c_at_lag2:.3f}")

    g, v = resample_last(np.array([0.0, 1.0, 5.0]), np.array([1.0, 2.0, 3.0]), 1.0)
    check("resample never back-fills before the first observation",
          g[0] == 0.0 and v.tolist() == [1, 2, 2, 2, 2, 3])


# ---------------------------------------------------------------------------
# pipeline-level
# ---------------------------------------------------------------------------
def run_world(tmp: Path, mode: str, sigma_pips: float, seed: int, hours: float):
    data = tmp / f"data_{mode}"
    out = tmp / f"out_{mode}"
    r = run(["collect_ticks.py", "--source", "synthetic", "--mode", mode,
             "--sigma-pips", str(sigma_pips), "--hours", str(hours),
             "--speed", "3600", "--seed", str(seed), "--quiet",
             "--no-clock-check",   # keeps the self-test entirely offline
             "--data-root", str(data)])
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return None, None
    r = run(["phase1_analyze.py", "--data-root", str(data), "--out", str(out)])
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        return None, None
    rep = json.loads(sorted(out.glob("phase1_*.json"))[-1].read_text(encoding="utf-8"))
    return rep, out


def gate_verdict_ignoring_synthetic(rep):
    """The gate is deliberately VOID on synthetic input. For the self-test we
    need the verdict it WOULD have returned, so re-evaluate the two criteria."""
    cr = rep["gate_g1"]["criteria"]
    return "PASS" if (cr["sigma_ok"] and cr["capture_ok"]) else "FAIL"


def test_derived_world(tmp):
    rep, out = run_world(tmp, "derived", 0.0, 1, 3.0)
    if not check("derived world: pipeline runs", rep is not None):
        return
    sd = rep["residual"]["resid_pips"]["sd"]
    cen = rep["deterministic_census"]
    check("derived world: sigma sits at the rounding/asynchrony floor",
          sd < 0.15, f"sigma={sd:.4f} pip")
    check("derived world: no deterministic arbitrage",
          cen["n_events"] == 0, f"{cen['n_events']} events")
    check("derived world: section 1.3 test fires",
          rep["residual"]["derived_cross_test"]["fraction_abs_below"] > 0.20,
          f"{rep['residual']['derived_cross_test']['fraction_abs_below']:.1%} "
          f"within 0.05 pip")
    check("derived world: gate would FAIL",
          gate_verdict_ignoring_synthetic(rep) == "FAIL")
    check("derived world: H1 and H2 confirmed",
          rep["hypotheses"]["H1"]["verdict"] == "CONFIRMED"
          and rep["hypotheses"]["H2"]["verdict"] == "CONFIRMED")
    check("derived world: gate is VOID on synthetic input",
          rep["gate_g1"]["verdict"] == "VOID_SYNTHETIC")


def test_independent_world(tmp):
    rep, out = run_world(tmp, "independent", 0.6, 7, 3.0)
    if not check("independent world: pipeline runs", rep is not None):
        return out
    sd = rep["residual"]["resid_pips"]["sd"]
    cen = rep["deterministic_census"]
    check("independent world: sigma recovers the injected 0.6 pip",
          abs(sd - 0.6) < 0.12, f"sigma={sd:.4f} pip")
    kd = rep["residual"].get("required_k_demeaned")
    kw = rep["residual"].get("required_k_within_hour")
    check("independent world: k reported on demeaned and within-hour sigma",
          kd is not None and kw is not None and kd > 0 and kw > 0,
          f"k_demeaned={kd}, k_within_hour={kw}")
    check("independent world: deterministic events appear",
          cen["n_events"] > 0, f"{cen['n_events']} events")
    hl = [s["ou_halflife_s"] for s in rep["residual"]["sampled"]
          if s["step_ms"] == 1000][0]
    check("independent world: OU half-life near the injected 4 s",
          hl is not None and 2.0 < hl < 8.0, f"{hl:.2f}s" if hl else "none")
    check("independent world: gate would PASS",
          gate_verdict_ignoring_synthetic(rep) == "PASS",
          f"capture {rep['gate_g1']['criteria']['best_capture_pips']} vs "
          f"required {rep['gate_g1']['criteria']['required_capture_pips']:.3f}")

    bins = [b for b in rep["reversion"]["curves"]["30.0"] if b.get("n", 0) >= 30
            and b.get("z_median") is not None and b["z_median"] > 0.4]
    caps = [b["capture_pips"] for b in sorted(bins, key=lambda b: b["z_median"])]
    check("independent world: capture rises monotonically with z",
          len(caps) >= 3 and all(y > x for x, y in zip(caps, caps[1:])),
          f"{[round(x, 2) for x in caps]}")
    return out


def test_gate_and_freeze(tmp, out_indep):
    data = tmp / "data_phase2"
    # Gate refusal: out_indep holds VOID_SYNTHETIC.
    r = run(["phase2_paper.py", "--dry-run", "--minutes", "5", "--speed", "600",
             "--out", str(out_indep), "--data-root", str(data)])
    check("phase 2 refuses to trade without a G1 PASS",
          r.returncode != 0 and "not PASS" in (r.stdout + r.stderr))

    # Missing gate file entirely.
    r = run(["phase2_paper.py", "--dry-run", "--minutes", "5",
             "--out", str(tmp / "out_empty"), "--data-root", str(data)])
    check("phase 2 refuses to trade before phase 1 has run",
          r.returncode != 0 and "phase1_analyze" in (r.stdout + r.stderr))

    # Pipeline-validation escape hatch works and is stamped.
    r = run(["phase2_paper.py", "--dry-run", "--hours", "2", "--speed", "3600",
             "--mode", "independent", "--sigma-pips", "0.6", "--seed", "11",
             "--sim-start", SIM_START, "--pipeline-validation", "--validation-trades", "25",
             "--out", str(out_indep), "--data-root", str(data)])
    ok = r.returncode == 0 and "PIPELINE VALIDATION ONLY" in r.stdout
    check("phase 2 runs under --pipeline-validation", ok,
          "" if ok else (r.stdout[-500:] + r.stderr[-800:]))

    tf = sorted((data / "trades").glob("*.jsonl")) if (data / "trades").exists() else []
    trades = []
    for f in tf:
        trades += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()
                   if x.strip()]
    check("trades were recorded", len(trades) > 0, f"{len(trades)} trades")
    if trades:
        t = trades[0]
        check("every trade carries all three cost bases",
              all(k in t["pnl_pips"] for k in
                  ("gross_mid_to_mid", "net_at_quoted_spread",
                   "net_with_measured_fills")))
        check("validation trades are stamped as non-evidential",
              all(x.get("validation_only") and x.get("simulated_fills")
                  for x in trades))
        check("gross exceeds net at the spread on every trade",
              all(x["pnl_pips"]["gross_mid_to_mid"]
                  > x["pnl_pips"]["net_at_quoted_spread"] for x in trades))
        check("exits respect the timeout",
              all(x["holding_s"] <= 31.0 for x in trades),
              f"max {max(x['holding_s'] for x in trades):.1f}s")

    # Parameter freeze: mutate a frozen field and expect refusal.
    cfg_txt = (HERE / "config.toml").read_text(encoding="utf-8")
    alt = tmp / "config_alt.toml"
    alt.write_text(cfg_txt.replace("k_in                = 3.0",
                                   "k_in                = 2.0"), encoding="utf-8")
    r = run(["phase2_paper.py", "--dry-run", "--minutes", "5", "--speed", "600",
             "--config", str(alt), "--out", str(out_indep),
             "--data-root", str(data), "--pipeline-validation"])
    check("parameter freeze blocks an undeclared change",
          r.returncode != 0 and "no revision was declared" in (r.stdout + r.stderr))
    r = run(["phase2_paper.py", "--dry-run", "--minutes", "2", "--speed", "3600",
             "--config", str(alt), "--out", str(out_indep),
             "--data-root", str(data), "--sim-start", SIM_START,
             "--pipeline-validation",
             "--declare-revision", "selftest"])
    check("a declared revision is accepted and logged",
          r.returncode == 0 and "revision declared" in r.stdout)
    ledger = json.loads((out_indep / "param_freeze.json").read_text(encoding="utf-8"))
    check("the freeze ledger records both parameter sets",
          len(ledger["entries"]) == 2)
    # A genuinely third parameter set -- reusing config.toml would just match the
    # first ledger entry and be accepted, which is correct behaviour, not a test.
    alt3 = tmp / "config_alt3.toml"
    alt3.write_text(cfg_txt.replace("k_in                = 3.0",
                                    "k_in                = 4.0"), encoding="utf-8")
    r = run(["phase2_paper.py", "--dry-run", "--minutes", "2", "--speed", "3600",
             "--config", str(alt3), "--out", str(out_indep),
             "--data-root", str(data), "--sim-start", SIM_START,
             "--pipeline-validation",
             "--declare-revision", "third"])
    check("a second revision is refused",
          r.returncode != 0 and "already records" in (r.stdout + r.stderr))
    r = run(["phase2_paper.py", "--dry-run", "--minutes", "2", "--speed", "3600",
             "--config", str(HERE / "config.toml"), "--out", str(out_indep),
             "--data-root", str(data), "--sim-start", SIM_START,
             "--pipeline-validation"])
    check("reverting to an already-declared parameter set is allowed",
          r.returncode == 0)
    return data


def test_phase3(tmp, out_indep, data):
    r = run(["phase3_report.py", "--out", str(out_indep), "--data-root", str(data)])
    ok = r.returncode == 0
    check("phase 3 report generates", ok, "" if ok else r.stderr[-800:])
    if not ok:
        return
    j = json.loads(sorted(out_indep.glob("phase3_report_*.json"))[-1]
                   .read_text(encoding="utf-8"))
    check("success criterion is NOT met on validation-only trades",
          not j["success_criterion"]["met"])
    check("validation trades are excluded from the evidential set",
          j["trades"]["n_evidential"] == 0)
    d = j["variance_decomposition"]
    shares = [d.get("share_grid_rounding"), d.get("share_asynchrony"),
              d.get("share_unexplained")]
    check("variance decomposition sums to one",
          all(s is not None for s in shares) and abs(sum(shares) - 1.0) < 1e-6,
          f"{[round(s, 3) for s in shares if s is not None]}")
    md = sorted(out_indep.glob("phase3_report_*.md"))[-1].read_text(encoding="utf-8")
    check("report warns that the data is not live",
          "non-live data" in md)
    check("report carries the practice-fill caveat",
          "last-look rejection" in md)


def test_source_isolation(tmp):
    """Synthetic and live ticks must never pool into one census."""
    import polars as pl
    d = tmp / "data_mixed" / "ticks" / "2026-08-24"
    d.mkdir(parents=True, exist_ok=True)
    src = sorted((tmp / "data_derived" / "ticks").glob("*/part-*.parquet"))[0]
    df = pl.read_parquet(src)
    df.write_parquet(d / "part-000000.parquet")
    df.with_columns(pl.lit("oanda").alias("source")).write_parquet(
        d / "part-000001.parquet")
    r = run(["phase1_analyze.py", "--data-root", str(tmp / "data_mixed"),
             "--out", str(tmp / "out_mixed")])
    check("mixing synthetic and live ticks is refused",
          r.returncode != 0 and "must never be pooled" in (r.stdout + r.stderr))


def test_reboot_split(tmp, out_indep):
    """A reboot restarts the monotonic clock, so two runs can share a monotonic
    range. On 2026-09-09 that interleaved the 1 Sept and 9 Sept runs tick by
    tick under a global recv_mono sort. Re-cut the independent world into two
    runs with deliberately OVERLAPPING monotonic ranges; the census must come
    out the same as on the unsplit data."""
    import polars as pl
    from fxlib import replay
    src = tmp / "data_independent"
    df = pl.read_parquet(sorted((src / "ticks").glob("*/part-*.parquet"))).sort("recv_mono")
    mid = float(df["recv_mono"].median())
    a = df.filter(pl.col("recv_mono") < mid)
    b = df.filter(pl.col("recv_mono") >= mid)
    uptime = 8.0 * 86400.0                       # run A: long-lived boot
    a = a.with_columns((pl.col("recv_mono") + uptime).alias("recv_mono"),
                       pl.lit("synthetic-bootA").alias("run_id"))
    # run B: fresh boot whose clock lands INSIDE run A's range
    b = b.with_columns((pl.col("recv_mono") - mid + uptime + 60.0).alias("recv_mono"),
                       pl.lit("synthetic-bootB").alias("run_id"))
    day = sorted(x.name for x in (src / "ticks").iterdir())[0]
    d = tmp / "data_reboot" / "ticks" / day
    d.mkdir(parents=True, exist_ok=True)
    a.write_parquet(d / "part-000000.parquet")
    b.write_parquet(d / "part-000001.parquet")

    loaded = replay.load_ticks(tmp / "data_reboot")
    rids = loaded["run_id"].to_list()
    switches = sum(1 for i in range(1, len(rids)) if rids[i] != rids[i - 1])
    check("reboot split: runs stay contiguous despite overlapping monotonic clocks",
          switches == 1, f"{switches} run switches")

    out = tmp / "out_reboot"
    r = run(["phase1_analyze.py", "--data-root", str(tmp / "data_reboot"),
             "--out", str(out)])
    if not check("reboot split: pipeline runs", r.returncode == 0,
                 (r.stdout + r.stderr)[-800:] if r.returncode else ""):
        return
    rep = json.loads(sorted(out.glob("phase1_*.json"))[-1].read_text(encoding="utf-8"))
    ref = json.loads(sorted(Path(out_indep).glob("phase1_*.json"))[-1].read_text(encoding="utf-8"))
    n0 = ref["deterministic_census"]["n_synchronized_observations"]
    n1 = rep["deterministic_census"]["n_synchronized_observations"]
    check("reboot split: synchronized sample survives the boundary",
          abs(n1 - n0) <= 0.01 * n0, f"{n1} vs {n0} unsplit")
    h0 = ref["deterministic_census"]["observed_hours"]
    h1 = rep["deterministic_census"]["observed_hours"]
    check("reboot split: observed hours exclude the between-run gap",
          abs(h1 - h0) <= 0.02 * h0, f"{h1:.3f} h vs {h0:.3f} h unsplit")
    inst = next(k for k in ref["latency"] if not k.startswith("_"))
    t0 = ref["latency"][inst]["ticks_per_min"]
    t1 = rep["latency"][inst]["ticks_per_min"]
    check("reboot split: tick rate unaffected by pre-reboot uptime",
          abs(t1 - t0) <= 0.02 * t0, f"{t1:.1f}/min vs {t0:.1f}/min unsplit")
    hl = [x["ou_halflife_s"] for x in rep["residual"]["sampled"]
          if x["step_ms"] == 1000][0]
    check("reboot split: OU half-life still recovered",
          hl is not None and 2.0 < hl < 8.0, f"{hl:.2f}s" if hl else "none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    print("=== unit ===")
    test_triangle_algebra(cfg)
    test_leg_inversion(cfg)
    test_phase1b_configs()
    test_sessions(cfg)
    test_stats()

    tmp = Path(tempfile.mkdtemp(prefix="fxstatarb-selftest-"))
    try:
        print("\n=== derived world (truth: no residual) ===")
        test_derived_world(tmp)
        print("\n=== independent world (truth: real residual) ===")
        out_indep = test_independent_world(tmp)
        print("\n=== gates, freeze and the trading arm ===")
        data = test_gate_and_freeze(tmp, out_indep)
        print("\n=== phase 3 ===")
        test_phase3(tmp, out_indep, data)
        print("\n=== reboot / overlapping runs ===")
        test_reboot_split(tmp, out_indep)
        print("\n=== provenance guards ===")
        test_source_isolation(tmp)
    finally:
        if args.keep:
            print(f"\nscratch kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    n = len(_results)
    bad = [r for r in _results if not r[1]]
    print(f"\n{n - len(bad)}/{n} checks passed")
    for name, _, detail in bad:
        print(f"  FAILED: {name} {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
