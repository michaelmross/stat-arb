"""Phase 0 step 4: characterize the feed before trusting anything measured on it.

    python phase0_latency.py
    python phase0_latency.py --dates 2026-08-25

Reports the distribution of (local receive - OANDA timestamp), tick
inter-arrival per pair, and the quote-age structure that determines how much of
the sample survives the tau filter. Run this after a first collection session
and before committing to a two-week Phase 1: if median latency is 800 ms or the
cross ticks once every three seconds, the synchronized sample will be too thin
and tau needs revisiting before the clock starts, not after.

Note on the latency figure: it compares the venue clock to the local wall clock,
so it includes any NTP offset on this machine. It is a diagnostic, never an
input to signal logic -- the engine uses the monotonic clock for ages (see
fxlib/book.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fxlib import phase1 as P
from fxlib.clocks import describe as describe_offset, measure_offset
from fxlib.config import instruments, load_config
from fxlib.replay import check_single_source, load_ticks, replay
from phase1_analyze import clock_anchors

HERE = Path(__file__).resolve().parent


def row(label, d, keys=("median", "p95", "p99", "max")):
    vals = " ".join(f"{k}={d.get(k, float('nan')):,.1f}" for k in keys)
    return f"  {label:<22} n={d.get('n', 0):>9,}  {vals}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="filter to session labels, e.g. overlap")
    ap.add_argument("--json", action="store_true", help="dump the full report")
    ap.add_argument("--no-clock-check", action="store_true",
                    help="skip the live NTP offset measurement")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.data_root:
        cfg["storage"]["root"] = args.data_root
    a, b, c = instruments(cfg)
    root = Path(cfg["_path"]).parent / cfg["storage"]["root"]

    ticks = load_ticks(root, args.dates, args.sessions)
    source = check_single_source(ticks)
    # Prefer the offset each collector run recorded at the time; fall back to a
    # live reading for tick logs collected before the offset was recorded.
    live = {} if args.no_clock_check else measure_offset()
    anchors = clock_anchors(root, ticks)
    lat = P.latency_report(ticks, anchors)
    snap = replay(cfg, ticks, a, b, c)
    ages = P.quote_age_report(snap)
    taus = P.tau_sensitivity(snap, cfg["signal"]["tau_sensitivity_ms"])

    if args.json:
        print(json.dumps({"source": source, "latency": lat, "quote_age": ages,
                          "tau_sensitivity": taus}, indent=2, default=str))
        return 0

    # Collection health FIRST -- a day with a hole in it should be the first
    # thing you see, not a footnote under the latency percentiles.
    days = args.dates or sorted({str(d)[:10] for d in ticks["recv_wall"].to_list()})
    health = P.collection_health(root, days, cfg["sessions"]["timezone"])
    print()
    print("=== collection health (intended window %s-%s %s) ==="
          % (health["window"][0], health["window"][1], health["timezone"]))
    if any(d.get("in_progress") for d in health["days"]):
        print("(a run is still collecting; re-run after collector.log shows "
              "'END ... exit=0', usually just after 17:00 but up to ~17:10 "
              "because the rollover silence delays the --until check)")
    for h in health["days"]:
        # NB: not `live` -- that name already holds the NTP measurement, and
        # shadowing it here made describe_offset() crash on a list.
        still_going = h.get("in_progress")
        ok = h.get("coverage_pct", 0) >= 99.0 and not h.get("killed_runs")
        flag = ".. " if still_going else ("OK " if ok else "** ")
        print("%s%s  %7d ticks  coverage %5.1f%%  uncovered %.0f min"
              % (flag, h["date"], h["n_ticks"], h.get("coverage_pct", float("nan")),
                 h.get("uncovered_minutes", 0)))
        for run in h.get("runs", []):
            mark = ("  <-- KILLED" if run["ended"].startswith("KILLED")
                    else "  <-- still collecting"
                    if run["ended"].startswith("IN PROGRESS") else "")
            print("      %s..%s  %6d rows  %s%s"
                  % (run["from"], run["to"], run["rows"], run["ended"], mark))
        for hole in h.get("holes", []):
            print("      NOT COLLECTED %s..%s  (%.0f min)"
                  % (hole["from"], hole["to"], hole["minutes"]))
    print()

    scope = []
    if args.dates:
        scope.append("dates=" + ",".join(args.dates))
    if args.sessions:
        scope.append("sessions=" + ",".join(args.sessions))
    print(f"source={source}  ticks={len(ticks):,}  snapshots={len(snap):,}"
          + ("  [" + "; ".join(scope) + "]" if scope else "  [all data]"))
    if live:
        print(describe_offset(live))
    print("\nLatency, local receive minus OANDA timestamp (ms) -- RAW:")
    for inst, d in P.instrument_rows(lat):
        print(row(inst, d["latency_ms"]))
    if any(d.get("latency_corrected_ms") for d in lat.values()):
        print("\nLatency CORRECTED for the local clock offset (ms). This is the "
              "real transport\nlatency; the raw block above is inflated by "
              "however far this machine's clock\nis from true time. Neither "
              "affects signal logic -- quote ages use the\nmonotonic clock.")
        for inst, d in P.instrument_rows(lat):
            if d.get("latency_corrected_ms"):
                print(row(inst, d["latency_corrected_ms"],
                          ("min", "median", "p95", "p99")))
    print("\nInter-arrival (s):")
    for inst, d in P.instrument_rows(lat):
        print(row(inst, d["inter_arrival_s"], ("median", "p95", "p99", "max")))
        print(f"  {'':<22} {d['ticks_per_min']:.1f} ticks/min, "
              f"spread median {d['spread_pips'].get('median', float('nan')):.2f} pip")
    print("\nQuote age at snapshot (ms):")
    for k, d in ages.items():
        print(row(k, d))
    print("\nSynchronization tolerance:")
    print(f"  {'tau (ms)':>9} {'n':>10} {'share':>8} {'resid sd (pip)':>16}")
    for t in taus:
        print(f"  {t['tau_ms']:>9} {t['n']:>10,} {t['fraction_of_ticks']:>7.1%} "
              f"{t['resid_sd_pips']:>16.4f}")
    print(f"\nConfigured tau = {cfg['signal']['tau_ms']} ms. If the share at that "
          f"tau is very low, Phase 1 will be starved of synchronized observations "
          f"-- change tau now and report the sensitivity, do not change it later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
