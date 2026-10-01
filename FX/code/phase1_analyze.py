"""Phase 1 driver: turn the tick log into the census.

    python phase1_analyze.py                       # everything collected so far
    python phase1_analyze.py --dates 2026-08-25 2026-08-26
    python phase1_analyze.py --sessions overlap    # London/NY overlap only

Writes out/phase1_<stamp>.json (the full census) and out/phase1_<stamp>.md
(readable summary), plus out/g1_verdict.json, which phase2_paper.py reads and
refuses to trade without.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fxlib import phase1 as P
from fxlib.config import instruments, load_config
from fxlib.replay import (check_single_source, load_ticks,
                          overlapping_runs, replay)

HERE = Path(__file__).resolve().parent


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "n/a"
        return f"{x:,.{nd}f}"
    return str(x)


def event_log_summary(root: Path, dates: list | None) -> dict:
    files = sorted(glob.glob(str(root / "events" / "*.jsonl")))
    if dates:
        files = [f for f in files if Path(f).stem in dates]
    kinds, gaps, total = {}, [], 0
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            k = rec.get("event", "?")
            kinds[k] = kinds.get(k, 0) + 1
            if k == "gap":
                gaps.append(rec)
    return {"files": [Path(f).name for f in files], "records": total,
            "by_kind": kinds, "n_gaps": len(gaps), "gaps": gaps[:200],
            "total_gap_seconds": round(sum(g.get("gap_s", 0.0) for g in gaps), 2)}


def clock_anchors(root, df) -> dict:
    root = Path(root)
    """run_id -> list of NTP anchors, each with a monotonic stamp.

    Manifests written before the mono stamp existed record only the measurement
    order (start of run, end of run). Those are pinned to the run's first and
    last tick, which is off by however long the measurement took -- a few
    seconds. At the slew rate actually observed (0.075 ms/s) that is well under
    a millisecond, so it is not worth discarding the anchor over.
    """
    bounds = {}
    if df is not None and len(df):
        g = df.group_by("run_id").agg([pl.col("recv_mono").min().alias("lo"),
                                       pl.col("recv_mono").max().alias("hi")])
        bounds = {r["run_id"]: (r["lo"], r["hi"]) for r in g.to_dicts()}
    out = {}
    for f in sorted(glob.glob(str(root / "runs" / "*.json"))):
        try:
            m = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = m.get("run_id")
        lo, hi = bounds.get(run_id, (None, None))
        anchors = []
        for key, fallback in (("clock_offset", lo), ("clock_offset_end", hi)):
            c = m.get(key) or {}
            if not c.get("ok") or c.get("offset_s") is None:
                continue
            mono = c.get("mono", fallback)
            if mono is None:
                continue
            anchors.append({"offset_s": float(c["offset_s"]),
                            "mono": float(mono), "source": key})
        if anchors:
            out[run_id] = anchors
    return out


def build_markdown(rep: dict) -> str:
    L = []
    A = L.append
    prov, lat, res, cen, ll, gate, hyp = (
        rep["provenance"], rep["latency"], rep["residual"],
        rep["deterministic_census"], rep["lead_lag"], rep["gate_g1"],
        rep["hypotheses"])

    A("# Triangular FX Stat-Arb — Phase 1 Census")
    A("")
    if prov["source"] != "oanda":
        A(f"> **SYNTHETIC DATA ({prov['source']}). Pipeline validation only — "
          f"this is not a measurement of any real feed, and Gate G1 is void.**")
        A("")
    A(f"- Generated: {rep['generated_utc']}")
    A(f"- Source: `{prov['source']}` · config hash `{prov['config_hash']}`")
    A(f"- Dates: {', '.join(prov['dates'])}")
    A(f"- Ticks: {prov['n_ticks']:,} · snapshots: {prov['n_snapshots']:,} · "
      f"synchronized (τ={prov['tau_ms']} ms): {prov['n_synchronized']:,} "
      f"({prov['synchronized_fraction']:.1%})")
    A(f"- Observed span: {cen['observed_hours']:.2f} h · stream gaps: "
      f"{rep['events']['n_gaps']} ({rep['events']['total_gap_seconds']} s total)")
    A("")

    A("## 0. Latency and arrival structure")
    A("")
    A("| pair | ticks | ticks/min | latency ms (p50) | (p95) | inter-arrival s (p50) | spread pips (p50) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for inst, d in P.instrument_rows(lat):
        A(f"| {inst} | {d['ticks']:,} | {d['ticks_per_min']:.1f} | "
          f"{_fmt(d['latency_ms'].get('median'),1)} | "
          f"{_fmt(d['latency_ms'].get('p95'),1)} | "
          f"{_fmt(d['inter_arrival_s'].get('median'),3)} | "
          f"{_fmt(d['spread_pips'].get('median'),2)} |")
    A("")
    qa = rep["quote_age"]
    A(f"Quote age at snapshot (ms): A p50 {_fmt(qa['age_a_ms'].get('median'),0)} / "
      f"B p50 {_fmt(qa['age_b_ms'].get('median'),0)} / "
      f"C p50 {_fmt(qa['age_c_ms'].get('median'),0)} · "
      f"max-age p50 {_fmt(qa['max_age_ms'].get('median'),0)}, "
      f"p95 {_fmt(qa['max_age_ms'].get('p95'),0)}")
    A("")
    bs = rep.get("burst_structure") or {}
    if bs.get("n_bursts"):
        A(f"**Feed cadence.** The stream is coalesced, not tick-by-tick: "
          f"{bs['n_ticks']:,} ticks arrive in {bs['n_bursts']:,} bursts "
          f"({bs['ticks_per_burst']:.2f} ticks/burst) at a median interval of "
          f"{_fmt(bs['burst_interval_ms'].get('median'),0)} ms "
          f"(~{bs['cadence_hz']:.1f} Hz). Only "
          f"{bs['fraction_bursts_all_three']:.1%} of bursts carry all three "
          f"legs, which is the ceiling on the synchronized sample for any τ "
          f"below the cadence. Burst-opening instrument: {bs['starts_burst']}.")
        A("")
        A("This bounds what the experiment can resolve: an opportunity shorter "
          "than one flush interval is invisible, and reversion horizons under "
          "~0.5 s contain one or two observations.")
        A("")

    vs = rep.get("venue_sync_check") or {}
    if vs.get("n_contaminated") is not None:
        A(f"**Venue-time cross-check.** {vs['fraction_contaminated']:.1%} of "
          f"receive-synchronized observations have OANDA timestamps more than "
          f"250 ms apart — they passed the τ test because local scheduling "
          f"delay compressed their receive times, not because the quotes were "
          f"contemporaneous. Residual σ on those is "
          f"{_fmt(vs.get('resid_sd_contaminated'))} pip against "
          f"{_fmt(vs.get('resid_sd_clean'))} pip on the rest. Median venue "
          f"spread overall is "
          f"{_fmt(vs['venue_spread_ms'].get('median'),0)} ms, so even a clean "
          f"'synchronized' observation is not a simultaneous snapshot.")
        A("")

    A("### τ sensitivity")
    A("")
    A("| τ ms | n | share of ticks | residual sd (pips) | mean (pips) | \\|resid\\| p95 |")
    A("|---:|---:|---:|---:|---:|---:|")
    for r in rep["tau_sensitivity"]:
        A(f"| {r['tau_ms']} | {r['n']:,} | {r['fraction_of_ticks']:.1%} | "
          f"{_fmt(r['resid_sd_pips'],4)} | {_fmt(r['resid_mean_pips'],4)} | "
          f"{_fmt(r['abs_resid_p95_pips'],4)} |")
    A("")

    A("## 1. Residual distribution (H2)")
    A("")
    rp = res["resid_pips"]
    A(f"resid = ln C − ln(A/B), in EUR/GBP pips. n = {rp['n']:,}")
    A("")
    A(f"- sd **{_fmt(rp['sd'])} pip** · mean {_fmt(rp['mean'])} · "
      f"excess kurtosis {_fmt(rp['excess_kurtosis'],2)}")
    A(f"- quantiles: p01 {_fmt(rp['p01'])} · p25 {_fmt(rp['p25'])} · "
      f"median {_fmt(rp['median'])} · p75 {_fmt(rp['p75'])} · p99 {_fmt(rp['p99'])}")
    A(f"- EUR/GBP spread: median {_fmt(res['spread_c_pips']['median'],2)} pip · "
      f"p95 {_fmt(res['spread_c_pips']['p95'],2)} pip")
    A("")
    dtst = res["derived_cross_test"]
    A(f"**Section 1.3 test — is the cross internally derived?** "
      f"{dtst['fraction_abs_below']:.1%} of synchronized observations have "
      f"|ε| < {dtst['threshold_pips']} pip; {dtst['fraction_exactly_zero']:.2%} "
      f"are exactly zero.")
    A("")
    A(f"The literal test above is not the informative one when the residual "
      f"carries a persistent wedge, and this one does: median "
      f"{_fmt(dtst.get('median_pips'),3)} pip, same sign in "
      f"{dtst.get('fraction_same_sign', float('nan')):.1%} of observations. "
      f"A venue can derive the cross from the majors *and* apply a markup, "
      f"which is a shifted derived cross, not an independent price. After "
      f"removing the wedge, "
      f"{dtst.get('fraction_abs_below_after_demeaning', float('nan')):.1%} of "
      f"observations fall within {dtst['threshold_pips']} pip — that is the "
      f"number that speaks to §1.3.")
    A("")
    A("| sampling | n | sd pips | AR(1) φ | OU half-life s | acf(1) | acf(5) |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    for s in res["sampled"]:
        hl = s["ou_halflife_s"]
        A(f"| {s['step_ms']} ms | {s['n']:,} | {_fmt(s['sd_pips'])} | "
          f"{_fmt(s['ar1_phi'],4)} | {'no mean reversion' if hl is None else _fmt(hl,2)} | "
          f"{_fmt(s['acf'][1],3)} | {_fmt(s['acf'][5],3)} |")
    A("")

    A("## 2. Deterministic triangular census (H1)")
    A("")
    A(f"- synchronized observations with R1>0 or R2>0: "
      f"**{cen['n_observations_with_positive_cycle']:,} / "
      f"{cen['n_synchronized_observations']:,} "
      f"({cen['fraction_of_observations']:.3%})**")
    A(f"- distinct opportunity events: **{cen['n_events']}** "
      f"({_fmt(cen['events_per_hour'],3)} per hour over "
      f"{cen['observed_hours']:.2f} h)")
    A(f"- best cycle return: median {_fmt(cen['best_cycle_bp'].get('median'),3)} bp, "
      f"max {_fmt(cen['best_cycle_bp'].get('max'),3)} bp")
    A(f"- three-leg cycle cost: median {_fmt(cen['cycle_cost_bp'].get('median'),3)} bp")
    if cen["n_events"]:
        A(f"- event magnitude: median {_fmt(cen['magnitude_bp'].get('median'),3)} bp, "
          f"max {_fmt(cen['magnitude_bp'].get('max'),3)} bp")
        A(f"- event duration: median {_fmt(cen['duration_s'].get('median'),3)} s, "
          f"p95 {_fmt(cen['duration_s'].get('p95'),3)} s "
          f"(n = {cen['duration_s'].get('n', 0)} uncensored of {cen['n_events']})")
        A(f"- closed by a tick in: {cen['closed_by']}")
        if cen.get("n_events_censored_by_desync"):
            A(f"- **{cen['n_events_censored_by_desync']} of {cen['n_events']} "
              f"events were censored by desynchronization**: the book went stale "
              f"before the cycle was observed to close, so their durations are "
              f"unknown and excluded above. A high share here means τ and the "
              f"cross's tick rate, not the market, are setting what can be seen.")
    A("")

    A("## 3. Lead-lag")
    A("")
    A(f"- |z| ≥ {ll['event_z']} crossings: {ll['n_crossings']:,}")
    A(f"- last pair to update when the residual formed: "
      f"{ll['residual_formed_by_last_update']}")
    A(f"- pair whose tick closed it: {ll['closed_by_tick_in']}")
    if ll.get("confounded_by_flush_order"):
        A(f"- **CONFOUNDED — do not read the line above as price discovery.** "
          f"{ll['confound_note']}")
        A(f"- synchronized-sample base rate: "
          f"{ {k: f'{v:.1%}' for k, v in ll['synchronized_base_rate'].items()} }")
        A(f"- lift of closed-by over base rate (1.0 = no information): "
          f"{ {k: round(v, 2) for k, v in ll['closed_by_lift_over_base_rate'].items()} }")
    if ll["time_to_close_s"].get("n"):
        A(f"- time to close: median {_fmt(ll['time_to_close_s'].get('median'),3)} s")
    if ll["hy_best_lag"]:
        A(f"- Hayashi–Yoshida peak correlation {_fmt(ll['hy_best_lag']['corr'],4)} "
          f"at lag {ll['hy_best_lag']['lag_ms']} ms — {ll['hy_interpretation']}")
    A("")

    A("## 4. Conditional reversion curve")
    A("")
    A("`capture` is the favourable move, in pips, for the trade the bin implies "
      "(short C when z>0, long C when z<0). CI is Newey–West.")
    A("")
    for h, bins in rep["reversion"]["curves"].items():
        rows = [b for b in bins if b.get("n", 0) >= 30]
        if not rows:
            continue
        A(f"**Δ = {h} s**")
        A("")
        A("| z bin | n | z̃ | g(x,Δ) on ε (pips) | capture (pips) | NW t | CI95 |")
        A("|---|---:|---:|---:|---:|---:|---|")
        for b in rows:
            lo = "-inf" if b["z_lo"] is None else f"{b['z_lo']:g}"
            hi = "inf" if b["z_hi"] is None else f"{b['z_hi']:g}"
            ci = b["capture_ci95"]
            A(f"| [{lo}, {hi}) | {b['n']:,} | {_fmt(b['z_median'],2)} | "
              f"{_fmt(b['g_eps_mean_pips'])} | {_fmt(b['capture_pips'])} | "
              f"{_fmt(b['d_resid_t'],2)} | "
              f"[{_fmt(ci[0])}, {_fmt(ci[1])}] |")
        A("")
    chk = rep["reversion"]["best_bin_bootstrap_check"]
    if chk:
        A(f"Block-bootstrap cross-check on the best bin at Δ={chk['horizon_s']} s: "
          f"capture {_fmt(chk['capture_pips'])} pip, "
          f"bootstrap CI95 [{_fmt(chk['block_bootstrap_ci95'][0])}, "
          f"{_fmt(chk['block_bootstrap_ci95'][1])}], "
          f"NW CI95 [{_fmt(chk['nw_ci95'][0])}, {_fmt(chk['nw_ci95'][1])}].")
        A("")

    A("## 5. Gate G1")
    A("")
    cr = gate["criteria"]
    A(f"**{gate['verdict']}** — {gate['action']}")
    A("")
    A(f"- σ = {_fmt(cr['sigma_pips'])} pip vs required > {cr['min_sigma_pips']} pip "
      f"→ {'ok' if cr['sigma_ok'] else 'FAILS'}")
    A(f"- best capture within {cr['capture_horizon_s']} s = "
      f"{_fmt(cr['best_capture_pips'])} pip vs required > "
      f"{_fmt(cr['required_capture_pips'])} pip "
      f"(25% of the {_fmt(cr['eurgbp_spread_median_pips'],2)} pip spread) "
      f"→ {'ok' if cr['capture_ok'] else 'FAILS'}")
    if gate["suggested_parameters"]:
        sp = gate["suggested_parameters"]
        A(f"- suggested k_in = {_fmt(sp['k_in'],2)} "
          f"(expected net {_fmt(sp['expected_net_pips'])} pip) — {sp['caveat']}")
    else:
        A("- no z bin has expected capture exceeding the round-trip cost, so no "
          "k_in is suggested.")
    A("")

    A("## 6. Pre-registered hypotheses")
    A("")
    for k in ("H1", "H2", "H3"):
        h = hyp[k]
        A(f"- **{k}: {h['verdict']}** — {h['claim']}")
    A("")
    A("---")
    A("")
    A("Cost-model note: no fills exist in Phase 1, so every net figure above "
      "assumes zero slippage and is an upper bound. Phase 2 reports gross, "
      "net-at-quoted-spread and net-with-measured-latency-slippage separately.")
    return "\n".join(L)


def _rel(path: str) -> str:
    """Config path relative to fx_statarb/, so outputs carry no local home dir."""
    try:
        return str(Path(path).resolve().relative_to(HERE.resolve()))
    except ValueError:
        return Path(path).name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None,
                    help="override [storage].root; does not affect the config hash")
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="filter to session labels, e.g. overlap")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gate", action="store_true",
                    help="do not write out/g1_verdict.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.data_root:
        cfg["storage"]["root"] = args.data_root
    a, b, c = instruments(cfg)
    root = Path(cfg["_path"]).parent / cfg["storage"]["root"]
    outdir = Path(args.out) if args.out else HERE / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    ticks = load_ticks(root, args.dates, args.sessions)
    source = check_single_source(ticks)
    print(f"[phase1] {len(ticks):,} ticks, source={source}", flush=True)
    overlaps = overlapping_runs(ticks)
    if overlaps:
        print("[phase1] WARNING: %d overlapping run pair(s) -- two "
              "collectors recorded the same market and the census is "
              "double-counting. Quarantine one in "
              "data/excluded_runs.json before trusting this."
              % len(overlaps), flush=True)
        for o in overlaps[:5]:
            print("           %s vs %s  %s..%s"
                  % (o["run_a"][-8:], o["run_b"][-8:], o["from"], o["to"]),
                  flush=True)

    snap = replay(cfg, ticks, a, b, c)
    sync = snap.filter(snap["synchronized"])
    print(f"[phase1] {len(snap):,} snapshots, {len(sync):,} synchronized", flush=True)
    if len(sync) < 100:
        raise SystemExit(
            f"Only {len(sync)} synchronized observations. Collect more data, or "
            f"raise [signal].tau_ms -- but report the sensitivity either way.")

    anchors = clock_anchors(root, ticks)
    if anchors:
        print("[phase1] clock anchors: "
              + ", ".join("%s n=%d" % (k[-8:], len(v))
                          for k, v in anchors.items()), flush=True)
    else:
        print("[phase1] no clock anchors recorded; latency is raw", flush=True)
    dates = sorted({str(d)[:10] for d in ticks["recv_wall"].to_list()})
    an = cfg["analysis"]

    residual = P.residual_report(sync, an["resample_ms"],
                                 an["derived_cross_pip_threshold"])
    census = P.deterministic_census(sync)
    lead = P.lead_lag_report(sync, float(an["event_study_z"]), a, b, c)
    curve = P.reversion_curve(sync, [float(h) for h in an["reversion_horizons_s"]],
                              int(an["bootstrap_iters"]))
    gate = P.gate_g1(cfg, residual, curve, source)
    hyp = P.hypothesis_verdicts(census, residual, gate)

    rep = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provenance": {
            "source": source,
            "config_path": _rel(cfg["_path"]),
            "config_hash": cfg["_hash"],
            "dates": dates,
            "sessions_filter": args.sessions,
            "n_ticks": len(ticks),
            "n_snapshots": len(snap),
            "n_synchronized": len(sync),
            "synchronized_fraction": len(sync) / max(len(snap), 1),
            "tau_ms": cfg["signal"]["tau_ms"],
            "instruments": {"A": a, "B": b, "C": c},
            # Median mid per leg, so the report's quote-grid rounding floor is
            # computed at the price levels actually observed.
            "price_levels": {
                key: float(np.median(
                    (ticks.filter(pl.col("instrument") == name)["bid"].to_numpy()
                     + ticks.filter(pl.col("instrument") == name)["ask"].to_numpy())
                    / 2.0))
                for key, name in (("A", a), ("B", b), ("C", c))
            },
        },
        "events": event_log_summary(root, args.dates),
        "overlapping_runs": overlaps,
        "clock_anchors": anchors,
        "latency": P.latency_report(ticks, anchors),
        "burst_structure": P.burst_structure(ticks),
        "venue_sync_check": P.venue_sync_report(sync, float(cfg["signal"]["tau_ms"])),
        "quote_age": P.quote_age_report(snap),
        "tau_sensitivity": P.tau_sensitivity(snap, cfg["signal"]["tau_sensitivity_ms"]),
        "residual": residual,
        "deterministic_census": census,
        "lead_lag": lead,
        "reversion": curve,
        "gate_g1": gate,
        "hypotheses": hyp,
    }

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp = outdir / f"phase1_{stamp}.json"
    mp = outdir / f"phase1_{stamp}.md"
    jp.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    mp.write_text(build_markdown(rep), encoding="utf-8")
    if not args.no_gate:
        (outdir / "g1_verdict.json").write_text(
            json.dumps({**gate, "generated_utc": rep["generated_utc"],
                        "report": jp.name, "dates": dates}, indent=2,
                       default=str), encoding="utf-8")

    print(f"[phase1] residual sd = {residual['resid_pips']['sd']:.4f} pip", flush=True)
    print(f"[phase1] deterministic events = {census['n_events']} "
          f"({census['fraction_of_observations']:.3%} of observations)", flush=True)
    print(f"[phase1] GATE G1: {gate['verdict']} -- {gate['action']}", flush=True)
    print(f"[phase1] wrote {jp.name} and {mp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
