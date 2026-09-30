"""Phase 1 analysis: the census itself.

Every number the report claims is produced here, from the raw tick log, through
the live TriangleBook. Nothing is fitted by grid search over PnL -- the
conditional reversion curve is estimated once, nonparametrically, and the
thresholds fall out of it. That is deliberate (spec section 2, Phase 1.4): a
threshold grid search over realized PnL would inflate significance through
multiple testing, and this experiment exists to avoid exactly that error.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from pathlib import Path

from . import clocks, stats
from .book import PIP

# z-bins for the conditional reversion curve. Fixed a priori, not tuned.
Z_EDGES = np.array([-np.inf, -5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5,
                    0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, np.inf])


# ---------------------------------------------------------------------------
# Phase 0: latency and arrival structure
# ---------------------------------------------------------------------------
def clock_error_ms(df: pl.DataFrame, anchors_by_run: dict | None = None):
    """Per-tick local clock error in ms, reconstructed run by run.

    Returns (error_ms_series_aligned_to_df, diagnostics_by_run). Runs with no
    usable anchor get NaN, so their latency is reported raw rather than
    corrected by a number we do not have.
    """
    n = len(df)
    err = np.full(n, np.nan)
    diags = {}
    if not anchors_by_run:
        return err, diags
    wall_s = df["recv_wall"].dt.epoch("ns").to_numpy().astype(np.float64) / 1e9
    mono = df["recv_mono"].to_numpy().astype(np.float64)
    runs = np.array(df["run_id"].to_list())
    for run_id, anchors in anchors_by_run.items():
        m = runs == run_id
        if not m.any():
            continue
        order = np.argsort(mono[m], kind="stable")
        idx = np.flatnonzero(m)[order]
        e, diag = clocks.error_trace(wall_s[idx], mono[idx], anchors)
        diags[run_id] = diag
        if e is not None:
            err[idx] = e * 1e3
    return err, diags


def _collected_s(df: pl.DataFrame, col: str = "recv_mono") -> float:
    """Seconds actually collected: the sum of each run's own span.

    Never max - min over several runs -- that counts the gaps between them and,
    on recv_mono, whatever uptime preceded a reboot.
    """
    if df.is_empty():
        return 0.0
    g = df.group_by("run_id").agg((pl.col(col).max() - pl.col(col).min()).alias("s"))
    return float(g["s"].sum())


def latency_report(df: pl.DataFrame, anchors_by_run: dict | None = None) -> dict:
    """Latency per pair, raw and corrected for local clock error.

    anchors_by_run maps run_id -> list of NTP anchors; see fxlib.clocks.
    Correction is per TICK, not per run: the clock is slewed continuously by the
    OS, so a single scalar is wrong everywhere within a long run.
    """
    out = {}
    err_ms, clock_diag = clock_error_ms(df, anchors_by_run)
    df = df.with_columns(pl.Series("_clock_err_ms", err_ms))
    for inst in sorted(df["instrument"].unique().to_list()):
        # Run first: recv_mono restarts at reboot, so it only orders ticks
        # within one process.
        d = df.filter(pl.col("instrument") == inst).sort(["run_id", "recv_mono"])
        lat_ms = ((d["recv_wall"].dt.epoch("ns").to_numpy().astype(np.float64)
                   - d["oanda_ts"].dt.epoch("ns").to_numpy().astype(np.float64))
                  / 1e6)
        corrected = None
        e = d["_clock_err_ms"].to_numpy()
        if np.isfinite(e).any():
            corrected = lat_ms - e
        # Inter-arrival WITHIN a run only. The idle stretch between two
        # collector processes is a collection gap, not a market gap, and
        # counting it reports a 3.4 h "inter-arrival" that never happened.
        rid = np.array(d["run_id"].to_list())
        mono_d = d["recv_mono"].to_numpy()
        gaps_s = np.diff(mono_d)[rid[1:] == rid[:-1]]
        out[inst] = {
            "ticks": int(len(d)),
            "latency_ms": stats.describe(lat_ms),
            "latency_corrected_ms": (stats.describe(corrected)
                                     if corrected is not None else None),
            "inter_arrival_s": stats.describe(gaps_s),
            # Over collected time only. Spanning runs with recv_mono counted
            # the whole uptime before a reboot: 2.0/min on 9 Sept, truly ~30.
            "ticks_per_min": (float(len(d)) / max(_collected_s(d) / 60.0, 1e-9)),
            "spread_pips": stats.describe(
                (d["ask"].to_numpy() - d["bid"].to_numpy()) / PIP),
        }
        # A corrected latency below zero is impossible; if it appears, the clock
        # reconstruction is wrong and the number must not be presented as fact.
        if corrected is not None:
            p01 = out[inst]["latency_corrected_ms"].get("p01")
            if p01 is not None and np.isfinite(p01) and p01 < -5.0:
                out[inst]["latency_correction_warning"] = (
                    f"corrected p01 is {p01:.0f} ms, which is physically "
                    f"impossible -- the clock anchors disagree with the "
                    f"wall-minus-monotonic trace; treat the raw figure as the "
                    f"reliable one")
    out["_clock"] = clock_diag
    return out


def burst_structure(df: pl.DataFrame, gap_ms: float = 10.0) -> dict:
    """Characterize the feed's delivery cadence.

    The OANDA v20 pricing stream is NOT tick-by-tick: it coalesces updates and
    flushes them on a fixed cadence, measured at ~265 ms (roughly 4 Hz) on the
    practice feed. Ticks arrive in bursts of one to three instruments less than
    a millisecond apart, then nothing until the next flush.

    This determines what the experiment can see, so it belongs in the census:

      * "all three quote ages < tau" is really "all three legs landed in the
        same burst" for any tau below the cadence, which is why the synchronized
        count is flat from tau=100 to tau=250 and then jumps at tau=500;
      * the deterministic census cannot resolve an opportunity shorter than one
        flush interval, and the interdealer literature puts cost-exceeding
        triangular deviations well under one second -- so a retail feed of this
        kind cannot observe them even in principle;
      * reversion horizons below ~0.5 s contain one or two observations and
        should not be read as a curve.
    """
    d = df.sort(["run_id", "recv_mono"])
    mono = d["recv_mono"].to_numpy().astype(np.float64)
    inst = np.array(d["instrument"].to_list())
    rid = np.array(d["run_id"].to_list())
    if mono.size < 10:
        return {"n_ticks": int(mono.size), "note": "too few ticks"}
    gaps_ms = np.diff(mono) * 1e3
    # A run boundary always starts a new burst, and the idle time between runs
    # is not a burst interval.
    same_run = rid[1:] == rid[:-1]
    gaps_ms = np.where(same_run, gaps_ms, np.inf)
    new_burst = np.concatenate(([True], gaps_ms >= gap_ms))
    burst_id = np.cumsum(new_burst) - 1
    n_bursts = int(burst_id[-1]) + 1
    sizes = np.bincount(burst_id)
    bstart = mono[new_burst]
    brun = rid[new_burst]
    burst_gap_ms = np.diff(bstart)[brun[1:] == brun[:-1]] * 1e3

    # How often does a single burst carry all three legs? That is the ceiling on
    # the synchronized sample at any tau below the cadence.
    seen = {}
    for bid_, name in zip(burst_id.tolist(), inst.tolist()):
        seen.setdefault(bid_, set()).add(name)
    distinct = np.zeros(n_bursts, dtype=np.int64)
    for k, v in seen.items():
        distinct[k] = len(v)

    return {
        "n_ticks": int(mono.size),
        "n_bursts": n_bursts,
        "ticks_per_burst": float(mono.size / n_bursts),
        "burst_definition_gap_ms": gap_ms,
        "intra_burst_gap_ms": stats.describe(
            gaps_ms[np.isfinite(gaps_ms) & (gaps_ms < gap_ms)]),
        "burst_interval_ms": stats.describe(burst_gap_ms),
        "burst_size_counts": {int(k): int(v) for k, v in
                              zip(*np.unique(sizes, return_counts=True))},
        "bursts_with_all_three": int((distinct >= 3).sum()),
        "fraction_bursts_all_three": float((distinct >= 3).mean()),
        "starts_burst": _counts(inst[new_burst].tolist()),
        "cadence_hz": (1000.0 / float(np.median(burst_gap_ms))
                       if burst_gap_ms.size else float("nan")),
    }


def venue_sync_report(sync: pl.DataFrame, tau_ms: float) -> dict:
    """Cross-check receive-clock synchronization against venue timestamps.

    The tau test uses quote AGES on the monotonic receive clock, per the spec.
    That test has one blind spot: if the process is descheduled -- a busy
    machine, a slow disk flush, another job on the box -- queued ticks drain in
    microseconds, so three legs that genuinely arrived hundreds of ms apart get
    near-identical receive stamps and sail through the tau filter while holding
    prices from different moments. Local CPU contention thus ADDS contaminated
    observations rather than merely losing clean ones.

    Venue timestamps do not share that failure mode. This reports the spread of
    OANDA timestamps within each receive-synchronized observation, and what the
    residual looks like with the contaminated tail removed.

    It is a robustness check, not a change to the pre-registered criterion: the
    census still defines synchronization exactly as the spec does, and reports
    this alongside so the reader can see how much it would matter.
    """
    v = sync["venue_spread_ms"].to_numpy()
    r = sync["resid_pips"].to_numpy()
    rows = []
    for thr in (tau_ms, 250.0, 500.0, 1000.0, 2000.0):
        m = v <= thr
        rows.append({
            "venue_spread_max_ms": thr,
            "n": int(m.sum()),
            "fraction_kept": float(m.mean()) if v.size else float("nan"),
            "resid_sd_pips": float(np.nanstd(r[m])) if m.sum() > 2 else float("nan"),
            "resid_mean_pips": float(np.nanmean(r[m])) if m.sum() else float("nan"),
        })
    contaminated = v > max(tau_ms, 250.0)
    return {
        "venue_spread_ms": stats.describe(v),
        "by_threshold": rows,
        "n_contaminated": int(contaminated.sum()),
        "fraction_contaminated": float(contaminated.mean()) if v.size else float("nan"),
        "resid_sd_contaminated": (float(np.nanstd(r[contaminated]))
                                  if contaminated.sum() > 2 else None),
        "resid_sd_clean": (float(np.nanstd(r[~contaminated]))
                           if (~contaminated).sum() > 2 else None),
        "note": "receive-synchronized observations whose venue timestamps are "
                "far apart indicate local scheduling delay, not a real "
                "simultaneous quote",
    }


def collection_health(root, dates, tz_name: str,
                      window=("03:00", "17:00")) -> dict:
    """Did we actually collect the whole session, and did every run end cleanly?

    A collector that is killed writes no `run_end` event -- it cannot, it is
    already gone -- so the only way to notice is to look for the absence. This
    happened three times in the first week: a console break, a closed window,
    and Windows Restart Manager shutting the process down for an MSI install.
    Each time the loss was found by a human noticing, not by the tooling.

    Reports, per date: coverage of the intended window, every uncovered gap over
    a minute, and which runs ended cleanly.
    """
    import datetime as _dt
    import glob as _glob
    import json as _json
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_name)
    h0, m0 = (int(x) for x in window[0].split(":"))
    h1, m1 = (int(x) for x in window[1].split(":"))

    ended = {}
    for f in sorted(_glob.glob(str(Path(root) / "events" / "*.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = _json.loads(line)
            except ValueError:
                continue
            if r.get("event") == "run_end":
                ended[r.get("run_id")] = r.get("reason")

    out = {"window": list(window), "timezone": tz_name, "days": []}
    for day in dates:
        files = _glob.glob(str(Path(root) / "ticks" / day / "*.parquet"))
        if not files:
            out["days"].append({"date": day, "n_ticks": 0,
                                "coverage_pct": 0.0, "note": "no ticks"})
            continue
        df = pl.read_parquet(files)
        d0 = _dt.datetime.fromisoformat(day).replace(tzinfo=tz)
        w0 = d0.replace(hour=h0, minute=m0)
        w1 = d0.replace(hour=h1, minute=m1)
        spans, runs = [], []
        for rid in sorted(df["run_id"].unique().to_list()):
            sub = df.filter(pl.col("run_id") == rid)
            lo = sub["recv_wall"].min().astimezone(tz)
            hi = sub["recv_wall"].max().astimezone(tz)
            spans.append((max(lo, w0), min(hi, w1)))
            # No run_end means one of two very different things: the collector
            # died, or it is still running and simply has not written one yet.
            # Reporting "KILLED" for a live run is a false alarm that trains the
            # reader to ignore the flag, so distinguish them by recency.
            if rid in ended:
                state = ended[rid]
            else:
                age_s = (_dt.datetime.now(tz) - hi).total_seconds()
                state = ("IN PROGRESS (no run_end yet)" if age_s < 300
                         else "KILLED (no run_end)")
            runs.append({"run_id": rid, "rows": len(sub),
                         "from": lo.strftime("%H:%M:%S"),
                         "to": hi.strftime("%H:%M:%S"),
                         "ended": state})
        spans = [(a, b) for a, b in spans if b > a]
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        covered = sum((b - a).total_seconds() for a, b in merged)
        total = (w1 - w0).total_seconds()
        holes, cursor = [], w0
        for a, b in merged:
            if (a - cursor).total_seconds() > 60:
                holes.append({"from": cursor.strftime("%H:%M:%S"),
                              "to": a.strftime("%H:%M:%S"),
                              "minutes": round((a - cursor).total_seconds() / 60, 1)})
            cursor = max(cursor, b)
        if (w1 - cursor).total_seconds() > 60:
            holes.append({"from": cursor.strftime("%H:%M:%S"),
                          "to": w1.strftime("%H:%M:%S"),
                          "minutes": round((w1 - cursor).total_seconds() / 60, 1)})
        out["days"].append({
            "date": day, "n_ticks": len(df),
            "coverage_pct": 100.0 * covered / total if total else float("nan"),
            "uncovered_minutes": round(sum(h["minutes"] for h in holes), 1),
            "holes": holes, "runs": runs,
            "killed_runs": [r["run_id"] for r in runs
                            if r["ended"].startswith("KILLED")],
            "in_progress": [r["run_id"] for r in runs
                            if r["ended"].startswith("IN PROGRESS")],
        })
    return out


def instrument_rows(lat: dict):
    """Iterate the per-instrument entries of a latency report, skipping the
    underscore-prefixed diagnostics that share the dict."""
    return [(k, v) for k, v in lat.items() if not k.startswith("_")]


def quote_age_report(snap: pl.DataFrame) -> dict:
    return {
        "age_a_ms": stats.describe(snap["age_a_ms"].to_numpy()),
        "age_b_ms": stats.describe(snap["age_b_ms"].to_numpy()),
        "age_c_ms": stats.describe(snap["age_c_ms"].to_numpy()),
        "max_age_ms": stats.describe(snap["max_age_ms"].to_numpy()),
    }


def tau_sensitivity(snap: pl.DataFrame, taus_ms: list) -> list:
    """How much of the sample survives each synchronization tolerance, and what
    the residual looks like there. Reported because tau is an arbitrary choice
    and the conclusion must not depend on it."""
    rows = []
    max_age = snap["max_age_ms"].to_numpy()
    resid_pips = snap["resid_pips"].to_numpy()
    n = len(snap)
    for tau in taus_ms:
        m = max_age < tau
        r = resid_pips[m]
        rows.append({
            "tau_ms": tau,
            "n": int(m.sum()),
            "fraction_of_ticks": float(m.mean()) if n else 0.0,
            "resid_sd_pips": float(np.nanstd(r)) if r.size > 2 else float("nan"),
            "resid_mean_pips": float(np.nanmean(r)) if r.size else float("nan"),
            "abs_resid_p95_pips": (float(np.nanquantile(np.abs(r), 0.95))
                                   if r.size else float("nan")),
        })
    return rows


# ---------------------------------------------------------------------------
# H2: residual structure, and the section 1.3 derived-cross test
# ---------------------------------------------------------------------------
def _k_bases(sync: pl.DataFrame, resid_pips: np.ndarray) -> dict:
    """k = median cross spread / sigma, with sigma demeaned and within-hour."""
    out = {"sigma_demeaned_pips": None, "required_k_demeaned": None,
           "sigma_within_hour_pips": None, "required_k_within_hour": None}
    if resid_pips.size < 3:
        return out
    spread = float(np.median(sync["c_spread_pips"].to_numpy()))
    dem = sync["z"].to_numpy() * sync["sigma_pips"].to_numpy()   # resid - EWMA mean
    dem = dem[np.isfinite(dem)]
    if dem.size > 2 and np.std(dem) > 0:
        out["sigma_demeaned_pips"] = float(np.std(dem))
        out["required_k_demeaned"] = spread / float(np.std(dem))
    g = (sync.with_columns(pl.col("recv_wall").dt.strftime("%Y-%m-%d %H").alias("_dh"))
             .group_by("_dh").agg(pl.col("resid_pips").var().alias("v"), pl.len().alias("n"))
             .filter(pl.col("n") >= 30))
    if len(g):
        w = float(np.sqrt(np.average(g["v"].to_numpy(), weights=g["n"].to_numpy())))
        if w > 0:
            out["sigma_within_hour_pips"] = w
            out["required_k_within_hour"] = spread / w
    return out


def residual_report(sync: pl.DataFrame, resample_ms: list,
                    derived_threshold_pips: float) -> dict:
    resid_pips = sync["resid_pips"].to_numpy()
    ts = sync["t_axis"].to_numpy()
    seg = np.array(sync["run_id"].to_list())

    rep = {
        "n_synchronized": int(len(sync)),
        "resid_pips": stats.describe(resid_pips),
        "eps_pips": stats.describe(-resid_pips),
        "spread_c_pips": stats.describe(sync["c_spread_pips"].to_numpy()),
        # THE cross-triangle comparable. sigma in pips is NOT comparable between
        # crosses -- one EUR/GBP pip is 1.16 bp, one EUR/CZK pip is 0.041 bp, so
        # reading sigma alone makes EUR/CZK look like a 22x larger residual when
        # in basis points it is smaller. k = round_trip / sigma is dimensionless
        # (the pip factor cancels) and IS the entry threshold breakeven demands
        # at full capture. Compare triangles on this, never on sigma.
        "required_k_full_capture": (
            float(np.median(sync["c_spread_pips"].to_numpy()))
            / float(np.nanstd(resid_pips))) if resid_pips.size > 2 else None,
        # k on two further sigma bases. Raw sigma above includes the wedge's
        # LEVEL drifting between hours and days, which no 30 s trade can
        # capture, and it grows with sample length (EUR/GBP: k 13.7 demeaned
        # but 10.5 raw over the fortnight). Comparing a 3-day arm's raw k with
        # a fortnight's demeaned k made AUD/NZD look like it beat EUR/GBP when
        # on every consistent basis it did not. Compare triangles on
        # required_k_demeaned: it is the residual the signal trades.
        **_k_bases(sync, resid_pips),
        "derived_cross_test": {
            "threshold_pips": derived_threshold_pips,
            # The spec's literal test: how much of the residual sits within
            # 0.05 pip of ZERO.
            "fraction_abs_below": float(
                np.mean(np.abs(resid_pips) < derived_threshold_pips))
            if resid_pips.size else float("nan"),
            "fraction_exactly_zero": float(np.mean(resid_pips == 0.0))
            if resid_pips.size else float("nan"),
            # ...and the test that actually answers the question. A venue can
            # derive the cross from the majors AND apply a markup, which leaves
            # a large constant wedge with almost no variance around it. The
            # literal test calls that "not derived" and is wrong: it is derived,
            # just shifted. Measured here at +0.41 pip mean with 0.12 pip sd,
            # positive in 99.5% of observations. What rules out an internally
            # derived cross is VARIANCE around the wedge, not distance from zero.
            "median_pips": float(np.median(resid_pips)) if resid_pips.size else None,
            "fraction_same_sign": (float(max(np.mean(resid_pips > 0),
                                             np.mean(resid_pips < 0)))
                                   if resid_pips.size else float("nan")),
            "fraction_abs_below_after_demeaning": float(
                np.mean(np.abs(resid_pips - np.median(resid_pips))
                        < derived_threshold_pips)) if resid_pips.size else float("nan"),
            "note": "fraction_abs_below_after_demeaning is the informative one "
                    "whenever the residual carries a persistent wedge",
        },
        "sampled": [],
    }

    for step_ms in resample_ms:
        step = step_ms / 1000.0
        g, v = stats.resample_last(ts, resid_pips, step, seg=seg)
        phi, hl = stats.ar1_halflife(v, step)
        a = stats.acf(v, nlags=20)
        rep["sampled"].append({
            "step_ms": step_ms,
            "n": int(v.size),
            "sd_pips": float(np.nanstd(v)) if v.size > 2 else float("nan"),
            "excess_kurtosis": stats.describe(v).get("excess_kurtosis"),
            "ar1_phi": phi,
            "ou_halflife_s": hl,
            "acf": [float(x) for x in a[:11]],
        })
    return rep


# ---------------------------------------------------------------------------
# H1: deterministic triangular arbitrage census
# ---------------------------------------------------------------------------
def deterministic_census(sync: pl.DataFrame) -> dict:
    r1 = sync["r1"].to_numpy()
    r2 = sync["r2"].to_numpy()
    ts = sync["t_axis"].to_numpy()
    runs = sync["run_id"].to_list()
    wall = sync["recv_wall"].to_list()
    tick_inst = sync["tick_instrument"].to_list()
    idx = sync["snap_idx"].to_numpy()
    best = np.maximum(r1, r2)
    arb = best > 0.0

    events = []
    n_desync = 0
    i, n = 0, len(sync)
    while i < n:
        if not arb[i]:
            i += 1
            continue
        # An event runs while the cycle stays positive AND the book stays
        # synchronized. If synchronization lapses, the event is truncated and
        # marked -- we did not observe it close, so we must not claim a duration
        # that spans the blind spot.
        j = i
        while (j + 1 < n and arb[j + 1] and idx[j + 1] == idx[j] + 1
               and runs[j + 1] == runs[j]):
            j += 1
        if j + 1 >= n:
            closer = "end_of_sample"
        elif runs[j + 1] != runs[j]:
            closer = "end_of_run"
        elif idx[j + 1] != idx[j] + 1:
            closer = "desync"
            n_desync += 1
        else:
            closer = tick_inst[j + 1]
        events.append({
            "start_utc": str(wall[i]),
            "duration_s": float(ts[j] - ts[i]),
            "duration_censored": closer in ("desync", "end_of_sample", "end_of_run"),
            "n_snapshots": int(j - i + 1),
            "max_magnitude_bp": float(best[i:j + 1].max() * 1e4),
            "direction": "D1" if r1[i:j + 1].max() >= r2[i:j + 1].max() else "D2",
            "closed_by_tick_in": closer,
        })
        i = j + 1

    mags = np.array([e["max_magnitude_bp"] for e in events]) if events else np.array([])
    # Only uncensored events have a duration we actually observed to completion.
    durs = np.array([e["duration_s"] for e in events
                     if not e["duration_censored"]]) if events else np.array([])
    # Sum of per-run spans. ts[-1] - ts[0] counted every overnight and weekend
    # gap between runs as observed time, understating events_per_hour.
    span_h = _collected_s(sync, "t_axis") / 3600.0 if n else 0.0
    return {
        "n_synchronized_observations": int(n),
        "n_observations_with_positive_cycle": int(arb.sum()),
        "fraction_of_observations": float(arb.mean()) if n else 0.0,
        "n_events": len(events),
        "events_per_hour": (len(events) / span_h) if span_h > 0 else float("nan"),
        "observed_hours": span_h,
        "magnitude_bp": stats.describe(mags),
        "duration_s": stats.describe(durs),
        "duration_note": "computed on uncensored events only; events truncated "
                         "by desynchronization or end-of-sample are excluded",
        "n_events_censored_by_desync": n_desync,
        "closed_by": _counts([e["closed_by_tick_in"] for e in events]),
        "best_cycle_bp": stats.describe(best * 1e4),
        "cycle_cost_bp": stats.describe(sync["cycle_cost_bp"].to_numpy()),
        "events": events[:500],
        "events_truncated": max(0, len(events) - 500),
    }


def _counts(xs: list) -> dict:
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Lead-lag: does the cross actually lag the majors?
# ---------------------------------------------------------------------------
def lead_lag_report(sync: pl.DataFrame, event_z: float,
                    a_name: str, b_name: str, c_name: str,
                    lag_grid_ms=(-2000, -1000, -500, -250, -100, -50, 0,
                                 50, 100, 250, 500, 1000, 2000)) -> dict:
    z = sync["z"].to_numpy()
    ts = sync["t_axis"].to_numpy()      # runs RUN_GAP_S apart: 60 s window stops at a run end
    seg = np.array(sync["run_id"].to_list())
    last_upd = sync["last_updated"].to_list()
    tick_inst = sync["tick_instrument"].to_list()

    # --- event study around |z| crossings ------------------------------
    ok = np.isfinite(z)
    az = np.abs(np.where(ok, z, 0.0))
    crossed = np.zeros(len(z), dtype=bool)
    crossed[1:] = (ok[1:] & ok[:-1] & (az[1:] >= event_z) & (az[:-1] < event_z)
                   & (seg[1:] == seg[:-1]))
    idx = np.flatnonzero(crossed)

    formed_by, closed_by, close_secs = [], [], []
    exit_z = event_z / 2.0
    for i in idx:
        formed_by.append(last_upd[i])
        j = i + 1
        limit = ts[i] + 60.0
        while j < len(z) and ts[j] <= limit:
            if np.isfinite(z[j]) and abs(z[j]) <= exit_z:
                closed_by.append(tick_inst[j])
                close_secs.append(float(ts[j] - ts[i]))
                break
            j += 1
        else:
            closed_by.append("not_closed_within_60s")

    # --- Hayashi-Yoshida cross-check -----------------------------------
    # ln C from cross ticks only; ln(A/B) from major ticks only. Each series
    # keeps its own arrival times -- no common grid, no interpolation.
    ln_c_all = np.log(sync["c_mid"].to_numpy())
    ln_imp_all = ln_c_all - sync["resid"].to_numpy()
    is_c = np.array([t == c_name for t in tick_inst])
    is_m = np.array([t in (a_name, b_name) for t in tick_inst])

    hy = []
    if is_c.sum() > 50 and is_m.sum() > 50:
        t_m, p_m = ts[is_m], ln_imp_all[is_m]
        t_c, p_c = ts[is_c], ln_c_all[is_c]
        for lag_ms in lag_grid_ms:
            _, corr = stats.hayashi_yoshida(t_m, p_m, t_c, p_c,
                                            lag_s=lag_ms / 1000.0,
                                            seg1=seg[is_m], seg2=seg[is_c])
            hy.append({"lag_ms": lag_ms, "corr": corr})
    best_lag = None
    if hy:
        fin = [h for h in hy if np.isfinite(h["corr"])]
        if fin:
            best_lag = max(fin, key=lambda h: h["corr"])

    # The synchronized subset is not a random sample of ticks. All three quotes
    # are only simultaneously fresh at the END of a delivery burst, so whichever
    # instrument the venue flushes last supplies almost every synchronized
    # observation -- 95.5% GBP_USD on the 2026-08-28 sample. "Which pair closed
    # the dislocation" then just recovers that base rate and says nothing about
    # price discovery. Report the base rate alongside, and the lift over it,
    # so the confound cannot be read as a result.
    base = _counts(tick_inst)
    total = max(len(tick_inst), 1)
    base_share = {k: v / total for k, v in base.items()}
    closed = _counts(closed_by)
    n_closed = max(sum(v for k, v in closed.items()
                       if k != "not_closed_within_60s"), 1)
    lift = {}
    for k, v in closed.items():
        if k == "not_closed_within_60s" or k not in base_share or not base_share[k]:
            continue
        lift[k] = (v / n_closed) / base_share[k]
    dominant = max(base_share.values()) if base_share else 0.0

    return {
        "event_z": event_z,
        "n_crossings": int(len(idx)),
        "residual_formed_by_last_update": _counts(formed_by),
        "closed_by_tick_in": closed,
        "synchronized_base_rate": base_share,
        "closed_by_lift_over_base_rate": lift,
        "confounded_by_flush_order": bool(dominant > 0.6),
        "confound_note": (
            "one instrument supplies %.1f%% of synchronized observations because "
            "the venue flushes it last in each burst; closed_by therefore mostly "
            "recovers that base rate. Judge lift over base rate, not raw counts, "
            "and treat Hayashi-Yoshida on raw tick times as the only "
            "unconfounded lead-lag instrument here." % (dominant * 100)
        ) if dominant > 0.6 else None,
        "time_to_close_s": stats.describe(np.array(close_secs)),
        "hayashi_yoshida": hy,
        "hy_best_lag": best_lag,
        "hy_best_lag_at_grid_edge": bool(
            best_lag is not None and best_lag["lag_ms"] in (min(lag_grid_ms), max(lag_grid_ms))),
        "hy_interpretation": (
            "NEGATIVE best lag_ms means the majors LEAD the quoted cross by that many "
            "ms (the cross had to be shifted earlier to line up); positive means the "
            "cross leads. Verified on synthetic ground truth. A maximum at the edge of "
            "the lag grid is a boundary value, not an estimate of the lag."),
        "share_of_ticks": _counts(tick_inst),
    }


# ---------------------------------------------------------------------------
# The conditional reversion curve -- the object that sets the thresholds
# ---------------------------------------------------------------------------
def reversion_curve(sync: pl.DataFrame, horizons_s: list,
                    bootstrap_iters: int = 1000) -> dict:
    # t_axis is sorted and puts runs RUN_GAP_S apart, so searchsorted is valid
    # and any pair spanning two runs fails the realized-gap check below.
    ts = sync["t_axis"].to_numpy()
    resid_pips = sync["resid_pips"].to_numpy()
    z = sync["z"].to_numpy()
    c_spread = sync["c_spread_pips"].to_numpy()
    n = len(ts)

    curves = {}
    for h in horizons_s:
        # First observation at or after t+h. Pairs that would straddle a stream
        # gap are dropped: a "30-second" move measured across a 5-minute hole is
        # not a 30-second move.
        j = np.searchsorted(ts, ts + h, side="left")
        valid = (j < n)
        j_safe = np.where(valid, np.minimum(j, n - 1), 0)
        realized = ts[j_safe] - ts
        valid &= (realized <= h + max(1.0, 0.5 * h))
        valid &= np.isfinite(z)

        d_resid = np.where(valid, resid_pips[j_safe] - resid_pips, np.nan)
        # g is defined on eps = -resid (spec 1.1); capture is what a trade earns.
        d_eps = -d_resid

        bins = []
        bidx = np.digitize(z, Z_EDGES) - 1
        for k in range(len(Z_EDGES) - 1):
            m = valid & (bidx == k)
            cnt = int(m.sum())
            lo, hi = Z_EDGES[k], Z_EDGES[k + 1]
            if cnt < 30:
                bins.append({"z_lo": _f(lo), "z_hi": _f(hi), "n": cnt})
                continue
            xz = z[m]
            dz = d_resid[m]
            centre = float(np.median(xz))
            mean_d, se, t = stats.nw_tstat(dz)
            # Capture: the favourable move for the trade this bin implies.
            # Enter short C when resid > 0, long C when resid < 0.
            side = -1.0 if centre > 0 else 1.0
            cap = side * mean_d
            # Cost of the one-leg trade at the spread QUOTED AT THE TIME: half
            # the cross spread at entry plus half at exit. The gate compares
            # capture with the MEDIAN spread, but the extreme bins sit on news
            # releases where the cross is quoted 2-6 pips wide -- on this
            # census the z>=5 bin captured 0.93 pip against a 2.2 pip cost.
            # Diagnostic only: it does not enter the pre-registered gate.
            cost = 0.5 * c_spread[m] + 0.5 * c_spread[j_safe[m]]
            bins.append({
                "z_lo": _f(lo), "z_hi": _f(hi), "n": cnt,
                "z_median": centre,
                "resid_median_pips": float(np.median(resid_pips[m])),
                "g_eps_mean_pips": float(np.mean(d_eps[m])),
                "d_resid_mean_pips": mean_d,
                "d_resid_nw_se": se,
                "d_resid_t": t,
                "capture_pips": cap,
                "capture_ci95": [cap - 1.96 * se, cap + 1.96 * se]
                if np.isfinite(se) else [float("nan")] * 2,
                "c_spread_median_pips": float(np.median(c_spread[m])),
                "cost_at_quoted_spread_pips": float(np.mean(cost)),
                "net_at_quoted_spread_pips": float(np.mean(side * dz - cost)),
                "fraction_net_positive": float(np.mean(side * dz - cost > 0)),
            })
        curves[str(h)] = bins

    # Block-bootstrap cross-check on the single most favourable bin at the
    # gate horizon, because that is the bin the decision actually rests on.
    check = None
    gate_h = str(max(horizons_s))
    scored = [b for b in curves[gate_h] if b.get("n", 0) >= 30
              and np.isfinite(b.get("capture_pips", np.nan))]
    if scored:
        best = max(scored, key=lambda b: b["capture_pips"])
        h = float(gate_h)
        j = np.searchsorted(ts, ts + h, side="left")
        valid = j < n
        j_safe = np.where(valid, np.minimum(j, n - 1), 0)
        valid &= (ts[j_safe] - ts) <= h + max(1.0, 0.5 * h)
        bidx = np.digitize(z, Z_EDGES) - 1
        k = int(np.flatnonzero((Z_EDGES[:-1] == _u(best["z_lo"])))[0])
        m = valid & (bidx == k) & np.isfinite(z)
        side = -1.0 if best["z_median"] > 0 else 1.0
        sample = side * (resid_pips[np.where(m, j, 0)] - resid_pips)[m]
        lo, hi = stats.circular_block_bootstrap_ci(sample, iters=bootstrap_iters)
        check = {"horizon_s": h, "z_lo": best["z_lo"], "z_hi": best["z_hi"],
                 "capture_pips": best["capture_pips"],
                 "block_bootstrap_ci95": [lo, hi],
                 "nw_ci95": best["capture_ci95"]}

    return {"z_edges": [_f(x) for x in Z_EDGES], "curves": curves,
            "best_bin_bootstrap_check": check}


def _f(x):
    return None if not np.isfinite(x) else float(x)


def _u(x):
    return -np.inf if x is None else x


# ---------------------------------------------------------------------------
# Gate G1 and the threshold that Phase 2 would freeze
# ---------------------------------------------------------------------------
def gate_g1(cfg: dict, residual: dict, curve: dict, source: str) -> dict:
    g = cfg["gate_g1"]
    sigma_pips = residual["resid_pips"]["sd"]
    spread_pips = residual["spread_c_pips"]["median"]
    horizon = str(float(g["capture_horizon_s"]))
    bins = curve["curves"].get(horizon) or []
    scored = [b for b in bins if b.get("n", 0) >= 30
              and np.isfinite(b.get("capture_pips", float("nan")))]
    best = max(scored, key=lambda b: b["capture_pips"]) if scored else None

    need_capture = g["min_capture_fraction"] * spread_pips
    cond_sigma = sigma_pips > g["min_sigma_pips"]
    cond_capture = bool(best) and best["capture_pips"] > need_capture

    # k_in = argmax over x of [capture - spread - 2 * slippage].  In Phase 1 no
    # fills exist, so measured slippage is zero by construction; the number is
    # therefore an UPPER bound on what Phase 2 can expect, and is labelled so.
    round_trip = spread_pips
    ranked = sorted(
        [b for b in scored if b["z_median"] is not None],
        key=lambda b: b["capture_pips"] - round_trip, reverse=True)
    suggested = None
    if ranked and ranked[0]["capture_pips"] - round_trip > 0:
        b = ranked[0]
        suggested = {
            "k_in": abs(b["z_median"]),
            "z_bin": [b["z_lo"], b["z_hi"]],
            "expected_capture_pips": b["capture_pips"],
            "round_trip_cost_pips": round_trip,
            "expected_net_pips": b["capture_pips"] - round_trip,
            "slippage_assumed_pips": 0.0,
            "caveat": "zero slippage assumed -- Phase 1 has no fills; this is an "
                      "upper bound, not a forecast",
        }

    verdict = "PASS" if (cond_sigma and cond_capture) else "FAIL"
    if source != "oanda":
        verdict = "VOID_SYNTHETIC"

    return {
        "verdict": verdict,
        "source": source,
        "config_hash": cfg["_hash"],
        "criteria": {
            "sigma_pips": sigma_pips,
            "min_sigma_pips": g["min_sigma_pips"],
            "sigma_ok": bool(cond_sigma),
            "eurgbp_spread_median_pips": spread_pips,
            "required_capture_pips": need_capture,
            "best_capture_pips": best["capture_pips"] if best else None,
            "best_capture_z_bin": [best["z_lo"], best["z_hi"]] if best else None,
            "capture_ok": bool(cond_capture),
            "capture_horizon_s": g["capture_horizon_s"],
        },
        "suggested_parameters": suggested,
        "action": ("proceed to Phase 2" if verdict == "PASS" else
                   "skip to Phase 3 and report; Phase 2 may run 2-3 days as "
                   "pipeline validation only"),
    }


def hypothesis_verdicts(census: dict, residual: dict, gate: dict) -> dict:
    sd = residual["resid_pips"]["sd"]
    return {
        "H1": {
            "claim": "P(R1>0 or R2>0 at synchronized executable quotes) ~ 0",
            "observed_fraction": census["fraction_of_observations"],
            "n_events": census["n_events"],
            "events_per_hour": census["events_per_hour"],
            "verdict": "CONFIRMED" if census["fraction_of_observations"] < 1e-4
                       else "NOT CONFIRMED",
        },
        "H2": {
            "claim": "stationary residual sigma <= 0.3 pip EUR/GBP-equivalent",
            "observed_sigma_pips": sd,
            "verdict": "CONFIRMED" if sd <= 0.3 else "NOT CONFIRMED",
        },
        "H3": {
            "claim": "E[net per trade] < 0 at all thresholds under the cost model",
            "note": "Phase 1 can only bound this: it is decided by the reversion "
                    "curve against the measured round trip, and confirmed or "
                    "refuted by realized fills in Phase 2.",
            "best_expected_net_pips": (gate["suggested_parameters"]
                                       ["expected_net_pips"]
                                       if gate["suggested_parameters"] else None),
            "verdict": ("CONFIRMED (no threshold has positive expected net)"
                        if not gate["suggested_parameters"]
                        else "NOT CONFIRMED IN PHASE 1 -- test in Phase 2"),
        },
    }
