"""Replay a stored tick log through the live TriangleBook.

The shadow measurement in Phase 1 and the live signal in Phase 2 must be the
same computation or the census describes a system that was never run. So the
analysis does not re-implement the residual: it feeds the recorded ticks, in
recorded order, with their recorded monotonic receive stamps, through the very
same fxlib.book.TriangleBook the trading engine uses.

Consequences worth stating explicitly:
  * quote ages in the replay are the ages the live engine would have seen;
  * a snapshot is produced on every PRICE tick, event-driven, exactly as live;
  * nothing is interpolated, forward-filled or smoothed anywhere.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

from .book import PIP, Quote, TriangleBook

SNAP_FIELDS = [
    "ts_mono", "eps", "resid", "resid_pips", "sigma", "sigma_pips", "mu",
    "z", "z_raw", "n_obs", "r1", "r2", "cycle_cost_bp",
    "age_a_ms", "age_b_ms", "age_c_ms", "max_age_ms", "c_spread_pips",
    "venue_spread_ms",
]


def tick_files(root: Path, dates: list | None = None) -> list:
    base = Path(root) / "ticks"
    if dates:
        out = []
        for d in dates:
            out += sorted(glob.glob(str(base / d / "*.parquet")))
        return out
    return sorted(glob.glob(str(base / "*" / "*.parquet")))


def load_exclusions(root: Path) -> dict:
    """Runs quarantined from analysis, keyed by run_id.

    The tick log is append-only: nothing is ever deleted from it. But a run can
    still be invalid -- most obviously when two collectors overlapped and
    recorded the same market twice. Quarantining by run_id keeps the raw record
    intact and the reason on the record, which is the honest version of
    "delete the bad rows".
    """
    p = Path(root) / "excluded_runs.json"
    if not p.exists():
        return {}
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {e["run_id"]: e for e in body.get("runs", []) if e.get("run_id")}


def overlapping_runs(df: pl.DataFrame) -> list:
    """Pairs of runs whose recv_wall intervals intersect.

    Two collectors running at once record the same ticks under different
    run_ids, which double-counts every observation in the census while leaving
    every file looking valid. Cheap to detect, invisible if you do not look.
    """
    spans = []
    for rid in sorted(df["run_id"].unique().to_list()):
        d = df.filter(pl.col("run_id") == rid)
        spans.append((rid, d["recv_wall"].min(), d["recv_wall"].max(), len(d)))
    out = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if a[1] <= b[2] and b[1] <= a[2]:
                out.append({"run_a": a[0], "rows_a": a[3],
                            "run_b": b[0], "rows_b": b[3],
                            "from": str(max(a[1], b[1])),
                            "to": str(min(a[2], b[2]))})
    return out


def load_ticks(root: Path, dates: list | None = None,
               sessions: list | None = None) -> pl.DataFrame:
    files = tick_files(root, dates)
    if not files:
        raise SystemExit(
            f"No tick files under {Path(root) / 'ticks'}. Run collect_ticks.py first.")
    df = pl.read_parquet(files)
    # Receive order is the only order the engine could ever have seen -- but
    # only WITHIN a process. The monotonic clock restarts at every reboot, so
    # two runs days apart can share a monotonic range; a global sort on it
    # interleaved the 1 Sept and 9 Sept runs tick by tick (18,142 run switches
    # across 20 runs). Runs are ordered by their wall-clock start (a calendar
    # question, which is what the wall clock is for), ticks within a run by
    # the monotonic clock.
    df = (df.with_columns(pl.col("recv_wall").min().over("run_id").alias("_run_start"))
            .sort(["_run_start", "run_id", "recv_mono"])
            .drop("_run_start"))
    excluded = load_exclusions(root)
    if excluded:
        before = len(df)
        df = df.filter(~pl.col("run_id").is_in(list(excluded)))
        dropped = before - len(df)
        if dropped:
            print("[replay] excluded %d rows from %d quarantined run(s): %s"
                  % (dropped, len(excluded),
                     "; ".join("%s (%s)" % (k, v.get("reason", "no reason given"))
                               for k, v in excluded.items())))
    if sessions:
        df = df.filter(pl.col("session").is_in(sessions))
    return df


def check_single_source(df: pl.DataFrame) -> str:
    if df.is_empty():
        raise SystemExit(
            "No ticks left after filtering. Check --dates and --sessions against "
            "what was actually collected; session labels are "
            "overlap / other / rollover / weekend, and 'overlap' means "
            "08:00-17:00 ET on a weekday.")
    srcs = sorted(df["source"].unique().to_list())
    if len(srcs) > 1:
        raise SystemExit(
            "Tick log mixes sources " + str(srcs) + ". Synthetic and live data "
            "must never be pooled -- move one aside and re-run.")
    return srcs[0]


def replay(cfg: dict, df: pl.DataFrame, a: str, b: str, c: str) -> pl.DataFrame:
    """Returns one row per PRICE tick, with the book state after that tick.

    The book is RESET at every run boundary. A new collector process starts with
    an empty TriangleBook, so a faithful replay must too -- otherwise the first
    ticks of a run are aged against quotes the live engine never held. On
    2026-09-01 a 3.4 h outage split the day into two runs and the naive
    single-book replay reported quote ages of 12,314 seconds, aging the 07:35
    run against the 04:08 one.

    Those observations fail the tau filter anyway, so the synchronized sample was
    never contaminated -- but the age and inter-arrival statistics were wrong,
    and "the replay sees exactly what the engine saw" has to be true literally,
    not approximately.
    """
    book = TriangleBook(cfg, a, b, c)
    run_ids = df["run_id"].to_list()
    current_run = run_ids[0] if run_ids else None

    inst = df["instrument"].to_list()
    bid = df["bid"].to_numpy()
    ask = df["ask"].to_numpy()
    ots = df["oanda_ts"].to_list()
    rmono = df["recv_mono"].to_numpy()
    rwall = df["recv_wall"].to_list()
    session = df["session"].to_list()
    eflag = df["event_flag"].to_list()

    n = len(inst)
    cols = {k: np.full(n, np.nan) for k in SNAP_FIELDS}
    sync = np.zeros(n, dtype=bool)
    majors_led = np.zeros(n, dtype=bool)
    last_upd = [""] * n
    c_mid = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        if run_ids[i] != current_run:
            book = TriangleBook(cfg, a, b, c)      # new process, empty book
            current_run = run_ids[i]
        book.update(inst[i], Quote(float(bid[i]), float(ask[i]), ots[i],
                                   float(rmono[i]), rwall[i]))
        snap = book.snapshot(float(rmono[i]), rwall[i], update_moments=True)
        if snap is None:
            continue
        valid[i] = True
        d = snap.as_dict()
        for k in SNAP_FIELDS:
            cols[k][i] = d[k]
        sync[i] = snap.synchronized
        majors_led[i] = snap.majors_led
        last_upd[i] = snap.last_updated
        c_mid[i] = 0.5 * (snap.c_bid + snap.c_ask)

    out = pl.DataFrame({
        # Position in the full snapshot sequence. Downstream code filters to the
        # synchronized subset, where "the next row" is not necessarily "the next
        # tick" -- an event whose rows are 1 apart is contiguous, one whose rows
        # are 40 apart had the book go stale in between.
        "snap_idx": np.arange(n, dtype=np.int64),
        "run_id": run_ids,
        # Use THIS for any time difference between snapshots, never ts_mono.
        "t_axis": run_time_axis(run_ids, rmono) if n else np.zeros(0),
        "recv_wall": rwall,
        "tick_instrument": inst,
        "session": session,
        "event_flag": eflag,
        "synchronized": sync,
        "majors_led": majors_led,
        "last_updated": last_upd,
        "c_mid": c_mid,
        **{k: cols[k] for k in SNAP_FIELDS},
    })
    return out.filter(pl.Series(valid))


# Gap inserted between consecutive runs on the analysis time axis. Larger than
# any horizon, window or duration any analysis uses, so nothing measured on
# t_axis can span two processes -- and every such check already rejects a pair
# whose realized separation is far beyond the horizon asked for.
RUN_GAP_S = 1.0e6


def run_time_axis(run_ids: list, mono: np.ndarray) -> np.ndarray:
    """Monotonic within each run, runs laid end to end RUN_GAP_S apart.

    Raw recv_mono is only comparable inside one process: it restarts at reboot
    and runs can overlap. Differences on this axis equal monotonic differences
    within a run and are >= RUN_GAP_S across runs. Input must already be in
    replay order (runs contiguous).
    """
    mono = np.asarray(mono, dtype=float)
    out = np.empty_like(mono)
    base, start = 0.0, 0
    for k in range(1, len(mono) + 1):
        if k == len(mono) or run_ids[k] != run_ids[start]:
            seg = mono[start:k]
            out[start:k] = seg - seg[0] + base
            base = out[k - 1] + RUN_GAP_S
            start = k
    return out


def pip_scale(df: pl.DataFrame) -> float:
    """Log-units per EUR/GBP pip at the sample's typical cross level."""
    c = float(np.nanmedian(df["c_mid"].to_numpy()))
    return PIP / c
