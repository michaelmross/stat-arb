"""Phase 0/1 collector: stream three pairs, log every tick, never hide a gap.

Collection does NOT stop at the rollover or weekend halt -- those halts govern
TRADING (spec section 3). The 5pm ET spread blowout is itself a measurement, so
it is recorded and tagged (session="rollover") for the analysis to condition on.

The residual is not stored. phase1_analyze.py replays this tick log through the
same fxlib.book.TriangleBook the live engine uses, so the shadow measurement and
the live signal cannot drift apart, and every number in the report is
reproducible from the raw log.

Usage
    python collect_ticks.py --source oanda --hours 8
    python collect_ticks.py --source synthetic --mode derived --minutes 20 --speed 60
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fxlib.book import Quote, TriangleBook
from fxlib.clocks import describe as describe_offset, measure_offset
from fxlib.config import (apply_account_override, instruments,
                          load_config, load_credentials)
from fxlib.oanda import OandaClient, StreamStalled, price_to_quote_fields
from fxlib.sessions import EventCalendar, SessionClock
from fxlib.storage import (JsonlLog, SingleWriterLock, TickWriter,
                           new_run_id, write_manifest)
from fxlib.synthetic import SyntheticFeed

_STOP = False


def raise_priority() -> str:
    """Nudge this process above ordinary background work.

    Done in-process rather than via cmd's `start /ABOVENORMAL`, because that
    launches a NEW console: the parent's `>> collector.log` then redirects only
    `start` itself (so the collector's output vanishes), and the child gets a
    visible window that kills the run if anyone closes it. Both happened on
    2026-09-01 and cost 3.4 hours.

    Best-effort: a failure here is never worth losing a session over.
    """
    if os.name != "nt":
        return "unchanged (not Windows)"
    try:
        import ctypes
        from ctypes import wintypes
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # restype MUST be set: a HANDLE is pointer-sized, and ctypes defaults to
        # c_int, which truncates the GetCurrentProcess pseudo-handle on 64-bit
        # and makes SetPriorityClass fail with ERROR_INVALID_HANDLE.
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.argtypes = []
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        if k32.SetPriorityClass(k32.GetCurrentProcess(),
                                ABOVE_NORMAL_PRIORITY_CLASS):
            return "AboveNormal"
        return "unchanged (SetPriorityClass failed, err=%d)" % ctypes.get_last_error()
    except Exception as exc:                                  # noqa: BLE001
        return "unchanged (%s)" % type(exc).__name__


def _handle_sigint(signum, frame):
    global _STOP
    _STOP = True
    print("\n[collector] stop requested; flushing...", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["oanda", "synthetic"], default="oanda")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None,
                    help="override [storage].root; does not affect the config hash")
    ap.add_argument("--hours", type=float, default=None,
                    help="duration of MARKET time to collect")
    ap.add_argument("--minutes", type=float, default=None,
                    help="duration of MARKET time to collect")
    ap.add_argument("--until", default=None, metavar="HH:MM",
                    help="collect until this local exchange time today, rather "
                         "than for a fixed duration. Use this whenever the run "
                         "may be auto-restarted: a restart with --hours would "
                         "start the full duration again and overrun the session.")
    ap.add_argument("--mode", choices=["derived", "independent"], default="derived",
                    help="synthetic only: is the cross internally derived (spec 1.3)?")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="synthetic only: virtual-time multiplier")
    ap.add_argument("--sigma-pips", type=float, default=0.25,
                    help="synthetic independent mode: stationary residual sigma")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-clock-check", action="store_true",
                    help="skip the NTP offset measurement at start-up")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGINT, _handle_sigint)

    cfg = load_config(args.config)
    if args.data_root:
        cfg["storage"]["root"] = args.data_root
    a, b, c = instruments(cfg)
    duration_s = (args.hours * 3600 if args.hours
                  else args.minutes * 60 if args.minutes else None)
    if args.until:
        # An absolute end time is restart-safe: however many times Task
        # Scheduler revives this process during the day, they all stop at the
        # same wall-clock moment instead of each running a fresh full duration.
        tz = ZoneInfo(cfg["sessions"]["timezone"])
        now_local = dt.datetime.now(tz)
        hh, mm = (int(x) for x in args.until.split(":"))
        target = now_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
        duration_s = (target - now_local).total_seconds()
        if duration_s <= 0:
            print("[collector] --until %s has already passed (local %s); "
                  "nothing to collect" % (args.until, now_local.strftime("%H:%M")),
                  flush=True)
            return 0
        print("[collector] collecting until %s local (%.2f h from now)"
              % (args.until, duration_s / 3600.0), flush=True)

    run_id = new_run_id("collect-" + args.source)
    root = Path(cfg["_path"]).parent / cfg["storage"]["root"]
    root.mkdir(parents=True, exist_ok=True)
    lock = SingleWriterLock(root)
    lock.acquire(run_id)
    writer = TickWriter(cfg, run_id, args.source)
    events = JsonlLog(root, "events")
    clock = SessionClock(cfg)
    calendar = EventCalendar(cfg)
    book = TriangleBook(cfg, a, b, c)

    # Spec 0.3 assumes an NTP-synced wall clock. Verify it rather than
    # assume it: an uncorrected offset lands entirely in the reported
    # latency. Never used for signal logic, only for the diagnostic.
    clock_offset = (measure_offset() if not args.no_clock_check
                    else {"ok": False, "offset_s": None,
                          "note": "skipped via --no-clock-check"})
    manifest_extra = {"kind": "collector", "source": args.source,
                      "argv": sys.argv[1:], "duration_s": duration_s,
                      "clock_offset": clock_offset}
    if args.source == "synthetic":
        manifest_extra["synthetic"] = {"mode": args.mode, "speed": args.speed,
                                       "sigma_pips": args.sigma_pips,
                                       "seed": args.seed}
    write_manifest(root, run_id, cfg, manifest_extra)
    events.write({"run_id": run_id, "event": "run_start", "source": args.source,
                  "config_hash": cfg["_hash"], "clock_offset": clock_offset})
    print("[collector] run_id=%s config_hash=%s source=%s"
          % (run_id, cfg["_hash"], args.source), flush=True)
    if args.source == "synthetic":
        print("[collector] SYNTHETIC DATA -- pipeline validation only, "
              "not a measurement", flush=True)
    print("[collector] " + describe_offset(clock_offset), flush=True)
    print("[collector] process priority: " + raise_priority(), flush=True)

    # Last-resort deadline guard. On 2026-09-04 a DNS outage left the collector
    # alive but doing nothing for 40 minutes past its end time; Task Scheduler
    # logged the instance as terminated but the python process survived, kept
    # the SingleWriterLock, and would have blocked the next session entirely.
    # Under S4U it sits in session 0, so it cannot even be killed without
    # elevation. Nothing outside the process can be relied on to end it, so it
    # must be able to end itself.
    if duration_s is not None:
        grace_s = 600.0

        def _deadline_guard():
            time.sleep(duration_s + grace_s)
            try:
                events.write({"run_id": run_id, "event": "run_end",
                              "reason": "deadline_guard_hard_exit",
                              "ticks": n_price, "gaps": n_gap,
                              "detail": "still alive %.0f min past the deadline; "
                                        "forcing exit so the lock is released"
                                        % (grace_s / 60.0)})
                writer.flush()
            finally:
                os._exit(3)

        threading.Thread(target=_deadline_guard, daemon=True).start()

    hb_timeout = float(cfg["stream"]["heartbeat_timeout_s"])
    backoff = list(cfg["stream"]["reconnect_backoff_s"])
    max_fail = int(cfg["stream"].get("collector_max_reconnect_failures",
                   cfg["stream"]["max_consecutive_reconnect_failures"]))

    client = None
    if args.source == "oanda":
        client = OandaClient(cfg, apply_account_override(cfg, load_credentials()))
        # The account summary is informational -- it prints the balance and
        # confirms the credentials. Collection does not need it, so it must
        # never be able to kill a session on its own.
        #
        # It did exactly that on 2026-08-31: a transient 401 twelve seconds into
        # the 03:00 run aborted the whole day, while the stream loop below would
        # have retried it with backoff. Both endpoints were fine minutes later.
        # Retry here, then carry on regardless: if the stream is genuinely
        # unreachable, the reconnect logic and the kill switch handle it with
        # the care this check was never given.
        acct = None
        for attempt in range(1, 6):
            try:
                acct = client.summary()
                break
            except Exception as exc:                      # noqa: BLE001
                wait = backoff[min(attempt - 1, len(backoff) - 1)]
                events.write({"run_id": run_id, "event": "account_check_failed",
                              "attempt": attempt,
                              "detail": "%s: %s" % (type(exc).__name__, exc),
                              "retry_in_s": wait})
                print("[collector] account check failed (%d/5): %s -- retrying "
                      "in %ss" % (attempt, exc, wait), flush=True)
                time.sleep(wait)
        if acct is not None:
            print("[collector] account %s balance %s %s"
                  % (acct.get("id"), acct.get("balance"), acct.get("currency")),
                  flush=True)
            events.write({"run_id": run_id, "event": "account",
                          "id": acct.get("id"), "currency": acct.get("currency")})
        else:
            print("[collector] account check failed 5 times -- proceeding to the "
                  "stream anyway; a real outage will surface there", flush=True)
            events.write({"run_id": run_id, "event": "account_check_abandoned"})

    # For a live stream, market time and wall time are the same thing. For the
    # synthetic feed they are not: --speed compresses wall time, and the feed
    # terminates itself after duration_s of MARKET time.
    wall_limited = args.source == "oanda"
    started = time.monotonic()
    consecutive_failures = 0
    n_price = n_hb = n_gap = 0
    last_msg_mono = time.monotonic()
    last_report = time.monotonic()
    exit_reason = "completed"

    try:
        while not _STOP:
            if wall_limited and duration_s is not None and \
                    time.monotonic() - started >= duration_s:
                break
            ended_cleanly = False
            try:
                if args.source == "synthetic":
                    feed = SyntheticFeed(cfg, mode=args.mode, seed=args.seed,
                                         sigma_pips=args.sigma_pips,
                                         speed=args.speed)
                    stream = feed.messages(duration_s if duration_s else 3600.0)
                else:
                    stream = client.stream_prices([a, b, c], hb_timeout)

                events.write({"run_id": run_id, "event": "connected"})
                consecutive_failures = 0

                for kind, msg in stream:
                    if _STOP:
                        break
                    now_mono = msg["_recv_mono"]
                    now_wall = msg["_recv_wall"]

                    # A silent hole in the stream is the one thing that must never
                    # happen; flag any inter-message gap beyond the heartbeat.
                    delta = now_mono - last_msg_mono
                    if delta > hb_timeout:
                        n_gap += 1
                        events.write({"run_id": run_id, "event": "gap",
                                      "gap_s": round(delta, 3),
                                      "detected_at": now_wall.isoformat()})
                    last_msg_mono = now_mono

                    # Check the deadline BEFORE the heartbeat filter. During the
                    # rollover vacuum no PRICE ticks arrive for minutes while
                    # heartbeats keep the loop inside this `for`, so a check
                    # placed after the filter cannot fire and the run overshoots
                    # its end time (17:04:56 on 2026-09-02).
                    if wall_limited and duration_s is not None and                             time.monotonic() - started >= duration_s:
                        break
                    if kind == "HEARTBEAT":
                        n_hb += 1
                        continue
                    if kind != "PRICE":
                        events.write({"run_id": run_id, "event": "stream_message",
                                      "type": kind, "raw": str(msg)[:300]})
                        continue

                    fields = price_to_quote_fields(msg)
                    if fields is None:
                        events.write({"run_id": run_id, "event": "one_sided_book",
                                      "instrument": msg.get("instrument")})
                        continue
                    bid, ask, oanda_ts, tradeable = fields
                    inst = msg["instrument"]

                    writer.append({
                        "instrument": inst, "bid": bid, "ask": ask,
                        "oanda_ts": oanda_ts, "recv_wall": now_wall,
                        "recv_mono": now_mono, "tradeable": tradeable,
                        "session": clock.session_label(now_wall),
                        "event_flag": calendar.flag(now_wall),
                    })
                    book.update(inst, Quote(bid, ask, oanda_ts, now_mono,
                                            now_wall, tradeable))
                    n_price += 1

                    if not args.quiet and time.monotonic() - last_report > 10.0:
                        last_report = time.monotonic()
                        snap = book.snapshot(now_mono, now_wall,
                                             update_moments=False)
                        extra = ""
                        if snap is not None:
                            extra = (" resid=%+.3fpip maxage=%.0fms sync=%s cost=%.2fbp"
                                     % (snap.resid_pips, snap.max_age_ms,
                                        snap.synchronized, snap.cycle_cost_bp))
                        print("[collector] %d ticks %d hb %d gaps%s"
                              % (n_price, n_hb, n_gap, extra), flush=True)

                    if wall_limited and duration_s is not None and \
                            time.monotonic() - started >= duration_s:
                        break
                ended_cleanly = True

            except StreamStalled as exc:
                consecutive_failures += 1
                events.write({"run_id": run_id, "event": "stalled",
                              "detail": str(exc),
                              "consecutive": consecutive_failures})
                print("[collector] stream stalled: %s" % exc, flush=True)
            except Exception as exc:                      # noqa: BLE001
                consecutive_failures += 1
                events.write({"run_id": run_id, "event": "stream_error",
                              "detail": "%s: %s" % (type(exc).__name__, exc),
                              "consecutive": consecutive_failures})
                print("[collector] stream error: %s: %s"
                      % (type(exc).__name__, exc), flush=True)

            if _STOP:
                break
            if ended_cleanly:
                if duration_s is not None or args.source == "synthetic":
                    break
                # An open-ended live stream that returns is a disconnect.
                consecutive_failures += 1
                events.write({"run_id": run_id, "event": "stream_ended",
                              "consecutive": consecutive_failures})

            if consecutive_failures >= max_fail:
                exit_reason = "kill_switch_reconnect_failures"
                events.write({"run_id": run_id, "event": "kill_switch",
                              "which": "reconnect_failures",
                              "count": consecutive_failures})
                print("[collector] KILL SWITCH: %d consecutive reconnect failures"
                      % consecutive_failures, flush=True)
                break
            wait = backoff[min(consecutive_failures - 1, len(backoff) - 1)]
            events.write({"run_id": run_id, "event": "reconnect_wait",
                          "seconds": wait})
            print("[collector] reconnecting in %ss (failure %d/%d)"
                  % (wait, consecutive_failures, max_fail), flush=True)
            time.sleep(wait)
    finally:
        writer.close()
        lock.release()
        if _STOP:
            exit_reason = "sigint"
        # run_end goes FIRST. It is the only signal distinguishing a clean
        # finish from a kill, and everything after it can block. On 2026-09-04 a
        # DNS outage made the end-of-run NTP read hang; Task Scheduler killed
        # the process 30 min later (event 111) and the run looked KILLED even
        # though it had left its loop normally at 17:00.
        events.write({"run_id": run_id, "event": "run_end", "reason": exit_reason,
                      "ticks": n_price, "heartbeats": n_hb, "gaps": n_gap,
                      "rows_written": writer.rows_written})
        print("[collector] done: %d rows, %d gaps, reason=%s"
              % (writer.rows_written, n_gap, exit_reason), flush=True)
        if client is not None:
            client.close()

        # Best-effort end anchor for the clock reconstruction. Bounded by a
        # thread join: socket timeouts do NOT bound getaddrinfo, so a broken
        # resolver can block indefinitely inside measure_offset.
        clock_offset_end = {"ok": False, "offset_s": None}
        if not args.no_clock_check:
            box = {}

            def _measure():
                try:
                    box["v"] = measure_offset()
                except Exception as exc:                      # noqa: BLE001
                    box["v"] = {"ok": False, "offset_s": None,
                                "note": "%s: %s" % (type(exc).__name__, exc)}

            th = threading.Thread(target=_measure, daemon=True)
            th.start()
            th.join(timeout=15.0)
            clock_offset_end = box.get(
                "v", {"ok": False, "offset_s": None,
                      "note": "timed out after 15s (resolver or network down)"})
        drift_ms = None
        if clock_offset.get("ok") and clock_offset_end.get("ok"):
            drift_ms = (clock_offset_end["offset_s"] - clock_offset["offset_s"]) * 1e3
            print("[collector] clock drift over the run: %+.0f ms" % drift_ms,
                  flush=True)
        events.write({"run_id": run_id, "event": "clock_offset_end",
                      "clock_offset": clock_offset_end, "drift_ms": drift_ms})
        try:
            mpath = root / "runs" / (run_id + ".json")
            m = json.loads(mpath.read_text(encoding="utf-8"))
            m["clock_offset_end"] = clock_offset_end
            m["clock_drift_ms"] = drift_ms
            mpath.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            events.write({"run_id": run_id, "event": "manifest_update_failed",
                          "detail": str(exc)})
    return 0 if exit_reason in ("completed", "sigint") else 1


if __name__ == "__main__":
    raise SystemExit(main())
