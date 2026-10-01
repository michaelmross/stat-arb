"""Phase 2: the one-leg paper trading arm.

    python phase2_paper.py --dry-run --hours 6 --speed 600   # no account touched
    python phase2_paper.py --hours 8                          # OANDA practice

Two hard preconditions before a single order is sent to an account:

 1. Gate G1 must have PASSED in out/g1_verdict.json. If it did not, the arm can
    only be run with --pipeline-validation, which caps the run at
    --validation-trades orders and stamps every record so the results can never
    be read as evidence about the strategy (spec section 2: "else run 2-3 days
    purely as pipeline validation").

 2. The parameters must be frozen. The config hash is written to
    out/param_freeze.json on first use; changing k_in, k_out, T, tau or the
    session windows afterwards is refused unless --declare-revision "reason" is
    passed, which appends the declaration to the ledger BEFORE the run and is
    allowed exactly once (spec section 2: "Freeze parameters for week 1; one
    revision allowed for week 2, declared in the log before use").

Kill switches: max trades/day, three consecutive reconnect failures, and any
observed position count outside {0, 1}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fxlib.book import PIP, Quote, TriangleBook
from fxlib.config import (apply_account_override, instruments,
                          load_config, load_credentials)
from fxlib.execution import LiveExecutor, SimExecutor
from fxlib.oanda import OandaClient, StreamStalled, price_to_quote_fields
from fxlib.sessions import EventCalendar, SessionClock
from fxlib.storage import JsonlLog, TickWriter, new_run_id, write_manifest
from fxlib.synthetic import SyntheticFeed

HERE = Path(__file__).resolve().parent
_STOP = False

# Only these fields are scientifically frozen. Storage paths and log verbosity
# can change freely without invalidating the pre-registration.
FROZEN_FIELDS = [("signal", "tau_ms"), ("signal", "k_in"), ("signal", "k_out"),
                 ("signal", "timeout_s"), ("signal", "ewma_halflife_s"),
                 ("signal", "demean"), ("signal", "majors_led_filter"),
                 ("signal", "majors_led_window_ms"), ("signal", "sigma_warmup_n"),
                 ("execution", "units"), ("sessions", "rollover_halt_start"),
                 ("sessions", "rollover_halt_end")]


def _handle_sigint(signum, frame):
    global _STOP
    _STOP = True
    print("\n[phase2] stop requested; flattening and flushing...", flush=True)


def frozen_params(cfg: dict) -> dict:
    return {f"{s}.{k}": cfg[s][k] for s, k in FROZEN_FIELDS}


def param_hash(cfg: dict) -> str:
    import hashlib
    blob = json.dumps(frozen_params(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def check_freeze(outdir: Path, cfg: dict, declare: str | None) -> dict:
    """Enforce the parameter freeze. Returns the ledger entry in force."""
    path = outdir / "param_freeze.json"
    ph = param_hash(cfg)
    entry = {"param_hash": ph, "params": frozen_params(cfg),
             "declared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "reason": declare or "initial freeze"}
    if not path.exists():
        path.write_text(json.dumps({"entries": [entry]}, indent=2), encoding="utf-8")
        print(f"[phase2] parameters frozen at {ph}", flush=True)
        return entry
    ledger = json.loads(path.read_text(encoding="utf-8"))
    entries = ledger["entries"]
    known = {e["param_hash"] for e in entries}
    if ph in known:
        return next(e for e in entries if e["param_hash"] == ph)
    if not declare:
        current = entries[-1]
        diffs = {k: (current["params"].get(k), v)
                 for k, v in frozen_params(cfg).items()
                 if current["params"].get(k) != v}
        raise SystemExit(
            "Parameters changed since the freeze but no revision was declared.\n"
            f"  frozen  {current['param_hash']}  ({current['declared_utc']})\n"
            f"  now     {ph}\n"
            f"  changed {json.dumps(diffs, indent=2)}\n"
            "Re-run with --declare-revision \"why\" to append the declaration "
            "to out/param_freeze.json before it is used, or revert config.toml.")
    if len(entries) >= 2:
        raise SystemExit(
            f"out/param_freeze.json already records {len(entries)} parameter "
            "sets. The protocol allows one revision (week 2). A further change "
            "means this is a new experiment: archive out/param_freeze.json "
            "under a new name and start a fresh pre-registration.")
    entries.append(entry)
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"[phase2] revision declared: {declare} (hash {ph})", flush=True)
    return entry


def check_gate(outdir: Path, allow_validation: bool) -> dict:
    path = outdir / "g1_verdict.json"
    if not path.exists():
        raise SystemExit(
            f"No {path}. Run phase1_analyze.py first -- Phase 2 may not trade "
            "before Gate G1 has been evaluated.")
    g = json.loads(path.read_text(encoding="utf-8"))
    if g.get("verdict") == "PASS":
        return g
    if not allow_validation:
        raise SystemExit(
            f"Gate G1 verdict is {g.get('verdict')}, not PASS.\n"
            f"  {g.get('action')}\n"
            "To exercise the plumbing anyway, re-run with --pipeline-validation; "
            "results will be stamped as validation-only and are not evidence "
            "about the strategy.")
    return g


def pnl_pips(side: int, entry: float, exit_: float) -> float:
    """side=+1 long EUR/GBP, -1 short. Returns pips earned."""
    return side * (exit_ - entry) / PIP


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic feed + local fill simulator; no account is touched")
    ap.add_argument("--mode", choices=["derived", "independent"], default="independent",
                    help="dry-run only: which section 1.3 world to simulate")
    ap.add_argument("--sigma-pips", type=float, default=0.6, help="dry-run only")
    ap.add_argument("--speed", type=float, default=1.0, help="dry-run only")
    ap.add_argument("--seed", type=int, default=0, help="dry-run only")
    ap.add_argument("--sim-start", default=None, metavar="ISO8601",
                    help="dry-run only: pin the synthetic clock (e.g. "
                         "2026-09-02T14:00:00+00:00). Without it the feed starts "
                         "at 'now', so a dry run during the rollover halt or at "
                         "a weekend correctly refuses to trade and records "
                         "nothing -- deterministic tests must pin this.")
    ap.add_argument("--sim-latency-pips", type=float, default=0.1,
                    help="dry-run only: adverse fill offset, a placeholder not a model")
    ap.add_argument("--pipeline-validation", action="store_true",
                    help="run despite a failed/void G1, capped and stamped")
    ap.add_argument("--validation-trades", type=int, default=20)
    ap.add_argument("--declare-revision", default=None,
                    help="declare the one permitted parameter revision")
    ap.add_argument("--log-ticks", action="store_true",
                    help="also write the tick log during the trading run")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_sigint)
    cfg = load_config(args.config)
    if args.data_root:
        cfg["storage"]["root"] = args.data_root
    a, b, c = instruments(cfg)
    outdir = Path(args.out) if args.out else HERE / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["_path"]).parent / cfg["storage"]["root"]

    gate = check_gate(outdir, args.pipeline_validation)
    freeze = check_freeze(outdir, cfg, args.declare_revision)
    validation_only = gate.get("verdict") != "PASS"

    sig = cfg["signal"]
    k_in, k_out = float(sig["k_in"]), float(sig["k_out"])
    timeout_s = float(sig["timeout_s"])
    units = int(cfg["execution"]["units"])
    max_trades = int(cfg["execution"]["max_trades_per_day"])
    if validation_only:
        max_trades = min(max_trades, args.validation_trades)

    sug = (gate.get("suggested_parameters") or {})
    if sug and abs(float(sug.get("k_in", k_in)) - k_in) > 1e-9:
        print(f"[phase2] NOTE: config k_in={k_in} differs from the Phase 1 "
              f"suggestion {sug['k_in']:.3f}. The config value is what runs; the "
              f"difference is recorded in the manifest.", flush=True)

    run_id = new_run_id("phase2" + ("-dry" if args.dry_run else ""))
    trades_log = JsonlLog(root, "trades")
    events = JsonlLog(root, "events")
    clock = SessionClock(cfg)
    calendar = EventCalendar(cfg)
    book = TriangleBook(cfg, a, b, c)
    writer = TickWriter(cfg, run_id, "synthetic" if args.dry_run else "oanda") \
        if args.log_ticks else None

    client = executor = feed = None
    if args.dry_run:
        start_wall = (dt.datetime.fromisoformat(args.sim_start)
                      if args.sim_start else None)
        feed = SyntheticFeed(cfg, mode=args.mode, seed=args.seed,
                             sigma_pips=args.sigma_pips, speed=args.speed,
                             start_wall=start_wall)
        executor = SimExecutor(book, c, latency_pips=args.sim_latency_pips)
    else:
        client = OandaClient(cfg, apply_account_override(cfg, load_credentials()))
        acct = client.summary()
        executor = LiveExecutor(client, c)
        standing = executor.open_units()
        if standing != 0:
            raise SystemExit(
                f"Account already holds {standing} units of {c}. The arm requires "
                "a flat book at start (kill switch: position count must be in "
                "{0,1} and owned by this run). Close it manually first.")
        print(f"[phase2] account {acct.get('id')} balance {acct.get('balance')} "
              f"{acct.get('currency')} -- flat in {c}", flush=True)

    write_manifest(root, run_id, cfg, {
        "kind": "phase2", "dry_run": args.dry_run,
        "validation_only": validation_only, "argv": sys.argv[1:],
        "param_hash": freeze["param_hash"], "frozen_params": freeze["params"],
        "gate_g1": {k: gate.get(k) for k in ("verdict", "generated_utc", "report")},
        "phase1_suggested": sug,
        "effective": {"k_in": k_in, "k_out": k_out, "timeout_s": timeout_s,
                      "units": units, "max_trades": max_trades},
    })
    banner = "DRY RUN (synthetic feed, simulated fills)" if args.dry_run \
        else "OANDA PRACTICE ACCOUNT"
    print(f"[phase2] run_id={run_id} {banner}", flush=True)
    print(f"[phase2] k_in={k_in} k_out={k_out} T={timeout_s}s units={units} "
          f"max_trades={max_trades} param_hash={freeze['param_hash']}", flush=True)
    if validation_only:
        print("[phase2] PIPELINE VALIDATION ONLY -- G1 verdict is "
              f"{gate.get('verdict')}; these trades are not evidence.", flush=True)

    duration_s = (args.hours * 3600 if args.hours
                  else args.minutes * 60 if args.minutes else None)
    wall_limited = not args.dry_run
    started = time.monotonic()
    hb_timeout = float(cfg["stream"]["heartbeat_timeout_s"])
    backoff = list(cfg["stream"]["reconnect_backoff_s"])
    max_fail = int(cfg["stream"]["max_consecutive_reconnect_failures"])

    position = None          # dict describing the open trade, or None
    n_trades = 0
    trade_day = None
    consecutive_failures = 0
    halt_reason = None
    n_ticks = 0
    cap_logged = False
    # The last clocks the book actually saw. Under --speed the synthetic feed
    # runs ahead of time.monotonic(), so the shutdown path must reuse these
    # rather than reading the wall clock and computing a negative holding time.
    last_mono = time.monotonic()
    last_wall = dt.datetime.now(dt.timezone.utc)

    def flatten(snap, reason):
        nonlocal position
        if position is None:
            return
        side = position["side"]
        try:
            fill = executor.market(-side * units, run_id + "-x")
        except Exception as exc:                            # noqa: BLE001
            events.write({"run_id": run_id, "event": "exit_order_failed",
                          "detail": str(exc)})
            print(f"[phase2] EXIT ORDER FAILED: {exc}", flush=True)
            return
        quoted_exit = snap.c_bid if side > 0 else snap.c_ask
        mid_exit = 0.5 * (snap.c_bid + snap.c_ask)
        rec = {
            "run_id": run_id, "kind": "trade", "validation_only": validation_only,
            "param_hash": freeze["param_hash"], "simulated_fills": executor.simulated,
            "side": "long" if side > 0 else "short", "units": units,
            "instrument": c,
            "entry": position["entry"],
            "exit": {
                "reason": reason,
                "signal_utc": snap.ts_wall.isoformat(),
                "z": snap.z, "sigma_pips": snap.sigma_pips,
                "quoted_price": quoted_exit, "mid": mid_exit,
                "fill_price": fill.price, "fill_utc": fill.time.isoformat(),
                "trade_id": fill.trade_id,
                "slippage_pips": side * (quoted_exit - fill.price) / PIP,
                "max_age_ms": snap.max_age_ms,
            },
            "holding_s": snap.ts_mono - position["ts_mono"],
            "pnl_pips": {
                "gross_mid_to_mid": pnl_pips(side, position["entry"]["mid"], mid_exit),
                "net_at_quoted_spread": pnl_pips(
                    side, position["entry"]["quoted_price"], quoted_exit),
                "net_with_measured_fills": pnl_pips(
                    side, position["entry"]["fill_price"], fill.price),
            },
            "cost_model_caveat":
                "practice fills are optimistic: no queue, no market impact, no "
                "last-look rejection. net_with_measured_fills is a lower bound on "
                "cost, not a production estimate.",
        }
        rec["pnl_ccy_quote"] = {k: v * PIP * units
                                for k, v in rec["pnl_pips"].items()}
        trades_log.write(rec)
        p = rec["pnl_pips"]
        print(f"[phase2] EXIT {rec['side']} {reason} "
              f"gross {p['gross_mid_to_mid']:+.2f} / quoted "
              f"{p['net_at_quoted_spread']:+.2f} / filled "
              f"{p['net_with_measured_fills']:+.2f} pips "
              f"({rec['holding_s']:.1f}s)", flush=True)
        position = None

    try:
        while not _STOP and halt_reason is None:
            if wall_limited and duration_s and time.monotonic() - started >= duration_s:
                break
            ended_cleanly = False
            try:
                if args.dry_run:
                    stream = feed.messages(duration_s if duration_s else 3600.0)
                else:
                    stream = client.stream_prices([a, b, c], hb_timeout)
                events.write({"run_id": run_id, "event": "connected",
                              "phase": "2"})
                consecutive_failures = 0

                for kind, msg in stream:
                    if _STOP or halt_reason:
                        break
                    if kind != "PRICE":
                        continue
                    fields = price_to_quote_fields(msg)
                    if fields is None:
                        continue
                    bid, ask, ots, tradeable = fields
                    now_mono, now_wall = msg["_recv_mono"], msg["_recv_wall"]
                    last_mono, last_wall = now_mono, now_wall
                    book.update(msg["instrument"],
                                Quote(bid, ask, ots, now_mono, now_wall, tradeable))
                    n_ticks += 1
                    if writer:
                        writer.append({
                            "instrument": msg["instrument"], "bid": bid, "ask": ask,
                            "oanda_ts": ots, "recv_wall": now_wall,
                            "recv_mono": now_mono, "tradeable": tradeable,
                            "session": clock.session_label(now_wall),
                            "event_flag": calendar.flag(now_wall)})

                    snap = book.snapshot(now_mono, now_wall, update_moments=True)
                    if snap is None:
                        continue

                    day = now_wall.strftime("%Y-%m-%d")
                    if day != trade_day:
                        trade_day, n_trades, cap_logged = day, 0, False

                    # ---- kill switch: position count must be in {0,1} -----
                    if not executor.simulated and n_ticks % 500 == 0:
                        held = executor.open_units()
                        expected = 0 if position is None else \
                            position["side"] * units
                        if held != expected:
                            halt_reason = "kill_switch_position_mismatch"
                            events.write({"run_id": run_id, "event": "kill_switch",
                                          "which": "position_mismatch",
                                          "held": held, "expected": expected})
                            break

                    # ---- exits --------------------------------------------
                    if position is not None:
                        held_s = snap.ts_mono - position["ts_mono"]
                        # Directional, not |z| <= k_out. With the spec's k_out = 0
                        # an absolute test can never fire (z is continuous), and
                        # it would also refuse to exit on an overshoot -- z going
                        # +3 -> -2 has fully reverted but still has |z| > k_out.
                        reverted = (snap.z <= k_out if position["side"] < 0
                                    else snap.z >= -k_out)
                        if held_s >= timeout_s:
                            flatten(snap, "timeout")
                        elif snap.synchronized and reverted:
                            flatten(snap, "signal_exit")
                        elif not clock.tradeable(now_wall):
                            flatten(snap, "session_halt")
                        continue

                    # ---- entries ------------------------------------------
                    if not clock.tradeable(now_wall):
                        continue
                    if not snap.synchronized or not tradeable:
                        continue
                    if snap.n_obs < int(sig["sigma_warmup_n"]) or snap.sigma <= 0:
                        continue
                    if not (abs(snap.z) >= k_in):
                        continue
                    if sig["majors_led_filter"] and not snap.majors_led:
                        continue
                    if n_trades >= max_trades:
                        if not cap_logged:
                            cap_logged = True
                            events.write({"run_id": run_id, "event": "kill_switch",
                                          "which": "max_trades_per_day",
                                          "count": n_trades})
                            print("[phase2] daily trade cap reached; no further "
                                  "entries today", flush=True)
                        continue

                    # z > 0: quoted cross is rich vs implied -> SELL C.
                    side = -1 if snap.z > 0 else 1
                    quoted_entry = snap.c_ask if side > 0 else snap.c_bid
                    decision = {
                        "signal_utc": snap.ts_wall.isoformat(),
                        "z": snap.z, "z_raw": snap.z_raw,
                        "sigma_pips": snap.sigma_pips,
                        "resid_pips": snap.resid_pips,
                        "quoted_price": quoted_entry,
                        "mid": 0.5 * (snap.c_bid + snap.c_ask),
                        "c_spread_pips": snap.c_spread_pips,
                        "max_age_ms": snap.max_age_ms,
                        "last_updated": snap.last_updated,
                        "majors_led": snap.majors_led,
                        "session": clock.session_label(now_wall),
                        "event_flag": calendar.flag(now_wall),
                    }
                    # The decision is recorded before the order call, so a crash
                    # between the two leaves evidence rather than a mystery.
                    events.write({"run_id": run_id, "event": "entry_decision",
                                  **decision, "side": side})
                    try:
                        fill = executor.market(side * units, run_id + "-e")
                    except Exception as exc:                # noqa: BLE001
                        events.write({"run_id": run_id, "event": "entry_order_failed",
                                      "detail": str(exc)})
                        print(f"[phase2] entry order failed: {exc}", flush=True)
                        continue
                    decision["fill_price"] = fill.price
                    decision["fill_utc"] = fill.time.isoformat()
                    decision["trade_id"] = fill.trade_id
                    decision["slippage_pips"] = side * (fill.price - quoted_entry) / PIP
                    position = {"side": side, "ts_mono": snap.ts_mono,
                                "entry": decision}
                    n_trades += 1
                    print(f"[phase2] ENTRY {'long' if side > 0 else 'short'} {c} "
                          f"z={snap.z:+.2f} @ {fill.price:.5f} "
                          f"(slip {decision['slippage_pips']:+.2f} pip) "
                          f"[{n_trades}/{max_trades}]", flush=True)

                    if wall_limited and duration_s and \
                            time.monotonic() - started >= duration_s:
                        break
                ended_cleanly = True

            except StreamStalled as exc:
                consecutive_failures += 1
                events.write({"run_id": run_id, "event": "stalled",
                              "detail": str(exc), "consecutive": consecutive_failures})
            except Exception as exc:                        # noqa: BLE001
                consecutive_failures += 1
                events.write({"run_id": run_id, "event": "stream_error",
                              "detail": f"{type(exc).__name__}: {exc}",
                              "consecutive": consecutive_failures})
                print(f"[phase2] stream error: {type(exc).__name__}: {exc}",
                      flush=True)

            if _STOP or halt_reason:
                break
            if ended_cleanly and (args.dry_run or duration_s):
                break
            if ended_cleanly:
                consecutive_failures += 1
            if consecutive_failures >= max_fail:
                halt_reason = "kill_switch_reconnect_failures"
                events.write({"run_id": run_id, "event": "kill_switch",
                              "which": "reconnect_failures",
                              "count": consecutive_failures})
                break
            wait = backoff[min(consecutive_failures - 1, len(backoff) - 1)]
            print(f"[phase2] reconnecting in {wait}s "
                  f"({consecutive_failures}/{max_fail})", flush=True)
            time.sleep(wait)
    finally:
        if position is not None:
            snap = book.snapshot(last_mono, last_wall, update_moments=False)
            if snap is not None:
                flatten(snap, halt_reason or "shutdown")
            else:
                events.write({"run_id": run_id, "event": "position_left_open",
                              "detail": "no book snapshot available at shutdown"})
                print("[phase2] WARNING: position may still be open -- check the "
                      "account", flush=True)
        if writer:
            writer.close()
        if client:
            client.close()
        events.write({"run_id": run_id, "event": "run_end",
                      "reason": halt_reason or ("sigint" if _STOP else "completed"),
                      "ticks": n_ticks, "trades": n_trades})
        print(f"[phase2] done: {n_ticks} ticks, {n_trades} entries, "
              f"reason={halt_reason or ('sigint' if _STOP else 'completed')}",
              flush=True)
    return 1 if halt_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
