"""Measure the local clock's offset from true time, so the latency figure means
something.

Spec section 0.3 says to stamp ticks with an "NTP-synced wall clock". That is an
assumption, and on a Windows desktop it is often wrong by hundreds of
milliseconds -- w32tm's default sync is loose. An uncorrected offset lands
entirely in (local receive - OANDA timestamp) and makes the feed look far slower
than it is. Measured on this machine at first run: the clock was 556 ms fast,
turning a 159 ms median latency into a reported 715 ms.

None of this touches signal logic -- quote ages use the monotonic clock, which
has no relationship to wall time (see fxlib/book.py). This module exists so the
Phase 0 diagnostic and the report can subtract a known error rather than
publishing a number that is mostly desktop clock drift.

Minimal SNTP client, no dependencies: one 48-byte UDP packet per sample.
"""
from __future__ import annotations

import socket
import statistics
import struct
import time

NTP_EPOCH_DELTA = 2_208_988_800     # seconds between 1900-01-01 and 1970-01-01
DEFAULT_SERVERS = ("time.windows.com", "pool.ntp.org", "time.google.com")


def sntp_sample(host: str, timeout: float = 3.0):
    """One SNTP exchange. Returns (offset_s, roundtrip_s).

    offset_s > 0 means the remote clock is AHEAD of this machine, i.e. the local
    clock is slow. Negative means the local clock is fast.

    Corrected latency = raw_latency + offset_s.
    """
    pkt = b"\x1b" + 47 * b"\0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        t1 = time.time()
        sock.sendto(pkt, (host, 123))
        data, _ = sock.recvfrom(1024)
        t4 = time.time()
    finally:
        sock.close()
    if len(data) < 48:
        raise OSError(f"short NTP reply from {host}: {len(data)} bytes")
    f = struct.unpack("!12I", data[:48])
    t2 = f[8] + f[9] / 2 ** 32 - NTP_EPOCH_DELTA     # server receive
    t3 = f[10] + f[11] / 2 ** 32 - NTP_EPOCH_DELTA   # server transmit
    offset = ((t2 - t1) + (t3 - t4)) / 2.0
    roundtrip = (t4 - t1) - (t3 - t2)
    return offset, roundtrip


def measure_offset(servers=DEFAULT_SERVERS, samples: int = 4,
                   timeout: float = 3.0) -> dict:
    """Median offset across several samples and servers.

    Returns a dict that is safe to serialize into a run manifest. Never raises:
    if every server is unreachable the result carries ok=False and the analysis
    reports raw latency with an explicit warning, rather than silently
    subtracting a number it does not have.
    """
    offsets, trips, errors, used = [], [], [], []
    for host in servers:
        got = 0
        for _ in range(samples):
            try:
                off, rtt = sntp_sample(host, timeout)
            except Exception as exc:                    # noqa: BLE001
                errors.append(f"{host}: {type(exc).__name__}: {exc}")
                break
            offsets.append(off)
            trips.append(rtt)
            got += 1
        if got:
            used.append(host)
        if len(offsets) >= samples:
            break
    if not offsets:
        return {"ok": False, "offset_s": None, "errors": errors,
                "note": "no NTP server reachable; latency is reported raw"}
    return {
        "ok": True,
        "offset_s": float(statistics.median(offsets)),
        # Monotonic stamp of the measurement, so error_trace can locate this
        # anchor on the same axis the ticks are recorded against.
        "mono": time.monotonic(),
        "wall_s": time.time(),
        "offset_spread_s": float(max(offsets) - min(offsets)),
        "roundtrip_median_s": float(statistics.median(trips)),
        "n_samples": len(offsets),
        "servers": used,
        "errors": errors,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "convention": "offset_s > 0 means the local clock is SLOW; "
                      "corrected_latency = raw_latency + offset_s",
    }


def error_trace(recv_wall_s, recv_mono, anchors):
    """Reconstruct the local clock error at every tick, in seconds.

    A single scalar offset per run is not enough. Windows Time does not step a
    sub-second error, it SLEWS it: on 2026-08-28 the clock error glided from
    610 ms to 76 ms across one two-hour run, decelerating as it converged.
    Correcting such a run with one number -- even the midpoint of a start and
    end reading -- is wrong everywhere, and produced a physically impossible
    NEGATIVE median latency of -29 ms before this function existed.

    The fix needs no extra measurement, because the tick log already records it.
    Let

        w(t) = recv_wall(t) - recv_mono(t)

    The monotonic clock is never adjusted, so w is constant unless the OS moves
    the wall clock, and w therefore traces every adjustment -- step or slew -- at
    tick resolution. True time also advances with the monotonic clock, so

        E(t) = wall(t) - true(t) = w(t) - K

    for an unknown K, and the NTP readings are only needed to pin K rather than
    to track the shape.

    K is not quite constant, though. The monotonic clock is
    QueryPerformanceCounter, whose crystal has its own frequency error (~1.6 ppm
    here), so true time and monotonic time drift apart slowly: negligible over
    two hours, 80 ms over fourteen. With two or more anchors K is therefore
    fitted as k0 + rate*(t - t_ref), and `monotonic_rate_error_ppm` reports the
    fitted rate. With three or more, `anchor_residual_ms` is the fit quality;
    with exactly two it is zero by construction.

    Args
      recv_wall_s : per-tick wall clock, seconds since the epoch
      recv_mono   : per-tick monotonic clock, seconds
      anchors     : list of {"offset_s": float, "mono": float} -- offset_s in the
                    sntp_sample convention (>0 means the local clock is SLOW).
                    Anchors without a usable "mono" are ignored.

    Returns (E_seconds_array, diagnostics). E > 0 means the local clock is FAST,
    so corrected_latency = raw_latency - E.
    """
    import numpy as np

    recv_wall_s = np.asarray(recv_wall_s, dtype=np.float64)
    recv_mono = np.asarray(recv_mono, dtype=np.float64)
    w = recv_wall_s - recv_mono

    ks, ts = [], []
    for a in anchors or []:
        if a is None or a.get("offset_s") is None or a.get("mono") is None:
            continue
        w_at = float(np.interp(float(a["mono"]), recv_mono, w))
        # offset_s > 0 means local is slow, so the local error E = -offset_s.
        ks.append(w_at + float(a["offset_s"]))
        ts.append(float(a["mono"]))
    if not ks:
        return None, {"ok": False,
                      "reason": "no usable clock anchors; latency reported raw"}

    # K is NOT constant. E(t) = w(t) - K assumes true time advances exactly with
    # the monotonic clock, but time.monotonic() is QueryPerformanceCounter, whose
    # crystal carries its own frequency error -- ~1.6 ppm on this machine. That
    # is invisible over a 2 h run and 80 ms over a 14 h one, and left uncorrected
    # it produced a physically impossible -25 ms minimum latency on 2026-09-02.
    #
    # Two anchors give two equations, so fit the rate as well as the offset:
    #     K(t) = k0 + rate * (t - t_ref)
    t_ref = float(ts[0])
    rate = 0.0
    if len(ks) >= 2 and (max(ts) - min(ts)) > 600.0:
        A = np.column_stack([np.ones(len(ts)),
                             np.asarray(ts, dtype=float) - t_ref])
        coef, *_ = np.linalg.lstsq(A, np.asarray(ks, dtype=float), rcond=None)
        k0, rate = float(coef[0]), float(coef[1])
        # A real crystal is within tens of ppm. Anything wilder means the anchors
        # are bad, not the oscillator; fall back rather than extrapolate nonsense
        # across a whole day.
        if abs(rate) > 1e-4:                     # 100 ppm
            k0, rate = float(np.median(ks)), 0.0
    else:
        k0 = float(np.median(ks))

    e = w - (k0 + rate * (recv_mono - t_ref))
    resid_ms = [abs(kk - (k0 + rate * (tt - t_ref))) * 1e3
                for kk, tt in zip(ks, ts)]
    return e, {
        "ok": True,
        "n_anchors": len(ks),
        "monotonic_rate_error_ppm": rate * 1e6,
        "anchor_residual_ms": max(resid_ms) if resid_ms else 0.0,
        "adjustment_during_run_ms": float(w[-1] - w[0]) * 1e3 if w.size else 0.0,
        "error_start_ms": float(e[0]) * 1e3 if e.size else None,
        "error_end_ms": float(e[-1]) * 1e3 if e.size else None,
        "method": "E(t) = (recv_wall - recv_mono) - (k0 + rate*(t - t_ref)); "
                  "the rate term absorbs the monotonic clock's own frequency "
                  "error, which a single constant cannot",
    }


def describe(off: dict) -> str:
    if not off or not off.get("ok"):
        return "clock offset unknown (no NTP server reachable)"
    ms = off["offset_s"] * 1000.0
    direction = "slow" if ms > 0 else "fast"
    return (f"local clock {abs(ms):.0f} ms {direction} vs "
            f"{'/'.join(off['servers'])} "
            f"(spread {off['offset_spread_s'] * 1000:.0f} ms, "
            f"n={off['n_samples']})")
