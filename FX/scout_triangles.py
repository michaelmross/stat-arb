"""Rank every available triangle by its COST, before spending a fortnight on one.

    python scout_triangles.py                 # all triangles, ranked
    python scout_triangles.py --top 25
    python scout_triangles.py --contains AUD_NZD,EUR_NOK,EUR_SEK

Why cost first. The EUR/USD-GBP/USD-EUR/GBP census gives a residual sigma of
about 0.10 pip against a 1.32 pip round trip -- so breakeven needs a 13-sigma
entry. Moving to a thinner cross changes BOTH sides of that ratio, and the
common intuition ("less traded, less efficient, more opportunity") is probably
backwards on a retail venue: a broker with no independent liquidity in a thin
cross synthesises the quote from the two majors and adds a markup, which shrinks
the residual toward rounding noise while widening the spread.

The spread half is measurable in one REST call. The residual half needs two
weeks of collection. So measure the cheap half first: a candidate whose round
trip is 5 pip rather than 1.3 needs a residual roughly 4x larger merely to be
equally hopeless, and can be discarded without collecting anything.

This script places no orders and touches nothing but /instruments and /pricing.

IMPORTANT: run it during liquid hours (the London/NY overlap). Spreads on a thin
session -- a US holiday, the 5pm rollover, the Asian afternoon -- are
unrepresentative and would rank USD-quoted candidates unfairly. The script
refuses to report if the venue says the instruments are not tradeable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fxlib.config import load_config, load_credentials
from fxlib.oanda import OandaClient

HERE = Path(__file__).resolve().parent
PIP_OF = {}          # instrument -> pip size, from the venue's pipLocation


def fx_instruments(client) -> dict:
    out = {}
    for i in client._get("/instruments")["instruments"]:
        if i.get("type") != "CURRENCY":
            continue
        out[i["name"]] = {
            "pip": 10.0 ** int(i.get("pipLocation", -4)),
            "display": int(i.get("displayPrecision", 5)),
            "margin": float(i.get("marginRate", 0.0) or 0.0),
        }
    return out


# Currencies whose value is pegged or tightly managed against a major. Their
# crosses price cheaply and look attractive on cost alone, but that is exactly
# why they are the WORST candidates: a currency held inside a narrow band cannot
# generate an independent residual. The ranking cannot see this, so it is
# annotated instead.
MANAGED = {"HKD": "pegged to USD (7.75-7.85 band)",
           "DKK": "pegged to EUR (ERM II, +/-2.25%)",
           "CNH": "managed float vs a USD-weighted basket"}


def triangles(names) -> list:
    """Every tradeable triangle, allowing either quote orientation per leg.

    A leg can be quoted the other way round -- the venue lists USD_NOK, not
    NOK_USD -- and an earlier version of this function only looked for one
    orientation. It therefore silently dropped every Scandi and CEE candidate,
    which were precisely the ones worth scouting. Spread in basis points is
    orientation-invariant, so the cost ranking is unaffected by which way a leg
    is quoted; but Phase 1b collection would have to invert that leg (reciprocal
    price, bid and ask swapped), so the orientation is recorded here.

    Returns (leg_a, leg_b, cross, inverted) where `inverted` lists the legs
    quoted opposite to the X_Z / Y_Z form the TriangleBook assumes.
    """
    have = set(names)
    out = []
    for cross in sorted(have):
        x, y = cross.split("_")
        for z in ("USD", "EUR", "GBP", "JPY", "AUD", "CHF"):
            if z in (x, y):
                continue
            inverted = []
            a = f"{x}_{z}"
            if a not in have:
                a = f"{z}_{x}"
                inverted.append(a)
            b = f"{y}_{z}"
            if b not in have:
                b = f"{z}_{y}"
                inverted.append(b)
            if a in have and b in have:
                out.append((a, b, cross, inverted))
                break
    return out


def sample_prices(client, names, rounds: int, pause: float) -> dict:
    """Median bid/ask per instrument over several rounds; spreads are noisy."""
    acc = {n: {"bid": [], "ask": [], "tradeable": []} for n in names}
    chunk = 30
    for r in range(rounds):
        for i in range(0, len(names), chunk):
            batch = names[i:i + chunk]
            try:
                resp = client.pricing(batch)
            except Exception as exc:                      # noqa: BLE001
                print("  pricing call failed: %s" % exc, flush=True)
                continue
            for p in resp.get("prices", []):
                n = p.get("instrument")
                if n not in acc or not p.get("bids") or not p.get("asks"):
                    continue
                acc[n]["bid"].append(float(p["bids"][0]["price"]))
                acc[n]["ask"].append(float(p["asks"][0]["price"]))
                acc[n]["tradeable"].append(bool(p.get("tradeable", False)))
        if r < rounds - 1:
            time.sleep(pause)
    out = {}
    for n, v in acc.items():
        if not v["bid"]:
            continue
        bid, ask = statistics.median(v["bid"]), statistics.median(v["ask"])
        mid = 0.5 * (bid + ask)
        out[n] = {
            "bid": bid, "ask": ask, "mid": mid,
            "spread_bp": 1e4 * (ask - bid) / mid if mid else float("nan"),
            "spread_pips": (ask - bid) / PIP_OF.get(n, 1e-4),
            "tradeable": all(v["tradeable"]),
            "n": len(v["bid"]),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rounds", type=int, default=5,
                    help="pricing samples per instrument (median is taken)")
    ap.add_argument("--pause", type=float, default=20.0,
                    help="seconds between rounds")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--contains", default=None,
                    help="only triangles whose cross is in this comma list")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="rank even outside the overlap (indicative only)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    client = OandaClient(cfg, load_credentials())
    base_a, base_b, base_c = (cfg["oanda"]["instrument_a"],
                              cfg["oanda"]["instrument_b"],
                              cfg["oanda"]["instrument_c"])

    meta = fx_instruments(client)
    PIP_OF.update({k: v["pip"] for k, v in meta.items()})
    tris = triangles(meta)
    if args.contains:
        want = {x.strip().upper() for x in args.contains.split(",")}
        tris = [t for t in tris if t[2] in want]
    needed = sorted({n for t in tris for n in t[:3]})
    print("%d FX instruments, %d complete triangles, %d instruments to price"
          % (len(meta), len(tris), len(needed)), flush=True)
    print("sampling %d rounds, %.0fs apart (~%.1f min)..."
          % (args.rounds, args.pause, args.rounds * args.pause / 60.0), flush=True)

    px = sample_prices(client, needed, args.rounds, args.pause)
    client.close()

    # tradeable=true is not the same as liquid. The venue reopens Sunday 17:00
    # ET and quotes all week, but Sunday evening, the Asian afternoon and a US
    # holiday all carry spreads that would rank USD-quoted candidates unfairly.
    from fxlib.sessions import SessionClock
    import datetime as _dt
    label = SessionClock(cfg).session_label(_dt.datetime.now(_dt.timezone.utc))
    if label != "overlap" and not args.force:
        print("\nREFUSING TO RANK: the current session is '%s', not 'overlap'. "
              "Spreads outside the London/NY overlap are not representative and "
              "would bias the ranking. Re-run 08:00-17:00 ET on a normal "
              "weekday, or pass --force to see indicative numbers anyway."
              % label)
        return 1

    untradeable = [n for n, v in px.items() if not v["tradeable"]]
    if len(untradeable) > len(px) * 0.2:
        print("\nREFUSING TO RANK: %d of %d instruments report tradeable=false. "
              "The venue is closed or in maintenance, and spreads sampled now "
              "are not representative. Re-run during the London/NY overlap."
              % (len(untradeable), len(px)))
        return 1

    rows = []
    for a, b, c, inv in tris:
        if not all(n in px for n in (a, b, c)):
            continue
        # Three-leg deterministic cycle: you cross half a spread on each leg,
        # twice over the round trip -- i.e. the full spread of all three.
        cycle_bp = sum(px[n]["spread_bp"] for n in (a, b, c))
        # One-leg statistical trade: in and out of the cross only.
        leg_bp = px[c]["spread_bp"]
        managed = sorted({cur for cur in c.split("_") if cur in MANAGED})
        rows.append({
            "cross": c, "legs": [a, b], "inverted_legs": inv,
            "managed": managed,
            "cycle_cost_bp": cycle_bp,
            "cross_round_trip_bp": leg_bp,
            "cross_round_trip_pips": px[c]["spread_pips"],
            "untradeable": not px[c]["tradeable"],
        })
    if not rows:
        print("no triangles priced")
        return 1

    base = next((r for r in rows if r["cross"] == base_c), None)
    ref = base["cross_round_trip_bp"] if base else min(
        r["cross_round_trip_bp"] for r in rows)
    for r in rows:
        # How much bigger the residual must be, versus the current triangle,
        # merely to be equally unprofitable.
        r["residual_multiple_needed"] = r["cross_round_trip_bp"] / ref if ref else float("nan")
    rows.sort(key=lambda r: r["cross_round_trip_bp"])

    print("\nRanked by the ONE-LEG round trip (the cost that matters for the")
    print("statistical arm). 'x need' = how many times larger the residual must")
    print("be than on %s just to be equally hopeless.\n" % base_c)
    print("  %-10s %10s %10s %9s   %s" % ("cross", "1-leg bp", "1-leg pip",
                                          "cycle bp", "x need"))
    for r in rows[:args.top]:
        mark = " <= current" if r["cross"] == base_c else ""
        if r["managed"]:
            mark += "  PEGGED/MANAGED - cheap but cannot carry a residual"
        elif r["inverted_legs"]:
            mark += "  (leg inversion needed)"
        print("  %-10s %10.2f %10.2f %9.2f   %5.1fx%s"
              % (r["cross"], r["cross_round_trip_bp"], r["cross_round_trip_pips"],
                 r["cycle_cost_bp"], r["residual_multiple_needed"], mark))

    if base:
        better = [r for r in rows if r["cross_round_trip_bp"] < ref]
        print("\n%d of %d crosses are CHEAPER to round-trip than %s."
              % (len(better), len(rows), base_c))
        print("A cheaper spread is necessary but nowhere near sufficient: the "
              "residual still has to exist. This ranking only says which "
              "candidates are not already dead on cost.")

    outp = Path(args.out) if args.out else HERE / "out" / "triangle_scout.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(
        {"reference_cross": base_c, "rounds": args.rounds, "rows": rows},
        indent=2), encoding="utf-8")
    print("\nwrote %s" % outp.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
