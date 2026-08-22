"""Fetch continuous front-month futures for the crack and crush spreads.

Three production spreads, each a genuine economic relationship rather
than an index-membership artifact:

  3-2-1 crack   CL=F, RB=F, HO=F    refinery margin
  board crush   ZS=F, ZM=F, ZL=F    soybean processing margin
  cattle crush  GF=F, LE=F, ZC=F    feedlot margin  (optional leg)

TWO THINGS THAT WILL BITE IF IGNORED
------------------------------------
1. ROLL DISCONTINUITIES. Yahoo's `=F` series are front-month continuous
   contracts stitched WITHOUT back-adjustment. At each roll the series
   jumps by the calendar spread. Those jumps are not price moves; in log
   space they are structural breaks that will read as spread dislocations
   and can manufacture or destroy apparent cointegration. Any serious
   spread work needs either back-adjusted continuous series or explicit
   roll handling. `roll_diagnostics()` below measures how bad it is here.

2. UNIT MISMATCH. The legs are not quoted in comparable units, so a
   naive log-price regression is not the crack or the crush:

     CL  $/barrel        RB, HO  $/gallon      (42 gal = 1 bbl)
     ZS  cents/bushel    ZM  $/short ton       ZL  cents/lb
     LE, GF  cents/lb    ZC  cents/bushel

   The classic constructions, in consistent dollars:
     3-2-1 crack = (2*RB + 1*HO) * 42 / 3  -  CL          [$/bbl]
     board crush = (11*ZM/100 * 2000/2000) + (9*ZL*60/100) - ZS/100 ...
   Conversion factors are supplied in UNITS below rather than hardcoded
   into a spread, because which construction you want depends on the
   question. This script only ACQUIRES the data.
"""

from __future__ import annotations

import paths

import argparse
import time
from pathlib import Path

import pandas as pd

SPREADS = {
    "crack_321": ["CL=F", "RB=F", "HO=F"],
    "board_crush": ["ZS=F", "ZM=F", "ZL=F"],
    "cattle_crush": ["GF=F", "LE=F", "ZC=F"],
}

# quoted units, for whoever builds the spread downstream
UNITS = {
    "CL=F": ("WTI crude", "USD/barrel", 1.0),
    "RB=F": ("RBOB gasoline", "USD/gallon", 42.0),      # x42 -> USD/bbl
    "HO=F": ("heating oil", "USD/gallon", 42.0),        # x42 -> USD/bbl
    "ZS=F": ("soybeans", "cents/bushel", 0.01),         # -> USD/bushel
    "ZM=F": ("soybean meal", "USD/short ton", 1.0),
    "ZL=F": ("soybean oil", "cents/lb", 0.01),          # -> USD/lb
    "GF=F": ("feeder cattle", "cents/lb", 0.01),
    "LE=F": ("live cattle", "cents/lb", 0.01),
    "ZC=F": ("corn", "cents/bushel", 0.01),
}


def fetch(tickers, out_dir: Path, start="2010-01-01", end=None, pause=1.0):
    import yfinance as yf

    out_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for i, t in enumerate(tickers, 1):
        dest = out_dir / f"{t.replace('=', '_').lower()}.csv"
        try:
            df = yf.Ticker(t).history(start=start, end=end,
                                      interval="1d", auto_adjust=False)
        except Exception as exc:
            failed.append((t, str(exc)[:90]))
            print(f"[{i}/{len(tickers)}] {t:6s} ERROR {str(exc)[:70]}")
            continue

        if df is None or df.empty:
            failed.append((t, "empty response"))
            print(f"[{i}/{len(tickers)}] {t:6s} EMPTY")
            continue

        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
        keep = [date_col, "Open", "High", "Low", "Close", "Volume"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].rename(columns={date_col: "Date"})
        # Keep non-positive settlements. WTI settled at -$37.63 on
        # 2020-04-20; that is a real observation, and dropping it also
        # fabricates a -60% two-day move across the resulting hole.
        # Price-space analysis handles it natively.
        df = df.dropna(subset=["Close"])
        df.to_csv(dest, index=False)
        ok.append(t)
        print(f"[{i}/{len(tickers)}] {t:6s} {len(df):5d} rows  "
              f"{df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}"
              f"   {UNITS[t][0]} ({UNITS[t][1]})")
        time.sleep(pause)
    return ok, failed


def roll_diagnostics(out_dir: Path, tickers, jump_sigma=5.0):
    """How badly do the un-back-adjusted rolls contaminate the series?

    Counts daily log moves beyond `jump_sigma` robust standard deviations.
    For a stitched front-month series these are dominated by roll gaps,
    not by market moves.
    """
    import numpy as np

    print(f"\nRoll-gap diagnostics (|log move| > {jump_sigma} robust sd):")
    print(f"  {'ticker':8s} {'n':>6s} {'robust sd':>10s} {'jumps':>6s} "
          f"{'max |chg|':>11s}  worst date")
    for t in tickers:
        f = out_dir / f"{t.replace('=', '_').lower()}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["Date"])
        # differences in quoted units, not log -- prices can be negative
        lr = np.diff(df["Close"].to_numpy())
        mad = np.median(np.abs(lr - np.median(lr)))
        sd = 1.4826 * mad
        big = np.abs(lr) > jump_sigma * sd
        worst = int(np.argmax(np.abs(lr)))
        print(f"  {t:8s} {len(df):6d} {sd:10.4f} {int(big.sum()):6d} "
              f"{np.abs(lr).max():11.3f}  {df['Date'].iloc[worst + 1].date()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.FUTURES))
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--spreads", nargs="*", default=list(SPREADS),
                    choices=list(SPREADS))
    args = ap.parse_args()

    tickers = [t for s in args.spreads for t in SPREADS[s]]
    out = Path(args.out)
    print(f"Fetching {len(tickers)} continuous front-month contracts "
          f"from Yahoo into {out}/\n")
    ok, failed = fetch(tickers, out, args.start, args.end)
    print(f"\n{len(ok)} ok, {len(failed)} failed")
    for t, why in failed:
        print(f"  {t}: {why}")
    roll_diagnostics(out, ok)


if __name__ == "__main__":
    main()
