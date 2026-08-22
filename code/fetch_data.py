"""Fetch daily OHLC CSVs for the candidate universe.

Two paths, because the two sources have very different access stories.

TIINGO (preferred -- gives adjClose, i.e. dividend-adjusted)
        python fetch_data.py --source tiingo --out data/
    The token is looked up in this order:
        1. TIINGO_API_KEY in the environment
        2. a token file, default .tiingo_token next to this script
    and is sent in an Authorization header, never in the URL -- URLs leak
    into server logs, shell history and Referer headers. The token file is
    read at run time and its contents are never printed.

STOOQ (no key, but see below)
    stooq.com now gates /q/d/l/ behind a JavaScript proof-of-work check,
    so scripted fetching does not work and defeating that check is not
    something this script will do. Download by hand instead -- either
    per ticker from
        https://stooq.com/q/d/l/?s=spy.us&i=d
    or the bulk daily US archive at
        https://stooq.com/db/h/
    and drop the CSVs in one directory. `--source stooq --out data/` then
    only verifies and normalises what is already there.

Either way the result is one CSV per ticker in --out, ready for scan.py.
"""

from __future__ import annotations

import paths

import argparse
import os
import sys
import time
from pathlib import Path

import requests

from universe import all_tickers
from data import load_prices

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
DEFAULT_TOKEN_FILE = Path(__file__).with_name(".tiingo_token")


def read_token(token_file: Path | None) -> str | None:
    """Env var first, then a token file. Never logs the value."""
    tok = os.environ.get("TIINGO_API_KEY")
    if tok and tok.strip():
        return tok.strip()
    path = token_file or DEFAULT_TOKEN_FILE
    if path.exists():
        # tolerate a bare token or a KEY=value line
        raw = path.read_text(encoding="utf-8-sig").strip()
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line and line.split("=", 1)[0].strip().upper().endswith("KEY"):
                line = line.split("=", 1)[1].strip()
            return line.strip().strip("'\"")
    return None


def fetch_tiingo(tickers, out_dir: Path, start: str, end: str | None,
                 token: str, pause: float = 0.6):
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Token {token}",
                         "Content-Type": "application/json"})
    ok, missing, failed = [], [], []
    for i, t in enumerate(tickers, 1):
        dest = out_dir / f"{t.lower()}.csv"
        if dest.exists() and dest.stat().st_size > 200:
            ok.append(t)
            print(f"[{i:3d}/{len(tickers)}] {t:6s} cached")
            continue
        params = {"startDate": start, "format": "csv",
                  "resampleFreq": "daily"}
        if end:
            params["endDate"] = end
        try:
            r = sess.get(TIINGO_URL.format(ticker=t.lower()),
                         params=params, timeout=30)
        except requests.RequestException as e:
            failed.append((t, str(e)[:80]))
            print(f"[{i:3d}/{len(tickers)}] {t:6s} ERROR {e}")
            continue

        if r.status_code == 404:
            missing.append(t)
            print(f"[{i:3d}/{len(tickers)}] {t:6s} not found on Tiingo")
        elif r.status_code == 401:
            sys.exit("Tiingo rejected the token (401). Check TIINGO_API_KEY.")
        elif r.status_code == 429 or r.text.lstrip().lower().startswith("error:"):
            # Tiingo signals the hourly cap with HTTP 200 + Content-Type
            # text/csv and an error sentence as the body. Writing that to
            # disk would produce a 136-byte "price file" that parses to
            # nothing -- so treat the body, not the status, as the truth.
            msg = r.text.strip()[:160] if r.text else f"HTTP {r.status_code}"
            print(f"[{i:3d}/{len(tickers)}] {t:6s} RATE LIMITED")
            print(f"\n  Tiingo says: {msg}\n"
                  f"  {len(ok)} tickers cached so far; re-run after the hourly\n"
                  f"  window resets and it will resume from {t}.")
            return ok, missing, failed + [(t, "rate limited")]
        elif r.ok and r.text.strip() and "date" in r.text[:200].lower():
            dest.write_text(r.text)
            n = r.text.count("\n") - 1
            ok.append(t)
            print(f"[{i:3d}/{len(tickers)}] {t:6s} {n:5d} rows")
        else:
            failed.append((t, f"HTTP {r.status_code}"))
            print(f"[{i:3d}/{len(tickers)}] {t:6s} HTTP {r.status_code}")
        time.sleep(pause)
    return ok, missing, failed


def verify(out_dir: Path):
    """Load every CSV present and print the loader's diagnostics."""
    files = sorted(out_dir.glob("*.csv"))
    if not files:
        print(f"no CSVs in {out_dir}")
        return
    print(f"\nVerifying {len(files)} files in {out_dir}:")
    short, unadj = [], []
    for f in files:
        try:
            _, info = load_prices(f)
        except Exception as e:
            print(f"  {f.name:16s} UNREADABLE: {e}")
            continue
        flag = "" if info.n_obs >= 756 else "  <-- short history"
        if info.n_obs < 756:
            short.append(info.ticker)
        if not info.dividend_adjusted:
            unadj.append(info.ticker)
        print(f"  {info.ticker:6s} {info.source:6s} {info.price_col:9s} "
              f"{info.n_obs:5d} obs  {info.start.date()} -> {info.end.date()}"
              f"{flag}")
        for w in info.warnings:
            if "dividend-adjusted" not in w:
                print(f"         ! {w}")
    if unadj:
        print(f"\n  {len(unadj)} series are NOT dividend-adjusted "
              f"(raw close): {', '.join(unadj[:12])}"
              f"{' ...' if len(unadj) > 12 else ''}")
        print("  Ex-dividend gaps will show up as jumps in the log spread.")
    if short:
        print(f"\n  {len(short)} series shorter than 756 obs "
              f"(will be skipped by scan.py): {', '.join(short)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["tiingo", "stooq"], default="tiingo")
    ap.add_argument("--out", default=str(paths.PRICES))
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="default: the whole candidate universe")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--token-file", default=None,
                    help=f"default: {DEFAULT_TOKEN_FILE.name} next to this script")
    args = ap.parse_args()

    out = Path(args.out)
    tickers = [t.upper() for t in (args.tickers or all_tickers())]

    if args.verify_only or args.source == "stooq":
        if args.source == "stooq" and not args.verify_only:
            print(__doc__.split("STOOQ")[1].split("Either way")[0].strip())
            print()
        verify(out)
        return

    token = read_token(Path(args.token_file) if args.token_file else None)
    if not token:
        sys.exit(
            "No Tiingo token found.\n"
            f"  Either write it to {DEFAULT_TOKEN_FILE}\n"
            "  or set $env:TIINGO_API_KEY in the shell you run this from.\n"
            "Token page: https://www.tiingo.com/account/api/token")
    print(f"token loaded ({len(token)} chars), sending as Authorization header")

    print(f"Fetching {len(tickers)} tickers from Tiingo "
          f"({args.start} -> {args.end or 'today'}) into {out}/\n")
    ok, missing, failed = fetch_tiingo(tickers, out, args.start, args.end, token)
    print(f"\n{len(ok)} ok, {len(missing)} not found, {len(failed)} failed")
    if missing:
        print(f"  not found: {', '.join(missing)}")
    if failed:
        for t, why in failed:
            print(f"  failed {t}: {why}")
    verify(out)


if __name__ == "__main__":
    main()
