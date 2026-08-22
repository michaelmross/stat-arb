"""Ground-truth tests for the real-data path (data.py / universe.py / scan.py).

Same discipline as validate.py: verify the machinery against data whose
answer we already know, before trusting it on data whose answer we don't.
Here the known quantity is a synthetic cointegrated pair round-tripped
through the exact on-disk layouts Stooq and Tiingo emit.

Run:  python test_data.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from synth import make_pair
from coint import engle_granger
from data import load_prices, load_panel, detect_source
from universe import candidate_pairs, all_tickers

PASS, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, cond, detail=""):
    print(f"[{PASS if cond else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def bdays(n, start="2010-01-04"):
    return pd.bdate_range(start=start, periods=n)


def write_stooq(path, dates, prices):
    """Date,Open,High,Low,Close,Volume  -- exactly as stooq.com/q/d/l emits."""
    p = np.asarray(prices)
    pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": np.round(p * 0.999, 4), "High": np.round(p * 1.004, 4),
        "Low": np.round(p * 0.996, 4), "Close": np.round(p, 4),
        "Volume": np.full(len(p), 1_000_000),
    }).to_csv(path, index=False)


def write_tiingo(path, dates, prices, adj_prices=None):
    """Tiingo /tiingo/daily/<t>/prices?format=csv column layout."""
    p = np.asarray(prices)
    a = np.asarray(adj_prices if adj_prices is not None else prices)
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": np.round(p, 4), "high": np.round(p * 1.004, 4),
        "low": np.round(p * 0.996, 4), "open": np.round(p * 0.999, 4),
        "volume": np.full(len(p), 1_000_000),
        "adjClose": np.round(a, 6), "adjHigh": np.round(a * 1.004, 6),
        "adjLow": np.round(a * 0.996, 6), "adjOpen": np.round(a * 0.999, 6),
        "adjVolume": np.full(len(p), 1_000_000),
        "divCash": np.zeros(len(p)), "splitFactor": np.ones(len(p)),
    }).to_csv(path, index=False)


def main():
    rng = np.random.default_rng(20260821)
    tmp = Path(tempfile.mkdtemp(prefix="statarb_test_"))
    print(f"scratch: {tmp}\n")

    # ---------------- 1. format detection + round-trip fidelity --------
    n = 3000
    dates = bdays(n)
    logp, logq, truth = make_pair("cointegrated", n=n, rng=rng)
    P, Q = np.exp(logp), np.exp(logq)

    write_stooq(tmp / "spy.us.csv", dates, P)
    write_tiingo(tmp / "ivv.csv", dates, Q)

    s_st, i_st = load_prices(tmp / "spy.us.csv")
    s_ti, i_ti = load_prices(tmp / "ivv.csv")

    check("stooq layout detected", i_st.source == "stooq", i_st.source)
    check("tiingo layout detected", i_ti.source == "tiingo", i_ti.source)
    check("ticker parsed from stooq '.us' filename", i_st.ticker == "SPY",
          i_st.ticker)
    check("tiingo uses adjClose", i_ti.price_col == "adjclose"
          and i_ti.dividend_adjusted)
    check("stooq flagged as NOT dividend-adjusted", not i_st.dividend_adjusted)
    check("stooq div warning surfaced",
          any("dividend-adjusted" in w for w in i_st.warnings))
    check("all rows survive clean load", i_st.n_obs == n and i_ti.n_obs == n,
          f"{i_st.n_obs}/{i_ti.n_obs} of {n}")
    check("stooq price round-trips to 4dp",
          np.allclose(s_st.to_numpy(), np.round(P, 4)))
    check("tiingo adj price round-trips to 6dp",
          np.allclose(s_ti.to_numpy(), np.round(Q, 6)))

    # ---------------- 2. beta recovery THROUGH the loader --------------
    panel = load_panel(tmp, tickers=["SPY", "IVV"])
    lp, lq, idx = panel.pair("SPY", "IVV")
    eg_disk = engle_granger(lp, lq)
    eg_mem = engle_granger(logp, logq)
    check("beta recovered through CSV round-trip",
          abs(eg_disk.beta - truth.beta) < 0.02,
          f"disk {eg_disk.beta:.4f} vs true {truth.beta}")
    check("disk vs in-memory beta agree",
          abs(eg_disk.beta - eg_mem.beta) < 1e-3,
          f"{eg_disk.beta:.6f} vs {eg_mem.beta:.6f}")
    check("cointegration still detected after round-trip",
          eg_disk.pvalue < 0.01, f"p={eg_disk.pvalue:.2e}")
    check("panel index preserved", len(idx) == n and idx[0] == dates[0])

    # ---------------- 3. dirty-data diagnostics ------------------------
    dirty = Path(tempfile.mkdtemp(prefix="statarb_dirty_"))
    d_dates, d_px = bdays(800), np.exp(logp[:800])

    # duplicate dates
    dd = d_dates.append(d_dates[-1:])
    write_stooq(dirty / "dup.csv", dd, np.append(d_px, d_px[-1]))
    _, i_dup = load_prices(dirty / "dup.csv")
    check("duplicate dates detected + dropped", i_dup.n_dupe_dates == 1
          and i_dup.n_obs == 800, f"dupes={i_dup.n_dupe_dates}")

    # non-positive price
    bad = d_px.copy()
    bad[100] = 0.0
    write_stooq(dirty / "zero.csv", d_dates, bad)
    _, i_zero = load_prices(dirty / "zero.csv")
    check("non-positive price detected + dropped", i_zero.n_nonpositive == 1
          and i_zero.n_obs == 799, f"nonpos={i_zero.n_nonpositive}")

    # stale run (illiquid ETF printing the same close)
    stale = d_px.copy()
    stale[200:260] = stale[199]
    write_stooq(dirty / "stale.csv", d_dates, stale)
    _, i_stale = load_prices(dirty / "stale.csv")
    check("stale-price run detected", i_stale.max_stale_run >= 59,
          f"max run {i_stale.max_stale_run}, {i_stale.stale_frac:.1%} flat")
    check("stale warning surfaced",
          any("unchanged close" in w for w in i_stale.warnings))

    # unequal calendars -> per-pair inner join must not truncate other pairs
    short_n = 500
    write_stooq(dirty / "shorty.csv", bdays(short_n), np.exp(logp[:short_n]))
    write_stooq(dirty / "longy.csv", d_dates, np.exp(logq[:800]))
    write_stooq(dirty / "longy2.csv", d_dates, np.exp(logp[:800]))
    p2 = load_panel(dirty, tickers=["SHORTY", "LONGY", "LONGY2"])
    check("outer join keeps union of dates", len(p2.log_prices) == 800,
          f"{len(p2.log_prices)} rows")
    a, b, _ = p2.pair("LONGY", "LONGY2", min_obs=100)
    c, d, _ = p2.pair("SHORTY", "LONGY", min_obs=100)
    check("short history does not truncate unrelated pairs",
          len(a) == 800 and len(c) == short_n,
          f"long pair {len(a)}, short pair {len(c)}")

    # ---------------- 4. universe sanity -------------------------------
    cands = candidate_pairs()
    keys = [tuple(sorted((c.a, c.b))) for c in cands]
    check("no duplicate candidate pairs", len(keys) == len(set(keys)))
    check("no self-pairs", all(c.a != c.b for c in cands))
    check("all four relation types present",
          {"same_index", "sector_parent", "hedged", "holdrs"}
          <= {c.relation for c in cands})
    check("controls are excludable",
          len(candidate_pairs(False)) < len(cands))
    check("ticker list covers every pair",
          set(all_tickers()) == {t for c in cands for t in (c.a, c.b)})

    # ---------------- 5. end-to-end scan on a known universe -----------
    uni = Path(tempfile.mkdtemp(prefix="statarb_uni_"))
    # SPY/IVV/VOO: genuinely cointegrated. GLD/XRT: independent walks.
    cp, cq, _ = make_pair("cointegrated", n=n, rng=rng)
    _, cr, _ = make_pair("cointegrated", n=n, rng=rng)
    r1, r2, _ = make_pair("random_walks", n=n, rng=rng)
    for tk, series in [("spy", cp), ("ivv", cq), ("voo", cr),
                       ("gld", r1), ("xrt", r2)]:
        write_tiingo(uni / f"{tk}.csv", dates, np.exp(series))

    split = dates[int(n * 0.6)].strftime("%Y-%m-%d")
    out = uni / "scan_results.json"
    proc = subprocess.run(
        [sys.executable, "scan.py", "--data-dir", str(uni),
         "--discovery-end", split, "--out", str(out)],
        capture_output=True, text=True, cwd=Path(__file__).parent)
    print("\n----- scan.py end-to-end -----")
    print(proc.stdout.strip() or proc.stderr.strip())
    print("------------------------------\n")
    check("scan.py runs clean", proc.returncode == 0, proc.stderr[-300:])
    if out.exists():
        import json
        res = json.loads(out.read_text())
        disc = {tuple(sorted((r["a"], r["b"]))): r for r in res["discovery"]}
        spy_ivv = disc.get(("IVV", "SPY"))
        gld_xrt = disc.get(("GLD", "XRT"))
        check("true pair SPY/IVV is a BH discovery",
              spy_ivv is not None and spy_ivv["bh"],
              f"p={spy_ivv['pvalue']:.2e}" if spy_ivv else "missing")
        check("control GLD/XRT is NOT a discovery",
              gld_xrt is not None and not gld_xrt["bh"],
              f"p={gld_xrt['pvalue']:.3f}" if gld_xrt else "missing")
        check("evaluation produced OOS results", len(res["evaluation"]) > 0,
              f"{len(res['evaluation'])} pairs evaluated")
        if res["evaluation"]:
            sh = np.median([e["sharpe"] for e in res["evaluation"]])
            check("OOS Sharpe positive on true cointegrated pairs", sh > 0.5,
                  f"median {sh:.2f}")

    for d in (tmp, dirty, uni):
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'ALL CHECKS PASSED' if not _failures else 'FAILURES: ' + ', '.join(_failures)}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
