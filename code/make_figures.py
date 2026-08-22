"""Regenerate all five figures of the note from archived artifacts.

Inputs (defaults assume repo root):
  --scan    scan_v2.json                      (merged-scan output)
  --census  fractional_census_results.json    (ELW census output)
  --data    data/                             (Tiingo CSVs, adjClose)
  --out     .                                 (where fig_*.pdf land)

Figures:
  fig_scatter.pdf   p-value vs edge ratio, initial (v1) cohort, 98 pairs
  fig_anatomy.pdf   SPY/IVV vs BND/SPAB spread + ACF, discovery window
  fig_oos.pdf       walk-forward OOS equity, the 7 discoveries that traded
  fig_crisis.pdf    BND/SPAB March 2020: frozen-parameter z + price anatomy
  fig_dcensus.pdf   fractional-d census with calibrated anchors

The d-census anchor bands are hardcoded below rather than recomputed on
every figure build. Their derivation is `fractional_census.anchors()`
(`python fractional_census.py --anchors`), NOT `validate()`: the OU
anchor uses a half-life and stationary width matched to the BH
discoveries' own medians (2.06d, 13.6bp, seed 555, no micro noise),
which is a different parameterisation from validate()'s SPY/IVV-scale
OU+noise cell and gives 0.068, not that cell's 0.317. Anchors reproduce
in distribution rather than to the digit, because the RNG stream
position differs between a standalone call and the same cell run inside
validate():

  OU anchor  hardcoded 0.066 +/- 0.053 ; anchors() 0.068 +/- 0.054
  RW anchor  hardcoded 0.543 +/- 0.042 ; anchors() 0.536 +/- 0.040

Both gaps are under 0.25 se of the mean.
"""

from __future__ import annotations

import paths
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from coint import engle_granger
from ou import fit_ou
from backtest import zscore_backtest
from statsmodels.tsa.stattools import acf

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

DISCOVERY_END = pd.Timestamp("2018-12-31")
CRISIS_FREEZE = "2020-02-15"          # freeze on data strictly before this
TRAIN = 504

FAM_COLORS = {"same_index": "tab:blue", "sector_parent": "tab:orange",
              "hedged": "tab:green", "holdrs": "tab:red", "control": "k"}
FAM_LABELS = {"same_index": "same-index", "sector_parent": "sector/parent",
              "hedged": "hedged/unhedged", "holdrs": "HOLDRs-style",
              "control": "negative control"}


def load_log(data_dir: Path, ticker: str) -> pd.Series:
    df = pd.read_csv(data_dir / f"{ticker.lower()}.csv",
                     parse_dates=["date"], index_col="date")
    return np.log(df["adjClose"])


def aligned(data_dir: Path, a: str, b: str) -> pd.DataFrame:
    pa, pb = load_log(data_dir, a), load_log(data_dir, b)
    idx = pa.index.intersection(pb.index)
    return pd.DataFrame({a: pa[idx], b: pb[idx]})


def fig_scatter(scan: pd.DataFrame, out: Path):
    v1 = scan[scan["cohort"] == "v1"]
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for rel, g in v1.groupby("relation"):
        m = "x" if rel == "control" else "o"
        ax.scatter(g["edge_ratio"], -np.log10(np.maximum(g["pvalue"], 1e-300)),
                   s=26, marker=m, c=FAM_COLORS[rel], alpha=.85,
                   label=FAM_LABELS[rel],
                   edgecolors="none" if m == "o" else None, zorder=3)
    ax.axhline(-np.log10(0.05), color="gray", lw=.8, ls="--")
    ax.text(250, -np.log10(0.05) + .12, "p = 0.05", color="gray", fontsize=8)
    ax.axvline(1.0, color="gray", lw=.8, ls=":")
    ax.text(1.06, 10.5, "cost-viability\nthreshold", color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("edge ratio (expected capture per round trip / round-trip cost)")
    ax.set_ylabel(r"$-\log_{10}$ Engle--Granger $p$-value (discovery window)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title(f"Cointegration strength vs. cost viability, {len(v1)} ETF pairs")
    fig.savefig(out / "fig_scatter.pdf")
    plt.close(fig)


def fig_anatomy(data_dir: Path, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    for col, (a, b, tag) in enumerate(
            [("SPY", "IVV", "noise-dominated"),
             ("BND", "SPAB", "genuinely mean-reverting")]):
        sub = aligned(data_dir, a, b)
        sub = sub[sub.index <= DISCOVERY_END]
        eg = engle_granger(sub[a].to_numpy(), sub[b].to_numpy())
        s = pd.Series(eg.spread * 1e4, index=sub.index)
        axes[0, col].plot(s.index, s.values, lw=.4, color="tab:blue")
        yearly = s.groupby(s.index.year).mean()
        mid = [pd.Timestamp(f"{y}-07-01") for y in yearly.index]
        axes[0, col].plot(mid, yearly.values, "r_-", ms=10, lw=1.2)
        axes[0, col].set_title(f"{a}/{b} spread ({tag})")
        if col == 0:
            axes[0, col].set_ylabel("bp")
        A = acf(eg.spread, nlags=40, fft=True)
        axes[1, col].bar(range(41), A, width=.8, color="tab:blue")
        axes[1, col].axhline(0, color="k", lw=.5)
        axes[1, col].set_ylim(-.1, 1.0)
        axes[1, col].set_xlabel("lag (days)")
        if col == 0:
            axes[1, col].set_ylabel("spread ACF")
    fig.tight_layout()
    fig.savefig(out / "fig_anatomy.pdf")
    plt.close(fig)


def fig_oos(data_dir: Path, out: Path):
    pairs = [("BND", "SPAB"), ("AGG", "BND"), ("AGG", "SPAB"),
             ("VTI", "SCHB"), ("LQD", "USIG"), ("SCHZ", "AGG"),
             ("VCIT", "IGIB")]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for a, b in pairs:
        sub = aligned(data_dir, a, b)
        bt = zscore_backtest(sub[a].to_numpy(), sub[b].to_numpy())
        post = np.asarray(sub.index > DISCOVERY_END)
        post[:TRAIN] = False
        eq = pd.Series(np.cumprod(1 + bt.daily_ret[post]) - 1,
                       index=sub.index[post])
        ax.plot(eq.index, eq.values * 100, lw=1.0, label=f"{a}/{b}")
    ax.axhline(0, color="k", lw=.5)
    ax.set_ylabel("cumulative return (%)")
    ax.set_title("Out-of-sample walk-forward equity, traded discoveries "
                 "(2 bp/leg costs)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.savefig(out / "fig_oos.pdf")
    plt.close(fig)


def fig_crisis(data_dir: Path, out: Path):
    sub = aligned(data_dir, "BND", "SPAB")
    pre = sub[sub.index < CRISIS_FREEZE].tail(TRAIN)
    eg = engle_granger(pre["BND"].to_numpy(), pre["SPAB"].to_numpy())
    ou = fit_ou(eg.spread)
    win = sub["2020-02-01":"2020-04-30"]
    z = pd.Series((win["BND"].to_numpy() - eg.alpha
                   - eg.beta * win["SPAB"].to_numpy() - ou.mu)
                  / ou.stationary_std, index=win.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2))
    ax1.plot(z.index, z.values, lw=1.1, color="navy")
    for lvl, c, ls in [(4, "r", "--"), (-4, "r", "--"),
                       (2, "gray", ":"), (-2, "gray", ":")]:
        ax1.axhline(lvl, color=c, ls=ls, lw=.8)
    ax1.annotate("+4$\\sigma$ stop", xy=(pd.Timestamp("2020-02-03"), 5.5),
                 fontsize=8, color="r")
    pk = z.abs().idxmax()
    ax1.annotate(f"peak {z.max():.0f}$\\sigma$\n{pk.date()}",
                 xy=(pk, z.max()),
                 xytext=(pd.Timestamp("2020-03-25"), 58), fontsize=8,
                 arrowprops=dict(arrowstyle="-", lw=.6))
    ax1.set_title("BND/SPAB spread $z$, parameters frozen 2020-02-14")
    ax1.set_ylabel(f"z-score (pre-crisis $\\sigma_\\infty$ = "
                   f"{ou.stationary_std*1e4:.1f} bp)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for lab in ax1.get_xticklabels():
        lab.set_rotation(30)

    w2 = sub["2020-03-05":"2020-03-20"]
    base = w2.iloc[0]
    for t, c in [("BND", "tab:blue"), ("SPAB", "tab:orange")]:
        ax2.plot(w2.index, (np.exp(w2[t] - base[t]) - 1) * 100, lw=1.2,
                 marker="o", ms=3, color=c, label=t)
    ax2.axvline(pd.Timestamp("2020-03-13"), color="k", lw=.6, ls=":")
    ax2.set_title("The anatomy: asynchronous NAV reconvergence")
    ax2.set_ylabel("cumulative return from Mar 5 (%)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for lab in ax2.get_xticklabels():
        lab.set_rotation(30)
    fig.tight_layout()
    fig.savefig(out / "fig_crisis.pdf")
    plt.close(fig)


def fig_dcensus(census: pd.DataFrame, out: Path):
    # anchors from fractional_census.validate(); see module docstring
    OU_ANCHOR, OU_BAND = 0.066, 0.11
    RW_ANCHOR, RW_BAND = 0.543, 0.08
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axhspan(OU_ANCHOR - OU_BAND, OU_ANCHOR + OU_BAND, color="tab:green",
               alpha=.15, label="OU+noise anchor (matched hl)")
    ax.axhspan(RW_ANCHOR - RW_BAND, RW_ANCHOR + RW_BAND, color="tab:orange",
               alpha=.18, label="RW+noise anchor (calibrated)")
    ax.axhline(1.0, color="tab:red", lw=.8, ls="--", label="pure random walk")
    x = -np.log10(np.maximum(census["pvalue"].to_numpy(), 1e-300))
    for flag, mk, col in [(False, "o", "tab:blue"), (True, "s", "gray")]:
        m = ((census["flag"] == flag) & (~census["bh"])).to_numpy()
        ax.scatter(x[m], census.loc[m, "d_hat"], s=22, marker=mk, c=col,
                   alpha=.6, label=("unflagged" if not flag else "flagged")
                   + " non-discovery")
    m = census["bh"].to_numpy()
    ax.scatter(x[m], census.loc[m, "d_hat"], s=60, marker="*", c="crimson",
               zorder=5, label="BH discovery")
    for r in census[census["bh"] & (census["d_hat"] < 0.47)].itertuples():
        ax.annotate(r.pair, (-np.log10(max(r.pvalue, 1e-300)), r.d_hat),
                    fontsize=7, xytext=(4, -9), textcoords="offset points")
    bnd = census[census["pair"] == "BND/SPAB"].iloc[0]
    ax.annotate("BND/SPAB", (-np.log10(max(bnd.pvalue, 1e-300)), bnd.d_hat),
                fontsize=7, xytext=(4, 5), textcoords="offset points")
    ax.set_xlabel(r"$-\log_{10}$ Engle--Granger $p$-value (discovery window)")
    ax.set_ylabel(r"memory parameter $\hat d$ "
                  r"(exact local Whittle, $m=n^{0.6}$)")
    ax.set_title(f"Fractional integration census of all {len(census)} spreads")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(out / "fig_dcensus.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=str(paths.data("scan_v2.json")))
    ap.add_argument("--census", default=str(paths.data("fractional_census_results.json")))
    ap.add_argument("--data", default=str(paths.PRICES))
    ap.add_argument("--out", default=str(paths.FIGURES))
    args = ap.parse_args()

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scan = pd.DataFrame(json.load(open(args.scan))["discovery"])

    fig_scatter(scan, out)
    fig_anatomy(data_dir, out)
    fig_oos(data_dir, out)
    fig_crisis(data_dir, out)
    if Path(args.census).exists():
        fig_dcensus(pd.read_json(args.census), out)
    else:
        print(f"note: {args.census} not found; skipping fig_dcensus")
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
