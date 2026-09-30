"""Figures for the companion paper. Derivable artifacts, not frozen bytes.

    python paper/paper_numbers.py     # first
    python paper/make_figures.py      # writes paper/fig_*.pdf
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import polars as pl                       # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N = json.loads((HERE / "paper_numbers.json").read_text(encoding="utf-8"))
NAMES = {"eurgbp": "EUR/GBP", "audnzd": "AUD/NZD", "eurczk": "EUR/CZK"}
COL = {"eurgbp": "#1f4e79", "audnzd": "#b5651d", "eurczk": "#3a7d44"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    if os.environ.get("FIG_PREVIEW"):          # PNG copies for eyeballing only
        (HERE / "preview").mkdir(exist_ok=True)
        fig.savefig(HERE / "preview" / f"{name}.png", dpi=130)


def census(name):
    p = HERE / "runs" / f"{name}.json"
    if not p.exists():
        p = Path(sorted(glob.glob(str(HERE / "runs" / name / "phase1_*.json")))[-1])
    return json.loads(p.read_text(encoding="utf-8"))


def fig_gate():
    """Capture vs z at 30 s, against the gate's bar and the cost actually quoted."""
    c = census("eurgbp_main")
    bins = [b for b in c["reversion"]["curves"]["30.0"]
            if b.get("z_median") is not None and b["n"] >= 30]
    z = np.array([b["z_median"] for b in bins])
    cap = np.array([b["capture_pips"] for b in bins])
    lo = np.array([b["capture_ci95"][0] for b in bins])
    hi = np.array([b["capture_ci95"][1] for b in bins])
    cost = np.array([b["cost_at_quoted_spread_pips"] for b in bins])
    bar = c["gate_g1"]["criteria"]["required_capture_pips"]
    spread = c["gate_g1"]["criteria"]["eurgbp_spread_median_pips"]

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    ax.axhline(bar, color="0.35", ls="--", lw=0.9)
    ax.text(0.3, bar + 0.05, f"Gate G1 bar: 25% of median spread ({bar:.3f} pip)",
            fontsize=7.5, color="0.3")
    ax.axhline(spread, color="0.6", ls=":", lw=0.9)
    ax.text(-1.2, spread - 0.13, f"median round trip ({spread:.2f} pip)", fontsize=7.5, color="0.45")
    ax.plot(z, cost, "-", color="#a33", lw=1.2, label="cost at the spread quoted at the time")
    ax.errorbar(z, cap, yerr=[cap - lo, hi - cap], fmt="o", ms=3.5, color=COL["eurgbp"],
                ecolor=COL["eurgbp"], elinewidth=0.8, capsize=2, label="30 s capture (95% HAC CI)")
    g = int(np.argmax(cap))
    ax.annotate("gate passes here\n(n = %d)" % bins[g]["n"], xy=(z[g], cap[g]),
                xytext=(z[g] - 2.6, cap[g] + 0.55), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))
    ax.set_xlabel("entry z (bin median)")
    ax.set_ylabel("pips of EUR/GBP")
    ax.set_ylim(-0.1, max(cost.max(), cap.max()) + 0.4)
    ax.legend(frameon=False, loc="upper center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_gate")
    plt.close(fig)


def fig_kbasis():
    """Pooled k on four sigma bases, and per-day demeaned k for the thin arms."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"width_ratios": [1.05, 1]})
    bases = [("k_raw", "raw"), ("k_within_hour", "within-hour"),
             ("k_demeaned", "demeaned"), ("k_spread_matched_median", "spread-matched")]
    x = np.arange(len(bases))
    for i, (tri, dx) in enumerate(zip(NAMES, (-0.2, 0.0, 0.2))):
        t = N["triangles"][tri]
        a.plot(x + dx, [t[k] for k, _ in bases], "o", color=COL[tri], ms=4.5, label=NAMES[tri])
    a.axhline(13.7, color="0.3", ls="--", lw=0.9)
    a.text(-0.35, 13.95, "pre-registered bar 13.7", fontsize=7, color="0.3")
    a.set_xticks(x, [lbl for _, lbl in bases], fontsize=7.5)
    a.set_ylabel("k = round trip / sigma")
    a.legend(frameon=False, fontsize=7, loc="lower right")
    a.set_title("pooled, by sigma basis", fontsize=8.5)

    for tri in ("audnzd", "eurczk", "eurgbp"):
        pdk = N["triangles"][tri]["k_per_day_demeaned"]
        days = sorted(pdk)
        b.plot(range(len(days)), [pdk[d] for d in days], "o-", ms=3, lw=0.9, color=COL[tri],
               label=NAMES[tri] + (" (census days)" if tri == "eurgbp" else ""))
    b.axhline(13.7, color="0.3", ls="--", lw=0.9)
    b.set_xlabel("session (in order)")
    b.set_title("per session, demeaned", fontsize=8.5)
    b.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_kbasis")
    plt.close(fig)


def fig_hourly():
    """Hourly sigma against hourly spread: high-sigma hours are wide-spread hours."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for ax, tri in zip(axes, NAMES):
        h = pl.read_csv(HERE / "fig_data" / f"hourly_{tri}.csv")
        spr, sd = h["spr"].to_numpy(), h["sd"].to_numpy()
        ax.plot(spr, sd, "o", ms=2.6, alpha=0.55, color=COL[tri])
        xmax, ymax = spr.max() * 1.25, sd.max() * 1.1
        xs = np.array([0.0, xmax])
        for k in (10, 15, 20):
            ax.plot(xs, xs / k, "-", lw=0.6, color="0.7")
            xl = min(xmax * 0.97, ymax * k * 0.93)      # where the line leaves the box
            ax.text(xl, xl / k, f"k={k}", fontsize=6, color="0.45", ha="right", va="bottom")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_title(NAMES[tri], fontsize=8.5)
        ax.set_xlabel("hourly median spread (pip)", fontsize=7.5)
    axes[0].set_ylabel("hourly demeaned sigma (pip)", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_hourly")
    plt.close(fig)


def fig_reboot():
    """Monotonic-clock range of every EUR/GBP collector run, by wall-clock start."""
    fs = sorted(glob.glob(str(ROOT / "data" / "ticks" / "*" / "*.parquet")))
    if not fs:
        # The only figure that needs raw ticks, which are not redistributed.
        print("fig_reboot: skipped (no data/ticks/ here; the committed PDF stands)")
        return
    df = pl.concat([pl.read_parquet(f, columns=["run_id", "recv_mono", "recv_wall"]) for f in fs])
    excl = json.loads((ROOT / "data" / "excluded_runs.json").read_text(encoding="utf-8"))
    bad = {e["run_id"] for e in excl.get("runs", [])}
    r = (df.filter(~pl.col("run_id").is_in(list(bad)))
           .group_by("run_id").agg(pl.col("recv_mono").min().alias("m0"),
                                   pl.col("recv_mono").max().alias("m1"),
                                   pl.col("recv_wall").min().alias("w0"),
                                   pl.len().alias("n"))
           .filter(pl.col("n") > 1000).sort("w0"))
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    for i, row in enumerate(r.iter_rows(named=True)):
        post = row["w0"].strftime("%m-%d") >= "09-09" and row["m0"] < 3 * 86400
        ax.plot([row["m0"] / 86400, row["m1"] / 86400], [i, i], lw=4,
                color="#a33" if post else COL["eurgbp"], solid_capstyle="butt")
        ax.text(row["m1"] / 86400 + 0.12, i, row["w0"].strftime("%d %b"), fontsize=6.5, va="center")
    ax.set_yticks([])
    ax.set_xlabel("receive-side monotonic clock (days since boot)")
    ax.set_title("each bar is one collector run, ordered by calendar start; red = after the 9 Sept reboot",
                 fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_reboot")
    plt.close(fig)


if __name__ == "__main__":
    fig_gate()
    fig_kbasis()
    fig_hourly()
    fig_reboot()
    print("wrote", ", ".join(sorted(p.name for p in HERE.glob("fig_*.pdf"))))
