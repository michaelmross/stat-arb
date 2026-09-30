"""Every number the companion paper cites, regenerated from the tick logs.

    python paper/paper_numbers.py          # ~10 min; writes paper/paper_numbers.json

Reads the census JSONs in paper/runs/ (produced by phase1_analyze.py, see
paper/README.md) and adds what the census does not emit on its own:

  * k on the spread-matched basis and per day, for all three triangles;
  * sigma and cost in basis points, so triangles compare in one unit;
  * the anatomy of the Gate G1 bin (when, at what spread, how stale);
  * the counterfactual cost of the cross-reboot ordering bug;
  * the NZD 07:00 NZT feed break, day by day.

Nothing here feeds back into the pre-registered analysis; it only describes it.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from fxlib.config import load_config            # noqa: E402
from fxlib.replay import load_ticks, replay     # noqa: E402

TZ = "America/New_York"
PHASE1B = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
           "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"]
# The Phase 1 census window as declared in run_collector.bat ("Phase 1 daily
# collector, 2026-08-31 .. 2026-09-11"). 27-28 Aug were Phase 0 calibration
# runs, declared throwaway; an internal analysis that pooled them read Gate G1
# as PASS, and is reported in the paper as that error, not as the result.
CENSUS = ["2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
          "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]
TRIANGLES = {
    "eurgbp": ("config.toml", CENSUS, "eurgbp_main"),
    "audnzd": ("config_audnzd.toml", PHASE1B, "audnzd"),
    "eurczk": ("config_eurczk.toml", PHASE1B, "eurczk"),
}


def census_json(name: str) -> dict:
    p = HERE / "runs" / f"{name}.json"
    if not p.exists():
        p = Path(sorted(glob.glob(str(HERE / "runs" / name / "phase1_*.json")))[-1])
    return json.loads(p.read_text(encoding="utf-8"))


def _q(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def describe_triangle(conf: str, dates, run_name: str) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    cfg = load_config(conf)
    o = cfg["oanda"]
    root = ROOT / cfg["storage"]["root"]
    ticks = load_ticks(root, dates)
    snap = replay(cfg, ticks, o["instrument_a"], o["instrument_b"], o["instrument_c"])
    s = snap.filter(snap["synchronized"]).with_columns(
        (pl.col("z") * pl.col("sigma_pips")).alias("dem"),
        pl.col("recv_wall").dt.convert_time_zone(TZ).alias("et"))
    s = s.with_columns(pl.col("et").dt.strftime("%Y-%m-%d %H").alias("dh"),
                       pl.col("et").dt.strftime("%Y-%m-%d").alias("d"))
    fin = s.filter(pl.col("dem").is_finite())

    pip = float(o.get("pip_size", 1e-4))
    c_mid = float(np.nanmedian(s["c_mid"].to_numpy()))
    pip_bp = pip / c_mid * 1e4
    spread = float(np.median(s["c_spread_pips"].to_numpy()))
    sd_dem = float(np.std(fin["dem"].to_numpy()))

    hourly = (fin.group_by("dh").agg(pl.len().alias("n"), pl.col("dem").std().alias("sd"),
                                     pl.col("c_spread_pips").median().alias("spr"))
                 .filter((pl.col("n") >= 30) & (pl.col("sd") > 0)).sort("dh"))
    kh = (hourly["spr"] / hourly["sd"]).to_numpy().astype(float)
    kh = kh[np.isfinite(kh)]
    daily = (fin.group_by("d").agg(pl.len().alias("n"), pl.col("dem").std().alias("sd"),
                                   pl.col("c_spread_pips").median().alias("spr")).sort("d"))
    per_day = {r["d"]: r["spr"] / r["sd"] for r in daily.iter_rows(named=True) if r["sd"]}

    out = {
        "config": conf,
        "instruments": [o["instrument_a"], o["instrument_b"], o["instrument_c"]],
        "inverted_legs": list(o.get("inverted_legs", []) or []),
        "pip_size": pip,
        "cross_mid_median": c_mid,
        "pip_in_bp": pip_bp,
        "spread_median_pips": spread,
        "spread_median_bp": spread * pip_bp,
        "sigma_demeaned_pips": sd_dem,
        "sigma_demeaned_bp": sd_dem * pip_bp,
        "sigma_as_pct_of_spread": 100.0 * sd_dem / spread,
        "k_spread_matched_median": float(np.median(kh)),
        "k_spread_matched_iqr": [_q(kh, 25), _q(kh, 75)],
        "k_spread_matched_hours": int(kh.size),
        "k_per_day_demeaned": per_day,
        "days_k_below_13_7": sum(1 for v in per_day.values() if v < 13.7),
    }
    return out, s, ticks


def gate_bin_anatomy(s: pl.DataFrame) -> dict:
    """Where the z >= 5 observations that decided Gate G1 come from."""
    ts = s["t_axis"].to_numpy()
    r = s["resid_pips"].to_numpy()
    z = s["z"].to_numpy()
    cs = s["c_spread_pips"].to_numpy()
    vs = s["venue_spread_ms"].to_numpy()
    n = len(ts)
    j = np.searchsorted(ts, ts + 30.0, side="left")
    ok = j < n
    js = np.where(ok, np.minimum(j, n - 1), 0)
    ok &= (ts[js] - ts) <= 45.0
    cap = np.where(ok, -np.sign(z) * (r[js] - r), np.nan)
    m = ok & np.isfinite(z) & (z >= 5.0)
    et = s["et"].to_list()
    idx = np.flatnonzero(m)

    def near(t, hh, mm, before_s=60, after_s=300):
        sec = t.hour * 3600 + t.minute * 60 + t.second
        tgt = hh * 3600 + mm * 60
        return tgt - before_s <= sec <= tgt + after_s

    release = [i for i in idx if near(et[i], 8, 30) or near(et[i], 10, 0)]
    rollover = [i for i in idx if (et[i].hour == 16 and et[i].minute >= 50)]
    med_spread = float(np.median(cs))
    clean = idx[vs[idx] < 250]
    return {
        "n": int(idx.size),
        "capture_mean_pips": float(np.mean(cap[idx])),
        "spread_at_entry_median_pips": float(np.median(cs[idx])),
        "spread_all_median_pips": med_spread,
        "share_spread_above_median": float(np.mean(cs[idx] > med_spread)),
        "share_within_release_window": len(release) / idx.size,
        "share_in_rollover_16_50_17_00": len(rollover) / idx.size,
        "distinct_dates": len({str(et[i])[:10] for i in idx}),
        "venue_spread_median_ms": float(np.median(vs[idx])),
        "venue_spread_lt250_n": int(clean.size),
        "venue_spread_lt250_capture_pips": float(np.mean(cap[clean])) if clean.size else None,
        "timestamps_et": sorted({et[i].strftime("%Y-%m-%d %H:%M") for i in idx}),
    }


def reboot_counterfactual(conf: str) -> dict:
    """Replay EUR/GBP with the ORIGINAL global recv_mono sort."""
    cfg = load_config(conf)
    o = cfg["oanda"]
    root = ROOT / cfg["storage"]["root"]
    fixed = load_ticks(root)
    buggy = fixed.sort("recv_mono")
    rf = fixed["run_id"].to_list()
    rb = buggy["run_id"].to_list()
    sw = lambda x: sum(1 for i in range(1, len(x)) if x[i] != x[i - 1])
    sf = replay(cfg, fixed, o["instrument_a"], o["instrument_b"], o["instrument_c"])
    sb = replay(cfg, buggy, o["instrument_a"], o["instrument_b"], o["instrument_c"])
    nf, nb = int(sf["synchronized"].sum()), int(sb["synchronized"].sum())

    day = fixed.filter(pl.col("recv_wall").dt.convert_time_zone(TZ).dt.strftime("%Y-%m-%d") == "2026-09-09")
    rates = {}
    for inst in sorted(day["instrument"].unique().to_list()):
        d = day.filter(pl.col("instrument") == inst)
        span_old = (d["recv_mono"].max() - d["recv_mono"].min())
        span_new = float(d.group_by("run_id").agg((pl.col("recv_mono").max() - pl.col("recv_mono").min()).alias("s"))["s"].sum())
        rates[inst] = {"old_ticks_per_min": len(d) / (span_old / 60.0),
                       "fixed_ticks_per_min": len(d) / (span_new / 60.0)}
    return {
        "runs": len(set(rf)),
        "run_switches_fixed": sw(rf),
        "run_switches_global_mono_sort": sw(rb),
        "synchronized_fixed": nf,
        "synchronized_global_mono_sort": nb,
        "synchronized_lost_fraction": 1.0 - nb / nf,
        "ticks_per_min_2026_09_09": rates,
    }


def nzd_break(conf: str) -> dict:
    cfg = load_config(conf)
    root = ROOT / cfg["storage"]["root"]
    out = {}
    for d in PHASE1B:
        # Sort by run THEN monotonic clock, and never difference across runs. The
        # first version of this function sorted a day on recv_mono alone and, on
        # 09-24 (machine crash and reboot mid-session), reported a 1,242,780 s
        # "gap": the very ordering trap the paper describes, reproduced here.
        t = pl.read_parquet(sorted(glob.glob(str(root / "ticks" / d / "*.parquet"))))
        t = (t.with_columns(pl.col("recv_wall").min().over("run_id").alias("_r0"))
              .sort(["_r0", "run_id", "recv_mono"]))
        g = t.filter(pl.col("instrument") == "NZD_USD")
        et = g["recv_wall"].dt.convert_time_zone(TZ).to_list()
        rid = g["run_id"].to_list()
        x = np.diff(g["recv_mono"].to_numpy())
        x = np.where([rid[k + 1] == rid[k] for k in range(len(rid) - 1)], x, -np.inf)
        i = int(np.argmax(x))
        out[d] = {"weekday": et[i + 1].strftime("%a"), "gap_s": float(x[i]),
                  "from_et": et[i].strftime("%H:%M:%S"), "to_et": et[i + 1].strftime("%H:%M:%S")}
    return out


def main() -> None:
    res = {"triangles": {}}
    hourly_for_fig = {}
    for name, (conf, dates, run_name) in TRIANGLES.items():
        print(f"[numbers] {name}: replaying", flush=True)
        desc, s, _ = describe_triangle(conf, dates, run_name)
        c = census_json(run_name)
        rs, cen = c["residual"], c["deterministic_census"]
        bins = [b for b in c["reversion"]["curves"]["30.0"] if b.get("z_median") is not None]
        desc.update({
            "dates": c["provenance"]["dates"],
            "n_ticks": c["provenance"]["n_ticks"],
            "n_synchronized": cen["n_synchronized_observations"],
            "synchronized_fraction": c["provenance"]["synchronized_fraction"],
            "observed_hours": cen["observed_hours"],
            "positive_cycles": cen["n_observations_with_positive_cycle"],
            "best_cycle_bp": cen["best_cycle_bp"]["max"],
            "cycle_cost_median_bp": cen["cycle_cost_bp"]["median"],
            "sigma_raw_pips": rs["resid_pips"]["sd"],
            "k_raw": rs["required_k_full_capture"],
            "k_demeaned": rs["required_k_demeaned"],
            "k_within_hour": rs["required_k_within_hour"],
            "best_bin_net_pips": max(b["net_at_quoted_spread_pips"] for b in bins if b["n"] >= 30),
            "gate": {"verdict_if_scored": ("PASS" if c["gate_g1"]["criteria"]["sigma_ok"]
                                           and c["gate_g1"]["criteria"]["capture_ok"] else "FAIL"),
                     **c["gate_g1"]["criteria"]},
            "hy_best_lag": c["lead_lag"].get("hy_best_lag"),
            "burst": {k: c["burst_structure"].get(k) for k in
                      ("cadence_hz", "ticks_per_burst", "fraction_bursts_all_three")},
            "venue_sync": {k: c["venue_sync_check"].get(k) for k in
                           ("fraction_contaminated", "resid_sd_contaminated", "resid_sd_clean")},
            "latency_corrected_median_ms": {k: v["latency_corrected_ms"]["median"]
                                            for k, v in c["latency"].items()
                                            if not k.startswith("_") and v.get("latency_corrected_ms")},
            "ticks_per_min": {k: v["ticks_per_min"] for k, v in c["latency"].items()
                              if not k.startswith("_")},
            "tau_sensitivity": c["tau_sensitivity"],
            "gaps_total_s": c["events"].get("total_gap_seconds"),
        })
        res["triangles"][name] = desc
        hourly_for_fig[name] = (s.filter(pl.col("dem").is_finite())
                                 .group_by("dh").agg(pl.len().alias("n"), pl.col("dem").std().alias("sd"),
                                                     pl.col("c_spread_pips").median().alias("spr"))
                                 .filter((pl.col("n") >= 30) & (pl.col("sd") > 0)).sort("dh"))
        if name == "eurgbp":
            res["gate_bin_anatomy"] = gate_bin_anatomy(s)

    pooled = census_json("eurgbp_all")
    res["eurgbp_pooled_with_calibration"] = {
        "dates": pooled["provenance"]["dates"],
        "n_synchronized": pooled["deterministic_census"]["n_synchronized_observations"],
        "positive_cycles": pooled["deterministic_census"]["n_observations_with_positive_cycle"],
        "best_cycle_bp": pooled["deterministic_census"]["best_cycle_bp"]["max"],
        "sigma_raw_pips": pooled["residual"]["resid_pips"]["sd"],
        "k_raw": pooled["residual"]["required_k_full_capture"],
        "k_within_hour": pooled["residual"]["required_k_within_hour"],
        "k_demeaned": pooled["residual"]["required_k_demeaned"],
        "gate": pooled["gate_g1"]["criteria"],
        "gate_bin": [b for b in pooled["reversion"]["curves"]["30.0"] if b["z_hi"] is None][0],
    }
    print("[numbers] reboot counterfactual", flush=True)
    res["reboot_counterfactual"] = reboot_counterfactual("config.toml")
    print("[numbers] NZD break", flush=True)
    res["nzd_break"] = nzd_break("config_audnzd.toml")

    (HERE / "paper_numbers.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    fig = HERE / "fig_data"
    fig.mkdir(exist_ok=True)
    for name, h in hourly_for_fig.items():
        h.write_csv(fig / f"hourly_{name}.csv")
    print("[numbers] wrote paper/paper_numbers.json", flush=True)


if __name__ == "__main__":
    main()
