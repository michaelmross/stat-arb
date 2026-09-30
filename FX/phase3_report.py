"""Phase 3: the census-style writeup.

    python phase3_report.py                    # newest phase1 JSON + all trades
    python phase3_report.py --phase1 out/phase1_20260901T120000Z.json

Combines the Phase 1 census with the Phase 2 trade log into one report, and adds
the two things neither phase produces on its own:

  * the cost decomposition -- gross, net at the quoted spread, and net at the
    fills actually received, side by side, so the practice-fill optimism is
    visible rather than buried;
  * the variance decomposition -- how much of the residual is quote-grid
    rounding, how much is cross-pair asynchrony, and how much is left over as a
    candidate for genuine cross-market lag. Without this, a feed artifact and an
    edge look identical.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fxlib import stats
from fxlib.config import load_config

HERE = Path(__file__).resolve().parent
PNL_KEYS = ["gross_mid_to_mid", "net_at_quoted_spread", "net_with_measured_fills"]
PNL_LABEL = {"gross_mid_to_mid": "gross (mid to mid)",
             "net_at_quoted_spread": "net at quoted spread",
             "net_with_measured_fills": "net at measured fills"}


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return f"{x:,}"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{x:,.{nd}f}" if np.isfinite(x) else "n/a"


def load_trades(root: Path) -> list:
    out = []
    for f in sorted(glob.glob(str(root / "trades" / "*.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "trade":
                out.append(rec)
    return out


def rounding_floor_pips(a: float, b: float, c: float, grid: float = 1e-5) -> float:
    """Residual sd implied by the quote grid alone, in EUR/GBP pips.

    Bid rounds down and ask rounds up onto the grid, so each mid carries an
    independent error with sd = grid*sqrt(2/12)/2. The three legs contribute in
    quadrature through eps = ln A - ln B - ln C.
    """
    per = grid * math.sqrt(2.0 / 12.0) / 2.0
    var = sum((per / p) ** 2 for p in (a, b, c))
    return math.sqrt(var) * c / 1e-4


def variance_decomposition(rep: dict) -> dict:
    """Split the observed residual variance into grid rounding, cross-pair
    asynchrony, and an unexplained remainder.

    Asynchrony is estimated by extrapolating sigma^2(tau) to tau = 0 across the
    tau-sensitivity grid: whatever variance disappears as the synchronization
    tolerance tightens was asynchrony, by construction. The remainder is the
    only part that could be genuine independent pricing of the cross -- and it
    is an upper bound on that, not a measurement of it.
    """
    rows = [r for r in rep["tau_sensitivity"]
            if r["n"] > 100 and np.isfinite(r["resid_sd_pips"])]
    total_sd = rep["residual"]["resid_pips"]["sd"]
    lv = rep["provenance"].get("price_levels") or {}
    a_px = float(lv.get("A") or 1.17)
    b_px = float(lv.get("B") or 1.36)
    c_px = float(lv.get("C") or 0.86)
    floor_pips = rounding_floor_pips(a_px, b_px, c_px)

    extrap = None
    if len(rows) >= 3:
        taus = np.array([r["tau_ms"] for r in rows], dtype=float)
        var = np.array([r["resid_sd_pips"] ** 2 for r in rows], dtype=float)
        A = np.column_stack([np.ones_like(taus), taus])
        coef, *_ = np.linalg.lstsq(A, var, rcond=None)
        extrap = float(max(coef[0], 0.0))

    # Force a genuine partition of the observed variance. The tau=0 intercept is
    # a regression extrapolation and the rounding floor is a theoretical value,
    # so neither is guaranteed to sit inside the observed total; clamping both
    # keeps the three shares non-negative and summing to one, and records when
    # the clamp bit so an over-extrapolation cannot hide.
    v_total = total_sd ** 2
    clamped = []
    v0 = extrap
    if v0 is not None:
        if v0 > v_total:
            clamped.append("tau0_intercept_above_total")
            v0 = v_total
        if v0 < 0.0:
            clamped.append("tau0_intercept_below_zero")
            v0 = 0.0
    v_round = min(floor_pips ** 2, v0) if v0 is not None else None
    if v_round is not None and v_round < floor_pips ** 2:
        clamped.append("rounding_floor_above_tau0_intercept")

    out = {
        "total_sd_pips": total_sd,
        "grid_rounding_sd_pips": floor_pips,
        "extrapolated_tau0_var": extrap,
        "clamped": clamped,
        "price_levels": {"A": a_px, "B": b_px, "C": c_px},
        "note": "sigma^2(tau) extrapolated linearly to tau=0; the slope is the "
                "asynchrony contribution, the intercept is everything else. The "
                "intercept still contains whatever asynchrony survives at tau=0 "
                "(quote ages are bounded by tau, never zero), so the unexplained "
                "term is an upper bound and is biased upward. Calibrate it "
                "against a synthetic derived-cross run, where the true "
                "unexplained component is zero by construction.",
    }
    if v0 is not None and v_total > 0:
        v_async = v_total - v0
        v_unexp = v0 - v_round
        out["asynchrony_sd_pips"] = math.sqrt(v_async)
        out["residual_after_rounding_sd_pips"] = math.sqrt(v_unexp)
        out["grid_rounding_sd_pips_used"] = math.sqrt(v_round)
        out["share_grid_rounding"] = v_round / v_total
        out["share_asynchrony"] = v_async / v_total
        out["share_unexplained"] = v_unexp / v_total
    return out


def trade_summary(trades: list) -> dict:
    if not trades:
        # Must still carry every key the report reads; an empty trade log is a
        # normal state (Gate G1 fails, so Phase 2 never runs), not an error.
        return {"n": 0, "n_evidential": 0, "by_group": {}}
    real = [t for t in trades if not t.get("validation_only")
            and not t.get("simulated_fills")]
    groups = {"all": trades, "evidential": real}
    out = {"n": len(trades), "n_evidential": len(real), "by_group": {}}
    for gname, g in groups.items():
        if not g:
            out["by_group"][gname] = {"n": 0}
            continue
        entry = {"n": len(g),
                 "holding_s": stats.describe(
                     np.array([t.get("holding_s", np.nan) for t in g])),
                 "sides": {},
                 "exit_reasons": {},
                 "slippage_entry_pips": stats.describe(np.array(
                     [t["entry"].get("slippage_pips", np.nan) for t in g])),
                 "slippage_exit_pips": stats.describe(np.array(
                     [t["exit"].get("slippage_pips", np.nan) for t in g])),
                 "pnl": {}}
        for t in g:
            entry["sides"][t["side"]] = entry["sides"].get(t["side"], 0) + 1
            r = t["exit"].get("reason", "?")
            entry["exit_reasons"][r] = entry["exit_reasons"].get(r, 0) + 1
        for k in PNL_KEYS:
            x = np.array([t["pnl_pips"].get(k, np.nan) for t in g], dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                entry["pnl"][k] = {"n": 0}
                continue
            mean = float(x.mean())
            se = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")
            t_stat = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
            lo, hi = stats.circular_block_bootstrap_ci(x, iters=2000, block=5) \
                if x.size >= 30 else (float("nan"), float("nan"))
            entry["pnl"][k] = {
                "n": int(x.size), "mean_pips": mean, "se_pips": se,
                "t": t_stat, "total_pips": float(x.sum()),
                "win_rate": float((x > 0).mean()),
                "bootstrap_ci95": [lo, hi],
            }
        entry["weekly"] = _weekly(g)
        out["by_group"][gname] = entry
    return out


def _weekly(trades: list) -> list:
    """Split by ISO week so the 'robust across the two weeks' criterion can be
    checked rather than asserted."""
    buckets = {}
    for t in trades:
        ts = t["entry"].get("signal_utc", "")[:10]
        try:
            wk = dt.date.fromisoformat(ts).isocalendar()
            key = f"{wk[0]}-W{wk[1]:02d}"
        except ValueError:
            key = "unknown"
        buckets.setdefault(key, []).append(t)
    rows = []
    for key in sorted(buckets):
        g = buckets[key]
        row = {"week": key, "n": len(g)}
        for k in PNL_KEYS:
            x = np.array([t["pnl_pips"].get(k, np.nan) for t in g], dtype=float)
            x = x[np.isfinite(x)]
            if x.size > 1:
                m = float(x.mean())
                se = float(x.std(ddof=1) / math.sqrt(x.size))
                row[k] = {"mean_pips": m, "t": m / se if se > 0 else float("nan")}
            elif x.size == 1:
                row[k] = {"mean_pips": float(x[0]), "t": float("nan")}
        rows.append(row)
    return rows


def success_criterion(summary: dict) -> dict:
    """Spec section 1.4: a positive surprise requires E[net] > 0 with t > 2 over
    at least 100 trades, robust across both calibration/validation weeks."""
    g = summary.get("by_group", {}).get("evidential", {})
    p = (g.get("pnl") or {}).get("net_with_measured_fills") or {}
    n = p.get("n", 0)
    mean = p.get("mean_pips", float("nan"))
    t = p.get("t", float("nan"))
    weeks = [w for w in (g.get("weekly") or [])
             if w.get("net_with_measured_fills")]
    week_means = [w["net_with_measured_fills"]["mean_pips"] for w in weeks]
    checks = {
        "n_trades_at_least_100": n >= 100,
        "mean_net_positive": bool(np.isfinite(mean) and mean > 0),
        "t_greater_than_2": bool(np.isfinite(t) and t > 2),
        "positive_in_every_week": bool(week_means) and all(m > 0 for m in week_means),
        "at_least_two_weeks": len(week_means) >= 2,
    }
    return {"checks": checks, "met": all(checks.values()),
            "n": n, "mean_net_pips": mean, "t": t,
            "week_means": dict(zip([w["week"] for w in weeks], week_means))}


def build_markdown(cfg, rep, trades, summary, crit, decomp) -> str:
    L = []
    A = L.append
    prov = rep["provenance"]
    cen = rep["deterministic_census"]
    res = rep["residual"]
    gate = rep["gate_g1"]
    hyp = rep["hypotheses"]
    synthetic = prov["source"] != "oanda"
    sim_trades = [t for t in trades if t.get("simulated_fills")
                  or t.get("validation_only")]

    A("# Triangular FX Statistical Arbitrage — Census Report")
    A("")
    A(f"EUR/USD (A) · GBP/USD (B) · EUR/GBP (C), identity C = A/B. "
      f"OANDA fxTrade practice, v20 pricing stream. Paper money, real-time data, "
      f"pre-registered hypotheses.")
    A("")
    if synthetic or sim_trades:
        A("> **This report contains non-live data.** "
          + (f"The census was computed from `{prov['source']}` ticks. " if synthetic else "")
          + (f"{len(sim_trades)} of {len(trades)} trade records are simulated or "
             f"validation-only. " if sim_trades else "")
          + "Nothing here is a measurement of the OANDA feed until it is re-run "
            "on live-collected data.")
        A("")
    A(f"- Report generated {dt.datetime.now(dt.timezone.utc).isoformat()}")
    A(f"- Census source `{prov['source']}`, config hash `{prov['config_hash']}`, "
      f"τ = {prov['tau_ms']} ms")
    A(f"- Dates {prov['dates'][0]} .. {prov['dates'][-1]} · "
      f"{prov['n_ticks']:,} ticks · {prov['n_synchronized']:,} synchronized "
      f"observations · {cen['observed_hours']:.1f} h")
    A("")

    A("## 1. Verdicts on the pre-registered hypotheses")
    A("")
    A("| | claim | result | verdict |")
    A("|---|---|---|---|")
    A(f"| H1 | no positive executable cycle | "
      f"{cen['n_events']} events, {cen['fraction_of_observations']:.4%} of "
      f"observations | **{hyp['H1']['verdict']}** |")
    A(f"| H2 | residual σ ≤ 0.3 pip | σ = {_fmt(res['resid_pips']['sd'],4)} pip | "
      f"**{hyp['H2']['verdict']}** |")
    A(f"| H3 | E[net per trade] < 0 at all thresholds | "
      f"{_fmt(crit.get('mean_net_pips'),3)} pip over {crit.get('n',0)} live trades "
      f"(t = {_fmt(crit.get('t'),2)}) | **{hyp['H3']['verdict']}** |")
    A("")
    A(f"Gate G1: **{gate['verdict']}** — {gate['action']}")
    A("")
    A(_reversion_table(rep, gate))
    A("")

    A("## 2. Is the cross independently priced? (spec §1.3)")
    A("")
    d = res["derived_cross_test"]
    A(f"- {d['fraction_abs_below']:.1%} of synchronized observations have "
      f"|ε| < {d['threshold_pips']} pip; {d['fraction_exactly_zero']:.2%} are "
      f"exactly zero.")
    A(f"- Residual σ = **{_fmt(res['resid_pips']['sd'],4)} pip** against a "
      f"EUR/GBP spread of {_fmt(res['spread_c_pips']['median'],2)} pip.")
    A("")
    A("### Variance decomposition")
    A("")
    A("| component | σ (pips) | share of variance |")
    A("|---|---:|---:|")
    A(f"| quote-grid rounding (theoretical floor) | "
      f"{_fmt(decomp.get('grid_rounding_sd_pips_used', decomp['grid_rounding_sd_pips']),4)} | "
      f"{_fmt(decomp.get('share_grid_rounding'),3)} |")
    A(f"| cross-pair asynchrony (σ²(τ) slope) | "
      f"{_fmt(decomp.get('asynchrony_sd_pips'),4)} | "
      f"{_fmt(decomp.get('share_asynchrony'),3)} |")
    A(f"| unexplained — upper bound on genuine cross pricing | "
      f"{_fmt(decomp.get('residual_after_rounding_sd_pips'),4)} | "
      f"{_fmt(decomp.get('share_unexplained'),3)} |")
    A(f"| **total observed** | **{_fmt(decomp['total_sd_pips'],4)}** | 1.000 |")
    A("")
    A("This is the decomposition the spec asks for in §3: separating the venue's "
      "price grid and the fact that three pairs do not tick at the same instant "
      "from anything that could be a real cross-market lag. The bottom row is an "
      "upper bound on the last of those, not a measurement of it — the τ→0 "
      "intercept still contains whatever asynchrony survives at τ→0, since quote "
      "ages are bounded by τ but never zero. Calibrate it against a synthetic "
      "derived-cross run, where the true unexplained component is zero.")
    if decomp.get("clamped"):
        A("")
        A(f"Clamping applied to keep the partition valid: "
          f"{', '.join(decomp['clamped'])}. The τ→0 extrapolation fell outside "
          f"the observed variance, so the split between asynchrony and the "
          f"remainder is less reliable than the total.")
    A("")

    A("## 3. Deterministic census (H1)")
    A("")
    A(f"- {cen['n_observations_with_positive_cycle']:,} of "
      f"{cen['n_synchronized_observations']:,} synchronized observations had "
      f"R1 > 0 or R2 > 0 ({cen['fraction_of_observations']:.4%}).")
    A(f"- {cen['n_events']} distinct events, "
      f"{_fmt(cen['events_per_hour'],3)} per hour.")
    A(f"- Three-leg cycle cost: median "
      f"{_fmt(cen['cycle_cost_bp'].get('median'),3)} bp. Best cycle return: "
      f"median {_fmt(cen['best_cycle_bp'].get('median'),3)} bp, max "
      f"{_fmt(cen['best_cycle_bp'].get('max'),3)} bp.")
    if cen["n_events"]:
        A(f"- Event magnitude median {_fmt(cen['magnitude_bp'].get('median'),3)} bp, "
          f"duration median {_fmt(cen['duration_s'].get('median'),3)} s over "
          f"{cen['duration_s'].get('n', 0)} events observed to completion.")
        A(f"- Closed by a tick in: {cen['closed_by']}")
        if cen.get("n_events_censored_by_desync"):
            A(f"- {cen['n_events_censored_by_desync']} of {cen['n_events']} events "
              f"were censored by desynchronization — the book went stale before "
              f"the cycle closed, so how long they lasted is not observable at "
              f"τ = {rep['provenance']['tau_ms']} ms with this cross tick rate.")
    A("")

    A("## 4. Lead-lag: does the cross lag the majors?")
    A("")
    ll = rep["lead_lag"]
    A(f"- {ll['n_crossings']:,} crossings of |z| ≥ {ll['event_z']}.")
    A(f"- Last pair to update when the residual formed: "
      f"{ll['residual_formed_by_last_update']}")
    A(f"- Pair whose tick closed it: {ll['closed_by_tick_in']}")
    if ll.get("hy_best_lag"):
        A(f"- Hayashi–Yoshida peak correlation "
          f"{_fmt(ll['hy_best_lag']['corr'],4)} at lag "
          f"{ll['hy_best_lag']['lag_ms']} ms ({ll['hy_interpretation']}).")
    A("")
    A("The trade-the-cross assumption in §1.2 stands or falls here: it requires "
      "the majors to lead and the cross to be the leg that closes the gap.")
    A("")

    A("## 5. Trading arm")
    A("")
    if not trades:
        A("No trades recorded. Either Gate G1 did not pass, or the arm has not "
          "been run.")
    else:
        for gname in ("all", "evidential"):
            g = summary["by_group"].get(gname) or {}
            if not g.get("n"):
                continue
            title = ("All records (including simulated and validation-only)"
                     if gname == "all" else
                     "Evidential records only (live fills, post-G1)")
            A(f"### {title} — n = {g['n']}")
            A("")
            A(f"sides {g['sides']} · exits {g['exit_reasons']} · holding median "
              f"{_fmt(g['holding_s'].get('median'),1)} s")
            A("")
            A("| cost basis | n | mean pips | t | win rate | total pips | bootstrap CI95 |")
            A("|---|---:|---:|---:|---:|---:|---|")
            for k in PNL_KEYS:
                p = g["pnl"].get(k) or {}
                if not p.get("n"):
                    continue
                ci = p.get("bootstrap_ci95", [None, None])
                A(f"| {PNL_LABEL[k]} | {p['n']:,} | {_fmt(p['mean_pips'])} | "
                  f"{_fmt(p['t'],2)} | {p['win_rate']:.1%} | "
                  f"{_fmt(p['total_pips'],1)} | "
                  f"[{_fmt(ci[0])}, {_fmt(ci[1])}] |")
            A("")
            A(f"Measured slippage: entry median "
              f"{_fmt(g['slippage_entry_pips'].get('median'))} pip, exit median "
              f"{_fmt(g['slippage_exit_pips'].get('median'))} pip.")
            A("")
            if g.get("weekly"):
                A("| week | n | gross | net at spread | net at fills |")
                A("|---|---:|---:|---:|---:|")
                for w in g["weekly"]:
                    cells = []
                    for k in PNL_KEYS:
                        v = w.get(k)
                        cells.append(_fmt(v["mean_pips"]) if v else "n/a")
                    A(f"| {w['week']} | {w['n']} | " + " | ".join(cells) + " |")
                A("")
    A("**Cost-model caveat.** Practice fills at OANDA's quoted price are "
      "optimistic: no queue position, no market impact, no last-look rejection. "
      "The three columns above are reported separately for exactly this reason, "
      "and even the rightmost is a lower bound on real-world cost.")
    A("")

    A("## 6. Success criterion")
    A("")
    A(f"Spec §1.4 requires E[net] > 0 with t > 2 over ≥ 100 trades, robust "
      f"across both weeks. **{'MET' if crit['met'] else 'NOT MET'}.**")
    A("")
    A("| check | result |")
    A("|---|---|")
    for k, v in crit["checks"].items():
        A(f"| {k.replace('_', ' ')} | {'yes' if v else 'no'} |")
    A("")

    A("## 7. Standing conclusion")
    A("")
    A(_conclusion(rep, crit, decomp, synthetic))
    A("")
    A("## 8. Reproduction")
    A("")
    A("```bash")
    A("python collect_ticks.py --source oanda --hours 9      # Phase 0/1")
    A("python phase1_analyze.py                              # census + Gate G1")
    A("python phase2_paper.py --hours 9                      # Phase 2, if G1 passed")
    A("python phase3_report.py                               # this report")
    A("```")
    A("")
    A(f"Config hash `{prov['config_hash']}`; frozen parameters and any declared "
      f"revision are in `out/param_freeze.json`. Every tick is in "
      f"`{cfg['storage']['root']}/ticks/`, every stream gap in "
      f"`{cfg['storage']['root']}/events/`.")
    A("")
    A("## 9. References")
    A("")
    for r in ["Aiba, Hatano, Takayasu, Marumo, Shimizu (2002), Triangular "
              "arbitrage as an interaction among foreign exchange rates, "
              "*Physica A*.",
              "Fenn, Howison, McDonald, Williams, Johnson (2009), The mirage of "
              "triangular arbitrage in the spot foreign exchange market, *IJTAF*.",
              "Foucault, Kozhan, Tham (2017), Toxic Arbitrage, *RFS*.",
              "Hasbrouck (1995) information shares; Hayashi–Yoshida (2005) "
              "asynchronous covariance."]:
        A(f"- {r}")
    return "\n".join(L)


def _gate_bin(rep, gate):
    """The 30 s reversion bin the gate verdict rests on, or None."""
    cr = gate.get("criteria", {})
    zb = cr.get("best_capture_z_bin")
    h = str(float(cr.get("capture_horizon_s", 30.0)))
    for b in rep.get("reversion", {}).get("curves", {}).get(h, []):
        if zb and [b.get("z_lo"), b.get("z_hi")] == list(zb):
            return b
    return None


def _reversion_table(rep, gate) -> str:
    """Capture against the spread quoted at the time, per z-bin.

    The gate compares mid-to-mid capture with 25% of the MEDIAN spread. The
    extreme bins sit on news releases, when the cross is quoted several pips
    wide, so a bin can clear the gate while losing money on every trade. This
    table puts both on the page; it does not alter the pre-registered verdict.
    """
    cr = gate.get("criteria", {})
    h = str(float(cr.get("capture_horizon_s", 30.0)))
    bins = [b for b in rep.get("reversion", {}).get("curves", {}).get(h, [])
            if b.get("n", 0) >= 30 and b.get("z_median") is not None]
    if not bins:
        return "_No reversion bins with n >= 30._"
    L = [f"Reversion at the gate horizon ({float(h):.0f} s). *capture* is "
         f"mid-to-mid, as the gate measures it; *cost* is half the cross spread "
         f"quoted at entry plus half at exit.", "",
         "| z bin | n | capture pip | spread at entry pip | cost pip | net pip | net > 0 |",
         "|---|---|---|---|---|---|---|"]
    zb = cr.get("best_capture_z_bin")
    for b in bins:
        lo = "−∞" if b["z_lo"] is None else f"{b['z_lo']:g}"
        hi = "∞" if b["z_hi"] is None else f"{b['z_hi']:g}"
        tag = " ← gate" if zb and [b["z_lo"], b["z_hi"]] == list(zb) else ""
        L.append(f"| [{lo}, {hi}){tag} | {b['n']:,} | {_fmt(b.get('capture_pips'))} | "
                 f"{_fmt(b.get('c_spread_median_pips'), 2)} | "
                 f"{_fmt(b.get('cost_at_quoted_spread_pips'))} | "
                 f"{_fmt(b.get('net_at_quoted_spread_pips'))} | "
                 f"{_fmt(b.get('fraction_net_positive'), 2)} |")
    nets = [b["net_at_quoted_spread_pips"] for b in bins
            if b.get("net_at_quoted_spread_pips") is not None]
    if nets:
        L += ["", f"Best bin net of the quoted spread: **{max(nets):+.3f} pip**. "
              f"Every bin is net negative." if max(nets) < 0 else
              f"Best bin net of the quoted spread: **{max(nets):+.3f} pip**."]
    return "\n".join(L)


def _conclusion(rep, crit, decomp, synthetic) -> str:
    if synthetic:
        return ("This run was computed from synthetic ticks and carries no "
                "conclusion about the OANDA feed. It demonstrates that the "
                "pipeline detects a residual when one exists and reports zero "
                "when it does not; re-run it on live-collected data for a "
                "result.")
    cen = rep["deterministic_census"]
    sd = rep["residual"]["resid_pips"]["sd"]
    spread = rep["residual"]["spread_c_pips"]["median"]
    parts = []
    if cen["fraction_of_observations"] < 1e-4:
        parts.append(
            f"The deterministic triangle is closed on this feed: "
            f"{cen['n_events']} cost-exceeding cycles in "
            f"{cen['observed_hours']:.0f} hours of synchronized quotes. H1 holds, "
            f"as the interdealer literature has implied since the late 2000s -- "
            f"at a {_fmt(cen['cycle_cost_bp'].get('median'),2)} bp three-leg cost "
            f"and retail latency, there is nothing to take.")
    else:
        parts.append(
            f"The deterministic triangle showed {cen['n_events']} cost-exceeding "
            f"cycles ({cen['events_per_hour']:.2f}/h), which contradicts H1 and "
            f"needs explaining before anything else in this report is trusted -- "
            f"start with quote staleness and one-sided books.")
    unexp = decomp.get("residual_after_rounding_sd_pips")
    if unexp is not None:
        parts.append(
            f"The residual has σ = {sd:.3f} pip against a {spread:.2f} pip "
            f"round trip. Of that, {decomp.get('share_grid_rounding', 0):.0%} is "
            f"the quote grid and {decomp.get('share_asynchrony', 0):.0%} is "
            f"cross-pair asynchrony, leaving at most {unexp:.3f} pip that could "
            f"be independent pricing of the cross.")
    if crit["met"]:
        parts.append(
            f"The trading arm cleared the pre-registered success criterion "
            f"({crit['mean_net_pips']:.3f} pip/trade, t = {crit['t']:.2f}, "
            f"n = {crit['n']}, positive in every week). That is a positive "
            f"surprise and should be treated as a hypothesis for a fresh "
            f"out-of-sample period, not as a result -- the fills are practice "
            f"fills, and the cost model has not been tested against a queue.")
    elif crit["n"] > 0:
        parts.append(
            f"The trading arm did not clear the success criterion "
            f"({crit['mean_net_pips']:.3f} pip/trade, t = {crit['t']:.2f}, "
            f"n = {crit['n']}). H3 stands.")
    elif rep.get("gate_g1", {}).get("verdict") == "PASS":
        b = _gate_bin(rep, rep["gate_g1"]) or {}
        net = b.get("net_at_quoted_spread_pips")
        parts.append(
            f"Gate G1 passed on its pre-registered wording, and that verdict "
            f"stands as recorded. It rests on one bin: z in "
            f"[{b.get('z_lo')}, {b.get('z_hi') or '∞'}), n = {b.get('n')}, "
            f"capturing {_fmt(b.get('capture_pips'))} pip in 30 s against a gate "
            f"bar set at 25% of the median spread. Those observations cluster "
            f"on scheduled releases, when the cross is quoted "
            f"{_fmt(b.get('c_spread_median_pips'), 2)} pip wide rather than "
            f"{spread:.2f}"
            + (f", so net of the spread quoted at the time the bin earns "
               f"{net:+.3f} pip per trade" if net is not None else "")
            + ". The gate's cost basis, not the market, is what passed. No "
              "Phase 2 trades have been recorded.")
    else:
        parts.append(
            "No live trades were taken, which is the correct outcome when the "
            "gate fails: the reversion available inside 30 seconds does not pay "
            "for one EUR/GBP spread, so there is no threshold worth trading.")
    parts.append(
        "The deliverable is the census, and the census is negative in the same "
        "direction as the ETF-pairs and futures-spread work: the structure is "
        "real, measurable, and smaller than the cost of touching it.")
    return " ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--phase1", default=None, help="path to a phase1_*.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.data_root:
        cfg["storage"]["root"] = args.data_root
    root = Path(cfg["_path"]).parent / cfg["storage"]["root"]
    outdir = Path(args.out) if args.out else HERE / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    p1 = Path(args.phase1) if args.phase1 else None
    if p1 is None:
        cands = sorted(glob.glob(str(outdir / "phase1_*.json")))
        if not cands:
            raise SystemExit(
                f"No phase1_*.json in {outdir}. Run phase1_analyze.py first.")
        p1 = Path(cands[-1])
    rep = json.loads(p1.read_text(encoding="utf-8"))

    trades = load_trades(root)
    summary = trade_summary(trades)
    crit = success_criterion(summary)
    decomp = variance_decomposition(rep)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = build_markdown(cfg, rep, trades, summary, crit, decomp)
    mp = outdir / f"phase3_report_{stamp}.md"
    jp = outdir / f"phase3_report_{stamp}.json"
    mp.write_text(md, encoding="utf-8")
    jp.write_text(json.dumps({"phase1_source": p1.name, "trades": summary,
                              "success_criterion": crit,
                              "variance_decomposition": decomp},
                             indent=2, default=str), encoding="utf-8")
    print(f"[phase3] {len(trades)} trade records "
          f"({summary.get('n_evidential', 0)} evidential)", flush=True)
    print(f"[phase3] success criterion: {'MET' if crit['met'] else 'NOT MET'}",
          flush=True)
    print(f"[phase3] wrote {mp.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
