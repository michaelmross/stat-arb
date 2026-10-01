# Companion paper: triangular FX residuals on a retail feed

`main.tex` is the draft. It compiles with any standard LaTeX (article class,
amsmath, booktabs, graphicx, hyperref, microtype) — none is installed on the
machine it was written on, so it has been linted (`lint_tex.py`) but not yet
compiled. Overleaf works.

## Regenerating every number and figure

Run from `fx_statarb/`, in this order. The census windows are passed
explicitly: **never run the EUR/GBP census without `--dates`**, because the
tooling defaults to every collected session and would pool the 27–28 Aug
calibration runs (the error Section 5.3 of the paper describes).

```bash
# 1. census runs (each ~1.5 min)
python phase1_analyze.py --no-gate --dates 2026-08-31 2026-09-01 2026-09-02 2026-09-03 2026-09-04 2026-09-07 2026-09-08 2026-09-09 2026-09-10 2026-09-11 --out paper/runs/eurgbp_main
python phase1_analyze.py --no-gate --out paper/runs/eurgbp_all            # the pooled run, kept as the documented error
python phase1_analyze.py --no-gate --config config_audnzd.toml --dates 2026-09-14 2026-09-15 2026-09-16 2026-09-17 2026-09-18 2026-09-21 2026-09-22 2026-09-23 2026-09-24 2026-09-25 --out paper/runs/audnzd
python phase1_analyze.py --no-gate --config config_eurczk.toml --dates 2026-09-14 2026-09-15 2026-09-16 2026-09-17 2026-09-18 2026-09-21 2026-09-22 2026-09-23 2026-09-24 2026-09-25 --out paper/runs/eurczk

# 2. variance decomposition for the declared EUR/GBP window
python phase3_report.py --phase1 paper/runs/eurgbp_main/phase1_<stamp>.json --out paper/runs/p3

# 3. everything the paper cites that the census does not emit (~10 min)
python paper/paper_numbers.py          # -> paper/paper_numbers.json, paper/fig_data/

# 4. figures, manifest, lint
python paper/make_figures.py           # -> paper/fig_*.pdf   (FIG_PREVIEW=1 also writes PNGs)
python paper/make_manifest.py          # -> paper/data_manifest.json
python paper/lint_tex.py
```

`paper_numbers.py` and `make_figures.py` read the newest
`paper/runs/<name>/phase1_*.json` for each of `eurgbp_main`, `eurgbp_all`,
`audnzd` and `eurczk`. Without the raw ticks, `make_figures.py` still rebuilds
three of the four figures; `fig_reboot` needs `data/ticks/` and is skipped.

## Before release

- `[TODO]` markers in `main.tex`: repository path, Zenodo DOI, OANDA's terms on
  redistributing practice-feed data, and an independent verification pass.
- `preview/` holds PNG renders for eyeballing only; do not archive it.
- `../out/g1_verdict.json` still records the pooled run's PASS. The paper's
  verdict is FAIL on the declared window.
