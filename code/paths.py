"""Single source of truth for repository layout.

Every path is derived from this file's own location, so scripts work from
any working directory: `python code/scan.py` and `cd code && python
scan.py` resolve identically.

    <root>/
      code/       every .py (flat -- no package, so plain imports)
      data/       archived JSON artifacts  [committed]
      figures/    generated figure PDFs    [committed]
      prices/     ETF daily CSVs           [NOT committed]
      futures/    futures daily CSVs       [NOT committed]

PRICE DATA IS NOT IN THE REPOSITORY. Vendor terms do not allow
redistributing Tiingo or Yahoo daily bars, so `prices/` and `futures/`
are gitignored and ship empty. Everything that consumes only JSON or
synthetic ground truth runs on a fresh clone; everything that needs bars
calls `require_prices()` and exits with the command that rebuilds them,
rather than dying on a FileNotFoundError three frames deep.

Override any location with an environment variable of the same name,
e.g. STATARB_PRICES=/mnt/bulk/prices.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _dir(name: str, default: str) -> Path:
    return Path(os.environ.get(f"STATARB_{name}", ROOT / default))


CODE = _dir("CODE", "code")
DATA = _dir("DATA", "data")            # JSON artifacts
FIGURES = _dir("FIGURES", "figures")
PRICES = _dir("PRICES", "prices")      # ETF CSVs, not committed
FUTURES = _dir("FUTURES", "futures")   # futures CSVs, not committed

TOKEN_FILE = ROOT / ".tiingo_token"

# archived artifacts, by name, so callers never hardcode a directory
SCAN_V1 = DATA / "scan_full.json"
SCAN_V2 = DATA / "scan_v2.json"
CENSUS = DATA / "fractional_census_results.json"
KALMAN = DATA / "kalman_results.json"
KALMAN_COSTS = DATA / "kalman_cost_attribution.json"
CRISIS = DATA / "crisis_2020.json"
FUTURES_RESULTS = DATA / "futures_results.json"
MANIFEST = DATA / "data_manifest.json"


def data(name: str) -> Path:
    """Resolve a JSON artifact by bare filename."""
    return DATA / Path(name).name


def require_prices(kind: str = "etf", min_files: int = 1,
                   directory=None) -> Path:
    """Return the price directory, or exit with how to populate it.

    `kind` is 'etf' or 'futures'. Pass `directory` when the caller was
    given an explicit path -- otherwise a caller pointed at its own data
    (a test fixture, say) would be blocked by the state of the default
    location, which it never intended to use. Exits non-zero with an
    actionable message rather than raising, because the overwhelmingly
    common cause is a fresh clone that has never run a fetch script.
    """
    default, cmd = ((PRICES, "python code/fetch_data.py --source tiingo "
                             "--start 2010-01-01")
                    if kind == "etf" else
                    (FUTURES, "python code/fetch_futures.py "
                              "--start 2010-01-01"))
    d = Path(directory) if directory is not None else default
    n = len(list(d.glob("*.csv"))) if d.exists() else 0
    if n >= min_files:
        return d
    if d != default:
        # caller supplied its own directory: report that, not the default
        sys.exit(f"\nNo CSVs in {d} ({n} found, need {min_files}).\n")
    if True:
        sys.exit(
            f"\nNo {kind} price data in {d}  ({n} CSVs found, need "
            f"{min_files}).\n\n"
            f"Daily bars are not redistributable, so they are not in the\n"
            f"repository. Rebuild them with:\n\n    {cmd}\n\n"
            + ("The Tiingo fetch needs a free token in .tiingo_token or\n"
               "$TIINGO_API_KEY.\n" if kind == "etf" else
               "The Yahoo fetch needs no key.\n")
            + "\nEverything that consumes only the archived JSON in "
              f"{DATA.name}/ runs without this.\n")
    return d


def describe() -> None:
    print(f"root      {ROOT}")
    for label, p in [("code", CODE), ("data (json)", DATA),
                     ("figures", FIGURES),
                     ("prices (etf)", PRICES), ("futures", FUTURES)]:
        if p.is_dir():
            n_json = len(list(p.glob("*.json")))
            n_csv = len(list(p.glob("*.csv")))
            n_py = len(list(p.glob("*.py")))
            n_pdf = len(list(p.glob("*.pdf")))
            bits = [f"{n}{s}" for n, s in
                    [(n_py, " py"), (n_json, " json"), (n_csv, " csv"),
                     (n_pdf, " pdf")] if n]
            print(f"  {label:14s} {p}  ({', '.join(bits) or 'empty'})")
        else:
            print(f"  {label:14s} {p}  (MISSING)")


if __name__ == "__main__":
    describe()
