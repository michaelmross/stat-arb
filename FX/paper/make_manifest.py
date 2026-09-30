"""Fingerprint every tick file the paper rests on, without redistributing it.

    python paper/make_manifest.py      # writes paper/data_manifest.json

OANDA's price data are not ours to republish. In their place the archive carries
SHA-256 and row counts per part file, grouped by triangle and session, so anyone
re-collecting or holding the originals can confirm they have the same bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WINDOWS = {
    "eurgbp": ("data", ["2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03",
                        "2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09",
                        "2026-09-10", "2026-09-11"]),
    "audnzd": ("data_audnzd", None),
    "eurczk": ("data_eurczk", None),
}
PHASE1B = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
           "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = {"note": "OANDA practice-feed ticks are not redistributed. Each entry is one "
                   "append-only parquet part file: SHA-256, row count, run_ids.",
           "triangles": {}}
    for tri, (root, dates) in WINDOWS.items():
        dates = dates or PHASE1B
        days = {}
        for d in dates:
            files = []
            for p in sorted((ROOT / root / "ticks" / d).glob("*.parquet")):
                df = pl.read_parquet(p, columns=["run_id"])
                files.append({"file": p.name, "sha256": sha256(p), "rows": len(df),
                              "run_ids": sorted(df["run_id"].unique().to_list())})
            days[d] = {"rows": sum(f["rows"] for f in files), "files": files}
        out["triangles"][tri] = {"root": root, "sessions": len(days),
                                 "rows": sum(v["rows"] for v in days.values()), "days": days}
    (HERE / "data_manifest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for tri, v in out["triangles"].items():
        print(f"{tri}: {v['sessions']} sessions, {v['rows']:,} rows")


if __name__ == "__main__":
    main()
