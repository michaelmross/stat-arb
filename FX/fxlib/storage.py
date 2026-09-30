"""Append-only tick / event / trade storage.

Ticks are written as immutable parquet *part files* under

    data/ticks/YYYY-MM-DD/part-000000.parquet

rather than one open ParquetWriter per day. A ParquetWriter that dies without
closing leaves a footer-less, unreadable file -- i.e. a crash silently destroys
the day. Part files are complete the moment they land, so a crash costs at most
one unflushed buffer, and the loss is visible as a gap in the events log.

Sidecar JSONL streams (never silently dropped, per spec section 0.2):
    data/events/YYYY-MM-DD.jsonl   stream gaps, reconnects, halts, kill switches
    data/trades/YYYY-MM-DD.jsonl   paper trade records (Phase 2)
    data/runs/<run_id>.json        run manifest incl. config hash
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TICK_SCHEMA = pa.schema([
    ("instrument", pa.string()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("oanda_ts", pa.timestamp("ns", tz="UTC")),   # venue time: cross-pair ordering ONLY
    ("recv_wall", pa.timestamp("ns", tz="UTC")),  # local NTP wall clock, for calendars
    ("recv_mono", pa.float64()),                  # local monotonic seconds, for ages
    ("tradeable", pa.bool_()),                    # OANDA's own tradeable flag
    ("session", pa.string()),                     # overlap | other | rollover | weekend
    ("event_flag", pa.string()),                  # NFP, FOMC, ... or ""
    ("source", pa.string()),                      # "oanda" | "synthetic"
    ("run_id", pa.string()),
])


def _utc(ts: dt.datetime | None) -> dt.datetime | None:
    if ts is None:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)


class TickWriter:
    """Buffers ticks and flushes complete parquet parts. Thread-safe."""

    def __init__(self, cfg: dict, run_id: str, source: str, root: str | Path | None = None):
        self.root = Path(root) if root else Path(cfg["_path"]).parent / cfg["storage"]["root"]
        self.part_rows = int(cfg["storage"]["part_rows"])
        self.part_flush_s = float(cfg["storage"]["part_flush_s"])
        self.run_id = run_id
        self.source = source
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self.rows_written = 0
        (self.root / "ticks").mkdir(parents=True, exist_ok=True)
        (self.root / "events").mkdir(parents=True, exist_ok=True)
        (self.root / "trades").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    def append(self, rec: dict) -> None:
        rec.setdefault("source", self.source)
        rec.setdefault("run_id", self.run_id)
        with self._lock:
            self._buf.append(rec)
            due = (len(self._buf) >= self.part_rows
                   or time.monotonic() - self._last_flush >= self.part_flush_s)
        if due:
            self.flush()

    def flush(self) -> int:
        with self._lock:
            buf, self._buf = self._buf, []
            self._last_flush = time.monotonic()
        if not buf:
            return 0
        # Split by UTC calendar day of the local receive clock so a part file
        # never straddles two daily directories.
        by_day: dict[str, list[dict]] = {}
        for r in buf:
            by_day.setdefault(_utc(r["recv_wall"]).strftime("%Y-%m-%d"), []).append(r)
        for day, rows in by_day.items():
            d = self.root / "ticks" / day
            d.mkdir(parents=True, exist_ok=True)
            n = len(list(d.glob("part-*.parquet")))
            tmp = d / f".part-{n:06d}.tmp"
            final = d / f"part-{n:06d}.parquet"
            cols = {f.name: [r.get(f.name) for r in rows] for f in TICK_SCHEMA}
            pq.write_table(pa.table(cols, schema=TICK_SCHEMA), tmp, compression="zstd")
            os.replace(tmp, final)   # atomic: readers never see a partial file
            self.rows_written += len(rows)
        return len(buf)

    def close(self) -> None:
        self.flush()


class JsonlLog:
    """Line-buffered, fsync-on-write append log. Used for events and trades."""

    def __init__(self, root: Path, kind: str):
        self.dir = Path(root) / kind
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, rec: dict) -> dict:
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **rec}
        path = self.dir / f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}.jsonl"
        line = json.dumps(rec, default=str)
        with self._lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return rec


class SingleWriterLock:
    """Refuse to run two collectors against one data root.

    Two concurrent collectors record the SAME market ticks under different
    run_ids. Nothing errors, the parquet files look fine, and the census then
    double-counts every observation -- inflating n, shrinking standard errors
    and silently corrupting the result. It is the kind of corruption that is
    invisible until someone recomputes it by hand.

    An OS-level file lock is used rather than a pid file, because the OS
    releases it when the process dies however it dies. A crashed collector
    therefore leaves no stale lock to clear by hand at 3am.
    """

    def __init__(self, root: Path, name: str = "collector.lock"):
        self.path = Path(root) / name
        self.info_path = Path(root) / (name + ".info")
        self._fh = None

    def acquire(self, run_id: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Byte 0, always. msvcrt.locking() locks a range starting at the file's
        # CURRENT position, so without an explicit seek two processes can lock
        # different bytes, both "succeed", and only collide later on a write --
        # which is how the first version of this appeared to work while
        # actually being broken.
        self._fh = open(self.path, "r+b" if self.path.exists() else "w+b")
        self._fh.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            held = ""
            # Holder details live in a SEPARATE file, so reading them can never
            # touch the locked byte.
            try:
                held = self.info_path.read_text(encoding="utf-8").strip()[:200]
            except OSError:
                pass
            raise SystemExit(
                f"Another collector is already writing to {self.path.parent}.\n"
                f"  lock holder: {held or 'unknown'}\n"
                "Two collectors would record the same ticks twice and corrupt "
                "the census -- silently, since both files look valid. Stop the "
                "running one first, or use --data-root to write elsewhere.")
        try:
            self.info_path.write_text(
                f"{run_id} pid={os.getpid()} "
                f"{dt.datetime.now(dt.timezone.utc).isoformat()}\n",
                encoding="utf-8")
        except OSError:
            pass

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def write_manifest(root: Path, run_id: str, cfg: dict, extra: dict) -> Path:
    p = Path(root) / "runs" / f"{run_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "run_id": run_id,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": cfg.get("_path"),
        "config_hash": cfg.get("_hash"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        **extra,
    }
    p.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return p


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
