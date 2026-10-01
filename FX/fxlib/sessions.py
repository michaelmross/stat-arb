"""Session windows, rollover halts, and scheduled-release flagging.

All decisions are made in the configured exchange timezone (America/New_York).
Wall-clock UTC is the input; the monotonic clock is never used here (it has no
calendar meaning) -- see fxlib/book.py for the monotonic-clock age logic.
"""
from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path
from zoneinfo import ZoneInfo

_WD = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
       "Friday": 4, "Saturday": 5, "Sunday": 6}


def _hhmm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


class SessionClock:
    def __init__(self, cfg: dict):
        s = cfg["sessions"]
        self.tz = ZoneInfo(s["timezone"])
        self.roll_start = _hhmm(s["rollover_halt_start"])
        self.roll_end = _hhmm(s["rollover_halt_end"])
        self.close_wd = _WD[s["week_close_weekday"]]
        self.close_t = _hhmm(s["week_close_time"])
        self.open_wd = _WD[s["week_open_weekday"]]
        self.open_t = _hhmm(s["week_open_time"])
        self.ov_start = _hhmm(s["overlap_start"])
        self.ov_end = _hhmm(s["overlap_end"])

    def local(self, when_utc: dt.datetime) -> dt.datetime:
        if when_utc.tzinfo is None:
            when_utc = when_utc.replace(tzinfo=dt.timezone.utc)
        return when_utc.astimezone(self.tz)

    def in_rollover_halt(self, when_utc: dt.datetime) -> bool:
        t = self.local(when_utc).time()
        return self.roll_start <= t < self.roll_end

    def in_weekend_halt(self, when_utc: dt.datetime) -> bool:
        """True from Friday close through Sunday open."""
        loc = self.local(when_utc)
        wd, t = loc.weekday(), loc.time()
        if wd == self.close_wd and t >= self.close_t:
            return True
        if wd == self.open_wd and t < self.open_t:
            return True
        # Saturday, plus any weekday strictly between close and open in the week cycle.
        if wd == 5:
            return True
        if self.close_wd < wd < self.open_wd:
            return True
        return False

    def tradeable(self, when_utc: dt.datetime) -> bool:
        return not (self.in_rollover_halt(when_utc) or self.in_weekend_halt(when_utc))

    def in_overlap(self, when_utc: dt.datetime) -> bool:
        """London/NY overlap -- an analysis label only; collection never stops."""
        loc = self.local(when_utc)
        return loc.weekday() < 5 and self.ov_start <= loc.time() < self.ov_end

    def session_label(self, when_utc: dt.datetime) -> str:
        if self.in_weekend_halt(when_utc):
            return "weekend"
        if self.in_rollover_halt(when_utc):
            return "rollover"
        if self.in_overlap(when_utc):
            return "overlap"
        return "other"


class EventCalendar:
    """Flags scheduled macro releases in the tick log.

    Two sources, both explicit -- nothing is guessed:
      * a deterministic rule for US Non-Farm Payrolls (first Friday, 08:30 ET),
        which is a published, rule-based schedule;
      * a user-maintained list in events.toml for everything whose dates are
        announced rather than derivable (CPI, FOMC, ECB, BoE).

    Flagged events are NEVER filtered out of the census (spec section 3); the flag
    exists so the analysis can condition on regime.
    """

    def __init__(self, cfg: dict, path: str | Path | None = None):
        self.tz = ZoneInfo(cfg["sessions"]["timezone"])
        self.window_min = 15
        self.entries: list[tuple[dt.datetime, dt.datetime, str]] = []
        p = Path(path) if path else Path(__file__).resolve().parent.parent / "events.toml"
        if p.exists():
            with open(p, "rb") as fh:
                data = tomllib.load(fh)
            self.window_min = int(data.get("window_minutes", 15))
            for e in data.get("event", []):
                when = dt.datetime.fromisoformat(e["when_local"]).replace(tzinfo=self.tz)
                w = dt.timedelta(minutes=int(e.get("window_minutes", self.window_min)))
                self.entries.append((when - w, when + w, e["name"]))

    def _is_nfp(self, loc: dt.datetime) -> bool:
        if loc.weekday() != 4 or loc.day > 7:
            return False
        release = loc.replace(hour=8, minute=30, second=0, microsecond=0)
        return abs((loc - release).total_seconds()) <= self.window_min * 60

    def flag(self, when_utc: dt.datetime) -> str:
        """Returns a comma-joined list of active event names, or "" if none."""
        loc = when_utc.astimezone(self.tz)
        names = []
        if self._is_nfp(loc):
            names.append("NFP")
        for lo, hi, name in self.entries:
            if lo <= loc <= hi:
                names.append(name)
        return ",".join(names)
