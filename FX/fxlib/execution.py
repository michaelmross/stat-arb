"""Order execution for the Phase 2 arm, behind one interface so the trading
logic is identical whether fills come from OANDA or from the local simulator.

The simulator exists so the entry/exit/kill-switch logic can be exercised end to
end without touching an account. It is NOT a fill model to draw conclusions
from: it fills at the touch, instantly, which is precisely the optimism the spec
warns about in section 2 (Phase 2, cost model caveat).
"""
from __future__ import annotations

import datetime as dt

from .book import PIP
from .oanda import parse_rfc3339


class Fill:
    __slots__ = ("price", "units", "time", "trade_id", "simulated", "raw")

    def __init__(self, price: float, units: int, time: dt.datetime,
                 trade_id: str | None, simulated: bool, raw: dict | None = None):
        self.price = price
        self.units = units
        self.time = time
        self.trade_id = trade_id
        self.simulated = simulated
        self.raw = raw or {}

    def as_dict(self) -> dict:
        return {"price": self.price, "units": self.units,
                "time": self.time.isoformat(), "trade_id": self.trade_id,
                "simulated": self.simulated}


class LiveExecutor:
    """Real market orders against the OANDA practice account."""

    simulated = False

    def __init__(self, client, instrument: str):
        self.client = client
        self.instrument = instrument

    def market(self, units: int, tag: str) -> Fill:
        resp = self.client.market_order(self.instrument, units, client_tag=tag)
        tx = resp.get("orderFillTransaction")
        if tx is None:
            reason = (resp.get("orderCancelTransaction", {}).get("reason")
                      or resp.get("errorMessage") or "unknown")
            raise RuntimeError("order not filled: " + str(reason))
        opened = tx.get("tradeOpened") or {}
        closed = (tx.get("tradesClosed") or [{}])[0]
        return Fill(price=float(tx["price"]), units=int(float(tx["units"])),
                    time=parse_rfc3339(tx["time"]),
                    trade_id=opened.get("tradeID") or closed.get("tradeID"),
                    simulated=False, raw=tx)

    def open_units(self) -> int:
        for p in self.client.open_positions():
            if p.get("instrument") == self.instrument:
                return int(round(float(p["long"]["units"])
                                 + float(p["short"]["units"])))
        return 0


class SimExecutor:
    """Local fill simulator: crosses the touch of the current book instantly.

    latency_pips is subtracted from the trader in both directions so the dry run
    at least acknowledges that a fill is not free; it is a placeholder, not a
    measurement.
    """

    simulated = True

    def __init__(self, book, instrument: str, latency_pips: float = 0.0):
        self.book = book
        self.instrument = instrument
        self.latency_pips = latency_pips
        self._units = 0
        self._seq = 0

    def market(self, units: int, tag: str) -> Fill:
        q = self.book.q.get(self.instrument)
        if q is None:
            raise RuntimeError("no quote for " + self.instrument)
        adverse = self.latency_pips * PIP
        price = (q.ask + adverse) if units > 0 else (q.bid - adverse)
        self._units += units
        self._seq += 1
        return Fill(price=price, units=units, time=q.recv_wall,
                    trade_id="sim-%d" % self._seq, simulated=True)

    def open_units(self) -> int:
        return self._units
