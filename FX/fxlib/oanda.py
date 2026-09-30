"""A thin hand-rolled OANDA v20 client over the two endpoints this experiment needs.

Deliberately not oandapyV20: three instruments and two endpoints do not justify a
dependency, and every byte on the wire should be auditable from this file (spec
section 3).

Endpoints used:
  GET  https://stream-fxpractice.oanda.com/v3/accounts/{id}/pricing/stream
  POST https://api-fxpractice.oanda.com/v3/accounts/{id}/orders
  GET  https://api-fxpractice.oanda.com/v3/accounts/{id}/summary | openPositions | openTrades
"""
from __future__ import annotations

import datetime as dt
import json
import time

import httpx


class OandaError(RuntimeError):
    pass


class StreamStalled(OandaError):
    """No PRICE or HEARTBEAT within the configured timeout."""


def parse_rfc3339(s: str) -> dt.datetime:
    """OANDA RFC3339 timestamps carry 9 fractional digits; fromisoformat handles
    at most 6 before 3.11 and is picky about 'Z'. Normalize both."""
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, rest = s.split(".", 1)
        frac, _, tz = rest.partition("+")
        tz = "+" + tz if tz else ""
        if not tz and "-" in rest[1:]:
            frac, _, tzr = rest.partition("-")
            tz = "-" + tzr
        s = f"{head}.{frac[:6]:0<6}{tz or '+00:00'}"
    d = dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


class OandaClient:
    def __init__(self, cfg: dict, cred: dict, timeout: float = 10.0):
        o = cfg["oanda"]
        if o["environment"] != "practice":
            raise OandaError("only the practice environment is permitted")
        self.account_id = cred["account_id"]
        self.api_base = f"https://{o['api_host']}"
        self.stream_base = f"https://{o['stream_host']}"
        self.headers = {
            "Authorization": f"Bearer {cred['api_token']}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }
        self._c = httpx.Client(headers=self.headers, timeout=timeout)

    # ---- REST -----------------------------------------------------------
    def _get(self, path: str) -> dict:
        r = self._c.get(f"{self.api_base}/v3/accounts/{self.account_id}{path}")
        if r.status_code >= 400:
            raise OandaError(f"GET {path} -> {r.status_code}: {r.text[:400]}")
        return r.json()

    def summary(self) -> dict:
        return self._get("/summary")["account"]

    def open_positions(self) -> list:
        return self._get("/openPositions").get("positions", [])

    def open_trades(self) -> list:
        return self._get("/openTrades").get("trades", [])

    def pricing(self, instruments: list) -> dict:
        q = ",".join(instruments)
        return self._get(f"/pricing?instruments={q}")

    def market_order(self, instrument: str, units: int,
                     client_tag: str | None = None) -> dict:
        """Signed units: positive buys the base currency, negative sells it.
        Fills come back synchronously in the transaction response."""
        body = {"order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }}
        if client_tag:
            body["order"]["clientExtensions"] = {"tag": client_tag,
                                                 "id": client_tag[:127]}
        r = self._c.post(
            f"{self.api_base}/v3/accounts/{self.account_id}/orders", json=body)
        if r.status_code >= 400:
            raise OandaError(f"POST /orders -> {r.status_code}: {r.text[:600]}")
        return r.json()

    def close(self) -> None:
        self._c.close()

    # ---- streaming ------------------------------------------------------
    def stream_prices(self, instruments: list, heartbeat_timeout_s: float):
        """Yields ('PRICE', dict) / ('HEARTBEAT', dict) with a local monotonic
        receive stamp attached as '_recv_mono' and wall clock as '_recv_wall'.

        Raises StreamStalled if nothing arrives within heartbeat_timeout_s. The
        caller owns reconnect policy so every gap is logged, never papered over.
        """
        url = (f"{self.stream_base}/v3/accounts/{self.account_id}/pricing/stream"
               f"?instruments={','.join(instruments)}")
        timeout = httpx.Timeout(connect=10.0, read=heartbeat_timeout_s,
                                write=10.0, pool=10.0)
        with httpx.Client(headers=self.headers, timeout=timeout) as c:
            with c.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise OandaError(
                        f"stream -> {resp.status_code}: {resp.text[:400]}")
                try:
                    for line in resp.iter_lines():
                        recv_mono = time.monotonic()
                        recv_wall = dt.datetime.now(dt.timezone.utc)
                        if not line.strip():
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg["_recv_mono"] = recv_mono
                        msg["_recv_wall"] = recv_wall
                        yield msg.get("type", "UNKNOWN"), msg
                except httpx.ReadTimeout as e:
                    raise StreamStalled(
                        f"no data for {heartbeat_timeout_s}s") from e


def price_to_quote_fields(msg: dict):
    """Extract (bid, ask, oanda_ts, tradeable) from a v20 PRICE message.

    v20 gives a ladder of bids/asks; index 0 is top of book, which is what a
    market order at this size touches. Returns None if either side is empty
    (OANDA sends one-sided books around the weekend gate)."""
    bids, asks = msg.get("bids") or [], msg.get("asks") or []
    if not bids or not asks:
        return None
    return (
        float(bids[0]["price"]),
        float(asks[0]["price"]),
        parse_rfc3339(msg["time"]),
        bool(msg.get("tradeable", True)),
    )
