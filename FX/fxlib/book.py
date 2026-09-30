"""The single in-memory quote book shared by the tick logger and the signal engine.

There is exactly one book per process and it only ever moves forward: a residual
computed at time t uses only quotes already received at t. The decision record is
built before any order call (spec section 3, "No look-ahead").

Clock discipline (spec section 3):
  * quote AGES use the local monotonic clock only;
  * cross-pair event ORDERING uses OANDA timestamps;
  * calendars/sessions use the NTP-synced wall clock.
These three are never mixed.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, asdict

PIP = 1e-4          # EUR/GBP pip
LN2 = math.log(2.0)


@dataclass
class Quote:
    bid: float
    ask: float
    oanda_ts: dt.datetime
    recv_mono: float
    recv_wall: dt.datetime
    tradeable: bool = True

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


def invert_quote(q: "Quote") -> "Quote":
    """Re-quote X_Y as Y_X.

    The venue lists USD_CZK, but the triangle EUR_CZK = EUR_USD / CZK_USD needs
    the CZK_USD orientation. Bid and ask SWAP as well as reciprocate: selling one
    CZK yields 1/ask_USDCZK dollars, buying one costs 1/bid_USDCZK. Getting this
    backwards would silently invert the spread and make every cycle look
    profitable by exactly the round trip.

    Relative spread is preserved -- (a-b)/(ab) against a mid of ~1/mid is the
    same fraction -- so the inverted leg stays economically identical, which is
    what makes it legitimate to trade through.
    """
    return Quote(bid=1.0 / q.ask, ask=1.0 / q.bid, oanda_ts=q.oanda_ts,
                 recv_mono=q.recv_mono, recv_wall=q.recv_wall,
                 tradeable=q.tradeable)


@dataclass
class Snapshot:
    """Everything the signal engine saw at one instant. Serialized verbatim into
    the trade log so a decision can be re-derived from the record alone."""
    ts_wall: dt.datetime
    ts_mono: float
    a_bid: float
    a_ask: float
    b_bid: float
    b_ask: float
    c_bid: float
    c_ask: float
    age_a_ms: float
    age_b_ms: float
    age_c_ms: float
    max_age_ms: float
    synchronized: bool
    eps: float                 # ln A - ln B - ln C     (spec 1.1)
    resid: float               # ln C - ln(A/B) = -eps  (spec 1.2)
    resid_pips: float
    sigma: float
    sigma_pips: float
    mu: float
    z: float
    z_raw: float
    n_obs: int
    r1: float                  # USD->EUR->GBP->USD executable cycle return
    r2: float                  # USD->GBP->EUR->USD executable cycle return
    cycle_cost_bp: float
    last_updated: str          # A, B or C -- ordered by OANDA timestamp
    majors_led: bool
    c_spread_pips: float
    venue_spread_ms: float   # max-min OANDA timestamp across the 3 legs

    def as_dict(self) -> dict:
        return asdict(self)


class EwmaMoments:
    """Time-decayed EWMA mean and variance. Half-life is in seconds, so the decay
    stays correct under irregular tick arrival -- which is the normal case here."""

    def __init__(self, halflife_s: float):
        self.hl = float(halflife_s)
        self.mean = 0.0
        self.var = 0.0
        self.n = 0
        self._last_t = None

    def update(self, x: float, t_mono: float) -> None:
        if self._last_t is None:
            self.mean, self.var, self.n, self._last_t = x, 0.0, 1, t_mono
            return
        dt_s = max(0.0, t_mono - self._last_t)
        self._last_t = t_mono
        alpha = 1.0 - math.exp(-LN2 * dt_s / self.hl) if self.hl > 0 else 1.0
        alpha = min(max(alpha, 1e-9), 1.0)
        d = x - self.mean
        self.mean += alpha * d
        self.var = (1.0 - alpha) * (self.var + alpha * d * d)
        self.n += 1

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.var, 0.0))


class TriangleBook:
    def __init__(self, cfg: dict, a: str, b: str, c: str):
        self.a, self.b, self.c = a, b, c
        s = cfg["signal"]
        self.tau_ms = float(s["tau_ms"])
        self.demean = bool(s.get("demean", True))
        self.warmup = int(s["sigma_warmup_n"])
        self.majors_window_ms = float(s["majors_led_window_ms"])
        # Legs the venue quotes the opposite way round to the X_Z / Y_Z form the
        # triangle identity assumes. Inverted on ingest, so every calculation
        # downstream -- residual, R1/R2, ages -- is untouched by the orientation.
        o = cfg.get("oanda", {})
        self.inverted = set(o.get("inverted_legs", []) or [])
        # Pip size of the CROSS, for reporting only. 1e-4 for most pairs, but a
        # JPY- or CZK-quoted cross can differ; take it from the venue metadata.
        self.pip = float(o.get("pip_size", PIP))
        self.q = {}
        self.moments = EwmaMoments(float(s["ewma_halflife_s"]))

    def update(self, instrument: str, q: Quote) -> None:
        if instrument in self.inverted:
            q = invert_quote(q)
        prev = self.q.get(instrument)
        # OANDA can redeliver or reorder across a reconnect; the book never rewinds.
        if prev is not None and q.oanda_ts < prev.oanda_ts:
            return
        self.q[instrument] = q

    def ready(self) -> bool:
        return all(k in self.q for k in (self.a, self.b, self.c))

    def ages_ms(self, now_mono: float) -> dict:
        return {k: (now_mono - v.recv_mono) * 1e3 for k, v in self.q.items()}

    def snapshot(self, now_mono: float, now_wall: dt.datetime,
                 update_moments: bool = True):
        if not self.ready():
            return None
        A, B, C = self.q[self.a], self.q[self.b], self.q[self.c]
        ages = self.ages_ms(now_mono)
        max_age = max(ages[self.a], ages[self.b], ages[self.c])
        sync = max_age < self.tau_ms

        eps = math.log(A.mid) - math.log(B.mid) - math.log(C.mid)
        resid = -eps

        # Only synchronized observations shape sigma; stale quotes would inflate
        # it with pure asynchrony noise rather than genuine dislocation.
        if update_moments and sync:
            self.moments.update(resid, now_mono)
        sigma = self.moments.sd
        mu = self.moments.mean if self.demean else 0.0
        n = self.moments.n

        if sigma > 0 and n >= self.warmup:
            z = (resid - mu) / sigma
            z_raw = resid / sigma
        else:
            z = z_raw = float("nan")

        r1 = (1.0 / A.ask) * C.bid * B.bid - 1.0
        r2 = (1.0 / B.ask) * (1.0 / C.ask) * A.bid - 1.0
        cost_bp = 1e4 * 0.5 * (A.spread / A.mid + B.spread / B.mid + C.spread / C.mid)

        order = sorted((self.a, self.b, self.c), key=lambda k: self.q[k].oanda_ts)
        last = order[-1]
        # Receive-clock ages can lie. If the process is descheduled, queued ticks
        # drain in microseconds and three legs that genuinely arrived hundreds of
        # ms apart all get near-identical recv_mono -- so they pass the tau test
        # while holding prices from different moments. The venue timestamps do
        # not have that failure mode, so carry their spread as a cross-check.
        # (Venue time is used here only to VALIDATE synchronization, never for
        # ages or signal timing -- see the clock discipline note above.)
        venue_spread_ms = (self.q[order[-1]].oanda_ts
                           - self.q[order[0]].oanda_ts).total_seconds() * 1e3
        last_label = {self.a: "A", self.b: "B", self.c: "C"}[last]
        majors_led = (last in (self.a, self.b)
                      and min(ages[self.a], ages[self.b]) <= self.majors_window_ms)

        return Snapshot(
            ts_wall=now_wall, ts_mono=now_mono,
            a_bid=A.bid, a_ask=A.ask, b_bid=B.bid, b_ask=B.ask,
            c_bid=C.bid, c_ask=C.ask,
            age_a_ms=ages[self.a], age_b_ms=ages[self.b], age_c_ms=ages[self.c],
            max_age_ms=max_age, synchronized=sync,
            eps=eps, resid=resid, resid_pips=resid * C.mid / self.pip,
            sigma=sigma, sigma_pips=sigma * C.mid / self.pip, mu=mu,
            z=z, z_raw=z_raw, n_obs=n,
            r1=r1, r2=r2, cycle_cost_bp=cost_bp,
            last_updated=last_label, majors_led=majors_led,
            c_spread_pips=C.spread / self.pip,
            venue_spread_ms=venue_spread_ms,
        )


def to_pips(log_resid: float, c_mid: float) -> float:
    """Convert a log-residual to EUR/GBP pips."""
    return log_resid * c_mid / PIP
