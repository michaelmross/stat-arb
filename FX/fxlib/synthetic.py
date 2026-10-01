"""A synthetic three-pair tick source, message-compatible with the OANDA stream.

Purpose: exercise the collector, the signal engine, the analysis and the paper
trading arm end to end WITHOUT credentials and without waiting two weeks. It is
pipeline validation only. Every row it produces is stamped source="synthetic",
and the analysis refuses to mix synthetic and live data or to evaluate a gate on
synthetic input.

Two modes, corresponding to the two possible answers to spec section 1.3:

  derived      EUR/GBP is constructed internally as A/B plus a spread, then
               rounded to the venue's 5-decimal grid. The mid-residual is then
               nothing but rounding noise -- sigma near 0.03 pip. This is the
               outcome the spec expects and the one that terminates at Phase 1.

  independent  EUR/GBP carries its own OU deviation from A/B with a configurable
               stationary sigma and half-life. This is the counterfactual, used
               to prove the analysis and trading code would actually detect and
               act on a real residual if one existed.
"""
from __future__ import annotations

import datetime as dt
import math
import time

import numpy as np

# OANDA display precision: both majors and the cross quote 5 decimals.
DECIMALS = {"EUR_USD": 5, "GBP_USD": 5, "EUR_GBP": 5}
# Typical practice-feed spreads, in price units.
SPREADS = {"EUR_USD": 0.00008, "GBP_USD": 0.00011, "EUR_GBP": 0.00013}
# Mean tick arrivals per second, per pair.
RATES = {"EUR_USD": 3.0, "GBP_USD": 2.5, "EUR_GBP": 1.5}
# Annualized vol of the majors.
ANNUAL_VOL = {"EUR_USD": 0.07, "GBP_USD": 0.08}
SECONDS_PER_YEAR = 252 * 86400


class SyntheticFeed:
    def __init__(self, cfg: dict, mode: str = "derived", seed: int = 0,
                 sigma_pips: float = 0.25, halflife_s: float = 4.0,
                 latency_ms: float = 120.0, latency_sd_ms: float = 40.0,
                 speed: float = 1.0, gap_per_hour: float = 0.0,
                 start_wall=None):
        if mode not in ("derived", "independent"):
            raise ValueError("mode must be 'derived' or 'independent'")
        self.a, self.b, self.c = (cfg["oanda"]["instrument_a"],
                                  cfg["oanda"]["instrument_b"],
                                  cfg["oanda"]["instrument_c"])
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.halflife = halflife_s
        self.theta = math.log(2.0) / halflife_s
        # Stationary sd of the OU deviation, converted from EUR/GBP pips to log units.
        self.sigma_x = sigma_pips * 1e-4 / 0.86
        self.latency_ms, self.latency_sd_ms = latency_ms, latency_sd_ms
        self.speed = max(speed, 1e-9)
        self.gap_hazard = gap_per_hour / 3600.0
        self.mid = {self.a: 1.17000, self.b: 1.36000}
        self.x = 0.0                      # OU deviation of ln C from ln(A/B)
        self.t = 0.0                      # virtual seconds since start
        # Pinning the start wall-clock makes downstream behaviour deterministic.
        # The trading arm refuses to trade during the rollover halt and at
        # weekends, so a test whose synthetic clock is "now" silently records
        # zero trades every evening after 16:55 ET -- which is exactly how this
        # parameter came to exist.
        self._t0_wall = start_wall or dt.datetime.now(dt.timezone.utc)
        self._t0_mono = time.monotonic()

    # ---- processes ------------------------------------------------------
    def _advance(self, dt_s: float) -> None:
        for k in (self.a, self.b):
            sd = ANNUAL_VOL[k] / math.sqrt(SECONDS_PER_YEAR) * math.sqrt(dt_s)
            self.mid[k] *= math.exp(self.rng.normal(0.0, sd))
        if self.mode == "independent":
            # Exact OU discretization -- correct for any dt, unlike Euler.
            decay = math.exp(-self.theta * dt_s)
            noise_sd = self.sigma_x * math.sqrt(max(0.0, 1.0 - decay * decay))
            self.x = decay * self.x + self.rng.normal(0.0, noise_sd)
        else:
            self.x = 0.0
        self.t += dt_s

    def _c_mid(self) -> float:
        return (self.mid[self.a] / self.mid[self.b]) * math.exp(self.x)

    def _quote(self, inst: float | str):
        mid = self._c_mid() if inst == self.c else self.mid[inst]
        half = SPREADS[inst] / 2.0
        d = DECIMALS[inst]
        # Round the way a venue does: bid down, ask up, onto the price grid.
        bid = math.floor((mid - half) * 10 ** d) / 10 ** d
        ask = math.ceil((mid + half) * 10 ** d) / 10 ** d
        return bid, ask

    # ---- stream ---------------------------------------------------------
    def messages(self, duration_s: float, heartbeat_s: float = 5.0):
        """Yields (type, msg) exactly like OandaClient.stream_prices.

        With speed > 1 the receive clocks are virtual: quote ages, holding times
        and horizons all stay internally consistent, but wall time compresses.
        """
        total_rate = sum(RATES.values())
        names = list(RATES.keys())
        probs = np.array([RATES[n] for n in names]) / total_rate
        next_hb = heartbeat_s
        real_start = time.monotonic()
        while self.t < duration_s:
            gap = float(self.rng.exponential(1.0 / total_rate))
            self._advance(gap)
            if self.speed <= 1000.0:
                behind = (self.t / self.speed) - (time.monotonic() - real_start)
                if behind > 0.002:
                    time.sleep(behind)
            while self.t >= next_hb:
                yield "HEARTBEAT", {"type": "HEARTBEAT",
                                    "time": self._venue_iso(next_hb),
                                    "_recv_mono": self._mono(next_hb),
                                    "_recv_wall": self._wall(next_hb)}
                next_hb += heartbeat_s
            if self.gap_hazard and self.rng.random() < self.gap_hazard * gap:
                return                      # simulated disconnect
            inst = names[int(self.rng.choice(len(names), p=probs))]
            bid, ask = self._quote(inst)
            lat = max(1.0, self.rng.normal(self.latency_ms, self.latency_sd_ms)) / 1e3
            yield "PRICE", {
                "type": "PRICE", "instrument": inst,
                "time": self._venue_iso(self.t),
                "bids": [{"price": f"{bid:.5f}", "liquidity": 10_000_000}],
                "asks": [{"price": f"{ask:.5f}", "liquidity": 10_000_000}],
                "tradeable": True,
                "_recv_mono": self._mono(self.t + lat),
                "_recv_wall": self._wall(self.t + lat),
            }

    def _venue_iso(self, t: float) -> str:
        return (self._t0_wall + dt.timedelta(seconds=t)).strftime(
            "%Y-%m-%dT%H:%M:%S.%f000Z")

    def _wall(self, t: float) -> dt.datetime:
        return self._t0_wall + dt.timedelta(seconds=t)

    def _mono(self, t: float) -> float:
        return self._t0_mono + t

    def current_quote(self, instrument: str):
        """Used by the paper-fill simulator in Phase 2 dry runs."""
        return self._quote(instrument)
