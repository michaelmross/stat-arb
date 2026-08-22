"""Production-margin spreads from stitched front-month futures.

PRICE SPACE, NOT LOG SPACE. A crack or crush margin is a linear
combination of dollar prices with FIXED technical weights. Nothing about
it needs logs, the anchor logic wants dollars (variable-cost floors are
dollar quantities), and log space cannot represent 2020-04-20, when WTI
settled at -$37.63.

Fixed weights also change the statistics for the better. There is no
hedge ratio to estimate, so:
  * ordinary ADF applies -- no MacKinnon correction, because nothing was
    fitted;
  * there is no spurious-regression channel for an OLS beta to absorb;
  * the d-census runs on the margin directly rather than on a residual.

UNADJUSTED STITCHING IS CORRECT HERE. An unadjusted front-month series
equals the true prompt contract price every day, so the margin built
from stitched fronts is the true prompt margin every day, with zero
cumulative distortion. The discontinuities at rolls are not errors: they
are genuine changes in the measured object (prompt margin of month k
versus month k+1). Back-adjustment would smooth those steps at the cost
of destroying the level, and the level is the anchor. Rolls are therefore
handled in the RETURN ACCOUNTING (see backtest_nleg), never by editing
the price series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Spread:
    name: str
    unit: str
    legs: tuple[str, ...]
    weights: tuple[float, ...]   # applied to the quoted price, in `unit`
    note: str = ""
    confirmatory: bool = True


# 3-2-1 crack, $/bbl: three barrels of crude -> two gasoline + one distillate.
# RB and HO are quoted $/gal, so x42 converts each to $/bbl; /3 puts the
# margin per barrel of crude input. CL is already $/bbl.
CRACK_321 = Spread(
    name="crack_321", unit="USD/bbl",
    legs=("CL_F", "RB_F", "HO_F"),
    weights=(-1.0, 2 * 42 / 3, 42 / 3),
    note="refinery margin, 3 crude -> 2 gasoline + 1 heating oil",
)

# Board crush, $/bu: a 60-lb bushel yields ~44 lb meal (0.022 short tons)
# and ~11 lb oil. ZM is $/short ton, ZL is cents/lb, ZS is cents/bushel.
BOARD_CRUSH = Spread(
    name="board_crush", unit="USD/bu",
    legs=("ZS_F", "ZM_F", "ZL_F"),
    weights=(-0.01, 0.022, 0.11),
    note="soybean processing margin, 1 bu -> 44 lb meal + 11 lb oil",
)

# Cattle crush: EXPLORATORY ONLY, deliberately outside the confirmatory
# set. The real feedlot margin is time-offset -- buy feeders and corn now,
# sell live cattle five to six months deferred. Front-month-only data
# cannot represent the deferred leg, so a contemporaneous front-month
# version is a different and economically muddled object. Kept so the
# choice is visible in code rather than silently dropped.
CATTLE_CRUSH = Spread(
    name="cattle_crush", unit="USD/cwt (contemporaneous proxy)",
    legs=("LE_F", "GF_F", "ZC_F"),
    weights=(0.01, -0.01 * 0.5, -0.01 * 0.02),
    note="NOT the true feedlot margin: front-month only, no deferred leg",
    confirmatory=False,
)

SPREADS = {s.name: s for s in (CRACK_321, BOARD_CRUSH, CATTLE_CRUSH)}


# ---------------------------------------------------------------------
# Contract calendars.
#
# Detection must be CALENDAR-driven, not jump-driven: the sample contains
# roughly 200 monthly rolls per energy leg, but only ~40-70 daily moves
# exceed 5 robust sd, so a jump threshold misses most rolls outright.
#
# Approximate, and deliberately so -- no exchange holiday calendar is
# assumed, so these land within a day of true expiry, which is why the
# masking window is +/- 2 business days rather than exact.
# ---------------------------------------------------------------------
GRAIN_EXPIRY_DAY = 15          # business day prior to the 15th

CONTRACT_MONTHS = {
    # Soybeans: F H K N Q U X  (no Oct, no Dec; Nov is the new-crop month)
    "ZS_F": (1, 3, 5, 7, 8, 9, 11),
    # Meal and oil: F H K N Q U V Z  (Oct and Dec, no Nov)
    "ZM_F": (1, 3, 5, 7, 8, 9, 10, 12),
    "ZL_F": (1, 3, 5, 7, 8, 9, 10, 12),
    # Corn: H K N U Z
    "ZC_F": (3, 5, 7, 9, 12),
    # Live cattle: even months.  Feeder cattle: Jan Mar Apr May Aug Sep Oct Nov
    "LE_F": (2, 4, 6, 8, 10, 12),
    "GF_F": (1, 3, 4, 5, 8, 9, 10, 11),
}


def _prev_bday(ts: pd.Timestamp, n: int = 1) -> pd.Timestamp:
    return ts - pd.tseries.offsets.BDay(n)


def expiries(ticker: str, start, end) -> pd.DatetimeIndex:
    """Approximate front-month expiry dates for one leg."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    months = pd.date_range(start - pd.offsets.MonthBegin(2),
                           end + pd.offsets.MonthBegin(2), freq="MS")
    for m in months:
        if ticker == "CL_F":
            # 3 business days before the 25th of the month PRECEDING delivery
            out.append(_prev_bday(m.replace(day=25), 3))
        elif ticker in ("RB_F", "HO_F"):
            # last business day of the month preceding delivery
            out.append(_prev_bday(m + pd.offsets.MonthBegin(1), 1))
        elif ticker in CONTRACT_MONTHS:
            if m.month not in CONTRACT_MONTHS[ticker]:
                continue
            out.append(_prev_bday(m.replace(day=GRAIN_EXPIRY_DAY), 1))
        else:
            raise ValueError(f"no calendar for {ticker}")
    idx = pd.DatetimeIndex(sorted(set(out)))
    return idx[(idx >= start) & (idx <= end)]


def roll_mask(index: pd.DatetimeIndex, legs, before: int = 0,
              after: int = 1) -> np.ndarray:
    """Boolean mask: True on days inside any leg's roll window.

    Window chosen empirically, not assumed. Sweeping (before, after) over
    both confirmatory spreads and scoring by large-jump capture per
    masked day:

        window     crack: %masked / %jumps    crush: %masked / %jumps
        -0/+0        3.2% /  1%                 1.2% /  3%
        -0/+1       12.1% / 40%   <- best       4.6% / 27%   <- best
        -1/+1       21.3% / 52%                 8.2% / 41%
        -2/+2       39.2% / 62%                15.3% / 46%

    -0/+0 catching almost nothing pins down the mechanism: the stitched
    series switches contract on the day AFTER expiry, not on expiry
    itself. Widening past +1 buys more jumps only by masking far more
    clean days -- and the extra jumps are mostly real margin moves
    (2020-04-20, the 2022 energy dislocation), which must NOT be masked.

    The union across legs matters for the crush in particular: ZS rolls
    on a different cycle from ZM/ZL (Nov versus Oct/Dec), so the margin
    changes reference contract on strictly more days than any one leg.
    """
    index = pd.DatetimeIndex(index)
    mask = np.zeros(len(index), dtype=bool)
    pos = pd.Series(np.arange(len(index)), index=index)
    for leg in legs:
        for e in expiries(leg, index[0], index[-1]):
            lo = _prev_bday(e, before) if before else e
            hi = e + pd.tseries.offsets.BDay(after) if after else e
            hit = pos[(index >= lo) & (index <= hi)]
            mask[hit.to_numpy()] = True
    return mask


@dataclass
class Margin:
    spread: Spread
    value: pd.Series          # the margin, in spread.unit
    legs: pd.DataFrame        # aligned leg prices, quoted units
    roll: np.ndarray          # boolean roll-window mask
    meta: dict = field(default_factory=dict)


          # exchange ratio, tick value $/contract, commission $/contract RT
COST_SPEC = {
    # legs traded per unit, $ per tick per contract, and the physical size
    # of one spread unit, so costs land in the margin's own units.
    "crack_321": dict(contracts={"CL_F": 3, "RB_F": 2, "HO_F": 1},
                      tick_value={"CL_F": 10.0, "RB_F": 4.20, "HO_F": 4.20},
                      unit_size=3000.0, unit="bbl"),      # 3 CL = 3000 bbl
    "board_crush": dict(contracts={"ZS_F": 10, "ZM_F": 11, "ZL_F": 9},
                        tick_value={"ZS_F": 12.50, "ZM_F": 10.0, "ZL_F": 6.0},
                        unit_size=50000.0, unit="bu"),    # 10 ZS = 50k bu
    "cattle_crush": dict(contracts={"LE_F": 2, "GF_F": 1, "ZC_F": 1},
                         tick_value={"LE_F": 10.0, "GF_F": 12.50, "ZC_F": 12.50},
                         unit_size=80000.0, unit="lb"),
}


def cost_per_side(spread_name: str, ticks: float = 1.0,
                  commission: float = 2.50) -> float:
    """One-way cost in the margin's own units ($/bbl, $/bu, ...).

    Ticks-plus-commission rather than basis points, because a futures
    spread has no natural notional to take bps of. One tick per leg per
    side is the honest retail default; `commission` is per contract per
    round turn and is halved here to charge one side.
    """
    spec = COST_SPEC[spread_name]
    tick_cost = sum(spec["contracts"][l] * spec["tick_value"][l] * ticks
                    for l in spec["contracts"])
    comm = sum(spec["contracts"].values()) * commission / 2.0
    return (tick_cost + comm) / spec["unit_size"]


def fit_ou_masked(x: np.ndarray, roll: np.ndarray, dt: float = 1 / 252):
    """OU fit on levels, excluding transitions that span a roll.

    A roll-day change is a change of contract, not margin dynamics, so it
    must not enter the AR(1) regression that sets the half-life.
    """
    s0, s1 = x[:-1], x[1:]
    keep = ~roll[1:]
    s0, s1 = s0[keep], s1[keep]
    A = np.column_stack([np.ones_like(s0), s0])
    coef, *_ = np.linalg.lstsq(A, s1, rcond=None)
    a, b = coef
    b = min(max(b, 1e-6), 1 - 1e-6)
    kappa = -np.log(b) / dt
    mu = a / (1.0 - b)
    resid = s1 - (a + b * s0)
    sigma = np.sqrt(resid.var(ddof=2) * 2.0 * kappa / (1.0 - b ** 2))
    return dict(mu=float(mu), half_life=float(np.log(2.0) / kappa / dt),
                stationary_std=float(sigma / np.sqrt(2.0 * kappa)))


def backtest_margin(margin: "Margin", train: int = 504, trade: int = 126,
                    z_entry: float = 2.0, z_exit: float = 0.5,
                    z_stop: float = 4.0, ticks: float = 1.0,
                    commission: float = 2.50,
                    min_half_life: float = 1.0, max_half_life: float = 60.0,
                    pval_gate: float = 0.05, score_from: int = None):
    """Walk-forward backtest of an N-leg margin, in price space.

    Roll handling (Decision 3): on a roll day the margin-change P&L is
    ZERO -- the level step is never realised, because holding through a
    roll means exiting the old months and entering the new ones -- the
    position is re-referenced at the new level, and one extra round of
    transaction costs is charged, since all legs really were traded.
    """
    from statsmodels.tsa.stattools import adfuller
    from synth import dejump

    x = margin.value.to_numpy(float)
    roll = margin.roll
    n = len(x)
    c_side = cost_per_side(margin.spread.name, ticks, commission)

    pnl = np.zeros(n)          # $ per unit spread, in margin units
    pos = np.zeros(n)
    zs = np.full(n, np.nan)
    windows = []
    round_trips = 0

    start = 0
    while start + train + 2 <= n:
        i1, j1 = start + train, min(start + train + trade, n)
        tr_x, tr_roll = x[start:i1], roll[start:i1]
        ou = fit_ou_masked(tr_x, tr_roll)
        # ADF on the de-jumped training series: contract offsets otherwise
        # make a random-walk margin reject far too often (validate_roll.py
        # measures 11.5% and 30.0% against a nominal 5%).
        pv = adfuller(dejump(tr_x, tr_roll), autolag="aic")[1]
        sd = max(ou["stationary_std"], 1e-12)
        tradable = (pv < pval_gate
                    and min_half_life < ou["half_life"] < max_half_life)
        windows.append(dict(start=start, pvalue=float(pv),
                            half_life=ou["half_life"], tradable=bool(tradable)))
        if tradable:
            state = 0.0
            for t in range(i1, j1):
                if state != 0.0:
                    if roll[t]:
                        # level step not realised; pay to re-establish
                        pnl[t] -= 2.0 * c_side
                    else:
                        pnl[t] += state * (x[t] - x[t - 1])
                pos[t] = state
                z = (x[t] - ou["mu"]) / sd
                zs[t] = z
                new = state
                if state == 0.0:
                    if z > z_entry:
                        new = -1.0
                    elif z < -z_entry:
                        new = +1.0
                elif abs(z) < z_exit or abs(z) > z_stop:
                    new = 0.0
                if new != state:
                    pnl[t] -= abs(new - state) * c_side
                    if new == 0.0:
                        round_trips += 1
                    state = new
        start += trade

    sl = np.zeros(n, bool)
    sl[(score_from if score_from is not None else train):] = True
    r = pnl[sl]
    sd_r = r.std(ddof=1)
    sharpe = float(r.mean() / sd_r * np.sqrt(252)) if sd_r > 0 else 0.0

    # comparability with the ETF study: same P&L over gross notional
    gross = np.abs(margin.legs.to_numpy() *
                   np.array(margin.spread.weights)).sum(axis=1)
    ret = np.divide(pnl, gross, out=np.zeros_like(pnl), where=gross > 0)
    rr = ret[sl]
    sd_rr = rr.std(ddof=1)
    return dict(
        sharpe=sharpe,
        ann_pnl=float(r.mean() * 252),
        pnl_sd=float(sd_r),
        notional_sharpe=float(rr.mean() / sd_rr * np.sqrt(252))
        if sd_rr > 0 else 0.0,
        ann_notional_ret=float(rr.mean() * 252),
        time_in_market=float((pos[sl] != 0).mean()),
        round_trips=round_trips,
        windows_tradable=float(np.mean([w["tradable"] for w in windows]))
        if windows else 0.0,
        cost_per_side=c_side, n_scored=int(sl.sum()),
        daily_pnl=pnl, positions=pos, z=zs, windows=windows)


def build_margin(frame: pd.DataFrame, spread: Spread,
                 before: int = 0, after: int = 1) -> Margin:
    """Construct one production margin from a PRICE-space panel."""
    missing = [l for l in spread.legs if l not in frame.columns]
    if missing:
        raise KeyError(f"{spread.name}: missing legs {missing}")
    sub = frame[list(spread.legs)].dropna()
    val = sum(w * sub[l] for l, w in zip(spread.legs, spread.weights))
    val.name = spread.name
    mask = roll_mask(sub.index, spread.legs, before=before, after=after)
    return Margin(spread=spread, value=val, legs=sub, roll=mask,
                  meta=dict(n=len(sub), roll_days=int(mask.sum()),
                            roll_frac=float(mask.mean())))
