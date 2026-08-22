"""Daily OHLC CSV loading for real-data runs (roadmap item 1).

Supports the two export formats this project targets.

Stooq  --  https://stooq.com/q/d/l/?s=spy.us&i=d
    Date,Open,High,Low,Close,Volume
    Split-adjusted but NOT dividend-adjusted. For ETF pairs that matters:
    two funds tracking the same index go ex-dividend on different dates,
    which stamps a sawtooth into the log spread. That sawtooth is a
    cash-flow artifact, not tradable mean reversion, and it will inflate
    both the AR(1) coefficient and apparent OU reversion.

Tiingo --  /tiingo/daily/<ticker>/prices?format=csv&startDate=YYYY-MM-DD
    date,close,high,low,open,volume,adjClose,adjHigh,adjLow,adjOpen,
    adjVolume,divCash,splitFactor
    `adjClose` is total-return adjusted -> the correct series for pairs.

Loader policy: prefer an adjusted close when the file has one, fall back to
raw close, and always record which was used so the caveat travels with the
result instead of getting lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

STOOQ_COLS = {"date", "open", "high", "low", "close", "volume"}
TIINGO_MARKERS = {"adjclose", "divcash", "splitfactor"}


@dataclass
class SeriesInfo:
    """What we loaded and what is wrong with it."""
    ticker: str
    source: str            # 'stooq' | 'tiingo'
    price_col: str         # column actually used
    dividend_adjusted: bool
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_dupe_dates: int = 0
    n_nonpositive: int = 0
    max_stale_run: int = 0     # longest run of identical closes
    stale_frac: float = 0.0    # fraction of days with zero price change
    warnings: list[str] = field(default_factory=list)


def detect_source(df: pd.DataFrame) -> str:
    cols = {c.strip().lower() for c in df.columns}
    if cols & TIINGO_MARKERS:
        return "tiingo"
    if STOOQ_COLS.issubset(cols):
        return "stooq"
    if {"date", "close"} <= cols:
        return "stooq"
    raise ValueError(f"unrecognised CSV layout, columns: {sorted(cols)}")


def load_prices(path: str | Path, ticker: str | None = None,
                source: str = "auto", prefer_adjusted: bool = True,
                allow_nonpositive: bool = False
                ) -> tuple[pd.Series, SeriesInfo]:
    """Read one daily CSV -> (price Series indexed by date, SeriesInfo)."""
    path = Path(path)
    ticker = ticker or path.stem.split(".")[0].upper()
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    src = detect_source(raw) if source == "auto" else source

    if src == "tiingo" and prefer_adjusted and "adjclose" in raw.columns:
        price_col, div_adj = "adjclose", True
    elif "close" in raw.columns:
        price_col, div_adj = "close", False
    else:
        raise ValueError(f"{path.name}: no close column")

    dates = pd.to_datetime(raw["date"], errors="coerce")
    s = pd.Series(pd.to_numeric(raw[price_col], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(dates), name=ticker)

    notes: list[str] = []
    n_bad_date = int(s.index.isna().sum())
    if n_bad_date:
        notes.append(f"{n_bad_date} unparseable dates dropped")
        s = s[~s.index.isna()]

    n_dupe = int(s.index.duplicated().sum())
    if n_dupe:
        notes.append(f"{n_dupe} duplicate dates -> kept last")
        s = s[~s.index.duplicated(keep="last")]

    s = s.sort_index()
    if allow_nonpositive:
        # Price space: a negative settlement is a legal observation, not
        # dirty data. Dropping WTI's 2020-04-20 close of -$37.63 would
        # also fabricate a -60% two-day move across the hole.
        n_nonpos = int((s <= 0).sum())
        s = s[s.notna()]
        if n_nonpos:
            notes.append(f"{n_nonpos} non-positive price(s) RETAINED "
                         f"(price-space load)")
    else:
        n_nonpos = int((~(s > 0)).sum())
        if n_nonpos:
            notes.append(f"{n_nonpos} non-positive/NaN prices dropped")
            s = s[s > 0]

    if src == "stooq" and prefer_adjusted:
        notes.append("stooq close is not dividend-adjusted; "
                     "ex-div gaps will show up in the spread")

    # Stale-price run: illiquid ETFs print the same close for days. That is
    # not mean reversion, but AR(1) cannot tell the difference.
    chg = s.diff().to_numpy()[1:]
    stale = chg == 0
    max_run = 0
    run = 0
    for flag in stale:
        run = run + 1 if flag else 0
        max_run = max(max_run, run)
    stale_frac = float(stale.mean()) if stale.size else 0.0
    if stale_frac > 0.05:
        notes.append(f"{stale_frac:.1%} of days have an unchanged close "
                     f"(max run {max_run}) - thin/illiquid")

    info = SeriesInfo(ticker=ticker, source=src, price_col=price_col,
                      dividend_adjusted=div_adj, n_obs=len(s),
                      start=s.index[0], end=s.index[-1],
                      n_dupe_dates=n_dupe, n_nonpositive=n_nonpos,
                      max_stale_run=max_run, stale_frac=stale_frac,
                      warnings=notes)
    return s, info


@dataclass
class Panel:
    """Date-aligned price panel plus the diagnostics behind it.

    `frame` holds log prices when `is_log` (the default, and what every
    equity caller wants) and raw prices otherwise. Futures margin work
    runs in price space: a crack or crush spread is a linear combination
    of dollar prices, nothing in it needs logs, and log space cannot
    represent 2020-04-20, when WTI settled at -$37.63. Accessing
    `.log_prices` on a price-space panel raises rather than silently
    handing back the wrong units.
    """
    frame: pd.DataFrame
    info: dict[str, SeriesInfo]
    n_dates_dropped: int
    any_dividend_unadjusted: bool
    is_log: bool = True

    @property
    def log_prices(self) -> pd.DataFrame:
        if not self.is_log:
            raise AttributeError(
                "this Panel holds raw prices (log_transform=False); "
                "use .frame")
        return self.frame

    @property
    def tickers(self) -> list[str]:
        return list(self.frame.columns)

    def pair(self, a: str, b: str, min_obs: int = 756):
        """Return (logp, logq, DatetimeIndex) for one pair, inner-joined.

        Aligned per pair rather than across the whole panel so that one
        short-history ETF cannot truncate every other pair's sample.
        """
        sub = self.frame[[a, b]].dropna()
        if len(sub) < min_obs:
            raise ValueError(f"{a}/{b}: only {len(sub)} overlapping obs "
                             f"(need {min_obs})")
        return (sub[a].to_numpy(), sub[b].to_numpy(), sub.index)


def load_panel(directory: str | Path, tickers: list[str] | None = None,
               start: str | None = None, end: str | None = None,
               source: str = "auto", prefer_adjusted: bool = True,
               how: str = "outer", log_transform: bool = True) -> Panel:
    """Load every CSV in `directory` into one aligned log-price panel.

    `how='outer'` keeps the union of dates (NaN-filled) so each pair can be
    inner-joined on its own overlap; 'inner' forces a common calendar.
    """
    directory = Path(directory)
    files = sorted(p for p in directory.glob("*.csv"))
    if tickers is not None:
        want = {t.upper() for t in tickers}
        files = [p for p in files if p.stem.split(".")[0].upper() in want]
    if not files:
        raise FileNotFoundError(f"no matching CSVs in {directory}")

    series: dict[str, pd.Series] = {}
    infos: dict[str, SeriesInfo] = {}
    for p in files:
        s, info = load_prices(p, source=source,
                              prefer_adjusted=prefer_adjusted,
                              allow_nonpositive=not log_transform)
        if info.n_obs == 0:
            warnings.warn(f"{p.name}: empty after cleaning, skipped")
            continue
        series[info.ticker] = s
        infos[info.ticker] = info

    px = pd.concat(series.values(), axis=1, join=how).sort_index()
    if start is not None:
        px = px[px.index >= pd.Timestamp(start)]
    if end is not None:
        px = px[px.index <= pd.Timestamp(end)]

    before = len(px)
    px = px.dropna(how="all")
    frame = np.log(px) if log_transform else px

    return Panel(frame=frame, is_log=log_transform, info=infos,
                 n_dates_dropped=before - len(px),
                 any_dividend_unadjusted=any(not i.dividend_adjusted
                                             for i in infos.values()))
