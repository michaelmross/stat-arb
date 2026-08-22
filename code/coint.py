"""Cointegration testing.

Engle-Granger two-step with MacKinnon (2010) response-surface p-values,
which are the correct null distribution when the cointegrating vector is
ESTIMATED (the ordinary ADF table is anti-conservative here).

Also provides Benjamini-Hochberg FDR control for pair scans: testing many
pairs at nominal 5% guarantees false discoveries without correction.
"""

from dataclasses import dataclass
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


@dataclass
class CointResult:
    tstat: float
    pvalue: float
    beta: float          # OLS hedge ratio: p = alpha + beta*q + spread
    alpha: float
    spread: np.ndarray


def engle_granger(logp: np.ndarray, logq: np.ndarray,
                  trend: str = "c") -> CointResult:
    """EG step 1: OLS of logp on logq. Step 2: unit-root test on residuals
    using MacKinnon cointegration critical values (via statsmodels.coint)."""
    X = sm.add_constant(logq)
    ols = sm.OLS(logp, X).fit()
    alpha, beta = ols.params
    spread = logp - alpha - beta * logq
    tstat, pvalue, _ = coint(logp, logq, trend=trend, autolag="aic")
    return CointResult(tstat=tstat, pvalue=pvalue, beta=beta,
                       alpha=alpha, spread=spread)


def adf_naive(spread: np.ndarray) -> float:
    """ADF p-value using the WRONG (single-series) null. Kept only to
    demonstrate, in validation, how anti-conservative it is on estimated
    residuals."""
    return adfuller(spread, autolag="aic")[1]


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Return boolean mask of discoveries under BH FDR control at level q."""
    p = np.asarray(pvalues)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    keep = np.zeros(m, dtype=bool)
    if below.any():
        kmax = np.max(np.nonzero(below)[0])
        keep[order[: kmax + 1]] = True
    return keep
