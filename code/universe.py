"""Candidate-pair universe of STRUCTURALLY related ETFs (roadmap item 1).

Why not all pairs? A 60-ticker universe has 1,770 pairs. BH-FDR controls
the false-discovery *rate* among rejections, but it cannot manufacture
prior plausibility: a "cointegrated" GLD/XRT is an artifact whatever its
q-value. Every pair here has an economic reason to share a stochastic
trend, so a rejection means something. This is the prior; BH is the
correction. You need both.

Four structural relations, in rough order of expected cointegration
strength (and inverse order of expected profitability -- see caveat 1 in
the README, which is the whole tension):

  SAME_INDEX   Different wrappers on the same benchmark. Near-arbitrage;
               spreads are basis points wide and costs eat them.
  SECTOR_PARENT A sector fund against its broad parent, or two funds on
               overlapping slices. Cointegration only holds while the
               sector's weight in the parent is stable -- it is not.
  HEDGED       Same equity basket, FX hedge on vs off. The spread IS the
               cumulative currency return: cointegrated only if the FX
               rate is mean-reverting, which is the actual bet.
  HOLDRS       The four surviving Merrill HOLDRs (terminated 2011, rolled
               into VanEck ETFs) against modern equivalents. Concentrated,
               unequal-weight baskets vs cap-weighted indices.
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Families: every within-family pair is a candidate.
# ---------------------------------------------------------------------
SAME_INDEX: dict[str, list[str]] = {
    "sp500":        ["SPY", "IVV", "VOO", "SPLG"],
    "nasdaq100":    ["QQQ", "QQQM"],
    "russell2000":  ["IWM", "VTWO", "IJR"],
    "dow30":        ["DIA"],
    "total_market": ["VTI", "ITOT", "SCHB"],
    "sp_midcap":    ["MDY", "IJH"],
    "dev_intl":     ["EFA", "IEFA", "VEA", "SCHF"],
    "emerging":     ["EEM", "IEMG", "VWO", "SPEM"],
    "agg_bond":     ["AGG", "BND", "SPAB"],
    "long_ust":     ["TLT", "VGLT", "SPTL"],
    "ig_corp":      ["LQD", "USIG"],
    "high_yield":   ["HYG", "JNK", "USHY"],
    "gold":         ["GLD", "IAU", "GLDM", "SGOL"],
    "silver":       ["SLV", "SIVR"],
    "reit":         ["VNQ", "IYR", "SCHH", "RWR"],
}

# ---------------------------------------------------------------------
# Fixed-income cohort (scan v2, added 2026-08-21).
#
# CONFIRMATORY, NOT EXPLORATORY. Scan v1 found that every credible
# discovery in an 84-ticker, equity-dominated universe was a bond pair,
# and attributed that to NAV-anchored creation-redemption being a real
# economic tether where shared index membership is not. This cohort tests
# that hypothesis on new pairs rather than re-cutting the same data. It
# is labelled separately so the write-up can say "identified in v1,
# tested in v2" instead of silently folding it into the original
# universe -- the date split protects the out-of-sample claim either way,
# but not the preregistration story.
#
# Duration discipline is load-bearing. A family here means same exposure
# AND same duration bucket: VTIP (0-5yr TIPS) sits with STIP, never with
# TIP/SCHP. Mixing durations inside a family manufactures exactly the
# slow-wandering spreads the noise_dominated flag exists to catch.
#
# Families already present in SAME_INDEX (aggregate, long_ust,
# high_yield) are restated here so their new members pair up correctly;
# de-duplication keeps the v1 label on any pair that already existed.
# ---------------------------------------------------------------------
FIXED_INCOME: dict[str, list[str]] = {
    "aggregate":  ["SCHZ", "AGG", "BND", "SPAB"],
    "long_ust":   ["TLT", "VGLT", "SPTL"],
    "int_ust":    ["IEF", "VGIT", "SPTI"],
    "short_ust":  ["SHY", "VGSH", "SCHO", "SPTS"],
    "tips_broad": ["TIP", "SCHP", "SPIP"],
    "tips_short": ["VTIP", "STIP"],
    "ig_int":     ["VCIT", "IGIB", "SPIB"],
    "ig_short":   ["VCSH", "IGSB", "SPSB"],
    "mbs":        ["MBB", "VMBS", "SPMB"],
    "em_usd_sov": ["EMB", "VWOB", "PCY"],
    "muni_broad": ["MUB", "VTEB"],
    "preferred":  ["PFF", "PGX", "PSK"],
    "high_yield": ["HYG", "JNK", "USHY"],
}

# Post-2017 SPDR renames. Tiingo usually stitches full history under the
# current ticker, but a series that starts in 2017 is a truncated rename,
# not a young fund -- it would silently fall out of the discovery window.
# fetch_data.py checks these explicitly.
SPDR_RENAMES: dict[str, str] = {
    "SPTL": "TLO", "SPTI": "ITE", "SPTS": "SST",
    "SPIP": "IPE", "SPSB": "SCPB", "SPIB": "ITR",
}

HEDGED: dict[str, list[str]] = {
    # (hedged, unhedged) on the same underlying basket
    "eafe":      ["HEFA", "DBEF", "EFA"],
    "eurozone":  ["HEZU", "EZU"],
    "japan":     ["HEWJ", "DBJP", "DXJ", "EWJ"],
    "germany":   ["HEWG", "EWG"],
}

HOLDRS: dict[str, list[str]] = {
    # ex-HOLDR basket  +  modern cap-weighted equivalents
    "semis":     ["SMH", "SOXX", "XSD"],
    "oilsvc":    ["OIH", "XES", "PXJ"],
    "retail":    ["RTH", "XRT"],
    "biotech":   ["BBH", "IBB", "XBI"],
}

# ---------------------------------------------------------------------
# Explicit cross-family structural pairs (sector vs parent, and the
# HOLDR-vs-parent-sector links that the family grouping above misses).
# ---------------------------------------------------------------------
SECTOR_PARENT: list[tuple[str, str]] = [
    # sector vs broad parent
    ("XLK", "SPY"), ("XLF", "SPY"), ("XLE", "SPY"), ("XLV", "SPY"),
    ("XLY", "SPY"), ("XLP", "SPY"), ("XLI", "SPY"), ("XLU", "SPY"),
    ("QQQ", "SPY"), ("QQQ", "XLK"),
    # same sector, different sponsor (tightest of this group)
    ("XLK", "VGT"), ("XLK", "IYW"), ("XLF", "VFH"), ("XLE", "VDE"),
    ("XLV", "VHT"), ("XLU", "VPU"), ("XLI", "VIS"), ("XLP", "VDC"),
    # sub-sector vs its sector
    ("XOP", "XLE"), ("OIH", "XLE"), ("KRE", "XLF"), ("KBE", "XLF"),
    ("KRE", "KBE"), ("IBB", "XLV"), ("XBI", "XLV"), ("SMH", "XLK"),
    ("XRT", "XLY"), ("ITB", "XLY"),
    # defensive vs cyclical (weak prior, kept as a negative control)
    ("XLP", "XLY"), ("XLU", "XLK"),
]

# Deliberate negative controls: no structural reason to cointegrate. If
# these show up as discoveries, the scan is too loose.
CONTROLS: list[tuple[str, str]] = [
    ("GLD", "XRT"), ("TLT", "XLE"), ("SLV", "XLF"), ("EWJ", "XOP"),
]


@dataclass(frozen=True)
class Candidate:
    a: str
    b: str
    relation: str      # same_index | sector_parent | hedged | holdrs | control
    family: str
    cohort: str = "v1"  # v1 = original scan; v2_fixed_income = follow-up


def _within(groups: dict[str, list[str]], relation: str,
            cohort: str = "v1") -> list[Candidate]:
    out = []
    for fam, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                out.append(Candidate(members[i], members[j], relation, fam,
                                     cohort))
    return out


def candidate_pairs(include_controls: bool = True,
                    include_fixed_income: bool = True) -> list[Candidate]:
    """Every structurally motivated candidate pair, de-duplicated.

    Order matters: v1 groups are emitted first, so a pair present in both
    cohorts (e.g. AGG/BND, which the fixed-income cohort restates to let
    SCHZ pair up) keeps its original v1 label and is not double-counted in
    the BH denominator.
    """
    cands = (_within(SAME_INDEX, "same_index")
             + _within(HEDGED, "hedged")
             + _within(HOLDRS, "holdrs")
             + [Candidate(a, b, "sector_parent", "cross")
                for a, b in SECTOR_PARENT])
    if include_controls:
        cands += [Candidate(a, b, "control", "control") for a, b in CONTROLS]
    if include_fixed_income:
        cands += _within(FIXED_INCOME, "same_index", "v2_fixed_income")

    seen, uniq = set(), []
    for c in cands:
        key = tuple(sorted((c.a, c.b)))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def all_tickers(include_controls: bool = True,
                include_fixed_income: bool = True) -> list[str]:
    ts = {t for c in candidate_pairs(include_controls, include_fixed_income)
          for t in (c.a, c.b)}
    return sorted(ts)


if __name__ == "__main__":
    from collections import Counter
    cands = candidate_pairs()
    print(f"{len(all_tickers())} tickers, {len(cands)} candidate pairs")
    for rel, n in Counter(c.relation for c in cands).most_common():
        print(f"  {rel:14s} {n:3d}")
