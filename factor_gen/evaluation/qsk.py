from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class FactorReport:
    ic: float
    rank_ic: float
    mean_ic: float
    median_ic: float
    ic_std: float
    ic_ir: float
    positive_ratio: float
    monotonicity: float
    turnover: float
    decay: dict[str, float]
    windows: int
    status: str

    def to_dict(self):
        return asdict(self)


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.unique(a[m]).size < 2 or np.unique(b[m]).size < 2:
        return np.nan
    v = spearmanr(a[m], b[m]).statistic
    return float(v) if np.isfinite(v) else np.nan


def cross_sectional_ics(frame, score_col, target_col="target"):
    vals = []
    for _, g in frame.groupby("eob", sort=True):
        v = _spearman(g[score_col], g[target_col])
        if np.isfinite(v): vals.append(v)
    return np.asarray(vals, float)


def _quantile_monotonicity(frame, score_col, target_col="target", groups=5):
    daily = []
    for _, g in frame.groupby("eob", sort=True):
        if len(g) < groups * 3: continue
        try: q = pd.qcut(g[score_col], groups, labels=False, duplicates="drop")
        except ValueError: continue
        means = g.assign(_q=q).groupby("_q")[target_col].mean().dropna().to_numpy()
        if len(means) >= 3:
            daily.append(spearmanr(np.arange(len(means)), means).statistic)
    return float(np.nanmean(daily)) if daily else np.nan


def _turnover(frame, score_col, groups=5):
    holdings = []
    for _, g in frame.groupby("eob", sort=True):
        if len(g) < groups * 3: continue
        q = pd.qcut(g[score_col], groups, labels=False, duplicates="drop")
        if q.isna().all(): continue
        holdings.append(set(g.loc[q == q.max(), "symbol"].astype(str)))
    if len(holdings) < 2: return np.nan
    vals = []
    for a, b in zip(holdings[:-1], holdings[1:]):
        vals.append(1.0 - len(a & b) / max(1, len(a | b)))
    return float(np.mean(vals))


def _decay(frame, score_col, horizons=(1, 3, 5, 10)):
    out = {}
    for h in horizons:
        target = f"target_{h}"
        if target not in frame: continue
        x = frame[[score_col, target]].dropna()
        out[str(h)] = _spearman(x[score_col], x[target])
    return {k: (None if not np.isfinite(v) else float(v)) for k, v in out.items()}


def evaluate(frame, score_col="factor_value", threshold=0.03, min_positive_ratio=0.60,
             horizons: Iterable[int] = (1, 3, 5, 10)) -> FactorReport:
    ics = cross_sectional_ics(frame, score_col)
    if len(ics) == 0:
        return FactorReport(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0,
                            np.nan, np.nan, {}, 0, "INSUFFICIENT_DATA")
    mean = float(np.mean(ics)); median = float(np.median(ics))
    std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    ir = float(mean / std) if std > 1e-12 else 0.0
    positive = float(np.mean(ics > threshold))
    mono = _quantile_monotonicity(frame, score_col)
    turnover = _turnover(frame, score_col)
    decay = _decay(frame, score_col, horizons)
    overall = _spearman(frame[score_col], frame["target"])
    status = "PASS" if mean > threshold and median > threshold and positive >= min_positive_ratio and ir >= 0.50 else "FAIL"
    return FactorReport(overall, overall, mean, median, std, ir, positive, mono, turnover, decay, len(ics), status)
