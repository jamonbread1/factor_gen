from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _safe_corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20: return np.nan
    x = x[m]; y = y[m]
    x -= x.mean(); y -= y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / den) if den > 1e-12 else np.nan


def cheap_screen(values, target, dates, threshold=0.01):
    """Cheap prefilter: global correlation + weekly directional agreement.
    It intentionally does not replace the final cross-sectional IC test.
    """
    values = np.asarray(values, float); target = np.asarray(target, float)
    corr = _safe_corr(values, target)
    m = np.isfinite(values) & np.isfinite(target)
    if m.sum() < 20:
        return -np.inf
    directional = float(np.mean(np.sign(values[m]) == np.sign(target[m])))
    return abs(corr) * 0.7 + max(0.0, directional - 0.5) * 0.6 if np.isfinite(corr) else -np.inf


def cross_sectional_ic_series(frame: pd.DataFrame, values, target_col="target") -> np.ndarray:
    """Daily cross-sectional Spearman IC using numpy sorting; avoids pandas groupby.rank."""
    x = np.asarray(values, float)
    y = frame[target_col].to_numpy(float)
    dates = frame["eob"].to_numpy()
    out = []
    for d in pd.unique(dates):
        idx = np.flatnonzero(dates == d)
        if idx.size < 10: continue
        a = x[idx]; b = y[idx]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10: continue
        ar = rankdata(a[m], method="average"); br = rankdata(b[m], method="average")
        out.append(_safe_corr(ar, br))
    return np.asarray(out, float)


def fast_factor_report(frame: pd.DataFrame, values, target_col="target", threshold=0.03, min_positive_ratio=0.60):
    ics = cross_sectional_ic_series(frame, values, target_col)
    if len(ics) == 0:
        return {"status": "INSUFFICIENT_DATA", "mean_ic": np.nan, "median_ic": np.nan,
                "ic_std": np.nan, "ic_ir": np.nan, "positive_ratio": 0.0, "windows": 0}
    mean = float(np.mean(ics)); median = float(np.median(ics)); std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    ir = float(mean / std) if std > 1e-12 else 0.0
    positive = float(np.mean(ics > threshold))
    status = "PASS" if mean > threshold and median > threshold and positive >= min_positive_ratio and ir >= 0.50 else "FAIL"
    return {"status": status, "ic": mean, "rank_ic": mean, "mean_ic": mean, "median_ic": median,
            "ic_std": std, "ic_ir": ir, "positive_ratio": positive, "windows": int(len(ics)),
            "ic_series": ics.tolist()}


def rolling_fold_scores(frame: pd.DataFrame, values, target_col="target", folds=4, threshold=0.03):
    """Walk-forward validation over contiguous time blocks."""
    dates = np.sort(frame["eob"].unique())
    if len(dates) < folds * 10: return []
    chunks = np.array_split(dates, folds)
    result = []
    for i, ds in enumerate(chunks):
        mask = frame["eob"].isin(ds).to_numpy()
        r = fast_factor_report(frame.loc[mask], np.asarray(values)[mask], target_col, threshold)
        r["fold"] = i + 1
        result.append(r)
    return result
