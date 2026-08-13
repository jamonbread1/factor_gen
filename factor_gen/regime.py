from __future__ import annotations

import numpy as np
import pandas as pd


def build_regime_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build point-in-time market regime labels from information available at EOD.

    No forward target is used. The labels are intentionally coarse so they can be
    used for conditional alpha-health diagnostics without becoming another fitted
    model.
    """
    required = {"eob", "symbol", "ret1", "ma_gap20"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing regime columns: {sorted(missing)}")
    x = frame[["eob", "symbol", "ret1", "ma_gap20"]].copy()
    rows = []
    for date, g in x.groupby("eob", sort=True):
        r = g["ret1"].to_numpy(float)
        gap = g["ma_gap20"].to_numpy(float)
        r = r[np.isfinite(r)]
        gap = gap[np.isfinite(gap)]
        if len(r) < 10:
            continue
        breadth = float(np.mean(gap > 0)) if len(gap) else np.nan
        dispersion = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        market_trend = float(np.nanmean(gap)) if len(gap) else np.nan
        rows.append((date, breadth, dispersion, market_trend))
    out = pd.DataFrame(rows, columns=["eob", "breadth", "dispersion", "market_trend"])
    if out.empty:
        return out.assign(regime=pd.Series(dtype=str))
    roll = out["dispersion"].rolling(60, min_periods=20)
    z = (out["dispersion"] - roll.mean()) / (roll.std(ddof=1) + 1e-8)
    out["dispersion_z"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["regime"] = np.select(
        [
            (out["market_trend"] > 0) & (out["breadth"] >= 0.55),
            (out["market_trend"] < 0) & (out["breadth"] < 0.45) & (out["dispersion_z"] >= 1.0),
            (out["market_trend"] < 0) & (out["breadth"] < 0.45),
        ],
        ["BULL_TREND", "STRUCTURAL_BULL", "BEAR_TREND"],
        default="MIXED",
    )
    return out


def attach_regime(frame: pd.DataFrame) -> pd.DataFrame:
    regimes = build_regime_frame(frame)
    return frame.merge(regimes[["eob", "regime", "breadth", "dispersion", "dispersion_z", "market_trend"]], on="eob", how="left", validate="many_to_one")


def recovery_score(frame: pd.DataFrame) -> dict:
    """Point-in-time recovery diagnostics; no future returns are consulted."""
    r = build_regime_frame(frame)
    if r.empty:
        return {"score": np.nan, "status": "INSUFFICIENT_DATA"}
    recent = r.tail(min(20, len(r)))
    price = float(np.clip(50 + 500 * recent["market_trend"].mean(), 0, 100))
    breadth = float(np.clip(100 * recent["breadth"].mean(), 0, 100))
    dispersion = float(np.clip(50 + 20 * recent["dispersion_z"].mean(), 0, 100))
    score = 0.45 * price + 0.40 * breadth + 0.15 * dispersion
    status = "DEFENSIVE" if score < 40 else "STABILIZING" if score < 55 else "RECOVERY" if score < 70 else "RE-ACCELERATION" if score < 85 else "FULL_RISK"
    return {"score": float(score), "status": status, "price": price, "breadth": breadth, "dispersion": dispersion, "latest_regime": str(r.iloc[-1]["regime"])}


def regime_ic_report(frame: pd.DataFrame, values, target_col="target") -> dict:
    """Cross-sectional IC by market regime, using only the supplied frame."""
    from .evaluation.fast import fast_factor_report

    x = attach_regime(frame)
    v = np.asarray(values, float)
    result = {}
    for regime in sorted(x["regime"].dropna().unique()):
        mask = x["regime"].to_numpy() == regime
        result[str(regime)] = fast_factor_report(x.loc[mask], v[mask], target_col)
    return result
