from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    horizon: int = 5
    buy_threshold: float = 0.03
    sell_threshold: float = -0.03
    flat_threshold: float = 0.01
    stop_loss: float = -0.04
    take_profit: float = 0.08


def add_weekly_targets(frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Add point-in-time forward-return targets. Horizon is trading bars."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    x = frame.copy().sort_values(["symbol", "eob"]).reset_index(drop=True)
    g = x.groupby("symbol", sort=False)
    close = x["close"].astype(float).clip(lower=1e-12)
    x[f"target_{horizon}"] = g["close"].shift(-horizon) / close - 1.0
    # Directional target is deliberately continuous; classification is only for
    # signal generation. This avoids throwing away information during training.
    x["target"] = x[f"target_{horizon}"]
    x["trend_up"] = (x["target"] > 0).astype("int8")
    x["trend_down"] = (x["target"] < 0).astype("int8")
    return x


def signal_from_score(score, buy_threshold=0.03, sell_threshold=-0.03):
    """Map a signed factor score into +1 buy, -1 sell, 0 neutral."""
    s = np.asarray(score, dtype=float)
    return np.where(s >= buy_threshold, 1, np.where(s <= sell_threshold, -1, 0)).astype(np.int8)


def make_signal_frame(frame: pd.DataFrame, score_col="factor_value", cfg: SignalConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SignalConfig()
    out = frame.copy()
    out["signal"] = signal_from_score(out[score_col], cfg.buy_threshold, cfg.sell_threshold)
    out["signal_strength"] = np.clip(np.abs(out[score_col].to_numpy(float)), 0, np.inf)
    return out


def signal_backtest_stats(frame: pd.DataFrame, score_col="factor_value", cfg: SignalConfig | None = None) -> dict:
    """Simple research diagnostics for weekly directional signals; no execution claims."""
    cfg = cfg or SignalConfig()
    x = frame[["eob", "symbol", "close", score_col, f"target_{cfg.horizon}"]].copy()
    x["signal"] = signal_from_score(x[score_col], cfg.buy_threshold, cfg.sell_threshold)
    active = x[x["signal"] != 0].copy()
    if active.empty:
        return {"active_ratio": 0.0, "buy_count": 0, "sell_count": 0, "mean_forward_return": np.nan,
                "buy_mean_return": np.nan, "sell_mean_return": np.nan, "hit_rate": np.nan}
    active["strategy_return"] = active["signal"] * active[f"target_{cfg.horizon}"]
    return {
        "active_ratio": float(len(active) / max(1, len(x))),
        "buy_count": int((active["signal"] == 1).sum()),
        "sell_count": int((active["signal"] == -1).sum()),
        "mean_forward_return": float(active["strategy_return"].mean()),
        "buy_mean_return": float(active.loc[active.signal == 1, f"target_{cfg.horizon}"].mean()) if (active.signal == 1).any() else np.nan,
        "sell_mean_return": float((-active.loc[active.signal == -1, f"target_{cfg.horizon}"].mean())) if (active.signal == -1).any() else np.nan,
        "hit_rate": float((active["strategy_return"] > 0).mean()),
    }
