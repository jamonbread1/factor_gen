from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .ai.transformer_v42 import V42Transformer
from .evaluation.fast import cheap_screen, fast_factor_report, rolling_fold_scores
from .regime import recovery_score, regime_ic_report
from .search.qsk_miner import Candidate, evaluate_spec
from .signal import SignalConfig, signal_backtest_stats

LOG = logging.getLogger("factor_gen.qsk_v52")
BASE_PANEL = ["ret1", "ret5", "ret10", "range", "body", "close_pos", "vol_z20", "ma_gap5", "ma_gap20", "volatility20", "trend20"]


def make_panel(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create point-in-time features and forward targets, always grouped by symbol."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    required = {"symbol", "eob", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    x = df.copy().sort_values(["symbol", "eob"]).reset_index(drop=True)
    g = x.groupby("symbol", group_keys=False, sort=False)
    c = x["close"].astype(float).clip(lower=1e-12)
    x["ret1"] = g["close"].pct_change()
    x["ret5"] = g["close"].pct_change(5)
    x["ret10"] = g["close"].pct_change(10)
    x["range"] = (x["high"] - x["low"]) / c
    x["body"] = (x["close"] - x["open"]) / c
    x["close_pos"] = (x["close"] - x["low"]) / (x["high"] - x["low"]).replace(0, np.nan)
    logv = np.log1p(x["volume"].astype(float).clip(lower=0))
    x["vol_z20"] = g["volume"].transform(lambda s: (np.log1p(s.clip(lower=0)) - np.log1p(s.clip(lower=0)).rolling(20, min_periods=10).mean()) / (np.log1p(s.clip(lower=0)).rolling(20, min_periods=10).std() + 1e-8))
    x["ma_gap5"] = g["close"].transform(lambda s: s / s.rolling(5, min_periods=5).mean() - 1)
    x["ma_gap20"] = g["close"].transform(lambda s: s / s.rolling(20, min_periods=20).mean() - 1)
    x["volatility20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=10).std())
    x["trend20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean()) / c - 1
    for h in (1, 3, 5, 10):
        x[f"target_{h}"] = g["close"].shift(-h) / c - 1
    x["target"] = x[f"target_{int(horizon)}"]
    out = x.replace([np.inf, -np.inf], np.nan).dropna(subset=BASE_PANEL + ["target"]).reset_index(drop=True)
    return out


def split_by_time(df: pd.DataFrame, train_ratio=.60, valid_ratio=.20, embargo: int = 5):
    """Time split with an embargo so overlapping forward-return labels cannot cross folds."""
    dates = np.sort(df.eob.unique())
    a = int(len(dates) * train_ratio)
    b = int(len(dates) * (train_ratio + valid_ratio))
    embargo = max(0, int(embargo))
    if a < 30 or b <= a or b >= len(dates) - embargo:
        raise ValueError("时间切分后样本不足，请增加历史K线长度")
    train_dates = dates[: max(0, a - embargo)]
    valid_dates = dates[min(a + embargo, len(dates)): max(a + embargo, b - embargo)]
    test_dates = dates[min(b + embargo, len(dates)):]
    if min(len(train_dates), len(valid_dates), len(test_dates)) == 0:
        raise ValueError("embargo 后存在空数据集，请增加历史K线长度")
    return tuple(df[df.eob.isin(part)].copy() for part in (train_dates, valid_dates, test_dates))


def add_ai_score(train, valid, test, cfg):
    model = V42Transformer(
        seq_len=int(cfg.get("seq_len", 64)), epochs=int(cfg.get("epochs", 5)),
        batch_size=int(cfg.get("batch_size", 32)), d_model=int(cfg.get("d_model", 32)),
        heads=int(cfg.get("heads", 4)), layers=int(cfg.get("layers", 2)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        infer_batch_size=int(cfg.get("infer_batch_size", 128)),
    )
    model.fit(train, BASE_PANEL, "target")
    parts = []
    for p in (train.copy(), valid.copy(), test.copy()):
        p["ai_score"] = model.predict(p, BASE_PANEL)
        parts.append(p.dropna(subset=["ai_score"]).reset_index(drop=True))
    return model, tuple(parts)


def candidate_values(frame, candidate: Candidate):
    return np.asarray(evaluate_spec(frame, (candidate.feature_idx, candidate.transforms, candidate.op, candidate.weights, candidate.expression)), dtype=np.float32)


def candidate_frame(frame, candidate):
    out = frame.copy()
    out["factor_value"] = candidate_values(frame, candidate)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor_value"] + BASE_PANEL + ["target"])


def orient_candidate(candidate, mean_ic):
    if not np.isfinite(mean_ic) or mean_ic >= 0:
        return candidate
    return replace(candidate, weights=-candidate.weights, expression=f"(-1*{candidate.expression})")


def corr_gate(selected_values, value, threshold=.85):
    value = np.asarray(value, float)
    for old in selected_values:
        a = pd.Series(old).rank(); b = pd.Series(value).rank(); m = a.notna() & b.notna()
        if m.sum() and abs(float(a[m].corr(b[m]))) >= threshold:
            return False
    return True


def _save_transformer(model, path):
    if model.net is None:
        return False
    import torch
    payload = {"state_dict": model.net.state_dict(), "mu": model.mu, "sd": model.sd, "seq_len": model.seq_len,
               "d_model": model.d_model, "heads": model.heads, "layers": model.layers,
               "learning_rate": model.learning_rate}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return True


def _export_factor(candidate, report, out_path, signal_cfg):
    expr = candidate.expression
    code = f'''# Auto-generated by factor_gen V5.2\n# Weekly trend signal factor; horizon={signal_cfg.horizon} bars\n# validation_mean_ic={report.get("mean_ic", float("nan")):.8f}\n# validation_ic_ir={report.get("ic_ir", float("nan")):.8f}\nimport numpy as np\n\ndef identity(x): return x\ndef neg(x): return -x\ndef abs(x): return np.abs(x)\ndef tanh(x): return np.tanh(np.clip(x, -8, 8))\ndef square(x): return np.sign(x) * np.square(np.clip(x, -3, 3))\n\ndef _factor_one(df, ai_score=None):\n    x = df.copy()\n    if "symbol" in x.columns:\n        x = x.sort_values(["symbol", "eob"]).copy()\n    c=x["close"].astype(float).clip(lower=1e-12); o=x["open"].astype(float); h=x["high"].astype(float); l=x["low"].astype(float); v=x["volume"].astype(float)\n    g=x.groupby("symbol", sort=False) if "symbol" in x.columns else None\n    def pct(n): return g["close"].pct_change(n) if g is not None else x["close"].pct_change(n)\n    ret1=pct(1); ret5=pct(5); ret10=pct(10); range=(h-l)/c; body=(c-o)/c; close_pos=(c-l)/(h-l).replace(0,np.nan)\n    if g is not None:\n        vol_z20=g["volume"].transform(lambda s:(np.log1p(s.clip(lower=0))-np.log1p(s.clip(lower=0)).rolling(20,min_periods=10).mean())/(np.log1p(s.clip(lower=0)).rolling(20,min_periods=10).std()+1e-8))\n        ma_gap5=g["close"].transform(lambda s:s/s.rolling(5,min_periods=5).mean()-1); ma_gap20=g["close"].transform(lambda s:s/s.rolling(20,min_periods=20).mean()-1)\n        volatility20=g["close"].pct_change().groupby(x["symbol"], sort=False).transform(lambda s:s.rolling(20,min_periods=10).std()); trend20=g["close"].transform(lambda s:s.rolling(20,min_periods=10).mean())/c-1\n    else:\n        lv=np.log1p(v.clip(lower=0)); vol_z20=(lv-lv.rolling(20,min_periods=10).mean())/(lv.rolling(20,min_periods=10).std()+1e-8); ma_gap5=c/c.rolling(5,min_periods=5).mean()-1; ma_gap20=c/c.rolling(20,min_periods=20).mean()-1; volatility20=ret1.rolling(20,min_periods=10).std(); trend20=c.rolling(20,min_periods=10).mean()/c-1\n    if ai_score is None: ai_score=x.get("ai_score")\n    if ai_score is None: raise ValueError("该因子依赖 Transformer ai_score")\n    locals_dict=locals()\n    score={expr}\n    return np.asarray(score, dtype=float)\n\ndef factor(df, ai_score=None):\n    if "symbol" not in df.columns:\n        return _factor_one(df, ai_score)\n    order=df.index\n    x=df.sort_values(["symbol", "eob"]).copy()\n    y=_factor_one(x, ai_score.loc[x.index] if hasattr(ai_score, "loc") else ai_score)\n    out=np.full(len(df), np.nan, dtype=float)\n    out[df.index.get_indexer(x.index)] = y\n    return out\n\ndef signal(df, ai_score=None):\n    score=factor(df, ai_score)\n    return np.where(score >= {signal_cfg.buy_threshold}, 1, np.where(score <= {signal_cfg.sell_threshold}, -1, 0)).astype(np.int8)\n'''
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(code, encoding="utf-8")


def _selection_key(item):
    valid = item["valid"]
    regimes = item.get("regime_valid", {})
    regime_means = [v.get("mean_ic", np.nan) for v in regimes.values() if np.isfinite(v.get("mean_ic", np.nan))]
    regime_floor = float(np.min(regime_means)) if regime_means else -np.inf
    return (valid.get("mean_ic", -np.inf), valid.get("ic_ir", -np.inf), regime_floor, valid.get("median_ic", -np.inf))


def run(train, valid, test, cfg=None):
    cfg = cfg or {}
    out = Path(cfg.get("output_dir", "generated_v52")); out.mkdir(parents=True, exist_ok=True)
    horizon = int(cfg.get("horizon", 5))
    signal_cfg = SignalConfig(horizon=horizon, buy_threshold=float(cfg.get("buy_threshold", .03)), sell_threshold=float(cfg.get("sell_threshold", -.03)), flat_threshold=float(cfg.get("flat_threshold", .01)), stop_loss=float(cfg.get("stop_loss", -.04)), take_profit=float(cfg.get("take_profit", .08)))
    model, (train, valid, test) = add_ai_score(train, valid, test, cfg)
    from .search.qsk_miner import search
    candidates = search(train, int(cfg.get("candidate_count", 1_000_000)), int(cfg.get("top_k", 3000)), int(cfg.get("seed", 42)), int(cfg.get("screen_batch", 100_000)))
    screened = []
    for c in candidates:
        values = candidate_values(valid, c)
        s = cheap_screen(values, valid["target"].to_numpy(float), valid["eob"].to_numpy(), float(cfg.get("cheap_threshold", .01)))
        if np.isfinite(s): screened.append((s, c, values))
    screened.sort(key=lambda z: z[0], reverse=True)
    exact_limit = int(cfg.get("validation_top_k", 200)); ranked = []
    for i, (_, c, values) in enumerate(screened[:exact_limit], 1):
        r = fast_factor_report(valid, values, "target", float(cfg.get("ic_threshold", .03)), float(cfg.get("min_positive_window_ratio", .60)))
        ranked.append((r.get("mean_ic", -np.inf), c, r, values))
        if i % 25 == 0: LOG.info("exact validation %d/%d", i, min(exact_limit, len(screened)))
    ranked.sort(key=lambda z: (z[2].get("mean_ic", -np.inf), z[2].get("ic_ir", -np.inf), z[2].get("median_ic", -np.inf)), reverse=True)
    frozen = []; frozen_values = []
    for _, c, vr, _ in ranked:
        c = orient_candidate(c, vr.get("mean_ic", np.nan)); vf = candidate_frame(valid, c)
        if len(vf) == 0 or not corr_gate(frozen_values, vf["factor_value"].to_numpy(float), float(cfg.get("correlation_gate", .85))): continue
        tr = fast_factor_report(train, candidate_values(train, c), "target", float(cfg.get("ic_threshold", .03)), float(cfg.get("min_positive_window_ratio", .60)))
        folds = rolling_fold_scores(vf, vf["factor_value"].to_numpy(float), "target", int(cfg.get("walk_forward_folds", 4)), float(cfg.get("ic_threshold", .03)))
        regime_valid = regime_ic_report(vf, vf["factor_value"].to_numpy(float), "target")
        frozen.append({"candidate": c, "train": tr, "valid": vr, "walk_forward": folds, "regime_valid": regime_valid})
        frozen_values.append(vf["factor_value"].to_numpy(float))
        if len(frozen) >= int(cfg.get("final_top_k", 20)): break
    if not frozen: raise RuntimeError("没有通过候选验证与相关性门控的候选因子")
    evaluated = []
    for item in frozen:
        tf = candidate_frame(test, item["candidate"])
        te = fast_factor_report(tf, tf["factor_value"].to_numpy(float), "target", float(cfg.get("ic_threshold", .03)), float(cfg.get("min_positive_window_ratio", .60))) if len(tf) else {"status": "INSUFFICIENT_DATA"}
        regime_test = regime_ic_report(tf, tf["factor_value"].to_numpy(float), "target") if len(tf) else {}
        evaluated.append({**item, "test": te, "regime_test": regime_test, "signal": signal_backtest_stats(tf, "factor_value", signal_cfg) if len(tf) else {}})
    best = sorted(evaluated, key=_selection_key, reverse=True)[0]
    c = best["candidate"]; factor_path = out / "weekly_signal_factor_v52.py"; _export_factor(c, best["valid"], factor_path, signal_cfg)
    model_path = out / "ai_transformer_v52.pt"; model_saved = _save_transformer(model, model_path)
    payload = {"version": "5.2-regime-aware-weekly-signal", "horizon": horizon, "status": best["test"].get("status"), "selection_basis": "validation_only_with_regime_stability", "expression": c.expression, "train": best["train"], "valid": best["valid"], "walk_forward": best["walk_forward"], "regime_valid": best["regime_valid"], "test": best["test"], "regime_test": best["regime_test"], "signal": best["signal"], "recovery": recovery_score(test), "model": str(model_path) if model_saved else None, "factor": str(factor_path), "search": {"candidates": len(candidates), "cheap_screened": len(screened), "exact_validated": min(exact_limit, len(screened))}, "top_candidates": [{"expression": x["candidate"].expression, "valid": x["valid"], "test": x["test"], "regime_valid": x["regime_valid"], "regime_test": x["regime_test"], "signal": x["signal"]} for x in evaluated]}
    (out / "factor_report_v52.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    LOG.info("FINAL %s horizon=%d testIC=%.5f mean=%.5f IR=%.3f active=%.2f recovery=%s", best["test"].get("status"), horizon, best["test"].get("ic", np.nan), best["test"].get("mean_ic", np.nan), best["test"].get("ic_ir", np.nan), best["signal"].get("active_ratio", np.nan), payload["recovery"].get("status"))
    return payload
