from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm


@dataclass
class Candidate:
    expression: str
    score: float
    complexity: int


BASE_FEATURES = {
    "ret1": lambda d: d.close.pct_change(),
    "ret5": lambda d: d.close.pct_change(5),
    "ret10": lambda d: d.close.pct_change(10),
    "range": lambda d: (d.high - d.low) / d.close,
    "body": lambda d: (d.close - d.open) / d.close,
    "close_pos": lambda d: (d.close - d.low) / (d.high - d.low).replace(0, np.nan),
    "vol_chg": lambda d: d.volume.replace(0, np.nan).pct_change(),
    "vol_z20": lambda d: (d.volume - d.volume.rolling(20).mean()) / d.volume.rolling(20).std(),
    "ma_gap5": lambda d: d.close / d.close.rolling(5).mean() - 1,
    "ma_gap20": lambda d: d.close / d.close.rolling(20).mean() - 1,
}
TRANSFORMS = ["identity", "neg", "abs", "square", "tanh"]
OPS = ["+", "-", "*"]


def ic(a: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(y)
    if mask.sum() < 50:
        return 0.0
    v = spearmanr(a[mask], y[mask]).statistic
    return float(v) if np.isfinite(v) else 0.0


def make_target(df: pd.DataFrame) -> pd.Series:
    return df.close.pct_change().shift(-1)


def apply_transform(x: np.ndarray, name: str) -> np.ndarray:
    if name == "identity": return x
    if name == "neg": return -x
    if name == "abs": return np.abs(x)
    if name == "square": return np.sign(x) * np.square(np.clip(x, -5, 5))
    if name == "tanh": return np.tanh(x)
    raise ValueError(name)


class MillionFactorSearch:
    """Exhaustive deterministic search over 1M+ generated expressions.

    Stage 1 uses vectorized Pearson screening in batches; Stage 2 computes exact
    Spearman IC on the strongest candidates, avoiding million-size DataFrames.
    """
    def __init__(self, df: pd.DataFrame, candidates: int = 1_000_000, top_k: int = 100, threshold: float = 0.03, seed: int = 42):
        self.df = df
        self.candidates = int(candidates)
        self.top_k = int(top_k)
        self.threshold = float(threshold)
        self.names = list(BASE_FEATURES)
        self.cache = {n: BASE_FEATURES[n](df).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32) for n in self.names}

    def _candidate(self, i: int):
        a = self.names[i % len(self.names)]
        b = self.names[(i // len(self.names)) % len(self.names)]
        c = self.names[(i // (len(self.names) ** 2)) % len(self.names)]
        ta = TRANSFORMS[(i // 7) % len(TRANSFORMS)]
        tb = TRANSFORMS[(i // 11) % len(TRANSFORMS)]
        op = OPS[(i // 17) % len(OPS)]
        wa = ((i * 1103515245 + 12345) % 2001 - 1000) / 1000.0
        wb = ((i * 214013 + 2531011) % 2001 - 1000) / 1000.0
        wc = ((i * 134775813 + 1) % 1001 - 500) / 1000.0
        if abs(wa) < 0.05: wa = 0.25
        if abs(wb) < 0.05: wb = -0.25
        if op == "+": expr = f"({wa:.3f}*{ta}({a})+{wb:.3f}*{tb}({b})+{wc:.3f}*{c})"
        elif op == "-": expr = f"({wa:.3f}*{ta}({a)}-{wb:.3f}*{tb}({b})+{wc:.3f}*{c})"
        else: expr = f"({wa:.3f}*{ta}({a)}*{tb}({b})+{wc:.3f}*{c})"
        return expr, a, b, c, ta, tb, op, wa, wb, wc

    def _eval(self, spec):
        _, a, b, c, ta, tb, op, wa, wb, wc = spec
        x = apply_transform(self.cache[a], ta)
        y = apply_transform(self.cache[b], tb)
        z = self.cache[c]
        if op == "+": v = wa * x + wb * y + wc * z
        elif op == "-": v = wa * x - wb * y + wc * z
        else: v = wa * x * y + wc * z
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def search(self) -> List[Candidate]:
        target = make_target(self.df).to_numpy(np.float32)
        valid = np.isfinite(target)
        y = target[valid]
        y0 = y - y.mean(); yn = np.sqrt(np.sum(y0 * y0)) + 1e-12
        batch_size = 4096
        stage1 = []
        logger.info(f"因子搜索空间: {self.candidates:,} 个唯一候选 | 预筛 batch={batch_size}")
        for start in tqdm(range(0, self.candidates, batch_size), desc="百万组合预筛"):
            end = min(start + batch_size, self.candidates)
            for i in range(start, end):
                spec = self._candidate(i)
                v = self._eval(spec)[valid]
                v0 = v - v.mean()
                pearson = abs(float(np.sum(v0 * y0) / ((np.sqrt(np.sum(v0 * v0)) + 1e-12) * yn)))
                stage1.append((pearson, i, spec))
            if len(stage1) > self.top_k * 20:
                stage1.sort(key=lambda q: q[0], reverse=True); stage1 = stage1[:self.top_k * 10]
        stage1.sort(key=lambda q: q[0], reverse=True)
        stage1 = stage1[:self.top_k * 10]
        logger.info(f"预筛完成: {self.candidates:,} -> {len(stage1)}")
        exact = []
        for _, _, spec in tqdm(stage1, desc="精确 Spearman IC"):
            value = self._eval(spec)
            score = abs(ic(value[valid], target[valid]))
            exact.append(Candidate(spec[0], score, 4))
        exact.sort(key=lambda q: q.score, reverse=True)
        return exact[:self.top_k]
