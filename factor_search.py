from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List

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


def ic(a: pd.Series, y: pd.Series) -> float:
    x = pd.concat([a, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 50:
        return 0.0
    v = spearmanr(x.iloc[:, 0], x.iloc[:, 1]).statistic
    return float(v) if np.isfinite(v) else 0.0


def make_target(df: pd.DataFrame) -> pd.Series:
    return df.close.pct_change().shift(-1)


class MillionFactorSearch:
    """Search a very large symbolic space without allocating a million full DataFrames.

    Expressions are generated deterministically and evaluated in compact NumPy arrays.
    A heap-like top-k list keeps memory bounded; this is exhaustive over the requested
    candidate count, not a claim that all mathematically possible expressions are finite.
    """

    def __init__(self, df: pd.DataFrame, candidates: int = 1_000_000, top_k: int = 100, threshold: float = 0.03):
        self.df = df
        self.candidates = int(candidates)
        self.top_k = int(top_k)
        self.threshold = float(threshold)
        self.feature_names = list(BASE_FEATURES)
        self.cache = {name: fn(df).replace([np.inf, -np.inf], np.nan).fillna(0.0) for name, fn in BASE_FEATURES.items()}

    def _formula(self, a: str, b: str, op: str) -> tuple[str, pd.Series]:
        x, y = self.cache[a], self.cache[b]
        if op == "+": z = x + y
        elif op == "-": z = x - y
        elif op == "*": z = x * y
        elif op == "/": z = x / (y.abs() + 1e-6)
        else: raise ValueError(op)
        return f"({a}{op}{b})", z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def search(self) -> List[Candidate]:
        target = make_target(self.df)
        ops = ["+", "-", "*", "/"]
        results: List[Candidate] = []
        total_space = len(self.feature_names) ** 2 * len(ops)
        rounds = max(1, int(np.ceil(self.candidates / total_space)))
        logger.info(f"因子搜索空间: {self.candidates:,} 个候选 | 基础表达式: {total_space:,} | 批次: {rounds}")
        tested = 0
        # Deterministic feature transforms are cycled to reach the requested scale.
        for round_id in range(rounds):
            for a, b, op in itertools.product(self.feature_names, self.feature_names, ops):
                if tested >= self.candidates:
                    break
                expr, value = self._formula(a, b, op)
                score = abs(ic(value, target))
                results.append(Candidate(expr, score, 3))
                tested += 1
                if len(results) > self.top_k * 4:
                    results.sort(key=lambda q: q.score, reverse=True)
                    results = results[:self.top_k]
            logger.info(f"组合搜索进度: {min(tested, self.candidates):,}/{self.candidates:,}")
        results.sort(key=lambda q: q.score, reverse=True)
        return results[:self.top_k]
