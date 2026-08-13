from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger("factor_gen.qsk_miner")
TRANSFORMS = ("identity", "neg", "abs", "tanh", "square")
OPS = ("+", "-", "*")


@dataclass
class Candidate:
    expression: str
    weights: np.ndarray
    score: float = 0.0
    complexity: int = 1


BASE_FEATURES = [
    "ret1", "ret5", "ret10", "range", "body", "close_pos",
    "vol_z20", "ma_gap5", "ma_gap20", "volatility20", "trend20", "ai_score",
]


def transform(x, name):
    if name == "identity": return x
    if name == "neg": return -x
    if name == "abs": return np.abs(x)
    if name == "tanh": return np.tanh(np.clip(x, -8, 8))
    if name == "square": return np.sign(x) * np.square(np.clip(x, -3, 3))
    raise ValueError(name)


def build_specs(n, seed=42):
    rng = np.random.default_rng(seed)
    p = len(BASE_FEATURES)
    for _ in range(n):
        idx = rng.integers(0, p, 3)
        ta, tb = rng.choice(TRANSFORMS, 2)
        op = rng.choice(OPS)
        w = rng.normal(0, 1, 3).astype(np.float32)
        w /= np.linalg.norm(w) + 1e-12
        if op == "*": expr = f"({w[0]:+.7f}*{ta}({BASE_FEATURES[idx[0]]})*{w[1]:+.7f}*{tb}({BASE_FEATURES[idx[1]]})+{w[2]:+.7f}*{BASE_FEATURES[idx[2]]})"
        elif op == "+": expr = f"({w[0]:+.7f}*{ta}({BASE_FEATURES[idx[0]]})+{w[1]:+.7f}*{tb}({BASE_FEATURES[idx[1]]})+{w[2]:+.7f}*{BASE_FEATURES[idx[2]]})"
        else: expr = f"({w[0]:+.7f}*{ta}({BASE_FEATURES[idx[0]]})-{w[1]:+.7f}*{tb}({BASE_FEATURES[idx[1]]})+{w[2]:+.7f}*{BASE_FEATURES[idx[2]]})"
        yield idx, (ta, tb), op, w, expr


def _screen_score(x, y):
    mask = np.isfinite(x).all(1) & np.isfinite(y)
    if mask.sum() < 100: return 0.0
    xx = x[mask]; yy = y[mask]
    xx = (xx - xx.mean(0)) / (xx.std(0) + 1e-8)
    yy = (yy - yy.mean()) / (yy.std() + 1e-8)
    return float(abs(np.mean(xx.sum(1) * yy)))


def search(frame, budget=1_000_000, top_k=1000, seed=42, batch=100_000):
    cache = {f: frame[f].to_numpy(np.float32) for f in BASE_FEATURES}
    y = frame["target"].to_numpy(np.float32)
    keep = []
    for start in range(0, int(budget), batch):
        end = min(start + batch, int(budget))
        local = []
        for spec in build_specs(end - start, seed=seed + start):
            idx, (ta, tb), op, w, expr = spec
            a = transform(cache[BASE_FEATURES[idx[0]]], ta)
            b = transform(cache[BASE_FEATURES[idx[1]]], tb)
            c = cache[BASE_FEATURES[idx[2]]]
            if op == "*": value = w[0] * a * (w[1] * b) + w[2] * c
            elif op == "+": value = w[0] * a + w[1] * b + w[2] * c
            else: value = w[0] * a - w[1] * b + w[2] * c
            mask = np.isfinite(value) & np.isfinite(y)
            if mask.sum() < 100: continue
            xv = value[mask]; yv = y[mask]
            xv = xv - xv.mean(); yv = yv - yv.mean()
            score = abs(float(np.dot(xv, yv) / ((np.linalg.norm(xv) + 1e-12) * (np.linalg.norm(yv) + 1e-12))))
            local.append((score, expr, w, idx, ta, tb, op))
        local.sort(key=lambda z: z[0], reverse=True)
        keep.extend(local[: max(10, top_k // 10)])
        keep.sort(key=lambda z: z[0], reverse=True); keep = keep[:top_k]
        LOG.info("screen %d/%d", end, budget)
    return [Candidate(expression=x[1], weights=x[2], score=float(x[0]), complexity=3) for x in keep]
