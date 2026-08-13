from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np

LOG = logging.getLogger("factor_gen.qsk_miner")
TRANSFORMS = ("identity", "neg", "abs", "tanh", "square")
OPS = ("+", "-", "*")
BASE_FEATURES = ["ret1", "ret5", "ret10", "range", "body", "close_pos", "vol_z20", "ma_gap5", "ma_gap20", "volatility20", "trend20", "ai_score"]

@dataclass
class Candidate:
    expression: str
    weights: np.ndarray
    feature_idx: tuple[int, int, int]
    transforms: tuple[str, str]
    op: str
    score: float = 0.0
    complexity: int = 3

def transform(x, name):
    if name == "identity": return x
    if name == "neg": return -x
    if name == "abs": return np.abs(x)
    if name == "tanh": return np.tanh(np.clip(x, -8, 8))
    if name == "square": return np.sign(x) * np.square(np.clip(x, -3, 3))
    raise ValueError(name)

def build_specs(n, seed=42):
    rng = np.random.default_rng(seed); p = len(BASE_FEATURES)
    for _ in range(n):
        idx = tuple(int(v) for v in rng.integers(0, p, 3))
        ta, tb = tuple(str(v) for v in rng.choice(TRANSFORMS, 2))
        op = str(rng.choice(OPS))
        w = rng.normal(0, 1, 3).astype(np.float32); w /= np.linalg.norm(w) + 1e-12
        if op == "*": expr = f"({w[0]:+.7f}*{ta}({BASE_FEATURES[idx[0]]})*{w[1]:+.7f}*{tb}({BASE_FEATURES[idx[1]]})+{w[2]:+.7f}*{BASE_FEATURES[idx[2]]})"
        elif op == "+": expr = f"({w[0]:+.7f}*{ta}({BASE_FEATURES[idx[0]]})+{w[1]:+.7f}*{tb}({BASE_FEATURES[idx[1]]})+{w[2]:+.7f}*{BASE_FEATURES[idx[2]]})"
        else: expr = f"({w[0]:+.7f}*{ta}({BASE_FEATURES[idx[0]]})-{w[1]:+.7f}*{tb}({BASE_FEATURES[idx[1]]})+{w[2]:+.7f}*{BASE_FEATURES[idx[2]]})"
        yield idx, (ta, tb), op, w, expr

def evaluate_spec(frame, spec):
    idx, (ta, tb), op, w, _ = spec
    a = transform(frame[BASE_FEATURES[idx[0]]].to_numpy(np.float32), ta)
    b = transform(frame[BASE_FEATURES[idx[1]]].to_numpy(np.float32), tb)
    c = frame[BASE_FEATURES[idx[2]]].to_numpy(np.float32)
    if op == "*": return w[0] * a * (w[1] * b) + w[2] * c
    if op == "+": return w[0] * a + w[1] * b + w[2] * c
    return w[0] * a - w[1] * b + w[2] * c

def search(frame, budget=1_000_000, top_k=1000, seed=42, batch=100_000):
    y = frame["target"].to_numpy(np.float32); keep = []
    for start in range(0, int(budget), batch):
        end = min(start + batch, int(budget)); local = []
        for spec in build_specs(end - start, seed=seed + start):
            value = np.nan_to_num(evaluate_spec(frame, spec), nan=0.0, posinf=0.0, neginf=0.0)
            mask = np.isfinite(value) & np.isfinite(y)
            if mask.sum() < 100: continue
            xv = value[mask] - value[mask].mean(); yv = y[mask] - y[mask].mean()
            score = abs(float(np.dot(xv, yv) / ((np.linalg.norm(xv) + 1e-12) * (np.linalg.norm(yv) + 1e-12))))
            local.append((score, spec))
        local.sort(key=lambda z: z[0], reverse=True); keep.extend(local[: max(10, top_k // 10)])
        keep.sort(key=lambda z: z[0], reverse=True); keep = keep[:top_k]
        LOG.info("screen %d/%d", end, budget)
    out = []
    for score, spec in keep:
        idx, tr, op, w, expr = spec
        out.append(Candidate(expr, w, idx, tr, op, float(score), 3))
    return out
