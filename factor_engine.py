from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import spearmanr
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

FEATURES = ["open", "high", "low", "close", "volume"]


class FactorConfig(PretrainedConfig):
    model_type = "gm3-factor-transformer"

    def __init__(self, input_dim=5, hidden_size=32, num_layers=2, nhead=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout


class FactorTransformer(PreTrainedModel):
    config_class = FactorConfig

    def __init__(self, config: FactorConfig):
        super().__init__(config)
        self.proj = nn.Linear(config.input_dim, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=config.nhead,
            dim_feedforward=config.hidden_size * 4, dropout=config.dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Sequential(nn.Linear(config.hidden_size, 16), nn.GELU(), nn.Linear(16, 1))
        self.post_init()

    def forward(self, x):
        z = self.proj(x)
        z = self.encoder(z)
        z = self.norm(z[:, -1])
        return self.head(z).squeeze(-1)


@dataclass
class ICResult:
    ic: float
    rank_ic: float
    windows: int
    mean_ic: float
    median_ic: float
    std_ic: float
    ic_ir: float
    positive_ratio: float
    status: str


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 20:
        return 0.0
    value = spearmanr(a[mask], b[mask]).statistic
    return float(value) if np.isfinite(value) else 0.0


def evaluate_ic(score: pd.Series, target: pd.Series, window: int = 252, threshold: float = 0.03) -> ICResult:
    frame = pd.DataFrame({"score": score, "target": target}).dropna()
    if len(frame) < 40:
        return ICResult(0, 0, 0, 0, 0, 0, 0, 0, "INSUFFICIENT_DATA")
    values = []
    for end in range(window, len(frame) + 1, window):
        chunk = frame.iloc[end - window:end]
        values.append(safe_spearman(chunk.score.to_numpy(), chunk.target.to_numpy()))
    remainder = len(frame) % window
    if remainder >= max(20, window // 4):
        chunk = frame.iloc[-remainder:]
        values.append(safe_spearman(chunk.score.to_numpy(), chunk.target.to_numpy()))
    if not values:
        values = [safe_spearman(frame.score.to_numpy(), frame.target.to_numpy())]
    arr = np.asarray(values, dtype=float)
    mean_ic = float(arr.mean())
    median_ic = float(np.median(arr))
    std_ic = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ic_ir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    positive_ratio = float((arr > threshold).mean())
    status = "PASS" if mean_ic > threshold and median_ic > threshold and positive_ratio >= 0.60 else "FAIL"
    overall = safe_spearman(frame.score.to_numpy(), frame.target.to_numpy())
    return ICResult(overall, overall, len(arr), mean_ic, median_ic, std_ic, float(ic_ir), positive_ratio, status)


def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    if "return" in df.columns:
        return pd.to_numeric(df["return"], errors="coerce")
    return df["close"].pct_change(horizon).shift(-horizon)


class FactorEngine:
    def __init__(self, df: pd.DataFrame, seq_len: int = 64, device: str = "auto"):
        self.df = df.copy()
        missing = [c for c in FEATURES if c not in self.df.columns]
        if missing:
            raise ValueError(f"K线缺少字段: {missing}; 需要 {FEATURES}")
        self.seq_len = seq_len
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        logger.info(f"计算设备: {self.device}")
        if self.device.type == "cuda":
            p = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {p.name} | 显存: {p.total_memory / 1024**3:.2f} GB")
        self.model = FactorTransformer(FactorConfig()).to(self.device)

    def normalized_features(self) -> np.ndarray:
        x = self.df[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        close = x.close.clip(lower=1e-12)
        feat = pd.DataFrame({
            "ret1": close.pct_change(),
            "range": (x.high - x.low) / close,
            "body": (x.close - x.open) / close,
            "vol_chg": x.volume.replace(0, np.nan).pct_change(),
            "close_pos": (x.close - x.low) / (x.high - x.low).replace(0, np.nan),
        }).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10, 10)
        return feat.to_numpy(np.float32)

    def build_sequences(self, features: np.ndarray, target: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = target.to_numpy(np.float32)
        xs, ys, idx = [], [], []
        for i in range(self.seq_len - 1, len(features)):
            if not np.isfinite(y[i]):
                continue
            xs.append(features[i - self.seq_len + 1:i + 1])
            ys.append(y[i])
            idx.append(i)
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(idx, np.int64)

    def train(self, epochs: int = 5, batch_size: int = 32, lr: float = 2e-4) -> pd.Series:
        features = self.normalized_features()
        target = make_target(self.df)
        X, y, idx = self.build_sequences(features, target)
        if len(X) < 100:
            raise ValueError("有效K线不足，至少需要约100条可训练样本")
        split = max(1, int(len(X) * 0.8))
        train_x, train_y = torch.from_numpy(X[:split]).to(self.device), torch.from_numpy(y[:split]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        loss_fn = nn.SmoothL1Loss()
        amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            for start in range(0, len(train_x), batch_size):
                xb, yb = train_x[start:start + batch_size], train_y[start:start + batch_size]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    loss = loss_fn(self.model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                total += float(loss.detach()) * len(xb)
            logger.info(f"Transformer训练 {epoch + 1}/{epochs} | loss={total / len(train_x):.6f}")
        self.model.eval()
        all_scores = np.full(len(self.df), np.nan, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start:start + batch_size]).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    pred = self.model(xb).float().cpu().numpy()
                all_scores[idx[start:start + batch_size]] = pred
        return pd.Series(all_scores, index=self.df.index, name="ai_factor")
