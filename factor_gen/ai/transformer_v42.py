from __future__ import annotations
import numpy as np


class V42Transformer:
    def __init__(self, seq_len=32, epochs=5, batch_size=32, d_model=32, heads=4, layers=2):
        self.seq_len, self.epochs, self.batch_size = seq_len, epochs, batch_size
        self.d_model, self.heads, self.layers = d_model, heads, layers
        self.net = None
        self.mu = None
        self.sd = None
        self.device = None

    def _windows(self, X, y=None):
        xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            xs.append(X[i-self.seq_len:i])
            if y is not None:
                ys.append(y[i])
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32) if y is not None else None

    def _get_device(self, torch):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Transformer] CUDA enabled: {torch.cuda.get_device_name(0)}")
            print(f"[Transformer] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return device
        print("[Transformer] WARNING: CUDA unavailable; using CPU")
        print(f"[Transformer] torch={torch.__version__} torch.version.cuda={torch.version.cuda}")
        return torch.device("cpu")

    def fit(self, frame, features, target):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.net = None
            print("[Transformer] PyTorch not installed; AI score disabled")
            return self

        X = frame[features].to_numpy(np.float32)
        y = frame[target].to_numpy(np.float32)
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0) + 1e-6
        X = np.nan_to_num((X - self.mu) / self.sd)
        wx, wy = self._windows(X, y)
        if len(wx) == 0:
            print("[Transformer] no training windows")
            return self

        self.device = self._get_device(torch)
        proj = nn.Linear(X.shape[1], self.d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.heads, batch_first=True)
        enc = nn.TransformerEncoder(enc_layer, self.layers)
        head = nn.Linear(self.d_model, 1)
        self.net = nn.ModuleList([proj, enc, head]).to(self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.HuberLoss()
        tx = torch.from_numpy(wx).to(self.device)
        ty = torch.from_numpy(wy).to(self.device)

        print(f"[Transformer] training samples={len(tx)} batch={self.batch_size} device={self.device}")
        for ep in range(self.epochs):
            self.net.train()
            total = 0.0
            batches = 0
            order = torch.randperm(len(tx), device=self.device)
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start+self.batch_size]
                h = self.net[0](tx[idx])
                h = self.net[1](h)
                # Use the final token as the sequence representation.
                pred = self.net[2](h[:, -1, :]).squeeze(-1)
                loss = loss_fn(pred, ty[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.detach().item()
                batches += 1
            print(f"[Transformer] epoch {ep+1}/{self.epochs} loss={total/max(1,batches):.6f} device={self.device}")
        return self

    def predict(self, frame, features):
        out = np.full(len(frame), np.nan, np.float32)
        if self.net is None:
            return out
        import torch
        X = frame[features].to_numpy(np.float32)
        X = np.nan_to_num((X - self.mu) / self.sd)
        wx, _ = self._windows(X)
        if len(wx) == 0:
            return out
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            h = self.net[0](torch.from_numpy(wx).to(device))
            h = self.net[1](h)
            pred = self.net[2](h[:, -1, :]).squeeze(-1).cpu().numpy()
        out[self.seq_len:] = pred
        return out
