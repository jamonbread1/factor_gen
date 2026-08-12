from __future__ import annotations
import numpy as np


class V42Transformer:
    def __init__(self, seq_len=32, epochs=5, batch_size=32, d_model=32, heads=4, layers=2, infer_batch_size=128):
        self.seq_len, self.epochs, self.batch_size = seq_len, epochs, batch_size
        self.d_model, self.heads, self.layers = d_model, heads, layers
        self.infer_batch_size = infer_batch_size
        self.net = None
        self.mu = None
        self.sd = None
        self.device = None

    def _windows(self, X, y=None):
        n = max(0, len(X) - self.seq_len)
        if n == 0:
            return np.empty((0, self.seq_len, X.shape[1]), np.float32), np.empty((0,), np.float32) if y is not None else None
        # Allocate only once; this is used for training where the training set is intentionally materialized.
        wx = np.lib.stride_tricks.sliding_window_view(X, (self.seq_len, X.shape[1]))[:, 0, :, :].copy().astype(np.float32, copy=False)
        wy = y[self.seq_len:].astype(np.float32, copy=False) if y is not None else None
        return wx, wy

    def _get_device(self, torch):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Transformer] CUDA enabled: {torch.cuda.get_device_name(0)}")
            print(f"[Transformer] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return device
        print("[Transformer] WARNING: CUDA unavailable; using CPU")
        print(f"[Transformer] torch={torch.__version__} torch.version.cuda={torch.version.cuda}")
        return torch.device("cpu")

    def _iter_predict_batches(self, X):
        """Yield (start_index, batch_windows) without materializing all inference windows."""
        for end in range(self.seq_len + self.infer_batch_size, len(X) + 1, self.infer_batch_size):
            start_window = end - self.infer_batch_size
            if start_window < self.seq_len:
                start_window = self.seq_len
            first = start_window - self.seq_len
            last = end - self.seq_len
            if last <= first:
                continue
            wx = np.empty((last - first, self.seq_len, X.shape[1]), dtype=np.float32)
            for j, i in enumerate(range(first, last)):
                wx[j] = X[i:i + self.seq_len]
            yield start_window, wx
        # Tail batch.
        first = max(0, len(X) - self.seq_len - ((len(X) - self.seq_len) % self.infer_batch_size))
        last = len(X) - self.seq_len
        if last > first:
            wx = np.empty((last - first, self.seq_len, X.shape[1]), dtype=np.float32)
            for j, i in enumerate(range(first, last)):
                wx[j] = X[i:i + self.seq_len]
            yield first + self.seq_len, wx

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

    def _predict_one(self, frame, features, out, index_positions=None):
        import torch
        X = frame[features].to_numpy(np.float32)
        X = np.nan_to_num((X - self.mu) / self.sd)
        if len(X) <= self.seq_len:
            return
        device = next(self.net.parameters()).device
        self.net.eval()
        total = len(X) - self.seq_len
        done = 0
        with torch.inference_mode():
            for start_pos, wx in self._iter_predict_batches(X):
                tx = torch.from_numpy(wx).to(device, non_blocking=True)
                h = self.net[0](tx)
                h = self.net[1](h)
                pred = self.net[2](h[:, -1, :]).squeeze(-1).cpu().numpy()
                begin = start_pos
                end = begin + len(pred)
                if index_positions is None:
                    out[begin:end] = pred
                else:
                    out[index_positions[begin:end]] = pred
                done += len(pred)
        print(f"[Transformer] inference {done}/{total} samples batch={self.infer_batch_size} device={device}")

    def predict(self, frame, features):
        out = np.full(len(frame), np.nan, np.float32)
        if self.net is None or len(frame) == 0:
            return out
        # Never let a sequence cross from one stock into another. This also makes inference memory bounded.
        if "symbol" in frame.columns:
            groups = frame.groupby("symbol", sort=False).indices
            for symbol, positions in groups.items():
                positions = np.asarray(positions, dtype=np.int64)
                # Preserve the frame's existing per-symbol order.
                sub = frame.iloc[positions]
                self._predict_one(sub, features, out, positions)
        else:
            self._predict_one(frame, features, out)
        return out
