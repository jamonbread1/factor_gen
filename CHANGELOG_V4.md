# V4 release history

## V4.1
- GM3-only market data path.
- Multi-symbol/index universe support.
- Train/validation/test time split.
- Cross-sectional daily Spearman IC.
- Batched candidate screening to avoid 1M x N memory allocation.

## V4.2 — Formal
- `main.py` is the sole root entry point.
- Train-only Transformer factor generation, CUDA/CPU fallback.
- One-million candidate analytical genome screen, followed by exact validation.
- Unseen test-set gate; no factor is marked PASS from training data alone.
- Formal PASS gate: validation mean IC > 0.03, test mean/median IC > 0.03, positive IC ratio >= 60%, IC-IR >= 0.50.
- GM3 index constituent universe; no CSV dependency.
- Explicit terminal output for status, IC, IR, positive ratio and formula.

Important: V4.2 does not fabricate a passing factor. If the unseen test does not meet the gate, status remains FAIL.
