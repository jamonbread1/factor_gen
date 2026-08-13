# factor_gen V5.1 — Weekly Trend Signal Research

V5.1 changes the research objective from generic factor mining to **signed trading-signal discovery** for approximately one trading week (default 5 bars).

## Research contract

1. All targets are forward returns; the default target is `target_5`.
2. Transformer learns the 5-bar forward-return target from point-in-time historical features.
3. One million candidate expressions can still be searched, but validation is now two-stage:
   - cheap vectorized pre-screen across all candidates;
   - exact cross-sectional daily Spearman IC on only `validation_top_k` candidates (default 200).
4. Candidate orientation is normalized so positive score means positive expected forward return. This makes `+1` naturally mean BUY and `-1` SELL.
5. Candidate selection is based on Train/Validation only. Test remains out-of-sample.
6. Walk-forward validation is recorded on contiguous validation folds.
7. Final reports include IC, IC-IR, positive-window ratio, OOS signal activity and directional hit rate.

## Why validation is faster

The previous pipeline performed expensive full validation for every shortlisted expression. V5.1 reduces this to a cheap numerical screen followed by exact IC validation for a small configurable shortlist. For a 3,000-candidate run, the exact validation workload is reduced to 200 by default.

## Signal semantics

- `signal = +1`: BUY candidate
- `signal = 0`: no position / neutral
- `signal = -1`: SELL candidate

Default research thresholds are +3% / -3% on the factor score. These are research defaults, not guaranteed optimal execution thresholds.

## Run

```bash
python run_v51.py --start 2020-01-01 --end 2026-08-01 --index SHSE.000300
```

The output directory contains the weekly signal factor, Transformer checkpoint and complete research JSON.

## Production-research roadmap

Before using any discovered factor for live trading, the project should additionally enforce point-in-time constituent membership, corporate-action/adjustment consistency, transaction-cost/slippage modeling, liquidity constraints, regime and sector exposure checks, multiple independent OOS periods, and a final untouched paper-trading period.
