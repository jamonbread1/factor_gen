import numpy as np
import pandas as pd

from factor_gen.evaluation.qsk import evaluate
from factor_gen.search.qsk_miner import search


def sample_frame(n_dates=30, n_symbols=30):
    rows=[]; rng=np.random.default_rng(7)
    for d in range(n_dates):
        for s in range(n_symbols):
            signal=rng.normal()
            rows.append({"eob":pd.Timestamp("2025-01-01")+pd.Timedelta(days=d),"symbol":f"S{s:03d}","ret1":signal,"ret5":signal+rng.normal(scale=.3),"ret10":rng.normal(),"range":abs(rng.normal()),"body":signal*.1+rng.normal(scale=.2),"close_pos":rng.random(),"vol_z20":rng.normal(),"ma_gap5":signal*.05+rng.normal(scale=.2),"ma_gap20":rng.normal(),"volatility20":abs(rng.normal()),"trend20":signal*.05+rng.normal(scale=.2),"ai_score":signal+rng.normal(scale=.2),"target":signal*.02+rng.normal(scale=.01),"target_1":signal*.02+rng.normal(scale=.01),"target_3":signal*.01+rng.normal(scale=.02),"target_5":signal*.008+rng.normal(scale=.02),"target_10":signal*.004+rng.normal(scale=.02)})
    return pd.DataFrame(rows)


def test_evaluator_has_required_metrics():
    r=evaluate(sample_frame())
    assert np.isfinite(r.mean_ic)
    assert "5" in r.decay
    assert np.isfinite(r.monotonicity)


def test_miner_returns_candidates():
    c=search(sample_frame(), budget=200, top_k=10, seed=42, batch=100)
    assert c
    assert c[0].expression
    assert len(c[0].feature_idx)==3
