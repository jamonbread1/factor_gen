import numpy as np
import pandas as pd

from factor_engine import evaluate_ic
from factor_search import MillionFactorSearch


def sample(n=600):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, .002, n)),
        "high": close * (1 + abs(rng.normal(0, .004, n))),
        "low": close * (1 - abs(rng.normal(0, .004, n))),
        "close": close,
        "volume": rng.lognormal(12, .3, n),
    })


def test_ic_api():
    d = sample()
    score = d.close.pct_change().fillna(0)
    result = evaluate_ic(score, d.close.pct_change().shift(-1))
    assert result.status in {"PASS", "FAIL", "INSUFFICIENT_DATA"}


def test_search_small():
    d = sample()
    result = MillionFactorSearch(d, candidates=200, top_k=5).search()
    assert len(result) == 5
    assert all(np.isfinite(x.score) for x in result)
