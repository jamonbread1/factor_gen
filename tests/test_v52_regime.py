import numpy as np
import pandas as pd

from factor_gen.qsk_v5 import make_panel, split_by_time
from factor_gen.regime import build_regime_frame, recovery_score


def bars(days=50, symbols=12):
    rows=[]
    for s in range(symbols):
        px=100.0+s
        for d in range(days):
            px *= 1.0 + 0.002 * np.sin(d / 5) + (0.0005 if s < symbols // 2 else -0.0002)
            rows.append({"eob": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d), "symbol": f"S{s}", "open": px, "high": px * 1.01, "low": px * .99, "close": px, "volume": 1000+s})
    return pd.DataFrame(rows)


def test_make_panel_never_calculates_returns_across_symbols():
    raw=bars(30, 4); panel=make_panel(raw, horizon=5)
    first=panel.groupby("symbol").head(1)
    assert first["ret1"].abs().max() < 0.01


def test_split_has_embargo():
    panel=make_panel(bars(80, 12), horizon=5)
    train, valid, test=split_by_time(panel, .6, .2, embargo=5)
    assert train.eob.max() < valid.eob.min()
    assert valid.eob.max() < test.eob.min()
    assert (valid.eob.min() - train.eob.max()).days >= 5
    assert (test.eob.min() - valid.eob.max()).days >= 5


def test_regime_uses_only_point_in_time_features():
    panel=make_panel(bars(60, 12), horizon=5)
    r=build_regime_frame(panel)
    assert {"breadth", "dispersion", "market_trend", "regime"}.issubset(r.columns)
    assert "target" not in r.columns
    assert len(r) > 0


def test_recovery_score_is_bounded():
    panel=make_panel(bars(60, 12), horizon=5)
    result=recovery_score(panel)
    assert 0 <= result["score"] <= 100
