import numpy as np
import pandas as pd

from factor_gen.signal import signal_from_score, SignalConfig, signal_backtest_stats
from factor_gen.evaluation.fast import fast_factor_report


def sample():
    rows=[]
    for d in range(20):
        for s in range(20):
            v=(s-10)/10 + np.sin(d/3)*0.1
            rows.append({"eob":pd.Timestamp("2025-01-01")+pd.Timedelta(days=d),"symbol":f"S{s}","factor_value":v,"target":v*0.05,"target_5":v*0.05})
    return pd.DataFrame(rows)


def test_signal_mapping():
    assert signal_from_score([-0.1,0,0.1],0.03,-0.03).tolist()==[-1,0,1]


def test_fast_report():
    df=sample(); r=fast_factor_report(df,df.factor_value.to_numpy())
    assert r["windows"]==20
    assert r["mean_ic"] > 0.9


def test_weekly_signal_stats():
    df=sample(); s=signal_backtest_stats(df,cfg=SignalConfig())
    assert s["buy_count"] > 0 and s["sell_count"] > 0
    assert np.isfinite(s["hit_rate"])
