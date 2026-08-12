"""GM3-only market data adapter.

Uses gm.api.history when running inside 掘金3. No CSV is required.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta

LOG = logging.getLogger("factor_gen.gm3")

class GM3DataAdapter:
    def __init__(self, symbols=None, frequency="1d", count=1200):
        self.symbols = symbols or ["SHSE.600000"]
        self.frequency = frequency
        self.count = count

    def load(self):
        try:
            from gm.api import history
        except ImportError as exc:
            raise RuntimeError("V4必须在掘金3环境运行：未找到 gm.api") from exc
        frames = []
        for i, symbol in enumerate(self.symbols, 1):
            LOG.info("GM3 history %d/%d: %s", i, len(self.symbols), symbol)
            df = history(symbol=symbol, frequency=self.frequency,
                         fields="symbol,eob,open,high,low,close,volume,amount",
                         count=self.count, adjust=ADJUST_NONE)
            if df is not None and len(df):
                frames.append(df)
        if not frames:
            raise RuntimeError("GM3未返回任何K线，请检查账号权限、symbol和回测时间范围")
        try:
            import pandas as pd
            return pd.concat(frames, ignore_index=True)
        except ImportError as exc:
            raise RuntimeError("需要pandas；请使用掘金Python环境自带或安装pandas") from exc

try:
    from gm.api import ADJUST_NONE
except Exception:
    ADJUST_NONE = "ADJUST_NONE"
