"""GM3 market-data adapter for execution inside the 掘金3 environment."""
from __future__ import annotations
import logging

LOG = logging.getLogger("factor_gen.gm3")
try:
    from gm.api import ADJUST_NONE, history
except Exception:
    ADJUST_NONE = None
    history = None

class GM3DataAdapter:
    def __init__(self, symbols=None, frequency="1d", count=1200):
        self.symbols = symbols or ["SHSE.600000"]
        self.frequency = frequency
        self.count = int(count)

    def load(self):
        if history is None:
            raise RuntimeError("必须在掘金3环境运行：未找到 gm.api")
        import pandas as pd
        frames = []
        for i, symbol in enumerate(self.symbols, 1):
            LOG.info("[GM3] history %d/%d: %s", i, len(self.symbols), symbol)
            kwargs = dict(symbol=symbol, frequency=self.frequency,
                          fields="symbol,eob,open,high,low,close,volume,amount",
                          count=self.count, df=True)
            if ADJUST_NONE is not None:
                kwargs["adjust"] = ADJUST_NONE
            df = history(**kwargs)
            if df is None or len(df) == 0:
                LOG.warning("[GM3] no bars returned: %s", symbol)
                continue
            frames.append(df)
        if not frames:
            raise RuntimeError("GM3未返回K线：请检查标的、权限以及回测时间范围")
        data = pd.concat(frames, ignore_index=True)
        if "eob" not in data.columns:
            raise RuntimeError("GM3返回数据缺少 eob 时间列")
        sort_cols = [c for c in ("symbol", "eob") if c in data.columns]
        return data.sort_values(sort_cols).reset_index(drop=True)
