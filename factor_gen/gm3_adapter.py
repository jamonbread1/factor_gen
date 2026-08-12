"""GM3 market-data adapter for execution inside 掘金3."""
from __future__ import annotations
import logging

LOG = logging.getLogger("factor_gen.gm3")
try:
    from gm.api import ADJUST_NONE, history, get_constituents
except Exception:
    ADJUST_NONE = None
    history = None
    get_constituents = None


class GM3DataAdapter:
    def __init__(self, index="SHSE.000300", symbols=None, frequency="1d", count=1200, max_symbols=300):
        self.index = index
        self.symbols = list(symbols) if symbols else None
        self.frequency = frequency
        self.count = int(count)
        self.max_symbols = int(max_symbols)

    def _resolve_symbols(self):
        if self.symbols:
            return self.symbols[: self.max_symbols]
        if get_constituents is None:
            raise RuntimeError("必须在掘金3环境运行：未找到 gm.api.get_constituents")
        try:
            result = get_constituents(index=self.index, df=True)
        except TypeError:
            result = get_constituents(index=self.index)
        if result is None:
            raise RuntimeError(f"GM3未返回指数成分股: {self.index}")
        if hasattr(result, "columns"):
            col = "symbol" if "symbol" in result.columns else result.columns[0]
            symbols = result[col].dropna().astype(str).tolist()
        elif isinstance(result, dict):
            symbols = result.get("symbol") or result.get("symbols") or []
        else:
            symbols = list(result)
        symbols = list(dict.fromkeys(str(s) for s in symbols if s))
        if not symbols:
            raise RuntimeError(f"GM3未返回有效指数成分股: {self.index}")
        return symbols[: self.max_symbols]

    def load(self):
        if history is None:
            raise RuntimeError("必须在掘金3环境运行：未找到 gm.api")
        import pandas as pd
        symbols = self._resolve_symbols()
        LOG.info("[GM3] universe=%s symbols=%d bars=%d", self.index, len(symbols), self.count)
        frames = []
        for i, symbol in enumerate(symbols, 1):
            LOG.info("[GM3] history %d/%d: %s", i, len(symbols), symbol)
            kwargs = dict(symbol=symbol, frequency=self.frequency,
                          fields="symbol,eob,open,high,low,close,volume,amount",
                          count=self.count, df=True)
            if ADJUST_NONE is not None:
                kwargs["adjust"] = ADJUST_NONE
            try:
                df = history(**kwargs)
            except TypeError:
                kwargs.pop("adjust", None)
                df = history(**kwargs)
            if df is None or len(df) == 0:
                LOG.warning("[GM3] no bars returned: %s", symbol)
                continue
            frames.append(df)
        if not frames:
            raise RuntimeError("GM3未返回K线：请检查标的、权限以及回测时间范围")
        data = pd.concat(frames, ignore_index=True)
        required = {"symbol", "eob", "open", "high", "low", "close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise RuntimeError(f"GM3返回数据缺少字段: {sorted(missing)}")
        return data.sort_values(["symbol", "eob"]).reset_index(drop=True)
