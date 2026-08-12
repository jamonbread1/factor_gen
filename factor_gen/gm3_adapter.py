"""GM3 market-data adapter for execution inside 掘金3."""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger("factor_gen.gm3")

try:
    from gm.api import ADJUST_NONE, history_n, stk_get_index_constituents, set_token
except Exception:
    ADJUST_NONE = None
    history_n = None
    stk_get_index_constituents = None
    set_token = None


class GM3DataAdapter:
    def __init__(self, index="SHSE.000300", symbols=None, frequency="1d", count=1200, max_symbols=300, token=None):
        self.index = index
        self.symbols = list(symbols) if symbols else None
        self.frequency = frequency
        self.count = int(count)
        self.max_symbols = int(max_symbols)
        self.token = token or os.getenv("GM_TOKEN") or os.getenv("GM3_TOKEN")

    def _configure_token(self):
        if self.token and set_token is not None:
            set_token(self.token)
            LOG.info("[GM3] token configured from GM_TOKEN/GM3_TOKEN")

    def _resolve_symbols(self):
        if self.symbols:
            return self.symbols[: self.max_symbols]
        if stk_get_index_constituents is None:
            raise RuntimeError("当前GM3 SDK未提供 stk_get_index_constituents，请升级GM3 SDK")
        try:
            result = stk_get_index_constituents(index=self.index)
        except Exception as exc:
            msg = str(exc)
            if "1000" in msg or "token" in msg.lower():
                raise RuntimeError("GM3返回1000：错误或无效的token。请在掘金3【系统设置→密钥管理】检查token。") from exc
            raise RuntimeError(f"GM3获取指数成分股失败: {exc}") from exc
        if result is None:
            raise RuntimeError(f"GM3未返回指数成分股: {self.index}")
        if hasattr(result, "columns"):
            if "symbol" not in result.columns:
                raise RuntimeError(f"stk_get_index_constituents返回字段异常: {list(result.columns)}")
            symbols = result["symbol"].dropna().astype(str).tolist()
        elif isinstance(result, dict):
            symbols = result.get("symbol") or result.get("symbols") or []
        else:
            symbols = list(result)
        symbols = list(dict.fromkeys(str(s) for s in symbols if s))
        if not symbols:
            raise RuntimeError(f"GM3未返回有效指数成分股: {self.index}")
        return symbols[: self.max_symbols]

    def load(self):
        if history_n is None:
            raise RuntimeError("必须在掘金3环境运行：未找到 gm.api.history_n")
        self._configure_token()
        import pandas as pd
        symbols = self._resolve_symbols()
        LOG.info("[GM3] universe=%s symbols=%d bars=%d", self.index, len(symbols), self.count)
        frames = []
        for i, symbol in enumerate(symbols, 1):
            LOG.info("[GM3] history_n %d/%d: %s", i, len(symbols), symbol)
            kwargs = dict(symbol=symbol, frequency=self.frequency,
                          count=self.count,
                          fields="symbol,eob,open,high,low,close,volume,amount",
                          df=True)
            if ADJUST_NONE is not None:
                kwargs["adjust"] = ADJUST_NONE
            try:
                df = history_n(**kwargs)
            except TypeError:
                # Some GM3 SDK builds differ on optional arguments. Retry with only
                # the stable history_n parameters before failing.
                kwargs.pop("adjust", None)
                try:
                    df = history_n(**kwargs)
                except TypeError:
                    kwargs.pop("df", None)
                    df = history_n(**kwargs)
            except Exception as exc:
                msg = str(exc)
                if "1000" in msg or "token" in msg.lower():
                    raise RuntimeError("GM3历史行情返回1000：token无效或未设置。请检查掘金3【系统设置→密钥管理】。") from exc
                raise
            if df is None or len(df) == 0:
                LOG.warning("[GM3] no bars returned: %s", symbol)
                continue
            if not hasattr(df, "columns"):
                raise RuntimeError("GM3 history_n 未返回DataFrame；请确认 df=True 可用。")
            frames.append(df)
        if not frames:
            raise RuntimeError("GM3未返回K线：请检查标的、token、权限以及回测时间范围")
        data = pd.concat(frames, ignore_index=True)
        required = {"symbol", "eob", "open", "high", "low", "close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise RuntimeError(f"GM3返回数据缺少字段: {sorted(missing)}")
        return data.sort_values(["symbol", "eob"]).reset_index(drop=True)
