from __future__ import annotations

import importlib
import os
from datetime import datetime, timedelta

import pandas as pd


class GM3DataAdapter:
    """GM3 market-data adapter with automatic history chunking."""

    MAX_RECORDS_PER_REQUEST = 30_000

    def __init__(self, token=None):
        self.gm = importlib.import_module("gm.api")
        token = token or os.getenv("GM_TOKEN")
        self.token = token
        if token and hasattr(self.gm, "set_token"):
            self.gm.set_token(token)

    def get_constituents(self, index="SHSE.000300", as_of=None):
        fn = getattr(self.gm, "stk_get_index_constituents", None)
        if fn is None:
            raise RuntimeError(
                "当前 gm.api 没有 stk_get_index_constituents，请升级 GoldMiner3/gm SDK。"
            )
        try:
            obj = fn(index=index, trade_date=as_of) if as_of is not None else fn(index=index)
        except Exception as exc:
            message = str(exc)
            if "1000" in message or "token" in message.lower():
                raise RuntimeError(
                    "GM3 成分股查询失败：token 无效或未配置。"
                    "请配置 GM_TOKEN 或 config.yaml 的 gm_token。原始错误: " + message
                ) from exc
            raise RuntimeError(f"GM3 stk_get_index_constituents 调用失败: {message}") from exc

        if obj is None:
            return []
        if isinstance(obj, pd.DataFrame):
            if "symbol" not in obj.columns:
                raise RuntimeError(f"返回结果缺少 symbol 字段: {list(obj.columns)}")
            return obj["symbol"].dropna().astype(str).drop_duplicates().tolist()
        if isinstance(obj, dict):
            return [str(x) for x in obj.get("symbol", obj.keys())]
        return [str(x) for x in obj]

    @staticmethod
    def _is_oversized_error(exc: Exception) -> bool:
        text = str(exc)
        return "1029" in text or "查询结果过大" in text or "超出最大限制" in text

    @staticmethod
    def _to_dt(value):
        return pd.Timestamp(value).to_pydatetime()

    def _history_once(self, symbols, start_date, end_date, frequency):
        fields = "symbol,eob,open,high,low,close,volume,amount"
        fn = getattr(self.gm, "history", None)
        if fn is None:
            raise RuntimeError("当前 gm.api 未找到 history，请检查 GM3 SDK 版本。")

        last_error = None
        for kw in (
            dict(symbol=list(symbols), frequency=frequency, start_time=start_date, end_time=end_date, fields=fields, df=True),
            dict(symbol=",".join(symbols), frequency=frequency, start_time=start_date, end_time=end_date, fields=fields, df=True),
        ):
            try:
                out = fn(**kw)
                if out is None:
                    return pd.DataFrame()
                return pd.DataFrame(out)
            except TypeError as exc:
                last_error = exc
                # Some older SDK builds do not accept df=True.
                kw.pop("df", None)
                try:
                    out = fn(**kw)
                    if out is None:
                        return pd.DataFrame()
                    return pd.DataFrame(out)
                except Exception as inner:
                    last_error = inner
            except Exception as exc:
                last_error = exc
        raise last_error or RuntimeError("GM3 history 请求失败")

    def _history_chunked(self, symbols, start_date, end_date, frequency, depth=0):
        """Fetch recursively. Split symbols first, then time range if needed."""
        if not symbols:
            return pd.DataFrame()

        try:
            return self._history_once(symbols, start_date, end_date, frequency)
        except Exception as exc:
            if not self._is_oversized_error(exc):
                raise

            if len(symbols) > 1:
                mid = len(symbols) // 2
                left = self._history_chunked(symbols[:mid], start_date, end_date, frequency, depth + 1)
                right = self._history_chunked(symbols[mid:], start_date, end_date, frequency, depth + 1)
                return pd.concat([left, right], ignore_index=True)

            start = self._to_dt(start_date)
            end = self._to_dt(end_date)
            if end <= start + timedelta(seconds=1):
                raise RuntimeError(
                    f"单标的、最小时间区间仍超过 GM3 history 限制: {symbols[0]} {start_date} ~ {end_date}"
                ) from exc

            midpoint = start + (end - start) / 2
            left_end = midpoint.strftime("%Y-%m-%d %H:%M:%S")
            right_start = midpoint.strftime("%Y-%m-%d %H:%M:%S")
            left = self._history_chunked(symbols, start.strftime("%Y-%m-%d %H:%M:%S"), left_end, frequency, depth + 1)
            right = self._history_chunked(symbols, right_start, end.strftime("%Y-%m-%d %H:%M:%S"), frequency, depth + 1)
            return pd.concat([left, right], ignore_index=True)

    def history(self, symbols, start_date, end_date, frequency="1d"):
        symbols = [str(s) for s in symbols if s]
        if not symbols:
            return pd.DataFrame()

        # The GM3 service documents an approximately 33k record request cap;
        # recursive splitting keeps each request comfortably below it.
        out = self._history_chunked(symbols, start_date, end_date, frequency)
        if out.empty:
            return out
        return out.drop_duplicates(subset=[c for c in ("symbol", "eob") if c in out.columns]).reset_index(drop=True)

    def fetch(self, symbols, start_date, end_date, frequency="1d"):
        df = self.history(symbols, start_date, end_date, frequency)
        if df.empty:
            raise ValueError("GM3 返回空数据，请检查日期/权限/股票代码")
        required = {"symbol", "eob", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"GM3 数据缺少字段: {sorted(missing)}")
        df["eob"] = pd.to_datetime(df["eob"])
        return df.sort_values(["symbol", "eob"]).reset_index(drop=True)
