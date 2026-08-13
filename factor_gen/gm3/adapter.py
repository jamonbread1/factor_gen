from __future__ import annotations

import importlib
import os

import pandas as pd


class GM3DataAdapter:
    """GM3 market-data adapter. Uses current stk_* APIs only."""

    def __init__(self, token=None):
        self.gm = importlib.import_module("gm.api")
        token = token or os.getenv("GM_TOKEN")
        self.token = token
        if token and hasattr(self.gm, "set_token"):
            self.gm.set_token(token)

    def get_constituents(self, index="SHSE.000300", as_of=None):
        """Return index constituents using the current GM3 API.

        GM3's legacy get_constituents() is deprecated. The current API is
        stk_get_index_constituents(index, trade_date=None).
        """
        fn = getattr(self.gm, "stk_get_index_constituents", None)
        if fn is None:
            raise RuntimeError(
                "当前 gm.api 没有 stk_get_index_constituents。"
                "请升级 GoldMiner3/gm SDK 后再运行 factor_gen V5。"
            )

        try:
            if as_of is not None:
                obj = fn(index=index, trade_date=as_of)
            else:
                obj = fn(index=index)
        except Exception as exc:
            message = str(exc)
            if "1000" in message or "token" in message.lower():
                raise RuntimeError(
                    "GM3 成分股查询失败：当前 gm.api token 无效或未配置。"
                    "请在掘金3项目环境配置有效 token，或在 config.yaml 设置 gm_token，"
                    "也可设置环境变量 GM_TOKEN。原始错误: " + message
                ) from exc
            raise RuntimeError(
                f"GM3 stk_get_index_constituents 调用失败: {message}"
            ) from exc

        if obj is None:
            return []
        if isinstance(obj, pd.DataFrame):
            if "symbol" not in obj.columns:
                raise RuntimeError(
                    f"stk_get_index_constituents 返回结果缺少 symbol 字段: {list(obj.columns)}"
                )
            return obj["symbol"].dropna().astype(str).drop_duplicates().tolist()
        if isinstance(obj, dict):
            return [str(x) for x in obj.get("symbol", obj.keys())]
        return [str(x) for x in obj]

    def history(self, symbols, start_date, end_date, frequency="1d"):
        fields = "symbol,eob,open,high,low,close,volume,amount"
        fn = getattr(self.gm, "history", None)
        if fn is None:
            raise RuntimeError(
                "当前 gm.api 未找到 history；请检查 GM3 SDK 版本。"
            )

        for kw in [
            dict(
                symbol=list(symbols),
                frequency=frequency,
                start_time=start_date,
                end_time=end_date,
                fields=fields,
            ),
            dict(
                symbol=",".join(symbols),
                frequency=frequency,
                start_time=start_date,
                end_time=end_date,
                fields=fields,
            ),
        ]:
            try:
                out = fn(**kw)
                if out is not None:
                    return pd.DataFrame(out)
            except (TypeError, ValueError):
                continue
        raise RuntimeError(
            "GM3 history 调用失败，请检查 SDK 函数签名、数据权限和 token。"
        )

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
