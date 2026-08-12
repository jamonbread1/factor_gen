from __future__ import annotations
import importlib
import pandas as pd

class GM3DataAdapter:
    """All market data comes from gm.api; no CSV dependency."""
    def __init__(self, token=None):
        self.gm = importlib.import_module('gm.api')
        self.token = token
        if token and hasattr(self.gm, 'set_token'):
            self.gm.set_token(token)

    def get_constituents(self, index='SHSE.000300', as_of=None):
        fn = getattr(self.gm, 'get_constituents', None)
        if fn is None:
            raise RuntimeError('gm.api 没有 get_constituents，请检查掘金 SDK 版本')
        try:
            obj = fn(index=index, date=as_of) if as_of is not None else fn(index=index)
        except TypeError:
            obj = fn(index, as_of) if as_of is not None else fn(index)
        if isinstance(obj, pd.DataFrame):
            col = 'symbol' if 'symbol' in obj.columns else obj.columns[0]
            return obj[col].dropna().astype(str).tolist()
        if isinstance(obj, dict):
            return [str(x) for x in obj.get('symbol', obj.keys())]
        return [str(x) for x in obj]

    def history(self, symbols, start_date, end_date, frequency='1d'):
        fields='symbol,eob,open,high,low,close,volume,amount'
        fn=getattr(self.gm,'history',None)
        if fn is None:
            raise RuntimeError('当前 gm.api 未找到 history；请根据本地 SDK 改 adapter')
        for kw in [
            dict(symbol=list(symbols),frequency=frequency,start_time=start_date,end_time=end_date,fields=fields),
            dict(symbol=','.join(symbols),frequency=frequency,start_time=start_date,end_time=end_date,fields=fields),
        ]:
            try:
                out=fn(**kw)
                if out is not None:
                    return pd.DataFrame(out)
            except (TypeError,ValueError):
                continue
        raise RuntimeError('GM3 history 调用失败，请检查 SDK 函数签名和数据权限')

    def fetch(self, symbols, start_date, end_date, frequency='1d'):
        df=self.history(symbols,start_date,end_date,frequency)
        if df.empty: raise ValueError('GM3 返回空数据，请检查日期/权限/股票代码')
        required={'symbol','eob','open','high','low','close','volume'}
        missing=required-set(df.columns)
        if missing: raise ValueError(f'GM3 数据缺少字段: {sorted(missing)}')
        df['eob']=pd.to_datetime(df['eob'])
        return df.sort_values(['symbol','eob']).reset_index(drop=True)
