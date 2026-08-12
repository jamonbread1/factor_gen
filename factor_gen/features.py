from __future__ import annotations
import numpy as np

def make_features(df, horizon=1):
    x = df.copy().sort_values(["symbol", "eob"] if "symbol" in df.columns else ["eob"])
    g = x.groupby("symbol", group_keys=False) if "symbol" in x.columns else [(None, x)]
    parts=[]
    if hasattr(g, "__iter__") and not isinstance(g, list):
        for _, z in g:
            z=z.copy()
            z["ret1"]=z.close.pct_change()
            z["range"]=(z.high-z.low)/z.close.replace(0,np.nan)
            z["body"]=(z.close-z.open)/z.open.replace(0,np.nan)
            z["vol_chg"]=z.volume.pct_change()
            z["mom5"]=z.close.pct_change(5)
            z["mom20"]=z.close.pct_change(20)
            z["vol20"]=z.ret1.rolling(20).std()
            z["target"]=z.close.shift(-horizon)/z.close-1
            parts.append(z)
        x=__import__('pandas').concat(parts)
    else:
        x["ret1"]=x.close.pct_change(); x["range"]=(x.high-x.low)/x.close
        x["body"]=(x.close-x.open)/x.open; x["vol_chg"]=x.volume.pct_change()
        x["mom5"]=x.close.pct_change(5); x["mom20"]=x.close.pct_change(20); x["vol20"]=x.ret1.rolling(20).std(); x["target"]=x.close.shift(-horizon)/x.close-1
    cols=["ret1","range","body","vol_chg","mom5","mom20","vol20"]
    x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=cols+["target"])
    return x[cols].to_numpy(dtype="float32"), x.target.to_numpy(dtype="float32")
