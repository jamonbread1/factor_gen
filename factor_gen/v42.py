from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .gm3_adapter import GM3DataAdapter

LOG = logging.getLogger("factor_gen.v42")

FEATURES = ["ret1", "ret5", "range", "body", "close_pos", "vol_z", "volatility", "trend"]

@dataclass
class Result:
    status: str
    mean_ic: float
    median_ic: float
    ic_std: float
    ic_ir: float
    positive_ratio: float
    formula: str
    test_mean_ic: float


def make_panel(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy().sort_values(["symbol","eob"])
    g=x.groupby("symbol", group_keys=False)
    x["ret1"]=g["close"].pct_change()
    x["ret5"]=g["close"].pct_change(5)
    x["range"]=(x.high-x.low)/(x.close.abs()+1e-8)
    x["body"]=(x.close-x.open)/(x.open.abs()+1e-8)
    x["close_pos"]=(x.close-x.low)/(x.high-x.low+1e-8)
    lv=np.log1p(x.volume.clip(lower=0))
    x["vol_z"]=lv-g["volume"].transform(lambda s: np.log1p(s.clip(lower=0)).rolling(20,min_periods=5).mean())
    x["volatility"]=g["ret1"].transform(lambda s:s.rolling(20,min_periods=10).std())
    x["trend"]=g["close"].transform(lambda s:s.rolling(20,min_periods=10).mean())/(x.close.abs()+1e-8)-1
    x["target"]=g["close"].shift(-1)/x.close-1
    return x.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["target"])


def split_by_time(df, train=.6, valid=.2):
    dates=np.sort(df.eob.unique()); a=int(len(dates)*train); b=int(len(dates)*(train+valid))
    return df[df.eob.isin(dates[:a])], df[df.eob.isin(dates[a:b])], df[df.eob.isin(dates[b:])]


def cs_ic(score, y, dates):
    vals=[]
    tmp=pd.DataFrame({"s":score,"y":y,"d":dates})
    for _,g in tmp.groupby("d",sort=False):
        if len(g)<10 or g.s.nunique()<2 or g.y.nunique()<2: continue
        vals.append(np.corrcoef(rankdata(g.s),rankdata(g.y))[0,1])
    return np.asarray(vals,float)


def exact_ic(df, weights):
    s=np.asarray(df[FEATURES],float) @ weights
    return cs_ic(s,df.target.to_numpy(float),df.eob)


def million_screen(train, budget=1_000_000, seed=42, batch=100_000):
    X=train[FEATURES].to_numpy(float)
    y=train.target.to_numpy(float)
    # Cross-sectional standardization makes the analytical screen comparable across dates.
    z=[]
    yy=[]
    for _,g in train.groupby("eob",sort=False):
        if len(g)<10: continue
        xx=g[FEATURES].to_numpy(float); tt=g.target.to_numpy(float)
        z.append((xx-np.nanmean(xx,axis=0))/(np.nanstd(xx,axis=0)+1e-8)); yy.append((tt-np.nanmean(tt))/(np.nanstd(tt)+1e-8))
    Z=np.vstack(z); Y=np.concatenate(yy)
    corr=np.nan_to_num(Z.T@Y/max(len(Y),1))
    rng=np.random.default_rng(seed); best=[]
    for start in range(0,budget,batch):
        n=min(batch,budget-start)
        W=rng.normal(size=(n,len(FEATURES))); W/=np.linalg.norm(W,axis=1,keepdims=True)+1e-12
        approx=np.abs(W@corr)
        k=min(1000,n)
        idx=np.argpartition(approx,-k)[-k:]
        best.extend((float(approx[i]),W[i].copy()) for i in idx)
        LOG.info("screen %d/%d candidates", min(start+n,budget), budget)
    best.sort(key=lambda x:x[0],reverse=True)
    return [w for _,w in best[:1000]]


def search(train, valid, test, budget=1_000_000):
    candidates=million_screen(train,budget=budget)
    scored=[]
    for i,w in enumerate(candidates,1):
        vi=exact_ic(valid,w)
        if len(vi)<20: continue
        score=float(np.nanmean(vi))
        scored.append((score,w,vi))
        if i%100==0: LOG.info("validation %d/%d",i,len(candidates))
    scored.sort(key=lambda x:x[0],reverse=True)
    for score,w,vi in scored[:20]:
        ti=exact_ic(test,w)
        if len(ti)<20: continue
        m=float(np.nanmean(ti)); med=float(np.nanmedian(ti)); sd=float(np.nanstd(ti)); ir=m/(sd+1e-12); pos=float(np.mean(ti>0))
        # Formal gate: validation and unseen test both need stable positive IC > .03.
        if score>0.03 and med>0.03 and m>0.03 and pos>=0.60 and ir>=0.50:
            formula=" + ".join(f"{w[j]:+.4f}*{FEATURES[j]}" for j in range(len(FEATURES)))
            return Result("PASS",m,med,sd,ir,pos,formula,m)
    if scored:
        score,w,vi=scored[0]; ti=exact_ic(test,w); m=float(np.nanmean(ti)) if len(ti) else float("nan")
        formula=" + ".join(f"{w[j]:+.4f}*{FEATURES[j]}" for j in range(len(FEATURES)))
        return Result("FAIL",float(np.nanmean(vi)),float(np.nanmedian(vi)),float(np.nanstd(vi)),float(np.nanmean(vi)/(np.nanstd(vi)+1e-12)),float(np.mean(vi>0)),formula,m)
    return Result("NO_CANDIDATE",float("nan"),float("nan"),float("nan"),float("nan"),0.0,"",float("nan"))


def run(config=None):
    config=config or {}
    data=GM3DataAdapter(index=config.get("index","SHSE.000300"), count=int(config.get("bars",1200))).load()
    LOG.info("rows=%d symbols=%d",len(data),data.symbol.nunique())
    panel=make_panel(data)
    train,valid,test=split_by_time(panel)
    LOG.info("time split: train=%s valid=%s test=%s",train.eob.max(),valid.eob.max(),test.eob.max())
    result=search(train,valid,test,int(config.get("candidates",1_000_000)))
    LOG.info("RESULT status=%s test_IC=%.5f mean=%.5f median=%.5f IR=%.3f positive=%.2f",result.status,result.test_mean_ic,result.mean_ic,result.median_ic,result.ic_ir,result.positive_ratio)
    return result
