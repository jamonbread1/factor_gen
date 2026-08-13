from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np

LOG = logging.getLogger("factor_gen.qsk_miner")
TRANSFORMS = ("identity", "neg", "abs", "tanh", "square")
OPS = ("+", "-", "*")
BASE_FEATURES = ["ret1", "ret5", "ret10", "range", "body", "close_pos", "vol_z20", "ma_gap5", "ma_gap20", "volatility20", "trend20", "ai_score"]

@dataclass
class Candidate:
    expression: str
    weights: np.ndarray
    feature_idx: tuple[int, int, int]
    transforms: tuple[str, str]
    op: str
    score: float = 0.0
    complexity: int = 3

def transform(x, name):
    if name == "identity": return x
    if name == "neg": return -x
    if name == "abs": return np.abs(x)
    if name == "tanh": return np.tanh(np.clip(x, -8, 8))
    if name == "square": return np.sign(x) * np.square(np.clip(x, -3, 3))
    raise ValueError(name)

def _feature_bank(frame):
    bank=[]; names=[]
    for f in BASE_FEATURES:
        x=frame[f].to_numpy(np.float32)
        for t in TRANSFORMS:
            bank.append(np.nan_to_num(transform(x,t),nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32)); names.append((f,t))
    return np.asarray(bank,dtype=np.float32), names

def _corr_vector(bank,y):
    y=np.nan_to_num(y.astype(np.float32)); y0=y-y.mean(); yn=np.linalg.norm(y0)+1e-12
    x0=bank-bank.mean(axis=1,keepdims=True); xn=np.linalg.norm(x0,axis=1)+1e-12
    return np.nan_to_num((x0@y0)/(xn*yn))

def _spec_value(frame,spec):
    idx,(ta,tb),op,w,_=spec
    a=transform(frame[BASE_FEATURES[idx[0]]].to_numpy(np.float32),ta); b=transform(frame[BASE_FEATURES[idx[1]]].to_numpy(np.float32),tb); c=frame[BASE_FEATURES[idx[2]]].to_numpy(np.float32)
    if op=="*": return w[0]*a*(w[1]*b)+w[2]*c
    if op=="+": return w[0]*a+w[1]*b+w[2]*c
    return w[0]*a-w[1]*b+w[2]*c

def evaluate_spec(frame,spec):
    return _spec_value(frame,spec)

def search(frame,budget=1_000_000,top_k=1000,seed=42,batch=100_000):
    y=frame["target"].to_numpy(np.float32); bank,names=_feature_bank(frame); corr=_corr_vector(bank,y); rng=np.random.default_rng(seed); keep=[]; p=len(names)
    for start in range(0,int(budget),batch):
        n=min(batch,int(budget)-start); ids=rng.integers(0,p,size=(n,3)); w=rng.normal(0,1,size=(n,3)).astype(np.float32); w/=np.linalg.norm(w,axis=1,keepdims=True)+1e-12; ops=rng.integers(0,3,size=n)
        approx=w[:,0]*corr[ids[:,0]]+w[:,2]*corr[ids[:,2]]
        approx+=np.where(ops==1,-w[:,1]*corr[ids[:,1]],w[:,1]*corr[ids[:,1]])
        score=np.abs(approx); k=min(max(10,top_k//10),n); idx=np.argpartition(score,-k)[-k:]
        for j in idx:
            i0,i1,i2=(int(v) for v in ids[j]); ta,tb=names[i0][1],names[i1][1]; op=("+","-","*")[int(ops[j])]; feats=(BASE_FEATURES.index(names[i0][0]),BASE_FEATURES.index(names[i1][0]),BASE_FEATURES.index(names[i2][0])); ww=w[j]
            if op=="*": expr=f"({ww[0]:+.7f}*{ta}({BASE_FEATURES[feats[0]]})*{ww[1]:+.7f}*{tb}({BASE_FEATURES[feats[1]]})+{ww[2]:+.7f}*{BASE_FEATURES[feats[2]]})"
            elif op=="+": expr=f"({ww[0]:+.7f}*{ta}({BASE_FEATURES[feats[0]]})+{ww[1]:+.7f}*{tb}({BASE_FEATURES[feats[1]]})+{ww[2]:+.7f}*{BASE_FEATURES[feats[2]]})"
            else: expr=f"({ww[0]:+.7f}*{ta}({BASE_FEATURES[feats[0]]})-{ww[1]:+.7f}*{tb}({BASE_FEATURES[feats[1]]})+{ww[2]:+.7f}*{BASE_FEATURES[feats[2]]})"
            spec=(feats,(ta,tb),op,ww.copy(),expr); keep.append((float(score[j]),spec))
        keep.sort(key=lambda z:z[0],reverse=True); keep=keep[:top_k]; LOG.info("screen %d/%d",min(start+n,int(budget)),budget)
    exact=[]
    valid=np.isfinite(y)
    for _,spec in keep:
        value=np.nan_to_num(_spec_value(frame,spec),nan=0.0,posinf=0.0,neginf=0.0); xv=value[valid]-value[valid].mean(); yv=y[valid]-y[valid].mean(); s=abs(float(np.dot(xv,yv)/((np.linalg.norm(xv)+1e-12)*(np.linalg.norm(yv)+1e-12))))
        idx,tr,op,ww,expr=spec; exact.append(Candidate(expr,ww,idx,tr,op,s,3))
    exact.sort(key=lambda x:x.score,reverse=True)
    return exact[:top_k]
