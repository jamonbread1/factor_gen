from __future__ import annotations
import numpy as np
from .evaluation import evaluate_factor

def search_combinations(X,y,max_candidates=1_000_000,seed=42):
    """Vectorized random symbolic-combination search. No fabricated IC: every candidate is evaluated."""
    rng=np.random.default_rng(seed); n=len(y); best=None
    names=["ret1","range","body","vol_chg","mom5","mom20","vol20"]
    for start in range(0,max_candidates,10000):
        k=min(10000,max_candidates-start)
        a=rng.integers(0,X.shape[1],k); b=rng.integers(0,X.shape[1],k); op=rng.integers(0,4,k)
        scores=np.empty((n,k),dtype="float32")
        for j in range(k):
            x=X[:,a[j]]; z=X[:,b[j]]
            scores[:,j]=x+z if op[j]==0 else x-z if op[j]==1 else x*z if op[j]==2 else np.tanh(x+z)
        for j in range(k):
            r=evaluate_factor(scores[:,j],y)
            if r.status=="PASS" and (best is None or r.mean_ic>best["report"].mean_ic):
                best={"expression":f"{names[a[j]]} {'+' if op[j]==0 else '-' if op[j]==1 else '*' if op[j]==2 else 'tanh+'} {names[b[j]]}","report":r}
        if (start+k)%100000==0: print(f"factor search {start+k:,}/{max_candidates:,}")
    return best or {"expression":"NO_PASS","report":evaluate_factor(np.zeros(n),y)}
