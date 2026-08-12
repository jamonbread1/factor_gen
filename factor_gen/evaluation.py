from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ICReport:
    mean_ic: float
    median_ic: float
    std_ic: float
    positive_ratio: float
    status: str

def _rankcorr(a,b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a,b).statistic)
    except Exception:
        ar=np.argsort(np.argsort(a)); br=np.argsort(np.argsort(b));
        return float(np.corrcoef(ar,br)[0,1])

def evaluate_factor(score,target,windows=5):
    mask=np.isfinite(score)&np.isfinite(target); score=np.asarray(score)[mask]; target=np.asarray(target)[mask]
    if len(score)<100: return ICReport(float("nan"),float("nan"),float("nan"),0.0,"INSUFFICIENT")
    chunks=np.array_split(np.arange(len(score)),windows)
    ics=np.array([_rankcorr(score[i],target[i]) for i in chunks if len(i)>10])
    mean=float(np.mean(ics)); med=float(np.median(ics)); std=float(np.std(ics)); pos=float(np.mean(ics>0.03))
    status="PASS" if mean>0.03 and med>0.03 and pos>=0.60 else "FAIL"
    return ICReport(mean,med,std,pos,status)
