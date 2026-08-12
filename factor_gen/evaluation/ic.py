import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def spearman_ic(score,target):
    a=pd.Series(score); b=pd.Series(target); m=a.notna()&b.notna()
    if m.sum()<10:return np.nan
    return float(spearmanr(a[m],b[m]).statistic)

def stability(ics,threshold=0.03):
    x=np.asarray(ics,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0:return {'status':'NO_DATA'}
    mean=float(x.mean()); med=float(np.median(x)); std=float(x.std(ddof=1)) if len(x)>1 else 0.0; ratio=float((x>threshold).mean())
    return {'status':'PASS' if mean>threshold and med>threshold and ratio>=.60 else 'FAIL','mean_ic':mean,'median_ic':med,'ic_std':std,'ic_ir':float(mean/std) if std>1e-12 else float('inf'),'positive_ratio':ratio,'n_windows':len(x)}
