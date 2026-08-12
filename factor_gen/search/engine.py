import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from .formulas import generate_formulas,eval_formula

def ic(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    return float(spearmanr(a[m],b[m]).statistic) if m.sum()>=10 else np.nan

class SearchEngine:
    def __init__(self,budget=1_000_000,topk=100,seed=7): self.budget,self.topk,self.seed=budget,topk,seed
    def search(self,frame,features,target_col):
        rng=np.random.default_rng(self.seed); n=min(len(frame),20000); sample=frame.iloc[rng.choice(len(frame),n,replace=False)]; y=sample[target_col].to_numpy(float); top=[]
        for f in tqdm(generate_formulas(features,self.budget,self.seed),total=self.budget,desc='AI组合搜索'):
            v=eval_formula(f,sample); z=ic(v,y)
            if np.isfinite(z):
                top.append((abs(z),z,f));
                if len(top)>self.topk*4: top=sorted(top,key=lambda x:x[0],reverse=True)[:self.topk]
        return sorted(top,key=lambda x:x[0],reverse=True)[:self.topk]
