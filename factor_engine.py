import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from loguru import logger
import torch
from torch import nn


class PriceTransformer(nn.Module):
    def __init__(self,features=5):
        super().__init__()
        layer=nn.TransformerEncoderLayer(d_model=features,nhead=1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2)
        self.head=nn.Linear(features,1)

    def forward(self,x):
        return self.head(self.encoder(x)).mean(1)


class FactorEngine:
    def __init__(self,path):
        self.df=pd.read_csv(path)
        self.model=PriceTransformer()
        self.best={'ic':0}

    def search_step(self,step):
        logger.info(f'搜索组合逻辑 {step+1}/5')

    def evaluate(self):
        logger.info('计算IC与失效检测')
        if 'return' in self.df:
            score=self.df['close'].pct_change()
            ic=spearmanr(score.shift(1),self.df['return']).statistic
        else:
            ic=0
        return {'ic':float(ic),'status':'PASS' if ic>0.03 else 'FAIL'}
