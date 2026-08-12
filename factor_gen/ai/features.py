import numpy as np
import pandas as pd

FEATURES=['ret1','ret5','range_pct','body_pct','close_pos','vol_chg','vol_z','mom20','vol20','gap']

def make_features(df):
    x=df.copy().sort_values(['symbol','eob']); g=x.groupby('symbol',group_keys=False); c=x['close'].replace(0,np.nan)
    x['ret1']=g['close'].pct_change(); x['ret5']=g['close'].pct_change(5)
    x['range_pct']=(x['high']-x['low'])/c; x['body_pct']=(x['close']-x['open'])/x['open'].replace(0,np.nan)
    x['close_pos']=(x['close']-x['low'])/(x['high']-x['low']).replace(0,np.nan)
    x['vol_chg']=g['volume'].pct_change(); vm=g['volume'].transform(lambda s:s.rolling(20,min_periods=10).mean()); vs=g['volume'].transform(lambda s:s.rolling(20,min_periods=10).std())
    x['vol_z']=(x['volume']-vm)/vs.replace(0,np.nan); x['mom20']=g['close'].pct_change(20)
    x['vol20']=g['ret1'].transform(lambda s:s.rolling(20,min_periods=10).std()); x['gap']=g['open'].pct_change()-g['close'].pct_change()
    return x.replace([np.inf,-np.inf],np.nan)

def make_target(df,horizon=1):
    x=df.copy(); x['target']=x.groupby('symbol')['close'].pct_change(horizon).shift(-horizon); return x
