import os,sys
from datetime import datetime
import numpy as np
from loguru import logger
from ..ai.features import make_features,make_target,FEATURES
from ..ai.trainer import TransformerTrainer
from ..evaluation.ic import spearman_ic
from ..search.engine import SearchEngine
from ..search.formulas import eval_formula
from .adapter import GM3DataAdapter

def run(symbols=None,index='SHSE.000300',start_date='2020-01-01',end_date=None,output='factor_gen_output',budget=1_000_000):
    logger.remove(); logger.add(sys.stdout,format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}'); logger.add(f'{output}/run.log')
    end_date=end_date or datetime.now().strftime('%Y-%m-%d'); logger.info('GM3 AI Factor Generator V3 启动')
    adapter=GM3DataAdapter(os.getenv('GM_TOKEN'))
    if symbols is None: logger.info(f'获取指数成分: {index}'); symbols=adapter.get_constituents(index,end_date); logger.info(f'股票数量: {len(symbols)}')
    logger.info(f'从 GM3 请求 K 线: {start_date} -> {end_date}'); raw=adapter.fetch(symbols,start_date,end_date); logger.info(f'GM3 数据完成: {len(raw):,} bars')
    df=make_target(make_features(raw),1).dropna(subset=FEATURES+['target']).copy(); X=[]; y=[]
    for _,g in df.groupby('symbol'):
        arr=g[FEATURES].to_numpy(float); tar=g.target.to_numpy(float)
        for i in range(32,len(g)): X.append(arr[i-32:i]); y.append(tar[i])
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float32); cut=max(1,int(.8*len(X))); logger.info(f'样本={len(X):,}; train={cut:,}; test={len(X)-cut:,}')
    trainer=TransformerTrainer(len(FEATURES)); trainer.fit(X[:cut],y[:cut]); pred=trainer.predict(X[cut:]); transformer_ic=spearman_ic(pred,y[cut:]); logger.info(f'Transformer test IC={transformer_ic:.5f}')
    test=df.iloc[int(.8*len(df)):].copy(); top=SearchEngine(budget=budget).search(test,FEATURES,'target'); rows=[]
    for _,_,f in top:
        z=spearman_ic(eval_formula(f,test),test.target); rows.append({'expression':f.expression(),'ic':z})
    rows=sorted(rows,key=lambda x:abs(x['ic']),reverse=True); best=rows[0] if rows else None; status='PASS_CANDIDATE' if best and best['ic']>0.03 else 'NO_PASS'
    os.makedirs(output,exist_ok=True)
    import json; json.dump({'status':status,'threshold':0.03,'transformer_test_ic':transformer_ic,'best':best,'candidates':rows},open(f'{output}/report.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    if best: open(f'{output}/best_factor.py','w',encoding='utf-8').write('def factor(df):\n    return '+repr(best['expression'])+'\n')
    logger.info(f'最终状态: {status}; best IC={best["ic"] if best else None}')
    return {'status':status,'best':best}
