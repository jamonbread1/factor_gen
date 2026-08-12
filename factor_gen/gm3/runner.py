import os,sys,json
from datetime import datetime
import numpy as np
from loguru import logger
from ..ai.features import make_features,make_target,FEATURES
from ..ai.trainer import TransformerTrainer
from ..evaluation.ic import spearman_ic,stability
from ..search.engine import SearchEngine
from ..search.formulas import eval_formula
from ..reporting.writer import write_factor
from .adapter import GM3DataAdapter

def _window_ics(df,score,target='target',windows=20):
    dates=sorted(df.eob.dropna().unique())
    if len(dates)<windows: return [spearman_ic(df[score],df[target])]
    chunks=np.array_split(dates,windows); out=[]
    for d in chunks:
        z=df[df.eob.isin(d)]; out.append(spearman_ic(z[score],z[target]))
    return out

def run(symbols=None,index='SHSE.000300',start_date='2020-01-01',end_date=None,output='factor_gen_output',budget=1_000_000):
    os.makedirs(output,exist_ok=True)
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
        score=eval_formula(f,test); tmp=test.copy(); tmp['_score']=score; ics=_window_ics(tmp,'_score'); st=stability(ics,0.03)
        rows.append({'expression':f.expression(),'current_ic':spearman_ic(score,test.target),'stability':st})
    rows=sorted(rows,key=lambda x:(x['stability'].get('mean_ic',-999),abs(x['current_ic'] or 0)),reverse=True); best=rows[0] if rows else None
    status='PASS' if best and best['stability'].get('status')=='PASS' else 'NO_PASS'
    report={'status':status,'threshold':0.03,'transformer_test_ic':transformer_ic,'best':best,'candidates':rows}
    json.dump(report,open(f'{output}/report.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    if best and status=='PASS': write_factor(output,best['expression'])
    logger.info(f'最终状态: {status}; current IC={best["current_ic"] if best else None}; mean IC={best["stability"].get("mean_ic") if best else None}')
    return report
