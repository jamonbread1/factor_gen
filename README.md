# factor_gen

基于掘金 GM3 K线数据的 Transformer 自动因子发现与 IC 评估系统。

## 功能

- 输入 OHLCV K线序列
- Transformer 自动学习价格行为模式
- 自动生成预测因子 score
- 计算 IC / RankIC
- 判断因子失效
- AI 搜索因子组合参数
- 输出稳定 IC 达标候选因子
- 实时日志和进度显示

目标筛选标准：`IC > 0.03`（需在真实数据回测验证）。

## 使用

```bash
pip install -r requirements.txt
python run.py --data data/kline.csv
```

输出：

- logs/run.log
- reports/factor_report.json
- factors/best_factor.py

