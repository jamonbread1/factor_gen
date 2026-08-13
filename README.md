# factor_gen V5.2 — GM3 Regime-Aware AI Factor Discovery

面向掘金 GM3 的 AI 自动因子发现框架。Transformer 从按股票严格排序的 K 线序列学习 `ai_score`，再进行候选组合搜索、横截面 IC 验证和周级趋势信号评估。

## V5.2 核心架构

```text
GM3 OHLCV
   ↓
point-in-time panel
   ├─ grouped-by-symbol features / forward targets
   └─ time split + embargo
   ↓
Transformer AI score
   ↓
candidate miner
   ├─ primitive feature bank
   ├─ cross-sectional IC proxy
   └─ exact candidate evaluation
   ↓
validation gates
   ├─ mean / median IC
   ├─ IC-IR / positive-window ratio
   ├─ walk-forward stability
   └─ factor correlation gate
   ↓
regime diagnostics
   ├─ breadth
   ├─ dispersion
   ├─ market trend
   ├─ BULL_TREND / STRUCTURAL_BULL / BEAR_TREND / MIXED
   └─ recovery score
   ↓
OOS test + signal diagnostics
```

### 为什么增加 Regime

2013 这类“指数弱、结构强”的年份不能简单归类成全市场熊市。V5.2 因此把市场状态和因子健康度分开：指数趋势弱但截面离散度高时，系统可以识别 `STRUCTURAL_BULL`，并在报告中查看因子在不同状态下的 IC，而不是把所有 alpha 一起判死。

### 防止常见隐性错误

- 所有收益率、均线、波动率和 Transformer window 均按 `symbol + eob` 计算；
- Transformer 每个股票独立建窗，不允许跨股票拼接时间序列；
- Train / Valid / Test 之间增加 embargo，避免 5-bar forward target 跨边界重叠；
- 候选预筛目标改为横截面 IC，而不是单纯全样本 Pearson correlation；
- Test 不参与候选搜索、排序、相关性门控或最终选择；
- 生成因子也按股票分组计算 rolling / pct_change，避免研究代码和交付代码口径不一致；
- 输出 market recovery / regime diagnostics，支持判断“市场修复”而不是只看指数涨跌。

## 运行

```bash
python run_v52.py --start 2022-01-01 --end 2026-08-01 --index SHSE.000300
```

配置位于 `config_v52.yaml`。研究产物默认写入 `generated_v52/`，该目录和模型文件已加入 `.gitignore`，避免污染源码工作区。

## 输出

```text
generated_v52/
  weekly_signal_factor_v52.py
  ai_transformer_v52.pt
  factor_report_v52.json
```

## 验收原则

默认研究门槛：mean IC > 0.03、median IC > 0.03、IC > 0.03 的窗口比例 >= 60%、IC-IR >= 0.50。门槛只用于研究筛选，不代表未来收益保证。

**Test 只用于最终 OOS 验证。** Walk-forward 和 regime report 用于稳定性诊断，不能把 Test 结果反向用于调参。

## 代码正确性检查

仓库提供 `pytest` 回归测试，覆盖：

- 跨股票收益率计算隔离；
- embargo 时间切分；
- regime 指标不读取未来 target；
- recovery score 范围；
- V5.1 原有 signal / evaluation 测试。

CI 会在 push / pull request 时执行 `pytest -q`。
