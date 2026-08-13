# factor_gen V5 — GM3 AI Factor Discovery

面向掘金 GM3 的 AI 自动因子发现框架。数据由 `gm.api` 提供，Transformer 从 K 线序列学习 `ai_score`，再进行百万级候选组合搜索。

V5 参考 QuantSkills 的因子挖掘/评估思路：

- Train / Valid / Test 严格时间隔离，Test 在候选冻结后才打开；
- 候选按 Valid IC 排名，相关性门控避免 Top-K 高度同质化；
- IC、RankIC、IC-IR、median IC、正 IC 比例；
- 分层单调性、Top 组换手率、1/3/5/10 日衰减；
- 保存 Top-K 候选和完整 JSON 研究报告；
- 无论 PASS / FAIL 都保存最佳候选因子，FAIL 不再丢失；
- 同时保存 Transformer checkpoint，最终因子可以组合 `ai_score` 使用。

## 运行

先在掘金3环境安装依赖，然后：

```bash
python run_v5.py --start 2022-01-01 --end 2026-08-01 --index SHSE.000300
```

也可以在 `config.yaml` 中指定 `symbols`，避免每次重新获取指数成分。

## 输出

```text
generated/
  ai_factor_v5.py
  ai_transformer_v5.pt
  factor_report_v5.json
```

`ai_factor_v5.py` 是可交付的因子文件；如果该因子包含 `ai_score`，运行时需要传入 Transformer 输出或先写入 `df["ai_score"]`。

## 验收原则

默认门槛：mean IC > 0.03、median IC > 0.03、IC > 0.03 的窗口比例 >= 60%、IC-IR >= 0.50。门槛只用于研究筛选，不代表未来收益保证。

最重要的是：**Test 只用于最终 OOS 验证，不参与候选搜索、排序、相关性门控或最终因子选择。**
