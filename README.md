# factor_gen V3 — GM3 AI Factor Discovery

面向掘金 GM3 环境的 AI 自动因子发现框架。**数据全部由掘金 `gm.api` 提供，不依赖 CSV。**

流程：GM3 历史 K 线 → 特征工程 → Transformer → IC / RankIC → 时间滚动稳定性 → 百万候选组合搜索 → 自动生成 GM3 因子模板。

默认筛选门槛：`mean IC > 0.03`，同时要求 median IC > 0.03、IC>0.03 窗口比例 >= 60%。没有达标候选时明确输出 `NO_PASS`，不会人为制造 PASS。

针对 RTX 3050 4GB：hidden_dim=32、layers=2、heads=4、seq_len=32、CUDA AMP。

## 在掘金3运行

```python
from factor_gen.gm3.runner import run
run()
```

也可运行 `examples/run_gm3.py`。GM3 数据适配集中在 `factor_gen/gm3/adapter.py`，以便兼容不同 SDK 版本。

## 输出

```text
factor_gen_output/
  run.log
  report.json
  best_factor.py
```

不包含 GM_TOKEN。请通过掘金环境或环境变量提供 token。

## 重要说明

`IC > 0.03` 是筛选条件，不是保证。真正有效性必须用未参与搜索的 out-of-sample / walk-forward 数据复核，避免未来数据泄漏、多重测试和过拟合。
