# factor_gen V2

基于掘金 GM3 K线数据的 AI 自动因子发现系统。目标不是“训练一个预测器就结束”，而是：**K线 → Transformer → 因子候选 → 百万级组合搜索 → 稳定 IC 验证 → 失效判断 → 自动生成 GM3 因子插件**。

## 针对 RTX 3050 4GB 的设计

你的 NVIDIA 驱动为 592.82，`nvidia-smi` 报告 CUDA 13.1、显存 4096 MiB。V2 使用小型 Transformer、batch 32、序列长度 64、FP16 AMP，避免把 4GB 显存当成大模型训练卡。

> `nvidia-smi` 的 CUDA Version 是驱动支持的 CUDA 上限，不要求 Python 环境必须安装 CUDA 13.1。PyTorch wheel 自带对应 CUDA runtime，安装时应选择与当前 PyTorch 官方 wheel 匹配的版本。

## 输入

CSV 至少包含：

```text
open,high,low,close,volume
```

可额外提供 `return`；否则系统自动使用下一根 K 线收益率作为预测目标。

## 一键运行

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -U pip
pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python run.py --data data/kline.csv
```

默认：

```yaml
candidate_count: 1000000
seq_len: 64
epochs: 5
batch_size: 32
ic_threshold: 0.03
```

## 四阶段流程

### 1. Transformer

只按时间顺序训练，不随机打乱，避免未来数据泄漏。模型通过 Hugging Face `transformers` 的 `PreTrainedModel`/`PretrainedConfig` 封装，并以 OHLCV 衍生特征序列作为输入。

### 2. 百万级组合搜索

搜索空间由：

- 10 个基础价量特征
- identity / neg / abs / square / tanh
- 加、减、乘
- 多组确定性权重

组成。系统分批生成候选，第一阶段快速相关性预筛，第二阶段对 Top 候选计算精确 Spearman IC，因此不会一次性创建一百万个 DataFrame。

### 3. 稳定性验证

不是只看一个 IC 数字。默认 PASS 条件：

```text
mean IC       > 0.03
median IC     > 0.03
IC > 0.03 窗口比例 >= 60%
```

同时输出 IC、RankIC、IC 标准差、IC-IR 和正 IC 窗口比例。

如果没有真实数据验证通过，程序明确输出 `FAIL`，不会伪造一个“有效因子”。

### 4. GM3 自动生成

最佳候选自动写到：

```text
generated/ai_factor_v2.py
generated/factor_report.json
generated/run.log
```

生成函数是纯 pandas Series 因子，可进一步接入 `gm3_factor/select/` 的注册机制。

## 日志与进度

终端会看到：

```text
GM3 AI 因子发现系统 V2 启动
K线加载完成: 120,000 rows
[1/4] Transformer 学习 K线序列规律
Transformer训练 1/5 | loss=...
AI因子当前IC=... | 稳定均值IC=... | 状态=...
[2/4] 百万级 AI/符号组合搜索
百万组合预筛  43%|████...
预筛完成: 1,000,000 -> 1000
[3/4] 候选因子稳定性复核
[4/4] 生成 GM3 插件和报告
```

同时保存完整日志到 `generated/run.log`。

## 重要说明

`IC > 0.03` 是筛选门槛，不是系统可以保证的结果。真正有效性必须在未参与搜索的 out-of-sample / walk-forward 数据上再次验证；尤其要防止数据窥视、多重测试和过拟合。
