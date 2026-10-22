# 2026-10-22：E3 真实模型上的官方 / 自定义 LoRA 对齐

## 1. 目标与实际范围

完成进入主实验前的实现控制：在相同真实权重、数据、种子及 200 次更新下，自定义 `official_square` 能否复现官方 Q/O LoRA？状态：**完成**。这不是新的能力提升实验，也没有更换数据或扩大预算。

## 2. 输入、配置与身份

- 官方参考：E2 已保存的 `artifacts/E2-step200`；本轮重新训练自定义组 `artifacts/E3-compat`。
- 原始 MiniMind-3 Dense，63,912,192 参数，基础权重 SHA256 `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`。Tokenizer、代码、环境完整身份见 [运行清单](../../results/evidence/2026-10-22-E3-compat/training/run.json)。Git 为 `170d095c23da50d5e91a8790f16ca912509f683f` 加未提交工作区。
- [配置](../../configs/easy_addsub_compat_r16.json)：只将 E2 的 mode 从 official 改成 custom；仍为 `official_square`、rank=alpha=16、dropout=0、seed=42、MPS / float32、AdamW lr=1e-4、weight_decay=.01、clip=1。实际注入 16 个 Q/O 投影，393,216 可训练参数。
- [冻结数据清单](../../data/manifests/easy-addsub-coverage-v2.json)：64 条训练、32 条验证；有效 batch=4×4=16，200 次更新、50 轮、3,200 次样本呈现、16,700 目标 token、173,500 非 padding token。训练与验证互斥。
- 生成：验证集全部 32 题，last 适配器，贪心、关闭思考、batch=1、上限 256 token、math-answer-v1 精确数值判分，失败计入分母。沿用 E2 的 last 主比较规则。

## 3. 实际步骤

1. 从原始基础权重训练自定义 Q/O 组 200 次更新，没有加载 E2 适配器。
2. 逐项核对两组输入身份、配置差异、实际注入清单，再比较每条训练日志及最终适配器所有张量。
3. 重载新适配器生成全部 32 道验证题，逐题比较 E2 与 E3 的 ID、原始文本、解析答案、正确性与失败类型。
4. 归档元数据、日志和原始预测；权重、优化器恢复状态仍只保存在忽略目录。

## 4. 代码与文件改动

| 文件 | 类型 | 内容、理由与验证 |
| --- | --- | --- |
| [check_compatibility.py](../../scripts/check_compatibility.py) | 新增检查 Python | 审计输入并比较真实训练日志 / 张量，避免只根据文件名声称兼容；本次实测全等 |
| [easy_addsub_compat_r16.json](../../configs/easy_addsub_compat_r16.json) | 新增配置 | 仅更换 mode，控制其余变量 |
| `mathlab/train.py`、`lora.py`、`evaluate.py`、`runtime.py` | 复用 | 本轮未修改训练器、LoRA 实现、生成器或判分器 |
| `vendor/minimind/model_lora.py` | 引用上游 | 保持 MiniMind 上游 `f659b55761b754d306bd140573493a6543cafd7f` 原实现 |
| [posttrain_campaign.py](../../scripts/posttrain_campaign.py)、[test_campaign.py](../../tests/test_campaign.py)、[use_math_model.py](../../scripts/use_math_model.py) | 新增，供后续阶段使用 | 固定主实验方案、验证集选择和使用 / 导出入口；本阶段仅执行其中选择逻辑测试，未把后续训练、导出算作 E3 已完成 |

## 5. 实际命令

工作目录为项目根目录；本次新目录已存在，复跑应更换目录名。

```bash
.venv/bin/python -m mathlab.train --config configs/easy_addsub_compat_r16.json --data data/processed/easy-addsub-coverage-v2 --out artifacts/E3-compat --stop-after-updates 200
.venv/bin/python -m scripts.check_compatibility artifacts/E2-step200 artifacts/E3-compat --out results/evidence/2026-10-22-E3-compat/compatibility.json
.venv/bin/python -m mathlab.evaluate --data data/processed/easy-addsub-coverage-v2/validation.jsonl --adapter artifacts/E3-compat/last --out results/runs/E3-compat-validation
.venv/bin/python -m pytest -q tests/test_campaign.py tests/test_lora.py
```

训练和生成分别重定向到 [train.log](../../results/evidence/2026-10-22-E3-compat/train.log)、[evaluate.log](../../results/evidence/2026-10-22-E3-compat/evaluate.log)。额外一次性 Python 归档命令复制了 JSON/JSONL 并比较逐题字段，结果保存在 [predictions_check.json](../../results/evidence/2026-10-22-E3-compat/predictions_check.json)，没有新增训练逻辑。

## 6. 验证与结果

| 本轮实际检查 | 结果 |
| --- | --- |
| [单元测试日志](../../results/evidence/2026-10-22-E3-compat/tests.log) | 7 passed，包含 2 项本轮 campaign 选择测试与 5 项已有 LoRA 测试；本轮确实重跑 |
| 200 条训练日志 | 逐字段完全相同，最大 loss 差 0 |
| 所有 last 适配器张量 | 逐位完全相同，最大绝对差 0 |
| 32 道验证原始输出与判分 | 全部一致；两组均 5/32 = 15.625%，无格式、截断或运行错误 |
| 本轮训练/验证累计耗时 | 42.454 秒，单次 MPS 观测，不能推导两种实现的普遍速度差 |

详见 [兼容性检查原始结果](../../results/evidence/2026-10-22-E3-compat/compatibility.json)、[原始预测](../../results/evidence/2026-10-22-E3-compat/validation/predictions.jsonl)、[指标](../../results/evidence/2026-10-22-E3-compat/validation/metrics.json)、[训练汇总](../../results/evidence/2026-10-22-E3-compat/training/summary.json)。历史 E2 参考来自当时真实保存的工件，本轮没有重新训练官方组。

## 7. 产物、完成项与局限

- 自定义 last：本地 `artifacts/E3-compat/last/adapter.pt`，SHA256 `005c3d9bfdf391b60d9f96b0ee302918179d4b965c81c8674e4f24587ae1c9ae`，与官方参考一致；[适配器元数据](../../results/evidence/2026-10-22-E3-compat/last_adapter.json)。恢复文件在同一训练目录。
- 对齐完成，可以继续注入模块 / rank 对照。结论限定为当前权重、Q/O、rank16、无 dropout、缩放 1、同设备 / 软件 / 种子 / 数据；不证明任意配置或跨硬件逐位相同。
- 没有能力提升结论，没有重新做主数据集最终测试，也没有本轮 Git 提交 / 推送。大权重继续忽略。
- 下一步为已提前注册的六组 500 步筛选及选定配置 3 轮训练；这些结果另写报告。
