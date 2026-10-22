# 2026-10-22：E4 主数据集上的模块与 Rank 筛选

> 状态：完成。六组各训练一轮并完成固定验证生成；最终三轮训练和测试单独记录在 E5，不能将筛选结果当作最终测试。

## 1. 目标与范围

回到 8,000 条主训练数据，完成官方 Q/O、Q/V、Q/K/V/O、Attention+FFN 及 rank8/16/32 的有界筛选。每组均从原始权重训练 500 次更新（一轮），不能把一轮筛选称为所有配置均已完成三轮训练。验证集选择结束后，最终三轮训练与测试另见 E5 报告。

## 2. 输入与预注册配置

- [冻结方案](../../results/evidence/2026-10-22-posttrain/plan.json)：在筛选前保存基础模型、数据、源码、配置 SHA256，固定 120 道验证题及 270 道最终测试题的 ID。
- [数据清单](../../data/manifests/v1.json)：math-data-v1，训练 8,000、验证 1,000、ID 测试 1,000、模板 / 范围 OOD 各 300，开发题 240。各划分全局问题键去重；本阶段仅用训练、验证。
- 基础权重 `/Users/xiexianpei/minimind/minimind-3`，63,912,192 参数；`model.safetensors` SHA256 `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`。没有加载 A1/E1/E2/E3 适配器。
- 代码身份 `170d095c23da50d5e91a8790f16ca912509f683f` 加未提交工作区，完整源码与环境哈希见方案和每组运行清单。设备 Mac M4 / MPS，float32；软件版本由实际运行清单记录。
- 六组 [配置目录](../../configs/campaign)：seed42、alpha=rank、dropout0、lr1e-4、AdamW weight_decay .01、clip1、恒定学习率、batch4×accum4=16、长度256；500 次更新，8,000 次样本呈现，各组样本顺序一致。
- 每组 best 按完整 1,000 题验证的目标 token loss 选择（允许 step0）；再用固定 120 题（每类40）贪心生成，以正确数降序、参数量升序、验证 loss 升序、组名升序选候选。协议为关闭思考、batch1、256 token、math-answer-v1，失败进入分母。

## 3. 实际执行步骤

1. E3 先验证官方 / 自定义 Q/O 的真实训练逐位相同，见 [E3 报告](2026-10-22_E3-official-custom-compatibility.md)。
2. 编排脚本 freeze 数据、源码、配置和选择规则；随后依次真实训练、生成验证、审计原始预测并归档。
3. 六组全部完成训练与生成，结果保存在 [screening.json](../../results/evidence/2026-10-22-posttrain/screening.json) 和 [驱动日志](../../results/evidence/2026-10-22-posttrain/screening_driver.log)。在任何最终测试前保存 [selection.json](../../results/evidence/2026-10-22-posttrain/selection.json)，选择 Q/V rank16。
4. 保存 `artifacts/SFT-screen-snapshot-qv_r16` 一轮完整快照，再将原训练目录交给 E5 续训。归档的一轮指标不会被三轮指标覆盖。
5. 运行 [一次性审计脚本](../../results/evidence/2026-10-22-posttrain/verify_screening.py)，核对六组更新1..500、每组8,000次样本呈现、有限loss/梯度、token数、冻结验证ID及实际适配器哈希；[审计结果](../../results/evidence/2026-10-22-posttrain/screening_verification.json) 全部通过。审计脚本随证据保存，不是新增训练器。

## 4. 代码 / 文件改动

| 文件 | 类型 | 作用与验证 |
| --- | --- | --- |
| [posttrain_campaign.py](../../scripts/posttrain_campaign.py) | 复用本工作轮 E3 时新增的编排代码 | 冻结身份、调用统一训练器、保存命令退出码、验证集选择；实际六组运行验证 |
| `configs/campaign/*.json` | 本阶段新增配置 | 模块 / rank 可比较，其他超参数和训练量一致 |
| [test_campaign.py](../../tests/test_campaign.py) | 复用本轮新增测试 | 确认选择优先级和禁止以 test 集选择；E3 中实际通过2项 |
| `mathlab/train.py`、`lora.py`、`evaluate.py`、`answers.py`、`runtime.py` | 复用 | 本轮未新增或修改训练/评测核心 Python 代码；沿用 mask、token 累积、严格加载与精确判分 |
| `vendor/minimind/model_lora.py` | 引用上游 | 原样保留 MiniMind 提交 `f659b55761b754d306bd140573493a6543cafd7f` 官方注入实现 |
| 本报告、证据目录、报告索引 | 新增 / 修改 | 保存实际结果和边界；与历史实验分开 |

## 5. 实际命令与验证

在项目根目录实际运行：

```bash
.venv/bin/python -m scripts.posttrain_campaign freeze
.venv/bin/python -m scripts.posttrain_campaign screening
.venv/bin/python -m mathlab.preflight
.venv/bin/python -m pytest -q tests/test_answers.py tests/test_data.py tests/test_evaluation.py tests/test_training.py tests/test_prediction_analysis.py tests/test_easy_math.py tests/test_coverage_experiment.py
.venv/bin/python -m results.evidence.2026-10-22-posttrain.verify_screening
```

编排所执行的每条训练 / 评测子命令、参数、工作目录、开始 / 结束时间和返回码逐条保存在 [commands](../../results/evidence/2026-10-22-posttrain/commands)，stdout/stderr 在 [logs](../../results/evidence/2026-10-22-posttrain/logs)。已有输出目录不能直接复跑覆盖。

本阶段回归测试 [48 passed](../../results/evidence/2026-10-22-posttrain/regression_tests.log)，加 E3 实际重跑的 7 项共55项。训练前检查 [preflight](../../results/evidence/2026-10-22-posttrain/preflight.json) 确认最大训练长度74；[检查命令记录](../../results/evidence/2026-10-22-posttrain/additional_checks.json) 区分本轮检查与历史测试。MPS 显存峰值未测，不能把 CUDA 的 null 写作 0 显存。

## 6. 结果

| 组（各500更新） | 可训练参数 | 验证正确/120 | 无除法正确/88 | 单步除法正确/22 | 最佳验证loss | 训练含验证/保存秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方 Q/O r16 | 393,216 | 23 | 1 | 22 | 0.934094 | 152.84 |
| Q/V r16 | 344,064 | 23 | 1 | 22 | 0.935571 | 153.51 |
| Q/K/V/O r16 | 688,128 | 23 | 1 | 22 | 0.915403 | 169.62 |
| Attention+FFN r8 | 958,464 | 23 | 1 | 22 | 0.902975 | 191.75 |
| Attention+FFN r16 | 1,916,928 | 23 | 1 | 22 | 0.888585 | 191.13 |
| Attention+FFN r32 | 3,833,856 | 21 | 0 | 21 | 0.864190 | 189.46 |

前五组均19.167%，rank32为17.5%；六组均120/120产生格式有效且未截断的答案。其余10道含除法的混合题不能由“单步除法”22题代替。所有正确单步除法均直接保留输入分子/分母，不代表学会约分或通用计算；协议按精确数值相等判分，未约分仍算正确。

六组每组均54,145目标token、504,202非padding token，共3,000更新 / 48,000次样本呈现；累计记录的训练含验证/保存时间1,048.31秒，不包括全部进程启动、加载、验证生成时间。单次顺序运行有温度和系统负载影响，不能把秒数当作稳定能效结论。

**选择：Q/V rank16**。五组正确数持平，依预注册规则选参数最少者，没有因rank32的loss更低而改选它。此规则用于选择后续训练配置，不证明Q/V总体最优。

原始预测和分析在 [各组归档](../../results/evidence/2026-10-22-posttrain/runs)。例如 [Q/V分析](../../results/evidence/2026-10-22-posttrain/runs/screen-eval-qv_r16/analysis.json)、[原始预测](../../results/evidence/2026-10-22-posttrain/runs/screen-eval-qv_r16/predictions.jsonl) 按固定顺序选择每种运算首个正确/错误案例：`94+55`输出55（应149），`48-51`输出-3正确，`29*11`输出392（应319）。这揭示普通计算仍是主要问题，不能将格式稳定或loss下降写作数学能力已经稳定提升。

## 7. 产物与未完成项

- 本地权重和恢复状态：`artifacts/SFT-<arm>`，被 Git 忽略；每组适配器哈希保存在运行目录及归档的 `best_adapter.json`、`last_adapter.json`。
- 轻量证据：[runs](../../results/evidence/2026-10-22-posttrain/runs)、配置、计划、逐题预测和日志。未提交或推送本工作轮代码 / 报告。
- 本阶段完成：六组训练、720条验证预测、候选选择、快照、预算与哈希审计。三轮最终模型、额外训练种子、最终测试、模型导出均不计为本报告已经完成，交由E5单独验收。

## 8. 局限与下一步

模块相同 rank 时参数量不同，不能当作等参数量实验。筛选仅一个种子、一轮预算；120 道验证已经用于开发，不是独立最终测试；多组选择存在验证集选择偏差。除法可直接写出分数，应同时报告非除法及逐运算结果。

下一步按预注册规则冻结候选，然后训练到三轮并用 seed42/123/2026 评测；发布种子预先固定42，不能按测试结果挑种子。最终测试严格使用预先保存的270个问题。跨版本审计见 [cross_version_audit.json](../../results/evidence/2026-10-22-posttrain/cross_version_audit.json)：这270题与E2训练题无重叠，且本轮模型均从原始权重训练。
