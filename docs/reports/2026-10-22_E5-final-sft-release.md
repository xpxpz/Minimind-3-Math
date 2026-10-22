# 2026-10-22：E5 三轮 LoRA-SFT、三种子最终测试与本地交付

> 状态：LoRA-SFT训练、预注册测试与本地模型交付完成。**普通数学计算能力的目标尚未达到**：发布版本38/270，其中37题是直接分数输出，无除法题仅1/183。不能将本报告表述为“数学推理能力显著提升成功”。

## 1. 目标、方案与实际范围

将 [E4](2026-10-22_E4-module-rank-screening.md) 仅凭验证集选出的 Q/V rank16 训练到三轮，独立复验 seed123/2026，再与原始模型比较。最终固定 ID、模板 OOD、范围 OOD 各90题，共270个独立问题；三个训练种子共810条测试预测，不能叫作810道独立题。发布种子预先固定42。

计划与哈希在 [plan.json](../../results/evidence/2026-10-22-posttrain/plan.json)，测试前的选择在 [selection.json](../../results/evidence/2026-10-22-posttrain/selection.json)。全部训练从原始基础权重开始；seed42恢复自己的一轮筛选训练，额外增加1,000更新；其他种子从原始模型训练完整1,500更新。没有加载E1/E2适配器，也没有用测试分数选配置、步数或种子。

## 2. 输入与配置

- 主数据 math-data-v1：8,000条训练、1,000题loss验证。训练 / 验证 / 测试全局问题键互斥；manifest SHA256 `e9e2af8bfe7aefd4b1d26bb52747131e5de486eaae02c9e14a272fb9e4766531`，见 [数据清单](../../data/manifests/v1.json)。
- 基础 MiniMind-3 Dense：63,912,192参数，本地 `/Users/xiexianpei/minimind/minimind-3`，基础权重SHA256 `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`。
- [seed42配置](../../configs/campaign/qv_r16.json)：Q/V、rank=alpha=16、dropout0，344,064可训练参数；AdamW lr1e-4、weight_decay .01、clip1、恒定学习率、batch4×accum4=16、长度256、float32/MPS。每个种子目标三轮、1,500更新、24,000次样本呈现。
- 检查点按完整1,000题验证目标token loss最小选择，包含step0；最终生成与原始模型统一贪心、关闭思考、batch1、256token、math-answer-v1。解析失败、截断、运行错误均计入分母。
- 训练时 Git `170d095c23da50d5e91a8790f16ca912509f683f` 加未提交工作区；Python3.12.14 / Torch2.6.0 / Transformers4.57.6 / macOS26.6.2 arm64。源码及Tokenizer完整身份在各运行清单与预注册方案。
- 本轮实际核对 [硬件](../../results/evidence/2026-10-22-posttrain/hardware.json)：Apple M4、16GiB统一内存；训练和主要评测MPS，导出数值检查CPU。未测MPS峰值，不能把CUDA空值当作零显存。

## 3. 实际执行步骤

1. 保留 `artifacts/SFT-screen-snapshot-qv_r16` 一轮权重 / 恢复状态后，运行编排的 final 阶段。
2. seed42已完成三轮，162,435目标token、1,512,606非padding token、记录的累计训练含验证/保存442.309秒。best与last均为SHA256 `1467af11668be5393921d80fe98b4e75a1b8400bdbcd5b2a754bc3a709c447d8`；best/last验证loss均0.861163904。
3. seed123和2026均完成三轮、1,500更新；每组同为162,435目标token、1,512,606非padding token。最佳loss分别0.862085546、0.860894878，训练含验证/保存432.093秒、425.364秒；[三组训练汇总](../../results/evidence/2026-10-22-posttrain/training_completed.json) 保存完整哈希。三个best均与各自last相同，因此实际发布的是1,500更新的权重，没有用较早checkpoint冒充三轮模型。
4. 完成原始模型和三个种子的全部预注册270题，正式预测共1,080条，另有12条预热不计入分母；按协议保留失败，生成9份配对比较。没有在测试中调参或更换种子。
5. 汇总脚本再次核对ID、原始预测及评分，复算比较结果；[最终汇总](../../results/evidence/2026-10-22-posttrain/final_summary.json) 记录题型 / 运算 / 分布及三种子均值和样本标准差。
6. 严格核对发布适配器与seed42实测适配器一致，打包到本地 `artifacts/release/adapter`。在CPU合并为完整HF模型，检查合并数值误差、保存并重载，再整理到 `artifacts/release/merged`。使用默认适配器入口做真实提问冒烟。
7. 更新模型卡、README、使用说明、报告索引；用 [交付审计脚本](../../results/evidence/2026-10-22-posttrain/verify_delivery.py) 核对预算、权重、36条子命令退出状态、结果及文档链接，结果见 [delivery_verification.json](../../results/evidence/2026-10-22-posttrain/delivery_verification.json)。

完整执行记录见 [final_driver.log](../../results/evidence/2026-10-22-posttrain/final_driver.log)、[完成标记](../../results/evidence/2026-10-22-posttrain/completed.json)，seed42的 [训练清单](../../results/evidence/2026-10-22-posttrain/runs/final-train-seed42/run.json)、[汇总](../../results/evidence/2026-10-22-posttrain/runs/final-train-seed42/summary.json)、[完整训练日志](../../results/evidence/2026-10-22-posttrain/runs/final-train-seed42/train.jsonl) 均已归档。

## 4. 代码与文件改动

| 文件 | 类型 | 内容与验证 |
| --- | --- | --- |
| [posttrain_campaign.py](../../scripts/posttrain_campaign.py) | 复用本工作轮新增编排 | final子命令按预注册配置训练和测试，逐条记录命令 / 返回码 |
| [summarize_campaign.py](../../scripts/summarize_campaign.py) | 本工作轮新增 | 审计固定测试ID、汇总运算类别和三个种子均值 / 样本标准差，复算9份配对比较；实际全量执行通过 |
| [release_campaign.py](../../scripts/release_campaign.py) | 本工作轮新增 | 只打包预先固定seed42，实际核对它与三个测试划分采用的适配器完全一致 |
| [use_math_model.py](../../scripts/use_math_model.py) | 复用本工作轮新增使用 / 导出入口 | 最终答案提示、严格加载、合并和重载；实际CPU导出与默认发布入口提问通过完整性验收 |
| `configs/campaign/final_seed*.json` | 新增配置 | 仅改变种子，其余沿用被选配置 |
| `mathlab/train.py`、`evaluate.py`、`compare.py`、`answers.py`、`lora.py`、`runtime.py` | 复用已有实现 | 本轮没有修改核心训练 / 判分 / 评测逻辑，历史功能不算作本轮新增 |
| `vendor/minimind/model_lora.py` | 引用上游 | 官方代码原样保留，上游SHA在运行环境中记录 |
| [使用文档](../posttraining_usage.md)、[模型卡](../model_card.md)、README、本报告、索引、轻量证据 | 新增 / 修改 | 说明实际结果、复现与使用，保留能力目标未达标的结论 |
| 证据中的 `verify_screening.py` / `verify_delivery.py` | 新增一次性审计 | 随证据保存实际检查逻辑，不是新的训练算法；真实工件集成审计通过 |

## 5. 实际命令

从项目根目录实际执行；已有目录不能直接覆盖复跑：

```bash
.venv/bin/python -m scripts.posttrain_campaign final
.venv/bin/python -m scripts.summarize_campaign
.venv/bin/python -m scripts.release_campaign
.venv/bin/python -m scripts.use_math_model export --adapter artifacts/SFT-qv_r16/best --device cpu --out artifacts/SFT-merged-seed42
.venv/bin/python -m scripts.use_math_model ask '计算 12 + 7。' --adapter artifacts/SFT-qv_r16/best --device cpu
.venv/bin/python -m scripts.use_math_model ask '计算 12 + 7。'
.venv/bin/python -m results.evidence.2026-10-22-posttrain.verify_delivery
```

每条实际子命令和日志在 [commands](../../results/evidence/2026-10-22-posttrain/commands)、[logs](../../results/evidence/2026-10-22-posttrain/logs)。恢复保持原始代码 / Git / 设备 / 环境 / 配置不变，没有改写旧运行清单绕过检查。

导出在其他种子训练期间于CPU执行，随后一次性Python用 `shutil.move("artifacts/SFT-merged-seed42", "artifacts/release/merged")` 整理目录；只移动文件，没有改变权重。导出命令和结果见 [export_command.json](../../results/evidence/2026-10-22-posttrain/export_command.json)、[export.log](../../results/evidence/2026-10-22-posttrain/export.log)。后半段评测使用随驱动PID结束的 `caffeinate -i -w 10931` 防止空闲休眠，[记录](../../results/evidence/2026-10-22-posttrain/keepawake.json) 没有将早前墙钟空档归因为已经证实的休眠。

## 6. 验证与结果

### 训练量与检验

E3/E4本工作轮已实际运行55项测试和数据 / mask / 长度检查，其日志在前序报告；E5未重复运行这55项，不写成新增测试。E5实际执行真实权重的训练、1,080条测试预测、汇总复算、导出、重载、发布入口及交付审计。

三个最终模型各24,000次样本呈现，合计72,000；seed42的前8,000次已计入E4。整个主campaign去掉这段重复后为7,000更新、112,000次样本呈现。三最终训练记录合计1,299.77秒；E4+E5去重为2,194.57秒，均为训练含验证/保存活动计时，不含全部加载、生成及墙钟空档，不是纯GPU算子性能。

### 预注册测试子集

| 模型 | ID /90 | 模板OOD /90 | 范围OOD /90 | 合计 /270 | 无除法 /183 | 格式有效 /270 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始模型 | 0 | 0 | 0 | 0（0%） | 0 | 0 |
| seed42，预先指定发布 | 18 | 12 | 8 | 38（14.074%） | 1 | 269 |
| seed123 | 16 | 13 | 14 | 43（15.926%） | 1 | 269 |
| seed2026 | 16 | 13 | 13 | 42（15.556%） | 2 | 267 |

三种子总体准确率均值15.185%，样本标准差0.980个百分点；这是3个训练种子的观测离散程度，不是均值置信区间。发布仍用预先固定的seed42，**没有事后换成得分更高的seed123**。

原始模型119条parse_failure、151条truncated；seed42为38正确、231数值错误、1解析失败；seed123为43正确、226数值错误、1解析失败；seed2026为42正确、225数值错误、3解析失败。所有模型均无runtime_error，三个SFT模型均无truncated。原始模型0分是本格式和上限约束下的分数，不能声称其所有数学能力为0。

发布模型相对原始模型的逐分布配对Bootstrap（5,000次、seed42）差值：ID +20.000pp，95%区间[12.222,28.889]；模板 +13.333pp，[6.667,21.111]；范围 +8.889pp，[3.333,14.444]。这衡量题目抽样不确定性，不能去除格式学习混杂，也不代替训练种子波动；未做多重比较校正。

### 数学计算诊断与原始证据

seed42的38个正确答案中，37个是直接保留单步除法分子/分母；seed123为41/43，seed2026为40/42。seed42无除法题仅1/183（0.546%），模板和范围的无除法题分别0/58、0/62。严格协议得分变高，主要不能归结为一般计算能力增强。

按冻结预测顺序每种运算首个正确/错误案例，发布模型的ID输出包括 `8*69 → 598`（应552）、`93+28 → 83`（应121）、`(42-8)/83 → -41/83`（应34/83）、`62-66 → -4` 正确；`54/71 → 54/71`为分数复制。完整案例及选择规则在 [ID分析](../../results/evidence/2026-10-22-posttrain/runs/test-seed42-test_id/analysis.json)，不是只挑成功样例。

发布模型原始预测：[ID](../../results/evidence/2026-10-22-posttrain/runs/test-seed42-test_id/predictions.jsonl)、[模板](../../results/evidence/2026-10-22-posttrain/runs/test-seed42-test_template/predictions.jsonl)、[范围](../../results/evidence/2026-10-22-posttrain/runs/test-seed42-test_range/predictions.jsonl)。原始模型与其余种子同样完整保存在 [runs目录](../../results/evidence/2026-10-22-posttrain/runs)，[汇总](../../results/evidence/2026-10-22-posttrain/final_summary.json) 包含全部分布、运算和9份比较。

### 模型文件完整性

[CPU导出检查](../../results/evidence/2026-10-22-posttrain/export_verification.json)：合并前后固定提示logits最大绝对差7.9155e-5，`rtol=atol=1e-4`通过；磁盘重载后logits完全相等。它只覆盖固定提示的数值检查，不证明任意生成逐位相同或能力提升。[CPU入口冒烟](../../results/evidence/2026-10-22-posttrain/ask_command.json) 对12+7输出10，明确保留失败；该例不计入270题，不用于选模型。发布默认入口输出见 [release_ask.log](../../results/evidence/2026-10-22-posttrain/release_ask.log)。

## 7. 产物与完成情况

| 产物 | 状态与路径 |
| --- | --- |
| 三种子最终训练与恢复 | 完成；本地 `artifacts/SFT-qv_r16`、`SFT-final-seed123`、`SFT-final-seed2026`；完整SHA见training_completed.json |
| 发布适配器 | 完成；`artifacts/release/adapter`；SHA256 `1467af11668be5393921d80fe98b4e75a1b8400bdbcd5b2a754bc3a709c447d8` |
| 完整HF模型 | 完成；`artifacts/release/merged`；model.safetensors SHA256 `c24414fd4b1304b64d4580caa69ae32b890e11963785a118b26f8e099193d37e` |
| 发布清单 | [release.json](../../results/evidence/2026-10-22-posttrain/release.json)，绑定预注册选择、实测权重、汇总哈希 |
| 实际测试与比较 | 1,080条预测、9份比较、三种子汇总完成；36条campaign子命令均成功 |
| 文档与入口 | 模型卡、使用说明、阶段报告、索引和真实提问 / 导出入口完成 |

所有基础模型、适配器、合并模型、优化器恢复状态继续留在被忽略的本地目录。本报告训练 / 测试结束时，本轮代码和报告尚未提交 / 推送；最终Git交付状态将在交付后补记。原先带“ 2”的用户副本未删除或用于Python导入，历史训练清单可能将这些文件纳入源码指纹，应保留用于旧工件审计。

## 8. 局限与下一步

这是合成基础数学任务的第一版LoRA-SFT，不含GRPO、教师蒸馏或全参数训练。三种子仅对选中的配置执行，筛选不是完整多种子析因实验；只评测预注册270题，不等于全部1,600题；分数允许数值等价的未约分分数。最终须同时解释格式学习与无除法题的计算表现。

本轮已完成后训练工程闭环与第一版模型交付，**数学能力目标未达标**。下一项研究应另立实验验证数位课程 / 过程监督等数据假设，并冻结新的确认测试集；不能用本轮已查看的测试结果反复调参后继续称为独立验证。这些新实验本轮未执行，不将其写成完成，也没有直接启动GRPO。
