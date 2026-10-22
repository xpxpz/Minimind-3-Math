# E1：一位数加减可学习性检查

状态：已完成。执行日期：2026-10-20。实际在 200 次更新达到训练题 32/32，按预设规则停止；问题留出的 32 道验证题答对 6 道。下节保留试验前固定的设计，后续各节记录实际执行与证据。

## 预先固定的问题与规则

问题：官方 Q/O LoRA 是否能拟合极小的一位数加减训练集，并在本轮 SFT 未见的底层问题上作答？从原始 MiniMind-3 开始，不加载此前 A1 数学适配器。

- 数据：`easy-addsub-v1`，训练 32 题、验证 32 题，均为 0～9 的整数加减，各半；加法按交换律归组，所有划分全局问题键不重叠。减法允许负数。其余小型 ID/模板/范围留出集只冻结，不用于本轮生成或选择。
- 模板、答案格式、Tokenizer、assistant mask、LoRA 目标/Rank/Alpha/Dropout、学习率、有效 batch、设备/精度和生成预算沿用此前方案。任务难度、训练集大小和重复次数与 A1 不同，因此 E1 是可学习性诊断，不是仅改变数值范围的公平消融。
- 配置：官方方阵匹配（实际 Q/O），rank=16、alpha=16、dropout=0，学习率 1e-4、有效 batch=16，MPS/float32、种子 42。
- 最大 1,000 次更新。先训练到 200 次；只按训练题的生成准确率决定是否增加到 500、再到 1,000 次。达到 **29/32（至少 90%）** 或达到最大预算即停止。预算选择不依据验证集生成准确率。
- 主可学习性指标使用各预算终点 `last` 的训练题生成准确率；它仅衡量拟合。验证题只在最终预算确定后评测；训练器原有验证 loss 可记录，但不据此改动停止规则或切换为 best。
- 原始模型先对固定 32 道验证题评测，最终模型用相同题目和协议比较。生成仍为贪心、关闭思考、256-token 上限，精确整数/有理数判分。
- 本轮只检查极小有限任务的拟合与问题留出，不代表通用数学泛化；无法排除原模型预训练中见过简单算术。

## 试验前确定的代码范围

新增 `scripts/generate_easy_math.py`、`configs/easy_addsub_official_r16.json`、`tests/test_easy_math.py`；复用原训练、生成评测、判分和结果分析入口。以下为实际结果，不把未执行的 500/1,000 次预算写成完成。

## 1. 实际执行了什么

1. 枚举 0～9 的单步加法/减法问题池。加法只保留 `a<=b`，共 55 个规范问题；减法保留方向，共 100 个。使用种子 42，先固定划分再训练。
2. 生成完整的小型数据版本：train=32、validation=32、test_id=16、test_template=12、test_range=32、smoke=12，共 136 题。训练和验证各有 16 道加法、16 道减法；范围留出的操作数为 10～19。所有划分全局问题键和文本无重复，训练/验证的 SFT 导出及文件哈希冻结。
3. 运行数据生成测试和原数据审计兼容测试，4 项通过。用实际 Tokenizer 编码训练/验证共 64 题，最长 55 token；assistant 目标未被截断。
4. 用原始 MiniMind-3 在固定 32 道验证题上生成 Baseline，使用原判分器，不从原始长输出中事后放宽规则提取数字。
5. 从同一原始基础权重新建官方 LoRA，不加载 A1 数学适配器。执行 200 次更新，microbatch=4、累积 4 次，有效 batch=16；32 个独立训练问题重复 100 个 epoch，共 3,200 次样本呈现。
6. 保存完整 `artifacts/E1-step200/` 快照，加载 **last** 对全部 32 个训练问题生成，得到 32/32。达到事先固定的 29/32 门槛，因此停止，不执行 500/1,000 次预算。
7. 在停止决定写入记录后，才用同一 last 对 32 个验证问题生成；获得 6/32。进行分运算分析、同题原始模型对比、全部正确样本核查和答案值覆盖率诊断。
8. 保存配置、数据清单、实际训练/验证数据、完整原始输出、训练日志、检查点元数据、停止决定与比较结果，并更新本报告、README 和索引。

训练器仍按原逻辑计算验证 loss 并保存 best，但本轮主模型**不使用验证 loss 选择的 best**。`summary.json` 中 `selected_adapter=best` 是通用训练器的默认推荐字段；本实验预先指定 last，实际生成命令、停止决定和权重哈希都证明使用的是 last。两者权重不同，不能混用结果。

## 2. 写了什么代码、改了什么文件

本轮未修改训练、生成评测或答案判分 Python 代码；新增低难度数据生成代码、配置与必要测试。

| 文件 | 类型 | 具体内容与原因 | 实际验证 |
| --- | --- | --- | --- |
| [scripts/generate_easy_math.py](../../scripts/generate_easy_math.py) | 新增 | 枚举有限问题池，消除加法交换变体，生成六个互不重叠的划分与冻结清单；复用原数据格式以保持训练链路不变 | 真实生成及原审计器验证成功 |
| [configs/easy_addsub_official_r16.json](../../configs/easy_addsub_official_r16.json) | 新增 | 保持 Q/O、rank/alpha、学习率、有效 batch 等配置，允许极小数据重复训练；最大 1,000 次更新 | 实际在 200 次按训练拟合门槛停止 |
| [tests/test_easy_math.py](../../tests/test_easy_math.py) | 新增 | 验证有限池数量、可重复生成、各运算比例、全局分组去重、数值范围、负数与拒绝覆盖 | 与原数据测试合计 4 passed |
| [mathlab/data.py](../../mathlab/data.py)、[sft.py](../../mathlab/sft.py)、[train.py](../../mathlab/train.py) | 复用 | 审计原协议、assistant mask、目标 token 累积、优化、保存和恢复 | 训练结束、原始模型和适配器身份核验 |
| [mathlab/evaluate.py](../../mathlab/evaluate.py)、[compare.py](../../mathlab/compare.py)、[scripts/analyze_predictions.py](../../scripts/analyze_predictions.py) | 复用 | 固定生成/判分、运算分类和同题配对比较 | 3 组生成完成；比较协议检查通过 |
| [数据清单](../../data/manifests/easy-addsub-v1.json)、本报告、README 和索引 | 新增/更新记录 | 可复现的数据身份、规则、停止决定、结果与局限 | 数字、哈希和相对链接核查 |

使用一次性 Python 完成快照复制、数据预检、停止决定、答案覆盖率统计、证据归档和结果一致性核查；未新增独立训练器或修改 LoRA 实现。既有 38 项工程测试与 7 项预测分析测试没有在本轮重新运行，本轮实际运行的是上述 4 项数据相关测试。

## 3. 输入、身份与实际命令

基础模型仍为 63,912,192 参数的 MiniMind-3，官方 LoRA 实际注入 16 个 Q/O 层，共 393,216 个可训练参数。基础权重 SHA256 为 `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`，只读使用；代码基于提交 `170d095`，工作区有未提交改动。

本轮数据版本、种子、生成器/数据哈希见 [数据清单](../../data/manifests/easy-addsub-v1.json)，配置与环境见 [训练运行清单](../../results/evidence/2026-10-20-E1-easy-addsub/training/run.json)。设备为 Apple M4、MPS、float32，Python 3.12.14、PyTorch 2.6.0、Transformers 4.57.6，原生 Qwen3 兼容权重与 Tokenizer 均未更换。

在仓库根目录使用 `.venv/bin/python`，主要实际命令如下，完整顺序见 [commands.txt](../../results/evidence/2026-10-20-E1-easy-addsub/commands.txt)：

```bash
python -m pytest -q tests/test_easy_math.py tests/test_data.py
python -m scripts.generate_easy_math --out data/processed/easy-addsub-v1
python -m mathlab.evaluate --data data/processed/easy-addsub-v1/validation.jsonl \
  --out results/runs/E1-base-validation
python -m mathlab.train --config configs/easy_addsub_official_r16.json \
  --data data/processed/easy-addsub-v1 --out artifacts/E1-easy-addsub --stop-after-updates 200
python -m mathlab.evaluate --data data/processed/easy-addsub-v1/train.jsonl \
  --adapter artifacts/E1-step200/last --out results/runs/E1-step200-train
# 此时训练题 32/32 达到门槛，记录停止决定，再评测 validation。
python -m mathlab.evaluate --data data/processed/easy-addsub-v1/validation.jsonl \
  --adapter artifacts/E1-step200/last --out results/runs/E1-step200-validation
python -m mathlab.compare results/runs/E1-base-validation \
  results/runs/E1-step200-validation --out results/runs/E1-comparison.json
```

输出目录已存在，复跑时应新建目录。本轮没有执行 `--resume` 继续到 500 或 1,000 次，也没有生成其他 E1 留出集或原 v1 正式测试集的答案。

## 4. 实测结果

| 指标 | 原始模型 | E1：200 次更新的 last |
| --- | --- | --- |
| 训练题生成准确率 | 未测 | 32/32（100%） |
| 训练题加法/减法 | 未测 | 各 16/16 |
| 问题留出验证准确率 | 0/32 | 6/32（18.75%） |
| 验证题加法 | 0/16 | 2/16（12.5%） |
| 验证题减法 | 0/16 | 4/16（25%） |
| 验证题合法且未截断 | 0/32 | 32/32 |
| 验证题解析失败 | 19/32 | 0/32 |
| 验证题截断 | 13/32 | 0/32 |
| 验证题数值错误 | 不能由格式失败直接判断 | 26/32 |
| 验证目标 token loss | 3.111573 | 0.892183 |

训练循环约 51.8 秒，含每个小 epoch 的验证和保存，不含 Baseline/后续生成、模型加载等全部开销。累计目标 token 为 16,500，非 padding token 为 173,300。本轮每个 epoch 只有两次更新，验证频率与大数据 A1 不同，不能直接用两轮总耗时推断效率优劣。

验证 loss 最低曾达到 0.533852，终点 last 为 0.892183；训练末端 loss 已接近零。结合训练 100% 与验证 18.75% 的差距，结果与明显的过拟合现象一致。没有用更低验证 loss 的 best 替换预先指定的 last 以美化生成结果。

原始模型 0/32 是**严格格式下的端到端分数**，其中输出包含长解释和非协议格式；不能解读为纯计算能力完全为零。对同一 32 题的端到端差值为 +18.75 个百分点，5,000 次配对 Bootstrap 区间为 [6.25, 31.25] 个百分点，只覆盖这批题目的抽样不确定性，不覆盖训练种子和模型选择/分布不确定性。

## 5. 全部正确验证样本与进一步核查

| 样本 ID | 表达式 | 标准答案与模型输出值 |
| --- | --- | --- |
| easy_validation_ef19e9abbc90211f | 3 - 3 | 0 |
| easy_validation_cab3ddccc1028b07 | 0 - 3 | -3 |
| easy_validation_fe1b8bbf9615f307 | 0 - 1 | -1 |
| easy_validation_5e364612d0044989 | 0 + 4 | 4 |
| easy_validation_4d689e338c56d18d | 5 - 6 | -1 |
| easy_validation_536a683ce977e33a | 0 + 0 | 0 |

这 6 个正确样本中，5 个含零操作数或同数相减；只有 `5-6` 不属于上述简单情形。全部原始输出均保留，不能只展示 32/32 的训练成绩或少量成功题来声称掌握了加减法。

此外，32 个验证问题中有 9 题的标准答案数值未在这 32 条训练标签中出现，这 9 题全部答错；其余 23 题中答对 6 题。这是有限数据覆盖不足的诊断线索，不是已经确认的唯一因果解释，也不意味着模型理论上无法生成未见数字。

## 6. 完成事项与产物

| 完成项 | 证据或位置 |
| --- | --- |
| 小数据生成、冻结和审计 | [数据清单](../../data/manifests/easy-addsub-v1.json)、[实际训练题](../../results/evidence/2026-10-20-E1-easy-addsub/train_data.jsonl)、[实际验证题](../../results/evidence/2026-10-20-E1-easy-addsub/validation_data.jsonl) |
| 200 次更新与小训练集拟合 | [训练日志](../../results/evidence/2026-10-20-E1-easy-addsub/training/train.jsonl)、[训练题原始输出](../../results/evidence/2026-10-20-E1-easy-addsub/training_generation/predictions.jsonl) |
| 按预设门槛停止 | [试验设计](../../results/evidence/2026-10-20-E1-easy-addsub/plan.json)、[停止决定](../../results/evidence/2026-10-20-E1-easy-addsub/stopping_decision.json) |
| 原始模型与新模型的同题验证 | [Baseline 输出](../../results/evidence/2026-10-20-E1-easy-addsub/baseline_validation/predictions.jsonl)、[微调输出](../../results/evidence/2026-10-20-E1-easy-addsub/validation_generation/predictions.jsonl)、[比较](../../results/evidence/2026-10-20-E1-easy-addsub/comparison.json) |
| 运算与覆盖诊断 | [验证分析](../../results/evidence/2026-10-20-E1-easy-addsub/validation_generation/analysis.json)、[完整成功案例和覆盖率](../../results/evidence/2026-10-20-E1-easy-addsub/diagnostics.json) |
| 适配器与恢复状态 | 本地 `artifacts/E1-easy-addsub/`、`artifacts/E1-step200/`；主模型为后者的 last，权重不进入 Git |

报告、数据清单、代码和小型证据写入本地工作区，尚未形成新的 Git 提交或推送。没有删除或覆盖之前的 A1 权重、原数据或实验记录。

## 7. 结论、局限与下一步

本轮完成了可学习性检查：当前基础模型、官方 LoRA 和训练链路能拟合这 32 道有数值答案的小任务；因此先前的低分不能简单归因于“LoRA 完全不更新”或“答案从未受到监督”。这不等于已排除所有潜在工程问题。

问题留出验证只有 6/32，多数正确题还属于简单情形，说明当前收益以训练题拟合/格式学习为主。无法据此区分唤醒已有能力、记忆有限映射与学习新算法；不将本轮数据宣传为掌握通用算术。只有一个种子、一个极小有限划分，且新数据大小、难度与重复次数都不同，不能和 A1 的 17.4% 做能力高低的直接比较。

下一阶段建议：固定同一批验证问题、LoRA 与总更新预算，单独扩充不同训练问题的覆盖率/多样性，并发布新数据版本，比较训练问题数量的影响；避免同时修改数据规模、监督过程和注入模块。若更多覆盖仍无助于泛化，再单独检查模块或过程监督。上述下一阶段尚未执行，当前不进入 GRPO。
