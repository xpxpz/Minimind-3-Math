# E2：固定验证题与预算的训练覆盖率对照

执行日期：2026-10-22。状态：已完成。E2 训练集 64/64，固定的 32 道验证题为 5/32，E1 同环境复评为 6/32；未观察到扩大这份训练集带来的验证改善。下节保留试验前方案，后文记录实际执行及局限。

## 固定的问题与实验规则

- 对照 E1：原始 MiniMind-3、官方 Q/O LoRA、rank=16、alpha=16、dropout=0、学习率 1e-4、有效 batch=16、种子 42、MPS/float32，固定 200 次更新、终点 last。
- E2 从同一原始模型开始，**不从 E1 适配器续训**。训练集从 32 增为 64 个不同问题，包含全部 E1 训练题，额外增加 16 道加法和 16 道减法；操作数仍为 0～9，提示词和答案监督不变。
- 新增训练问题只按运算类型、种子抽样，不依据验证答案、模型成功/失败或答案值覆盖率挑选。训练与验证问题键无重叠。
- E1 的 32 道验证题逐字节保留，ID、顺序、表达式和标准答案不变；沿用贪心、关闭思考、256-token 上限与原精确判分。
- 固定 200 次更新，无论 E2 训练正确率如何都不延长预算。两组均有 3,200 次样本呈现：E1 每题重复 100 次，E2 每题重复 50 次。固定的是更新数/样本呈现数，不声称完全匹配 token 或墙钟预算。
- 一位数加法只有 55 个交换律规范问题；保留 32 个训练加法和 16 个验证加法后，仅余 7 个。因此新版本保留小型备用划分，允许将 E1 **从未进行生成评测/选择的**备用测试或 smoke 问题重新划入训练，并在清单中逐项记录。这些历史备用题不能再充当 E2 的未见测试。
- 验证集已用于此前诊断，本轮是探索性对照，不是新的确认性测试。主指标是同一 32 题上的准确率；另报告加/减、格式、训练拟合和答案值覆盖率。
- 新版本整个 manifest 必然不同。新增独立同题比较入口：逐个验证各自真实 manifest，要求验证文件哈希、ID、协议和环境一致，明确记录两份 manifest 的差异；不改写旧元数据或放宽原比较工具的默认检查。

## 试验前确定的代码范围

新增数据扩充脚本、相同验证集比较脚本与必要测试；训练/生成/判分代码及 E1 配置复用。以下补充实际执行、代码、结果、产物和下一步。

## 1. 实际执行了什么

1. 读取并审计冻结的 E1 数据，保留全部 32 条原训练记录；在排除原训练/验证规范问题键后，按种子 42 和运算类型抽取 32 条新训练题。新训练集加法、减法各 32 题。
2. 新建 `easy-addsub-coverage-v2`，逐字节复制 E1 的 32 条验证记录。验证 SHA256 在两版本中均为 `c8391e1465a346139d79ca5a2f2d0a4d36ab2d15810631b5f44b004766eacf0a`；其他备用划分缩小为 ID=4、模板=4、smoke=4，范围留出保留 32 题，总计 140 题。
3. 明确记录 20 个旧 E1 备用问题转为 E2 训练问题：来自旧 test_id 9 题、test_template 8 题、smoke 3 题。它们此前未做 E1 生成评测，新版本中已经退役为训练数据，不能再被称为 E2 未见测试。
4. 执行新增 6 项与既有 2 项数据相关测试，共 8 项通过；确认数据可重复、旧训练记录保留、验证字节一致、新划分内部无重复，以及比较工具仍拒绝验证/协议/清单错配。
5. 从原始 MiniMind-3 新建官方 LoRA，使用与 E1 完全相同的配置文件，在新 64 题上训练 200 次更新。实际为 50 个 epoch、3,200 次样本呈现，每个问题出现 50 次；不根据训练/验证结果追加预算。
6. 将完整训练目录另存为 `artifacts/E2-step200/`，评测终点 **last**，未切换为验证 loss 最优的 best。
7. 在当前环境重新加载保存的 E1 last，对原 32 题复评；ID、全部原始输出和正确标签与 2026-10-20 的历史结果逐题一致，仍为 6/32。
8. 对 E2 的 64 条训练题和相同 32 条验证题生成，分别得到 64/64 与 5/32；按运算分类，执行同题配对比较，核查全部正确题和发生变化的题。
9. 额外审计 E2 与主数据 v1 的规范问题重叠，归档配置、源文件/数据身份、训练日志、全部预测、清单与报告。训练代码、原判分器和旧实验元数据均未改写。

## 2. 本轮代码与文件改动

本轮未修改训练/生成评测/答案判分 Python 代码；新增的是数据构建及结果比较代码。

| 文件 | 本轮类型 | 内容与原因 | 实际验证 |
| --- | --- | --- | --- |
| [scripts/expand_easy_coverage.py](../../scripts/expand_easy_coverage.py) | 新增 | 保留 E1 原训练题及验证字节，按种子加入 32 条新题，发布新划分与旧备用题退役记录 | 实际生成、原审计器及重复生成检查通过 |
| [scripts/compare_shared_validation.py](../../scripts/compare_shared_validation.py) | 新增 | 各自审计真实数据清单，要求验证字节/ID/基础模型/协议/环境一致，仅允许两版本整体 manifest 不同；输出整体与加减分组的配对比较 | 正向比较通过；错配拒绝测试通过 |
| [tests/test_coverage_experiment.py](../../tests/test_coverage_experiment.py) | 新增 | 数据保留/扩充、显式同题比较、严格默认策略保持、篡改/缺行/协议/清单拒绝测试 | 新增 6 项通过；与 E1 数据测试合计 8 passed |
| [mathlab/train.py](../../mathlab/train.py)、[sft.py](../../mathlab/sft.py)、[lora.py](../../mathlab/lora.py) | 复用 | 相同优化、mask、官方注入与保存路径 | 200 次更新完成，训练题 64/64，权重身份核验 |
| [mathlab/evaluate.py](../../mathlab/evaluate.py)、[answers.py](../../mathlab/answers.py)、[scripts/analyze_predictions.py](../../scripts/analyze_predictions.py) | 复用 | 原生成与判分、分运算和预测审计 | 三组生成完成，训练与验证分析完成 |
| [mathlab/compare.py](../../mathlab/compare.py) | 原样保留 | 默认仍要求整体 manifest 一致，不为本实验关闭其检查 | 单元测试确认原工具拒绝跨清单比较 |
| [配置](../../configs/easy_addsub_official_r16.json) | 原样复用 | 与 E1 同一 JSON 文件、相同超参数与最大预算，实际都在 200 次停止 | 配置展开后逐项相等 |
| [新数据清单](../../data/manifests/easy-addsub-coverage-v2.json)、本报告、README 和索引 | 新增/更新记录 | 记录实验身份、完整结果及限制 | 数字、哈希和相对链接核查 |

另使用一次性 Python 执行配置/验证身份检查、快照复制、E1 历史输出一致性核验、答案值覆盖率与跨版本问题重叠统计、证据归档。既有 38 项工程测试和 7 项分析测试属于历史记录，本轮没有重新运行；本轮实际测试命令为后文的 8 项数据/比较测试。

## 3. 输入与代码身份

基础模型为 63,912,192 参数的 MiniMind-3 Dense，权重 SHA256 仍为 `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`。训练 393,216 个 LoRA 参数，实际注入 8 层 Q/O，共 16 个线性投影。

设备为 Apple M4、MPS、float32；种子 42，Python 3.12.14、PyTorch 2.6.0、Transformers 4.57.6。代码基于 `170d095`，工作区有未提交改动。详细身份见 [训练清单](../../results/evidence/2026-10-22-E2-coverage/training/run.json)，新增脚本/配置/数据哈希见 [试验方案](../../results/evidence/2026-10-22-E2-coverage/plan.json) 与 [数据清单](../../data/manifests/easy-addsub-coverage-v2.json)。

E1/E2 的训练超参数、基础权重/Tokenizer 和原训练源文件保持一致。训练数据长度不同，因此每个 epoch 的样本排列和重复次数不同；验证 loss 计算次数也由 E1 每 2 次更新一次变为 E2 每 4 次更新一次。由于主模型固定为 last，验证次数差异不用于模型选择，但会影响墙钟开销。

## 4. 实际命令

从仓库根目录使用 `.venv/bin/python`，完整执行顺序保存在 [commands.txt](../../results/evidence/2026-10-22-E2-coverage/commands.txt)。主要命令：

```bash
python -m pytest -q tests/test_coverage_experiment.py tests/test_easy_math.py
python -m scripts.expand_easy_coverage --source data/processed/easy-addsub-v1 \
  --out data/processed/easy-addsub-coverage-v2
python -m mathlab.train --config configs/easy_addsub_official_r16.json \
  --data data/processed/easy-addsub-coverage-v2 --out artifacts/E2-coverage --stop-after-updates 200
python -m mathlab.evaluate --data data/processed/easy-addsub-v1/validation.jsonl \
  --adapter artifacts/E1-step200/last --out results/runs/E2-reference-E1-validation
python -m mathlab.evaluate --data data/processed/easy-addsub-coverage-v2/train.jsonl \
  --adapter artifacts/E2-step200/last --out results/runs/E2-step200-train
python -m mathlab.evaluate --data data/processed/easy-addsub-coverage-v2/validation.jsonl \
  --adapter artifacts/E2-step200/last --out results/runs/E2-step200-validation
python -m scripts.compare_shared_validation results/runs/E2-reference-E1-validation \
  results/runs/E2-step200-validation \
  --baseline-data data/processed/easy-addsub-v1/validation.jsonl \
  --candidate-data data/processed/easy-addsub-coverage-v2/validation.jsonl \
  --out results/runs/E2-vs-E1-shared-validation.json
```

还实际执行了原预测分析脚本，分别输出训练/验证的 `analysis.json`。快照和归档用一次性文件操作完成；目录已存在，复跑应使用新目录。

## 5. 实测结果

| 指标 | E1：32 道训练题 | E2：64 道训练题 |
| --- | --- | --- |
| 更新数 / 有效 batch | 200 / 16 | 200 / 16 |
| 样本呈现数 | 3,200 | 3,200 |
| 每题重复次数 | 100 | 50 |
| 训练题正确率 | 32/32（100%） | 64/64（100%） |
| 相同 32 题的验证正确率 | 6/32（18.75%） | 5/32（15.625%） |
| 验证加法 | 2/16 | 3/16 |
| 验证减法 | 4/16 | 2/16 |
| 验证合法且未截断 | 32/32 | 32/32 |
| 终点 last 验证 loss | 0.892183 | 0.699942 |
| 累计目标 token | 16,500 | 16,700 |
| 累计非 padding token | 173,300 | 173,500 |

E2 剩余 27 道验证题都是可解析但数值错误；解析失败、截断与运行错误均为 0。训练循环（含验证与保存）约 43.95 秒，但两组验证频率不同，不把这个耗时与 E1 的 51.76 秒当作等条件速度比较。

E2 的最小验证 loss 为 0.439404，终点 last 为 0.699942；训练器另存的 best 与 last 不同。本实验两组都按预设规则评测 last，没有改用较优的 best 选择生成结果。初始验证 loss 均为 3.111573，与从同一基础模型开始一致。

同题总准确率差值为 **-3.125 个百分点**；5,000 次配对 Bootstrap、种子 42，95% 百分位区间为 **[-12.5, 6.25] 个百分点**，包含零。只有一个训练种子、32 个反复使用的验证问题，不能据此确认负效果，也不能认为数据覆盖率在其他条件下无用。加减子集区间只作探索性诊断，不作多重比较后的确定性结论。

## 6. 逐题变化与覆盖率诊断

以下为全部发生正确性变化的 3 题：

| 表达式 | 标准答案 | E1 输出 | E2 输出 | 变化 |
| --- | --- | --- | --- | --- |
| 3 - 3 | 0 | 0 | -3 | 对 → 错 |
| 5 - 6 | -1 | -1 | -2 | 对 → 错 |
| 0 + 3 | 3 | 9 | 3 | 错 → 对 |

E2 全部 5 道正确验证题为 `0-3`、`0-1`、`0+4`、`0+3`、`0+0`，均涉及零操作数。两组都能拟合各自训练题，但在新问题上没有显示稳定的普通加减规律。

训练标签覆盖的不同答案值由 20 增为 26，新增值为 -4、-2、6、10、13、16；验证题中“标准答案数值未在本轮训练标签出现”的题数由 9 减为 1。覆盖率确实增加，但本次验证正确率没有随之提高，因此不能把上一轮的失败单独归因于答案值缺失。

## 7. 防泄漏审计与边界

本轮 E2 内部的训练与固定验证问题键不重叠；新加的训练题也不包含任何 E1 验证题。两版验证文件逐字节相同，双方预测 ID、设备、精度、生成参数、基础权重及判分代码均已核对；两份完整 manifest 按各自原值审计并在比较结果中保留，没有伪造相同清单。

额外跨数据集审计发现，E2 的训练问题与主数据 `math-data-v1` 存在规范键重合：原 train 11 题、validation 1 题、test_id 2 题、test_template 1 题、test_range/smoke 均为 0。本次没有用主 v1 题目做模型生成，也未据此重新挑选训练题；这些偶然重合不影响 E1/E2 这组自身互斥验证的对照，但**重合的旧 v1 测试题不能再作为 E2 的未见测试**。A1 历史权重和其训练数据未被更改；未来若测试 E2 的跨任务泛化，必须先跨版本去重并固定新的留出规则。

E1 的小型历史备用题有 20 题已转为 E2 训练题，具体 ID 在新数据清单中明确记录，不能混用旧 split 标签。范围、模板和其他备用题未用于本轮生成或模型选择。

## 8. 完成事项与产物

| 完成项 | 证据或位置 |
| --- | --- |
| 固定方案、数据扩充、冻结与审计 | [plan.json](../../results/evidence/2026-10-22-E2-coverage/plan.json)、[数据清单](../../data/manifests/easy-addsub-coverage-v2.json)、[训练数据](../../results/evidence/2026-10-22-E2-coverage/train_data.jsonl)、[验证数据](../../results/evidence/2026-10-22-E2-coverage/validation_data.jsonl) |
| 200 次更新与训练集拟合 | [训练日志](../../results/evidence/2026-10-22-E2-coverage/training/train.jsonl)、[汇总](../../results/evidence/2026-10-22-E2-coverage/training/summary.json)、[训练题输出](../../results/evidence/2026-10-22-E2-coverage/training_generation/predictions.jsonl) |
| E1 复评与 E2 同题验证 | [E1 输出](../../results/evidence/2026-10-22-E2-coverage/reference_validation/predictions.jsonl)、[E2 输出](../../results/evidence/2026-10-22-E2-coverage/validation_generation/predictions.jsonl) |
| 严格同题比较、变化案例和跨数据审计 | [比较结果](../../results/evidence/2026-10-22-E2-coverage/comparison.json)、[诊断汇总](../../results/evidence/2026-10-22-E2-coverage/diagnostics.json) |
| 权重和恢复检查点 | 本地 `artifacts/E2-coverage/`、`artifacts/E2-step200/`；主模型为 last，权重不进入 Git |

报告、代码、数据清单和小型证据保存在本地工作区，尚未形成新 Git 提交或推送。历史实验、原始权重及 E1 数据文件均保留。

## 9. 结论与下一步

固定 200 次更新与 3,200 次样本呈现后，把本份训练集从 32 个问题扩到 64 个，没有观察到验证准确率改善；两组依然是训练集完全拟合、问题留出泛化较弱。终点验证 loss 更低也没有对应到更高的生成准确率。

这一单种子、小样本、已反复使用验证集的探索性结果，不足以证明“更多数据有害”，也不能推广为更大规模、多样化训练无效。尚未执行其他抽样种子、过程监督、LoRA 模块/Rank 对照或新的确认性测试。

下一步优先在这份固定数据和预算上建立**自定义 Q/O 与官方 Q/O 的实际兼容组**，核验实现差异后再只改变 LoRA 注入位置。这样逐项完成原项目的模块对照，避免在数据、监督和模块上同时改动。若未来评价泛化，先建立跨版本无问题键重合的新测试；当前不进入 GRPO。上述下一阶段尚未执行，应另写报告。
