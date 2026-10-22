# 本地验证记录：2026-10-20

本记录区分工程验证与能力实验。代码已跑通，**尚未验证正式数学能力提升**。全部结果来自用户已下载的 MiniMind-3；未下载或替换基础权重。

## 本阶段写了什么代码、完成了什么

以下为首版工程阶段的回顾，不是后续 A1 试验再次新增了这些代码。基础模型结构来自 MiniMind/Qwen3，官方 LoRA 源码保留上游来源；新增实现由 Codex 协助编写。

| 文件 | 类型 | 具体实现与对应验证 |
| --- | --- | --- |
| [mathlab/data.py](../mathlab/data.py)、[io.py](../mathlab/io.py) | 新增 | 数学题生成、分组去重与划分、SFT 导出和冻结清单；生成 10,840 题并检查可重复性、泄漏与篡改检测 |
| [mathlab/answers.py](../mathlab/answers.py) | 新增 | 受限 AST 精确求值、规范问题键与最终答案解析；验证整数、分数、非法表达式及含糊输出 |
| [mathlab/lora.py](../mathlab/lora.py) | 新增 | 非方阵 LoRA、配置注入、冻结、加载与合并；验证形状、初始等价、梯度、保存加载、兼容组和合并数值一致 |
| [vendor/minimind/model_lora.py](../vendor/minimind/model_lora.py) | 上游原样引用 | 官方方阵注入对照，保留许可证并核对源码一致 |
| [mathlab/runtime.py](../mathlab/runtime.py)、[preflight.py](../mathlab/preflight.py) | 新增 | 本地权重加载、身份/环境记录、设备选择、适配器管理与预检；实际加载 MiniMind-3 并检查参数与模板 |
| [mathlab/sft.py](../mathlab/sft.py)、[train.py](../mathlab/train.py) | 新增 | assistant mask、目标 token 加权累积、验证、保存与恢复；测试累积梯度和 CPU 恢复轨迹，并运行 MPS 短训练 |
| [mathlab/evaluate.py](../mathlab/evaluate.py)、[compare.py](../mathlab/compare.py) | 新增 | 固定协议生成、逐题保存、错误分母统计与配对比较；离线测试及真实权重开发评测 |
| [tests](../tests)、[configs](../configs)、[pyproject.toml](../pyproject.toml)、[uv.lock](../uv.lock) | 新增 | 必要测试、可执行配置及依赖定义/锁定；最终 38 项测试通过 |
| [README](../README.md)、[docs](.)、[来源说明](../THIRD_PARTY_NOTICES.md) | 新增/整理文档 | 运行说明、数据与评测协议、贡献边界、原始规划参考及实测记录 |

本阶段已完成：首版工程、依赖安装、数据生成/审计、预检、单元测试、官方/自定义 LoRA 短训练、保存/恢复和开发题对比。产物包括 [冻结数据清单](../data/manifests/v1.json)、[首版证据](../results/evidence/2026-10-20) 及本地忽略目录下的权重/检查点。正式完整训练和消融仍未完成。首版代码已以 `170d095` 提交并推送；本报告的后续补充需要单独提交，不能视为已同步。

## 环境与输入

- Apple M4、16 GB 内存、MPS、float32；Python 3.12.14、PyTorch 2.6.0、Transformers 4.57.6。
- 本地基础模型为 Qwen3 兼容 Dense MiniMind-3，实际统计 **63,912,192 参数**。权重 SHA256：`3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`。
- 本地上游与官方仓库固定提交均为 `f659b55761b754d306bd140573493a6543cafd7f`。官方 LoRA 源码复制后逐字比较一致。
- 已生成 10,840 道互不重复的底层表达式问题，划分文件哈希见 [冻结清单](../data/manifests/v1.json)。最长训练样本 74 token，配置上限 256；assistant 监督文本检查为 `答案：56<|im_end|>\n`，不含题目和空思考前缀。
- Q/V rank16 实际可训练参数 344,064；Q 为 768→768，V 为 768→384。

## 已运行的检查

`uv sync --locked --extra dev` 成功；`python -m pytest -q`：**38 passed**。

覆盖答案和非法表达式、分组泄漏、数据可重复生成与篡改检测、非方阵 LoRA、零初始化等价、基础冻结、A/B 梯度、严格保存加载、合并前后数值一致、官方兼容组、assistant mask/padding、按 token 加权累积、CPU tiny Qwen3 中断恢复逐位一致、离线评测和比较协议拒绝错配。

真实权重也运行了两个训练短实验：

| 实验 | 配置 | 更新数 | 可训练参数 | 初始验证 loss | 最佳验证 loss |
| --- | --- | --- | --- | --- | --- |
| 官方 LoRA 方阵匹配 | official_smoke.json | 2 | 98,304 | 3.145605 | 3.064031 |
| 自定义 Q/V | smoke.json | 6 | 86,016 | 3.145605 | 2.840610 |

各实验只使用 16 条训练题和 12 条验证题，rank=4。两者更新预算不同，**不能据此比较方案优劣**。自定义训练先执行 2 次更新并保存，再通过 `--resume` 接续至第 6 次，实际验证 MPS 保存/恢复入口可用。CPU 单元测试另外验证了连续/恢复轨迹完全相同。

```bash
uv run python -m mathlab.train --config configs/official_smoke.json \
  --out artifacts/official-smoke
uv run python -m mathlab.train --config configs/smoke.json \
  --out artifacts/smoke-verified --stop-after-updates 2
uv run python -m mathlab.train --config configs/smoke.json \
  --out artifacts/smoke-verified --resume
```

上述目录已经存在；复跑时使用新目录。初始验证时本地仓库还没有代码提交，运行清单记录了这一事实，并保存源码文件哈希。之后已连接 GitHub 初始 README 历史，并创建本地分支 `codex/math-posttraining`；在代码/提交/环境变化后恢复旧训练会被严格拒绝，应新建运行。

## 开发评测结果与限制

最初 64-token、文件前 12 题试跑全被截断，仅用于发现生成预算和小样本题型偏斜，未用于正式结论。正式训练开始前把开发默认预算改为 256，有限子集改为按题型轮流选取。下面是在相同 12 道 smoke 题（每类 4 道）、256-token、关闭思考、贪心协议下，重载已保存适配器的结果。

| 模型 | N | 严格最终答案准确率 | 解析失败 | 截断 | 运行错误 |
| --- | --- | --- | --- | --- | --- |
| 原始模型 | 12 | 0/12 | 6/12 | 6/12 | 0/12 |
| Q/V 6 次更新 best | 12 | 0/12 | 8/12 | 4/12 | 0/12 |

此处零分是冻结格式和生成预算下的端到端结果，不能解释为“所有数学题都不会做”。人工查看输出同时发现格式未遵守、长篇重复和数值错误，例如将 `51×86` 回答为 432。6 次更新不足以证明任务适配有效，验证 loss 下降也没有带来可确认的生成准确率提升。

配对比较已运行，准确率差值 0 个百分点；12 题且全错的 Bootstrap 区间没有证明方案等效的意义。正式测试集尚未进行模型生成评测，没有据其结果调参。

可审查证据随仓库保留在 [results/evidence/2026-10-20](../results/evidence/2026-10-20)：原始输出、指标、比较结果、训练日志、环境、数据与源码身份。大权重、优化器恢复状态和完整工作目录位于被忽略的 artifacts/results/runs；不要把这份短运行当作正式实验报告或简历提升数据。

## 接下来的执行顺序

1. 用 `official_lora.json` 跑 50 次更新后暂停，查看验证 loss、输出格式、训练时间和资源；如不学习，优先检查数据与简化题型，而不是立即叠加算法。
2. 在开发题/验证集上确定生成预算、格式协议和正式训练预算。当前 256-token 下仍有截断，尚不应把这套初始设置称为完成验证的最优评测方案；如调整协议，原始模型需使用相同协议重新评测。
3. 冻结协议后跑完整原始模型 Baseline，完成 A1 官方 LoRA 与 A2 兼容组，再做 Q/V、Q/K/V/O、Attention+FFN 和 Rank 消融。控制变量并保留失败结果。
4. 最终测试和真实提升结论完成后，再判断是否有 GRPO 组内奖励信号。

短预算入口：

```bash
uv run python -m mathlab.train --config configs/official_lora.json \
  --out artifacts/A1-pilot --stop-after-updates 50
uv run python -m mathlab.evaluate --data data/processed/v1/validation.jsonl \
  --adapter artifacts/A1-pilot/best --limit 24 --out results/runs/A1-pilot-validation
```

协议/配置不变时，可以用相同训练命令添加 `--resume` 继续。正式命令与实验控制说明见 [实验指南](experiments.md)。

上述 50 次更新试验现已执行完成，结果及原始输出见 [A1 后续试验记录](A1_pilot_2026-10-20.md)。本文件中的 2/6 次更新数据保留为较早的工程验证记录。
