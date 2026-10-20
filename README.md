# MiniMind-3-Math

基于 **MiniMind-3 Dense** 的可复现数学后训练实验。第一阶段研究整数/有理数四则运算、带括号的混合运算及简单应用题，比较官方 LoRA 与可配置 LoRA 的注入位置。这里的模型是 [MiniMind](https://github.com/jingyaogong/minimind)，与 MiniMax 是不同项目。

**当前状态：** 已实现数据、评测、LoRA-SFT、断点恢复和对比工具。正式微调、消融及最终测试结果尚未完成；短运行只验证工程链路，不证明数学能力提升。实际验证记录见 [本地验证](docs/local_validation.md)。

## 先做什么

从已经下载的对话权重开始：**固定数据与 Baseline → 官方 LoRA-SFT → 自定义兼容组 → Q/V、Q/K/V/O、Attention+FFN 对照 → Rank 消融**。SFT 是训练目标，LoRA 是参数微调方式；现在做的是 LoRA-SFT。暂不加入 GRPO，先验证监督训练有效且评测可靠。完整实验顺序见 [实验指南](docs/experiments.md)。

## 本地开始

在 VS Code 打开本仓库，在项目根目录运行。推荐 Python 3.12；`uv.lock` 固定依赖，Mac MPS 默认使用 float32。CUDA 可在配置中显式选择 bfloat16。

```bash
uv sync --extra dev
cp configs/local.example.json configs/local.json
# 编辑 configs/local.json，将 model 改成已有 minimind-3 模型目录的绝对路径
# 本机已有 configs/local.json 时，无需再次复制覆盖

uv run python -m pytest -q
uv run python -m mathlab.data generate --out data/processed/v1
uv run python -m mathlab.data validate data/processed/v1
uv run python -m mathlab.preflight
```

如果数据已生成，直接执行 validate。生成器拒绝覆盖已有数据；更改数据或划分时使用新目录版本。支持的权重输入是包含 `config.json`、Tokenizer 和 `model.safetensors` 的**本地 Transformers Qwen3 格式 MiniMind-3 Dense 目录**；原生 `.pth`、Ollama GGUF、MiniMind-2 及 MoE 不能直接混用。不自动下载权重。

先验证小规模链路（12 道开发题、16 条训练样本、6 次更新）：

```bash
uv run python -m mathlab.evaluate --data data/processed/v1/smoke.jsonl \
  --limit 12 --out results/runs/base-smoke
uv run python -m mathlab.train --config configs/smoke.json \
  --out artifacts/smoke
uv run python -m mathlab.evaluate --data data/processed/v1/smoke.jsonl \
  --adapter artifacts/smoke/best --limit 12 --out results/runs/lora-smoke
uv run python -m mathlab.compare results/runs/base-smoke results/runs/lora-smoke \
  --out results/runs/smoke-comparison.json
```

每次实验使用新的输出目录，防止误覆盖。已存在的训练目录只允许显式 `--resume`；同一配置、模型、数据、代码、环境和设备才可恢复。

## 正式实验入口

先在固定评测集上保存原始模型结果。以下先展示 ID 集；对 `test_template.jsonl` 和 `test_range.jsonl` 使用独立输出目录重复执行。

```bash
uv run python -m mathlab.evaluate --data data/processed/v1/test_id.jsonl \
  --out results/runs/A0-id

uv run python -m mathlab.train --config configs/official_lora.json --out artifacts/A1
uv run python -m mathlab.train --config configs/lora_compat.json --out artifacts/A2
uv run python -m mathlab.train --config configs/lora_qv_r16.json --out artifacts/A3

# 用验证集比较候选配置。测试集只在选择结束后用于最终比较。
uv run python -m mathlab.evaluate --data data/processed/v1/validation.jsonl \
  --adapter artifacts/A3/best --out results/runs/A3-validation

# 完成选择后，使用相同协议运行最终测试
uv run python -m mathlab.evaluate --data data/processed/v1/test_id.jsonl \
  --adapter artifacts/A3/best --out results/runs/A3-id
uv run python -m mathlab.compare results/runs/A0-id results/runs/A3-id \
  --out results/runs/A3-vs-A0-id.json
```

`configs/lora_qkvo_r16.json` 和 `configs/lora_attention_ffn_r16.json` 用于后续模块消融。Rank 消融复制配置，将 rank 和 alpha 同时设为 4、8、16、32，保持 alpha/rank=1。配置加载器拒绝未知字段，避免拼错参数仍悄悄运行。

## 已实现的约束

- 8,000 训练、1,000 验证、1,000 ID 测试、各 300 模板/范围 OOD、240 开发烟雾题；所有集合全局问题键去重，冻结 JSONL 哈希与生成代码哈希。
- 三类任务，精确 `Fraction` 运算；白名单 AST 求值，不执行模型输出。统一 `答案：数值` 最终答案格式。
- 贪心、关闭思考、batch=1、保存逐题原始输出；解析失败、截断、运行错误均计入分母。对齐模型、数据、判分代码与解码协议后才能对比。
- 官方注入函数原样保留；自定义 LoRA 支持非方阵、目标模块、Rank/Alpha/Dropout、冻结、严格加载及合并。每次输出实际匹配和排除清单。
- 只监督 assistant 回答及结束标记；不监督提示词、空思考前缀、padding。按目标 token 数加权梯度累积，正确处理末次不足一个有效 batch 的更新。
- 保存适配器、基础权重和 Tokenizer 哈希、配置、数据版本、源码哈希、环境及训练进度。支持在优化器更新边界恢复，包含优化器及 CPU/设备随机状态。

## 文件入口

| 文件 | 用途 |
| --- | --- |
| `mathlab/data.py` | 生成、划分、审计、冻结及上游 conversations 格式导出 |
| `mathlab/answers.py` | 精确运算、规范问题键和最终答案解析 |
| `mathlab/preflight.py` | 权重、模板、Token 长度和 LoRA 注入检查 |
| `mathlab/train.py` / `mathlab/sft.py` | LoRA-SFT、assistant mask、保存与恢复 |
| `mathlab/lora.py` | 可配置非方阵 LoRA、官方对照、加载及合并 |
| `mathlab/evaluate.py` / `mathlab/compare.py` | 逐题评测和配对 Bootstrap |
| `docs/` | 数据卡、评测协议、实验指南、贡献和实测记录 |

原始规划保留为 [项目规划参考](docs/project_plan.md)，其中后续阶段尚未实现的内容不代表本仓库已经支持。大权重和完整预测默认不进入 Git；保留结果清单并按需归档。上游代码、权重及依赖来源见 [第三方说明](THIRD_PARTY_NOTICES.md)。

首次使用 Git 和 VS Code 源代码管理，可参考 [Git 操作入门](docs/git_quickstart.md)。
