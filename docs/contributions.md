# 来源与贡献

| 内容 | 来源 | 本仓库处理 |
| --- | --- | --- |
| MiniMind-3 结构、已训练权重和 Tokenizer | jingyaogong/minimind 与其模型发布 | 使用本地 HF Qwen3 格式权重；未重新实现模型主干 |
| 官方 LoRA 注入 | 上游 model/model_lora.py，固定 f659b55761b754d306bd140573493a6543cafd7f | 原样保留在 vendor/minimind/model_lora.py，运行 official 控制路径 |
| Qwen3 模型运行与 Attention | Hugging Face Transformers | 使用官方库加载，固定版本，不复制其实现 |
| 数学数据、划分、规范键和清单 | 本项目新增，Codex 协助编写 | 见 mathlab/data.py、answers.py 和测试 |
| 非方阵 LoRA、模块选择、元数据与合并 | 本项目新增，标准 LoRA 方法扩展 | 见 mathlab/lora.py；标准算法，不宣称方法原创 |
| 统一训练、mask、累积、恢复 | 本项目新增，Codex 协助编写 | 见 mathlab/train.py、sft.py |
| 评测与配对比较 | 本项目新增，Codex 协助编写 | 见 mathlab/evaluate.py、compare.py |
| 个人规划参考 | 用户提供的 2026-10-16 规划 | 原文保留在 project_plan.md，不把计划当作已完成证据 |

上游核心模型和官方 LoRA 并非个人从零原创。实验尚未完成；简历、README 和报告只能列出真实运行结果。新增代码需要由项目作者理解、复核，并在实际研究迭代中维护。
