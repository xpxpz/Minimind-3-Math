# 实验执行指南

先建立可复查的 SFT/LoRA 闭环，再考虑强化学习。Tiny 模型测试和短训练只能证明实现能运行；能力是否提高，需要固定测试和足够训练预算回答。

每次实际执行实验步骤后，都要按 [报告模板](reports/_template.md) 写独立 Markdown 报告，并更新 [报告索引](reports/README.md)。报告必须说明具体操作、代码/文件改动、实际完成项、执行命令、验证、产物与局限；未修改代码也要明确说明复用了哪些入口。失败或未完成的实验同样记录。

## 1. 固定输入

运行 `mathlab.preflight`。核对基础参数量、模型/Tokenizer 哈希、最长训练样本、实际注入层和 supervised_text。当前版本只接受本地 Transformers Qwen3 格式 Dense MiniMind-3，以 `trust_remote_code=False` 加载，不执行模型目录中的任意 Python 源码。

`configs/local.json` 是本机路径配置，默认被 Git 忽略。也可以设置 `MINIMIND_MODEL`，或在每条命令添加 `--model /absolute/path`。模型原目录只读，训练写到新建的 `artifacts/<run>`。

数据默认由种子 42 生成。`data/processed/v1/manifest.json` 固定全部划分的文件哈希。源码改变后若题目或协议改变，应发布 v2；保持旧版本结果可解释。

## 2. 最小工程验证

运行 README 的开发题 Baseline、smoke 训练和适配器重载评测。检查：

- preflight 中答案未被截断、用户题目被 mask；实际最大长度应小于 max_length。
- `train.jsonl` 中 loss、gradient_norm 有限，适配器更新；单元测试覆盖零增量等价、基础冻结、A/B 梯度和非方阵。
- 训练目录包含 `best/adapter.json`、`last/adapter.pt`、`checkpoint.pt`、`summary.json`。best 按验证集目标 token loss 选择，**也允许选择 step 0**；若训练后未改善，best 可能仍是原始输出。
- `predictions.jsonl` 保留每道题，包括失败。检查解析失败和截断率，人工查看 raw_output。

测试包含 CPU tiny Qwen3 的连续训练/中断恢复逐位一致验证，以及按 token 加权累积与合批梯度一致验证。MPS/CUDA 的跨硬件或跨软件版本逐位一致不作保证，正式恢复必须保持设备和环境。

```bash
# 在完整计划中先跑两次更新后保存退出；检查性能和预计预算
uv run python -m mathlab.train --config configs/official_lora.json \
  --out artifacts/A1 --stop-after-updates 2
# 原配置继续，不会从头重复样本
uv run python -m mathlab.train --config configs/official_lora.json \
  --out artifacts/A1 --resume
```

`--stop-after-updates` 是本次调用额外运行的更新数；配置内 `max_updates` 是整个实验的更新上限。只有保存边界可以恢复；突然中断会回到最近一个 checkpoint。训练日志可能含该 checkpoint 之后已写入的记录，恢复时会按 checkpoint 的 updates 截去这些记录。

## 3. 完整对照顺序

| ID | 配置 | 目的 |
| --- | --- | --- |
| A0 | 无适配器，evaluate | 固定原始模型 Baseline |
| A1 | official_lora.json | 原样调用上游方阵自动匹配 LoRA |
| A2 | lora_compat.json | 相同注入清单、初始化、缩放及数据顺序验证兼容 |
| A3 | lora_qv_r16.json | 加入非方阵 V 投影，研究 Q/V |
| A4 | lora_qkvo_r16.json | 完整 Attention 投影 |
| A5 | lora_attention_ffn_r16.json | Attention + FFN |

A1 使用官方注入函数，但训练器是本项目的统一训练器；不是声称原封不动运行了整个上游 train_lora.py。这样所有方案共用相同 mask、token 加权、验证和保存协议。A2 的 alpha/rank=1、dropout=0，且用同一随机初始化顺序。官方逻辑实际匹配哪些层，以每次 run.json 为准；当前基础模型中通常为 Q/O。

各配置固定有效 batch=16、学习率 1e-4、3 epoch、256 长度、恒定学习率、AdamW weight_decay=0.01。默认总更新数约为 3×ceil(8000/16)=1500。末批保留。先测量少量更新，不保证这是最优超参数。正式对照必须统一设备、dtype 和实际训练预算。

先单种子筛选，再对关键组使用 3 个训练种子，分别保存目录。模块相同 Rank 时可训练参数量不同；必须报告参数量和资源差异，不直接得出“更高效”的结论。Rank 消融同时调整 alpha=rank，避免改变缩放；Dropout 等变量单独比较。

best 默认按验证 loss 选 epoch/保存点。用 validation 生成准确率进一步比较运行方案；需在最终测试之前确定选择规则。训练器本身不在训练中计算生成准确率。`scripts/posttrain_campaign.py` 在外层编排预先固定的六组候选与验证生成，按冻结规则选配置；实际选择见 [E4报告](reports/2026-10-22_E4-module-rank-screening.md)，复现和交付入口见 [使用文档](posttraining_usage.md)。

## 4. 读结果

`run.json` 保存输入身份、源码哈希、环境和注入清单。`summary.json` 保存更新数、目标/非 padding token 数、训练与验证总耗时及 CUDA allocated/reserved 峰值（MPS/CPU 为 null，不能伪写显存）。此处吞吐明确包含验证开销，不等同纯训练算子吞吐。中断后恢复的墙钟时间累计各段已归档训练时间。

评测在预热后逐题计时，包含模板、生成、解码和设备同步。保存延迟 P50/P95 和总体生成 token/s；不混用合并/未合并形态的速度结论。评测 CLI 使用未合并适配器；`scripts/use_math_model.py export` 提供合并导出，核对单条固定提示的合并前后logits与磁盘重载一致性。导出检查不代替能力测试，也没有测量合并模型的整体吞吐收益。

同一测试集上的两组输出交给 `mathlab.compare`；它核对基础模型、数据、题目 ID、解码、dtype、设备及判分器代码。配对 Bootstrap 区间表示题目抽样不确定性，不能代替不同训练种子的波动。正式报告要列 N、分题型准确率、ID/模板 OOD/范围 OOD，以及失败率。

除主指标外，可用以下命令诊断运算类型，避免总准确率掩盖“单步除法直接写成分数”的情况。分析脚本只读冻结数据和预测，不重新生成、不改变判分规则；应在生成评测全部完成后运行。

```bash
python -m scripts.analyze_predictions results/runs/A1-step50-validation-full \
  --data data/processed/v1/validation.jsonl \
  --out results/runs/A1-step50-validation-full/analysis.json
```

## 5. 什么时候考虑 GRPO

先完成 SFT 关键对照、保存可重复的正/负结果；然后检查同题多次采样是否出现正确/错误混合组。若几乎全错，组内奖励没有差异，先降低难度或补充监督数据。

本版本未实现 GRPO、全参数 SFT、教师蒸馏或工具调用训练。后续加入 GRPO 时需单独固定 rollout、答案奖励、KL、组大小、零方差组比例及额外计算预算，和其起始 SFT 模型比较。不能把短训练的 loss 下降当作强化学习启动证据。
