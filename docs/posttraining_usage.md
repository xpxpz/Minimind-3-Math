# 后训练模型的使用与复现

本项目的第一版交付范围为 **MiniMind-3 Dense 的 LoRA 监督微调（SFT）**。完整训练、生成评测和本地导出分别验收；训练 loss 不代表未见题正确率。具体结果以 [实验报告索引](reports/README.md) 和模型卡为准。

## 在本机使用已完成的模型

在 VS Code 打开项目根目录，使用内置终端。发布阶段完成后，适配器位于 `artifacts/release/adapter`，基础模型仍使用本机 `configs/local.json` 中的路径。

```bash
.venv/bin/python -m scripts.use_math_model ask "计算 12 + 7。"
.venv/bin/python -m scripts.use_math_model ask "每盒装12个球，3盒共装多少个球？"
```

脚本会加上训练时相同的最终答案格式提示，关闭思考并贪心生成。答案仍可能出错，请通过精确计算核对。任意聊天、复杂证明、竞赛题不在训练和验收范围内。

## 导出完整 Hugging Face 模型

```bash
.venv/bin/python -m scripts.use_math_model export
```

默认写入 `artifacts/release/merged`；若该目录已存在会拒绝覆盖。导出会检查 LoRA 合并前后单条固定提示的 logits 在 `rtol=1e-4, atol=1e-4` 下接近，并重载磁盘模型检查 logits 完全相等。`export_verification.json` 保存误差与文件哈希。该检查验证导出过程，能力结论仍来自测试集。

导出的目录含 `config.json`、Tokenizer 和 `model.safetensors`，可以用 Transformers 的 `AutoTokenizer.from_pretrained` / `AutoModelForCausalLM.from_pretrained` 加载。调用 `apply_chat_template(..., open_thinking=False)` 并使用相同格式提示；若换提示或生成参数，需要重新评测。适配器 `.pt` 是本项目格式，不能直接冒充 PEFT 格式加载。

## 从干净工作区复现这轮训练

基础模型和依赖身份必须与运行清单一致。下面的命令适用于**尚未存在该轮输出的干净工作区**；当前本机已有工件，直接重复会拒绝覆盖。需要保留历史工件并使用新的 campaign 路径才能另开实验。

```bash
uv sync --extra dev
# 在 configs/local.json 配置本地基础模型；不要覆盖已有配置。
.venv/bin/python -m mathlab.data generate --out data/processed/v1
.venv/bin/python -m mathlab.data validate data/processed/v1
.venv/bin/python -m scripts.posttrain_campaign freeze
.venv/bin/python -m scripts.posttrain_campaign screening
.venv/bin/python -m scripts.posttrain_campaign final
.venv/bin/python -m scripts.summarize_campaign
.venv/bin/python -m scripts.release_campaign
.venv/bin/python -m scripts.use_math_model export
```

六个筛选组为官方 Q/O rank16、自定义 Q/V rank16、Q/K/V/O rank16、Attention+FFN rank8/16/32。筛选阶段每组从相同原始权重训练一轮 8,000 条 / 500 次更新，验证集选模块和 rank；选中配置训练至三轮 24,000 次样本呈现 / 1,500 次更新，额外种子 123、2026 从头训练同样预算。发布种子预先固定为 42，不按测试结果挑种子。

检查点按完整 1,000 题验证的目标 token loss 选 best，包括 step 0；候选组按固定 120 题验证的生成正确数、较少参数量、较低验证 loss、组名依次打破平局。最后固定 ID、模板 OOD、范围 OOD 各 90 题，270 个问题、每题 3 个训练种子，不能称作 810 道独立测试题。

`results/evidence/2026-10-22-posttrain/plan.json` 在筛选前冻结数据、源码、配置、基础权重与评测 ID；`selection.json` 在测试前冻结配置选择。每条子命令的开始 / 完成时间和退出码保存在同目录 `commands/`，stdout/stderr 在 `logs/`，小型运行清单、指标、原始预测在 `runs/`。

## 保存什么到 GitHub

代码、配置、报告和轻量证据可进入 Git。基础权重、适配器、完整合并模型、恢复检查点、虚拟环境保留在 `.gitignore` 指定的本地目录。适配器依赖完全匹配的基础权重和 Tokenizer，校验失败时应修正路径或使用正确版本，不要修改哈希绕过检查。

当前恢复实现会严格检查代码、Git 提交、环境、数据、设备和配置。需要续训时先保留这些身份；提交新代码以后，不要直接修改旧运行清单来强行恢复。实际实验若中途失败，要保留失败日志并写报告，再处理恢复。
