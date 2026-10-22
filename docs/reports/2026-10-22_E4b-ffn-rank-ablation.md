# 2026-10-22：E4b Attention+FFN Rank 消融

## 1. 目标与实际范围

在模块保持 Attention+FFN 的条件下，比较 rank8/16/32。状态：完成。三组就是 [E4](2026-10-22_E4-module-rank-screening.md) 同批运行的三个 FFN 组，本报告单独解释 rank 这一因素，**没有额外再训练三次，也不能把它们重复计入总训练量**。原计划每组500更新，实际均完成500更新与120题验证生成。

## 2. 输入、配置与身份

- MiniMind-3 Dense 原始权重，基础参数63,912,192，SHA256 `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`。Tokenizer、软件、源码、数据哈希见 [预注册方案](../../results/evidence/2026-10-22-posttrain/plan.json)。Git `170d095c23da50d5e91a8790f16ca912509f683f` 加未提交工作区，上游LoRA来自 `f659b55761b754d306bd140573493a6543cafd7f`。
- 复用 [math-data-v1](../../data/manifests/v1.json)：训练8,000、完整loss验证1,000；本次生成验证120题，每类40。未使用最终测试选择rank。
- [r8](../../configs/campaign/ffn_r8.json)、[r16](../../configs/campaign/ffn_r16.json)、[r32](../../configs/campaign/ffn_r32.json)：模块同为 Q/K/V/O + gate/up/down，实际每组56个投影；仅rank和alpha一起变动，保持alpha/rank=1。其他配置相同：seed42、dropout0、float32/MPS、batch4×accum4=16、lr1e-4、AdamW weight_decay .01、clip1、长度256、恒定学习率。
- 每组一轮、500更新、8,000次样本呈现、54,145目标token、504,202非padding token。每组从相同基础权重独立初始化，没有相互续训。
- 每组按完整验证loss选择best；生成统一贪心、关闭思考、batch1、256token、math-answer-v1精确数值判分，所有失败进入分母。

## 3. 实际步骤与执行命令

编排顺序为 r8→r16→r32，每组训练后立即生成同一固定验证子集、核对预测并保存证据。实际入口为：

```bash
.venv/bin/python -m scripts.posttrain_campaign screening
```

该入口还运行了其余模块对照；本报告只分析三组FFN。精确子命令及返回码见 [r8训练](../../results/evidence/2026-10-22-posttrain/commands/screen-train-ffn_r8.json)、[r16训练](../../results/evidence/2026-10-22-posttrain/commands/screen-train-ffn_r16.json)、[r32训练](../../results/evidence/2026-10-22-posttrain/commands/screen-train-ffn_r32.json)，生成命令在相同目录对应 `screen-eval-ffn_r*.json`。工作目录为项目根目录；已有输出不能覆盖。

## 4. 代码 / 文件改动

本轮未新增或修改训练/评测 Python 代码。本子阶段复用本工作轮已新增的 `scripts/posttrain_campaign.py` 和既有 `mathlab/train.py`、`lora.py`、`evaluate.py`、`answers.py`、`scripts/analyze_predictions.py`。变化是三个独立配置；本报告是新增的rank阶段记录。官方引用仅作为E4控制来源，FFN三组均用自定义非方阵实现。

## 5. 验证与结果

三组实际原始预测、清单、训练日志和诊断分别在 [r8](../../results/evidence/2026-10-22-posttrain/runs/screen-eval-ffn_r8)、[r16](../../results/evidence/2026-10-22-posttrain/runs/screen-eval-ffn_r16)、[r32](../../results/evidence/2026-10-22-posttrain/runs/screen-eval-ffn_r32)；训练归档同目录前缀 `screen-train-`。预算、有限loss/梯度、题目ID和本地适配器哈希通过 [E4实际审计](../../results/evidence/2026-10-22-posttrain/screening_verification.json)。没有为本子报告再次重跑测试，不能重复计数E3/E4的55项。

| Rank | 参数 | 最佳验证loss | 正确/120 | 无除法/88 | 单步除法/22 | 含验证/保存秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 958,464 | 0.902975 | 23 | 1 | 22 | 191.75 |
| 16 | 1,916,928 | 0.888585 | 23 | 1 | 22 | 191.13 |
| 32 | 3,833,856 | 0.864190 | 21 | 0 | 21 | 189.46 |

所有组120/120格式有效、无截断 / 运行错误；正确的单步除法均直接输出输入分子/分母。r8/r16准确率19.167%，r32为17.5%。示例按各分析文件中的固定顺序首个正确 / 错误案例提供，未经人工筛选成功样例。

## 6. 产物、完成情况与局限

完成：三个固定预算训练、360条验证预测、rank比较和哈希审计。适配器与恢复状态位于本地 `artifacts/SFT-ffn_r8`、`artifacts/SFT-ffn_r16`、`artifacts/SFT-ffn_r32`，SHA256记录在各训练归档的best/last_adapter.json；没有上传权重。本子阶段没有Git提交 / 推送。

参数量变为四倍时loss下降，但本次生成正确数没有提升，不能据此推荐更大rank；也不能从一次种子、120道复用验证题断言r32普遍更差。时间是单次顺序MPS观测，包含验证 / 保存，并非纯算子性能或稳定效率排名。不同rank的初始化随机抽样形状不同；本实验不是同一初始低秩子空间下的嵌套模型比较。

## 7. 下一步

E4统一选择在所有模块 / rank中取参数最少的并列最高分组Q/V r16；随后E5完成这个配置的三轮三种子与独立测试。本rank消融没有执行多种子、长预算或更广rank搜索，这些不能写成已完成。
