# 2026-10-22：训练配置与资源需求审阅

## 1. 目标与实际范围

回答当前后训练是否需要租服务器、配置字段含义、模型参数与内存需求。本轮为只读配置/历史证据审阅和文档整理，不是新训练实验；没有启动训练、生成评测、测试或内存基准。

## 2. 输入与身份

- 当前设备：系统信息显示 Apple M4、10 核 CPU、10 核 GPU、16 GB 统一内存。
- 权重目录：`/Users/xiexianpei/minimind/minimind-3`，由本地 `configs/local.json` 指定。
- 实际读取该目录的 `config.json` 和 `model.safetensors` 文件头。文件头统计 63,912,192 个存储参数，全部 F16；文件共 127,834,168 字节。
- 历史权重 SHA256：`3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`，来自 [预检清单](../../results/evidence/2026-10-22-posttrain/preflight.json)。本轮没有重新计算完整权重哈希。
- 历史实验代码身份：`170d095c23da50d5e91a8790f16ca912509f683f` 加未提交工作区；本轮 git status 同样显示工作区有未提交改动。
- 历史环境：Python 3.12.14、PyTorch 2.6.0、Transformers 4.57.6；MPS、float32。来源同上，不代表本轮重新运行训练。
- 数据为 math-data-v1，[清单](../../data/manifests/v1.json)；训练 8,000、验证 1,000。数据/源码/基础模型的历史哈希见 [冻结计划](../../results/evidence/2026-10-22-posttrain/plan.json)。

## 3. 实际步骤与命令

在仓库根目录执行 `rg --files`、`rg -n`、`cat` 和 `git status --short`，读取训练器、运行时、LoRA、collate、配置和历史报告。执行 `system_profiler SPHardwareDataType SPDisplaysDataType` 查看硬件；不在本报告保存设备序列号等标识。

执行一次 `.venv/bin/python - <<'PY'` 的标准库分析：用 `struct.unpack('<Q', ...)` 读取 safetensors 文件头长度，用 JSON 解析各张量 shape，以 `sum(math.prod(shape))` 统计存储参数；读取既有 summary/run JSON，计算 FP32/BF16 基础权重大小。此临时脚本未保存为仓库代码，也未把权重张量载入 GPU。

查看 PyTorch 官方 MPS 内存 API 文档，确认张量分配量和 Metal 驱动分配量的统计口径不同。未调用这些 API 对已有其他进程进行测量。

## 4. 模型和训练配置

模型为 MiniMind-3 Dense，使用 Transformers 的 Qwen3ForCausalLM 格式。8 层、hidden_size 768、FFN intermediate_size 2432、8 个 query heads、4 个 KV heads、head_dim 96、词表 6400、共享输入/输出 embedding。配置最大位置数 32768 是模型字段，不表示本项目已经验证该长度的数学训练或生成能力。

当前阶段为 LoRA-SFT：冻结基础参数，只训练注入的 LoRA 分支；只监督 assistant 答案和结束标记。不是全参数微调，也没有执行 GRPO。

| 字段 | 实际值与含义 |
| --- | --- |
| 配置来源 | [official_r16](../../configs/campaign/official_r16.json) 与其余 campaign JSON；遗漏字段由 [train.py](../../mathlab/train.py) DEFAULTS 补齐 |
| rank / alpha / dropout | 官方组 16 / 16 / 0；rank 决定分支容量与参数量，缩放 alpha/rank=1 |
| seed | 42；最终阶段计划额外使用 123、2026，不代表已完成 |
| epochs | 3；8000 条数据完整遍历三次，共 24000 次样本呈现 |
| batch_size / accumulation_steps | 4 / 4；每次前后向 4 条，累积 4 次才更新，有效 batch 16 |
| learning_rate | 0.0001；固定学习率，无 scheduler / warmup |
| optimizer / weight_decay | AdamW / 0.01 |
| grad_clip | 1.0，梯度范数裁剪阈值 |
| max_length | 256；题目、模板和答案的总 token 上限，超长报错；按 batch 内最长样本动态补齐 |
| dtype | 默认 float32；CPU/MPS 仅接受 float32，CUDA 可显式改为 bfloat16 |
| max_updates / save_every | 1500 / 100；每100步验证并保存，轮末/停止时也会执行 |
| train_limit / validation_limit | 默认 null，使用全部 8000 / 1000 条 |
| 筛选实际预算 | 编排命令额外传 `--stop-after-updates 500`，所以每组先跑一轮；JSON 的3轮不代表筛选已跑3轮 |

注入范围和参数量：

| 组 | 可训练参数 | 证据口径 |
| --- | ---: | --- |
| 官方 Q/O rank16 | 393216 | 历史实际运行清单 |
| 自定义 Q/V rank16 | 344064 | 历史实际运行清单 |
| 自定义 Q/K/V/O rank16 | 688128 | 历史实际运行清单 |
| Attention+FFN rank8 | 958464 | 根据8层投影形状及 r×(输入维度+输出维度) 计算 |
| Attention+FFN rank16 | 1916928 | 同上，为结构计算值 |
| Attention+FFN rank32 | 3833856 | 同上，为结构计算值 |

官方算法匹配方阵线性层；本模型实际对应 Q/O。Attention+FFN 为 Q/K/V/O 加 gate/up/down。

## 5. 历史训练耗时与结果边界

| 组 | 更新数 | 历史训练循环含验证/保存 | 整条训练命令墙钟 |
| --- | ---: | ---: | ---: |
| 官方 rank16 | 500 | 152.84 秒 | 约238秒 |
| Q/V rank16 | 500 | 153.51 秒 | 约158秒 |
| Q/K/V/O rank16 | 500 | 169.62 秒 | 约174秒 |

来源：[screening.json](../../results/evidence/2026-10-22-posttrain/screening.json)、[官方 summary](../../results/evidence/2026-10-22-posttrain/runs/screen-train-official_r16/summary.json)、[命令记录目录](../../results/evidence/2026-10-22-posttrain/commands)。训练循环计时不包含全部权重加载、哈希和数据准备；生成评测另计，不能把上述数字当成完整项目耗时。

预检的训练样本平均63.02525 token、最大74 token，所以这些耗时不能推广到每条都长256 token的训练。三个注意力组1500步训练循环粗略外推约8～9分钟；这是估算，不是三轮完成记录。六组筛选、三种子训练和所有生成评测的总耗时未在本轮测量。

历史三个组均为23/120验证正确；这不是本轮评测，也不能据此确认数学能力稳定提升。当前E4最终状态以其自身报告/实时产物为准。

## 6. 内存需求：可计算部分与未测部分

- 基础权重 FP32：63,912,192 × 4 字节 = 243.81 MiB；BF16/FP16 为121.90 MiB。磁盘存储是F16，但当前加载器转为FP32，不能按磁盘文件大小冒充训练总占用。
- 官方rank16分支：参数、梯度、AdamW两份状态按FP32合计约6 MiB；Attention+FFN rank32同口径约58.5 MiB。未计临时张量和优化器实现额外分配。
- 训练还需要各层激活、反向中间值、logits、注意力矩阵、框架/Metal缓存以及进程CPU内存。冻结基础参数不意味着无需保存反向所需激活。
- 当前代码使用 eager attention，没有量化、激活检查点或FlashAttention设置。长度和microbatch增大会提高占用；梯度累积不等同于一次并行处理16条。
- **精确训练峰值未知**：训练器仅保存CUDA峰值，本机MPS对应字段为null；null不是0，也无法从完成日志反推精确峰值。
- 对当前小模型、batch4、短序列LoRA，如需提前给CUDA设备留预算，可先按8 GB显存量级规划。这是带余量的工程预算，不是实测最小需求或对所有长度的保证；无需据此购买/租用设备。M4的16 GB统一内存由CPU/GPU/系统共享，不等于16 GB独立显存。
- 后续若要求精确数值，需要另开内存测量实验，并保存采样轨迹；张量量和驱动量分别参考 [current_allocated_memory](https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory) 与 [driver_allocated_memory](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory)。瞬时采样本身也不必然捕获真正峰值。

## 7. 本轮改动与产物

本轮未新增或修改训练/评测 Python 代码。静态审阅复用 [train.py](../../mathlab/train.py)、[runtime.py](../../mathlab/runtime.py)、[lora.py](../../mathlab/lora.py)、[sft.py](../../mathlab/sft.py)；官方来源为既有 [上游LoRA](../../vendor/minimind/model_lora.py)，本轮未修改。

本轮新增本报告，并在 [报告索引](README.md) 追加入口。未产生新权重、适配器、预测或训练指标；没有提交或推送。本轮检查数字的计算及报告相对链接，不重新执行历史测试。

## 8. 结论与下一步

当前仓库的63.9M MiniMind-3 LoRA-SFT不需要租服务器：本机16 GB M4已经实际跑完多个500步主数据组。可以继续本地完成配置对照和当前规模的最终训练。

更大模型、长推理轨迹、大量种子/数据或未来GRPO可能改变资源需求，需要按届时实现重新估算。当前关键问题仍包括未见题准确率低；租服务器不能替代数据、训练目标和泛化问题的诊断。精确MPS峰值及CUDA性能是未测项，本轮没有执行这些后续实验。
