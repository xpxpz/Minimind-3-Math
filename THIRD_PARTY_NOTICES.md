# Third-party sources

- MiniMind: https://github.com/jingyaogong/minimind, commit `f659b55761b754d306bd140573493a6543cafd7f`, jingyaogong and contributors, Apache-2.0. `vendor/minimind/model_lora.py` is an unmodified copy of `model/model_lora.py`; its license is included at `vendor/minimind/LICENSE`. The repository root LICENSE uses Apache-2.0 for the new project code.
- Transformers: https://github.com/huggingface/transformers, Apache-2.0. Used as an installed dependency, including Qwen3 model implementation.
- PyTorch: https://github.com/pytorch/pytorch, BSD-style license. Used as an installed dependency.
- LoRA: Hu et al., https://arxiv.org/abs/2106.09685. Standard method; project contribution is implementation/configuration and experimental evaluation.
- MiniMind-3 model distribution: https://huggingface.co/jingyaogong/minimind-3 and https://modelscope.cn/models/gongjy/minimind-3. User-provided weights/tokenizer are not redistributed in this repository. Consult the chosen model distribution for its applicable terms; code licensing does not automatically establish weight/data licensing.
- Generated math dataset: deterministic synthetic templates and exact programmatic labels in this project; no external benchmark data included.

The full upstream training repository is not vendored. Official LoRA control uses the upstream injection function in this project's common training harness, rather than claiming to reproduce every behavior of upstream train_lora.py.
