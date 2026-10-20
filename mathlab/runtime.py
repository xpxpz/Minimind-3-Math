"""Load explicitly identified local Qwen3-format MiniMind-3 weights."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import sha256, write_json
from .lora import adapter_state, inject, load_adapter_state

UPSTREAM = "f659b55761b754d306bd140573493a6543cafd7f"


def resolve_model(value=None):
    if value is None:
        value = os.environ.get("MINIMIND_MODEL")
    if value is None and Path("configs/local.json").exists():
        value = json.loads(Path("configs/local.json").read_text())["model"]
    if value is None:
        raise ValueError("set --model, MINIMIND_MODEL, or configs/local.json to a local HF model directory")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"local model directory does not exist: {path}")
    return path


def model_identity(path):
    # Includes tokenizer/template and generation config; excludes unrelated images/readme.
    patterns = ("config.json", "generation_config.json", "*token*.json", "chat_template*", "*.model",
                "model*.safetensors*", "pytorch_model*.bin*")
    files = sorted({p for pattern in patterns for p in path.glob(pattern) if p.is_file()})
    return {p.name: sha256(p) for p in files}


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def environment():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        revision = "uncommitted-initial-repository"
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
    return {"python": platform.python_version(), "platform": platform.platform(), "git_revision": revision,
            "git_dirty": dirty, "upstream_revision": UPSTREAM,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "tokenizers", "safetensors", "numpy")},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def load_model(path, device, dtype="float32"):
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError("v1 accepts local HF Qwen3-format Dense MiniMind-3 only; convert native .pth separately")
    if dtype not in ("float32", "bfloat16") or (dtype == "bfloat16" and device.type != "cuda"):
        raise ValueError("use float32 on CPU/MPS; bfloat16 is supported only on CUDA")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"weight compatibility failure: {loading}")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer needs explicit pad/eos token IDs")
    if len(tokenizer) > model.config.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds model vocabulary")
    model.to(device=device, dtype=getattr(torch, dtype))
    return model, tokenizer


def prompt_ids(tokenizer, conversation):
    return tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True,
                                         open_thinking=False)


def save_adapter(model, directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state(model), directory / "adapter.pt")
    write_json(directory / "adapter.json", {"format": "mathlab-lora-v1", "base_files": identity,
                                             "lora": model._mathlab_lora,
                                             "weights_sha256": sha256(directory / "adapter.pt")})


def load_adapter(model, directory, identity):
    directory = Path(directory)
    metadata = json.loads((directory / "adapter.json").read_text())
    if metadata.get("format") != "mathlab-lora-v1" or metadata["base_files"] != identity:
        raise ValueError("adapter format/base model/tokenizer identity mismatch")
    if sha256(directory / "adapter.pt") != metadata["weights_sha256"]:
        raise ValueError("adapter checksum mismatch")
    cfg = metadata["lora"]
    actual = inject(model, **{key: cfg[key] for key in ("mode", "targets", "rank", "alpha", "dropout")})
    if actual != cfg:
        raise ValueError("adapter injection manifest mismatch")
    load_adapter_state(model, torch.load(directory / "adapter.pt", map_location="cpu", weights_only=True))
    return metadata


def new_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
