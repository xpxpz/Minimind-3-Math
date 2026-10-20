"""Configurable rectangular LoRA, with an unchanged upstream control path."""
import math
import torch
from torch import nn

from vendor.minimind.model_lora import apply_lora as upstream_apply_lora

TARGETS = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attention_ffn": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


class Branch(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # Same random draws and initialization as the official control.
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora = Branch(base.in_features, base.out_features, rank, alpha, dropout).to(base.weight)

    def forward(self, x):
        return self.base(x) + self.lora(x)


def inject(model, mode="custom", targets="qv", rank=16, alpha=None, dropout=0.0):
    if hasattr(model, "_mathlab_lora") or any(hasattr(m, "lora") for m in model.modules()):
        raise ValueError("model already has adapters")
    alpha = rank if alpha is None else alpha
    if rank <= 0 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("invalid rank/alpha/dropout")
    if mode not in ("custom", "official"):
        raise ValueError("mode must be custom or official")
    square = targets == "official_square" or mode == "official"
    wanted = TARGETS.get(targets, targets) if isinstance(targets, str) else targets
    if not square and (not isinstance(wanted, list) or not wanted or not all(isinstance(t, str) and t for t in wanted)):
        raise ValueError("targets must be a preset or a nonempty list of module names/suffixes")
    if mode == "official" and (alpha != rank or dropout != 0):
        raise ValueError("official control requires alpha=rank, dropout=0")
    matched, excluded = [], []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        match = module.in_features == module.out_features if square else any(
            name == t or name.endswith("." + t) for t in wanted)
        entry = {"name": name, "in_features": module.in_features, "out_features": module.out_features}
        (matched if match else excluded).append(entry)
    if not matched:
        raise ValueError("no Linear matched targets")
    if mode == "official" and any(rank == entry["in_features"] for entry in matched):
        raise ValueError("official rank equals a square layer width and would recursively inject its own branch")
    if not square:
        missing = [t for t in wanted if not any(m["name"] == t or m["name"].endswith("." + t) for m in matched)]
        if missing:
            raise ValueError(f"unmatched requested targets: {missing}")
    if mode == "official":
        upstream_apply_lora(model, rank=rank)
        for entry in matched:
            layer = model.get_submodule(entry["name"])
            layer.lora.to(layer.weight)
    else:
        for entry in matched:
            name = entry["name"]
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha, dropout))
    model.requires_grad_(False)
    for entry in matched:
        model.get_submodule(entry["name"]).lora.requires_grad_(True)
    model._mathlab_lora = {"mode": mode, "targets": targets, "rank": rank, "alpha": alpha,
                          "dropout": dropout, "matched": matched, "excluded": excluded}
    return model._mathlab_lora


def adapter_state(model):
    return {f"{entry['name']}.lora.{key}": value.detach().cpu().clone()
            for entry in model._mathlab_lora["matched"]
            for key, value in model.get_submodule(entry["name"]).lora.state_dict().items()}


def load_adapter_state(model, state):
    expected = adapter_state(model)
    if expected.keys() != state.keys() or any(expected[k].shape != state[k].shape for k in expected):
        raise ValueError("adapter keys/shapes do not match injection manifest")
    for entry in model._mathlab_lora["matched"]:
        prefix = f"{entry['name']}.lora."
        model.get_submodule(entry["name"]).lora.load_state_dict(
            {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)


@torch.no_grad()
def merge(model):
    if model.training or not hasattr(model, "_mathlab_lora"):
        raise ValueError("merge requires an eval model with active adapters")
    for entry in model._mathlab_lora["matched"]:
        name = entry["name"]
        layer = model.get_submodule(name)
        branch = layer.lora
        base = layer.base if isinstance(layer, LoRALinear) else layer
        delta = (branch.B.weight.float() @ branch.A.weight.float()) * getattr(branch, "scale", 1.0)
        base.weight.add_(delta.to(base.weight))
        if isinstance(layer, LoRALinear):
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, base)
        else:
            del base.lora
            base.forward = nn.Linear.forward.__get__(base, nn.Linear)
    del model._mathlab_lora
    return model
