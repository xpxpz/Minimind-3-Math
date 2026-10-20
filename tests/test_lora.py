import copy

import pytest
import torch
from torch import nn

from mathlab.lora import adapter_state, inject, load_adapter_state, merge
from mathlab.runtime import load_adapter, save_adapter


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 4)
        self.down_proj = nn.Linear(4, 8)

    @property
    def device(self):
        return self.q_proj.weight.device

    def forward(self, x):
        return self.q_proj(x) + self.down_proj(self.v_proj(x))


def test_rectangular_zero_delta_freeze_gradient_and_update():
    torch.manual_seed(4)
    model = Tiny()
    x = torch.randn(2, 3, 8)
    original = model(x).detach()
    base = {k: v.detach().clone() for k, v in model.named_parameters()}
    inject(model, targets="qv", rank=2, alpha=2)
    torch.testing.assert_close(model(x), original, rtol=0, atol=0)
    assert model.v_proj.lora.B.weight.shape == (4, 2)
    assert all((".lora." in name) == param.requires_grad for name, param in model.named_parameters())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(x).square().mean().backward()
    assert model.v_proj.lora.B.weight.grad.abs().sum() > 0
    assert model.v_proj.lora.A.weight.grad.abs().sum() == 0
    optimizer.step(); optimizer.zero_grad()
    model(x).square().mean().backward()
    assert model.v_proj.lora.A.weight.grad.abs().sum() > 0
    optimizer.step()
    for name, value in base.items():
        current = name.replace("q_proj.", "q_proj.base.").replace("v_proj.", "v_proj.base.")
        torch.testing.assert_close(dict(model.named_parameters())[current], value, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["official", "custom"])
def test_save_load_and_merge(mode, tmp_path):
    model = Tiny()
    other = copy.deepcopy(model)
    inject(model, mode=mode, targets="official_square" if mode == "official" else "qv", rank=2)
    for key, param in model.named_parameters():
        if ".lora.B" in key:
            nn.init.normal_(param)
    model.eval()
    x = torch.randn(3, 8)
    expected = model(x)
    identity = {"model.safetensors": "example-base-hash"}
    save_adapter(model, tmp_path, identity)
    load_adapter(other, tmp_path, identity)
    other.eval()
    torch.testing.assert_close(other(x), expected)
    merge(other)
    torch.testing.assert_close(other(x), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="active adapters"):
        merge(other)
    with pytest.raises(ValueError, match="mismatch"):
        load_adapter(Tiny(), tmp_path, {"wrong": "base"})


def test_official_compatibility_same_initialization_and_training():
    a = Tiny(); b = copy.deepcopy(a)
    torch.manual_seed(7)
    official = inject(a, mode="official", rank=2)
    torch.manual_seed(7)
    custom = inject(b, targets="official_square", rank=2, alpha=2, dropout=0)
    assert official["matched"] == custom["matched"]
    x = torch.randn(4, 8)
    for _ in range(2):
        optimizers = [torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1) for m in (a, b)]
        for model, optimizer in zip((a, b), optimizers):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
        for key, tensor in adapter_state(a).items():
            torch.testing.assert_close(tensor, adapter_state(b)[key], rtol=0, atol=0)


def test_invalid_targets_and_double_injection():
    with pytest.raises(ValueError, match="recursively"):
        inject(Tiny(), mode="official", rank=8)
    with pytest.raises(ValueError, match="no Linear"):
        inject(Tiny(), targets=["nonexistent"])
    with pytest.raises(ValueError, match="unmatched"):
        inject(Tiny(), targets=["q_proj", "nonexistent"])
    model = Tiny()
    inject(model)
    with pytest.raises(ValueError, match="already"):
        inject(model)
    state = adapter_state(model)
    state.pop(next(iter(state)))
    with pytest.raises(ValueError, match="keys/shapes"):
        load_adapter_state(model, state)
