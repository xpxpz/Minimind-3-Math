import argparse
import json

import pytest
import torch

from mathlab.data import generate
from mathlab.io import read_jsonl, write_json
from mathlab.runtime import load_model, select_device
from mathlab.sft import collate, encode_sample, target_tokens
from mathlab.train import train


def test_mask_padding_and_truncation(tiny_hf_model):
    _, tokenizer = tiny_hf_model
    a = {"conversations": [{"role": "user", "content": "计算 1 + 2。"},
                            {"role": "assistant", "content": "答案：3"}]}
    b = {"conversations": [{"role": "user", "content": "计算 11 + 22。"},
                            {"role": "assistant", "content": "答案：33"}]}
    rows = [encode_sample(r, tokenizer, 256) for r in (a, b)]
    target = tokenizer.decode([i for i in rows[0]["labels"] if i != -100])
    assert "答案" in target and "计算" not in target and "think" not in target
    batch = collate(rows, tokenizer.pad_token_id)
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    with pytest.raises(ValueError, match="no truncation"):
        encode_sample(a, tokenizer, 3)


def test_accumulation_matches_combined_token_loss(tiny_hf_model):
    path, tokenizer = tiny_hf_model
    model, _ = load_model(path, select_device("cpu"))
    from mathlab.lora import inject
    inject(model, rank=2, alpha=2)
    rows = [encode_sample({"conversations": [{"role": "user", "content": "计算 1 + 2。"},
            {"role": "assistant", "content": f"答案：{answer}"}]}, tokenizer, 256) for answer in ("3", "12345")]
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    full = collate(rows, tokenizer.pad_token_id)
    model(**full).loss.backward()
    expected = [p.grad.clone() for p in params]
    model.zero_grad()
    total = target_tokens(full)
    for row in rows:
        batch = collate([row], tokenizer.pad_token_id)
        (model(**batch).loss * target_tokens(batch) / total).backward()
    for param, gradient in zip(params, expected):
        torch.testing.assert_close(param.grad, gradient, rtol=1e-4, atol=1e-7)


def test_resume_matches_uninterrupted_training(tiny_hf_model, tmp_path):
    path, _ = tiny_hf_model
    data = tmp_path / "data"
    generate(data, train=7, validation=3, test=3, ood=3, smoke=3)
    config_path = tmp_path / "config.json"
    write_json(config_path, {"epochs": 2, "batch_size": 2, "accumulation_steps": 2,
                             "rank": 2, "alpha": 2, "dropout": 0.2, "save_every": 1})
    def args(out, resume=False, stop=None):
        return argparse.Namespace(config=str(config_path), data=str(data), model=str(path), device="cpu",
                                  out=str(out), resume=resume, stop_after_updates=stop)
    a, b = tmp_path / "continuous", tmp_path / "resumed"
    train(args(a))
    train(args(b, stop=1))
    train(args(b, resume=True))
    left = torch.load(a / "last/adapter.pt", weights_only=True)
    right = torch.load(b / "last/adapter.pt", weights_only=True)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert [r["loss"] for r in read_jsonl(a / "train.jsonl")] == [r["loss"] for r in read_jsonl(b / "train.jsonl")]
    assert json.loads((a / "summary.json").read_text())["updates"] == 4
