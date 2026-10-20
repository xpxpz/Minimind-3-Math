import json

import pytest

from mathlab.data import generate, verify_manifest
from mathlab.io import read_jsonl, write_jsonl


def test_reproducible_disjoint_groups(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = generate(a, train=48, validation=12, test=12, ood=12, smoke=12)
    second = generate(b, train=48, validation=12, test=12, ood=12, smoke=12)
    assert first == second
    assert verify_manifest(a) == first
    seen = set()
    for split in first["splits"]:
        keys = {r["problem_key"] for r in read_jsonl(a / f"{split}.jsonl")}
        assert not seen & keys
        seen |= keys
    with pytest.raises(ValueError, match="not empty"):
        generate(a)


def test_tamper_and_leak_detection(tmp_path):
    generate(tmp_path, train=12, validation=6, test=6, ood=6, smoke=6)
    path = tmp_path / "validation.jsonl"
    rows = read_jsonl(path)
    leaked = read_jsonl(tmp_path / "train.jsonl")[0]
    leaked.update(split="validation", distribution="validation")
    rows[0] = leaked
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="duplicate"):
        verify_manifest(tmp_path)
