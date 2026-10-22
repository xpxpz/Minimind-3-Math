import ast
from collections import Counter

import pytest

from mathlab.data import verify_manifest
from mathlab.io import read_jsonl
from scripts.generate_easy_math import COUNTS, generate, problem_pool


def test_finite_problem_space_canonicalizes_addition():
    assert len(problem_pool(0, 9, "+")) == 55
    assert len(problem_pool(0, 9, "-")) == 100


def test_easy_data_is_frozen_reproducible_and_disjoint(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first, second = generate(a), generate(b)
    assert first == second
    assert verify_manifest(a) == first
    seen = set()
    for split, n in COUNTS.items():
        rows = read_jsonl(a / f"{split}.jsonl")
        assert len(rows) == n
        assert Counter("+" if "+" in r["expression"] else "-" for r in rows) == {"+": n // 2, "-": n // 2}
        keys = {r["problem_key"] for r in rows}
        assert not keys & seen
        seen |= keys
        low, high = (10, 19) if split == "test_range" else (0, 9)
        for row in rows:
            tree = ast.parse(row["expression"], mode="eval")
            assert all(low <= node.value <= high for node in ast.walk(tree) if isinstance(node, ast.Constant))
    assert any(r["answer"].startswith("-") for r in read_jsonl(a / "train.jsonl"))
    with pytest.raises(ValueError, match="not empty"):
        generate(a)
