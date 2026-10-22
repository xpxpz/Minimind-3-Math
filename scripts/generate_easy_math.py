"""Freeze a finite, group-disjoint single-digit add/sub learnability probe."""
import argparse
import hashlib
from pathlib import Path
import random

from mathlab.answers import evaluate_expression, problem_key
from mathlab.data import SPLITS, SUFFIX, audit
from mathlab.io import sha256, write_json, write_jsonl

VERSION = "easy-addsub-v1"
COUNTS = {"train": 32, "validation": 32, "test_id": 16, "test_template": 12, "test_range": 32, "smoke": 12}


def problem_pool(low, high, operation):
    expressions = [f"{a} {operation} {b}" for a in range(low, high + 1) for b in range(low, high + 1)
                   if operation != "+" or a <= b]
    keys = [problem_key(e) for e in expressions]
    if len(keys) != len(set(keys)):
        raise ValueError("pool contains equivalent problem keys")
    return expressions


def generate(directory, seed=42):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("output is not empty; choose a new dataset version")
    rng = random.Random(seed)
    pools = {}
    for domain, bounds in (("id", (0, 9)), ("range", (10, 19))):
        for operation in ("+", "-"):
            pool = problem_pool(*bounds, operation)
            rng.shuffle(pool)
            pools[domain, operation] = pool
    for split in SPLITS:
        rows = []
        for operation in ("+", "-"):
            pool = pools["range" if split == "test_range" else "id", operation]
            n = COUNTS[split] // 2
            if len(pool) < n:
                raise ValueError("split counts exceed unique finite problem space")
            expressions = pool[:n]
            del pool[:n]
            for expression in expressions:
                key = problem_key(expression)
                answer = str(evaluate_expression(expression))
                prompt = (f"表达式 {expression} 的值是多少？" if split == "test_template" else f"计算 {expression}。") + SUFFIX
                rows.append({"id": f"easy_{split}_{hashlib.sha256(key.encode()).hexdigest()[:16]}",
                             "task": "arithmetic", "template_id": f"easy_{'heldout' if split == 'test_template' else 'direct'}_{operation}",
                             "problem_key": key, "split": split,
                             "distribution": split.removeprefix("test_") if split.startswith("test_") else split,
                             "expression": expression, "answer": answer, "generator_version": VERSION,
                             "conversations": [{"role": "user", "content": prompt},
                                               {"role": "assistant", "content": f"答案：{answer}"}]})
        rng.shuffle(rows)
        write_jsonl(directory / f"{split}.jsonl", rows)
        if split in ("train", "validation"):
            write_jsonl(directory / f"{split}_sft.jsonl", ({"conversations": r["conversations"]} for r in rows))
    manifest = {"version": VERSION, "seed": seed, "generator_sha256": sha256(__file__),
                "grouping": "one canonical AST key globally; addition a<=b removes swapped duplicates",
                "purpose": "finite learnability probe; train accuracy measures fitting, not generalization",
                "ranges": {"id_leaf_operands": [0, 9], "range_ood_leaf_operands": [10, 19]},
                "splits": audit(directory),
                "sft_sha256": {f"{s}_sft.jsonl": sha256(directory / f"{s}_sft.jsonl") for s in ("train", "validation")}}
    write_json(directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed/easy-addsub-v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate(args.out, args.seed)
    print({"version": manifest["version"], "counts": {s: v["count"] for s, v in manifest["splits"].items()}})


if __name__ == "__main__":
    main()
