"""Expand E1 training coverage while preserving validation bytes exactly."""
import argparse
import hashlib
from pathlib import Path
import random
import shutil

from mathlab.answers import evaluate_expression, problem_key
from mathlab.data import SUFFIX, audit, verify_manifest
from mathlab.io import read_jsonl, sha256, write_json, write_jsonl
from scripts.generate_easy_math import problem_pool

VERSION = "easy-addsub-coverage-v2"


def make_row(expression, split):
    key = problem_key(expression)
    operation = "+" if "+" in expression else "-"
    prompt = (f"表达式 {expression} 的值是多少？" if split == "test_template" else f"计算 {expression}。") + SUFFIX
    answer = str(evaluate_expression(expression))
    return {"id": f"coverage_{split}_{hashlib.sha256(key.encode()).hexdigest()[:16]}",
            "task": "arithmetic", "template_id": f"easy_{'heldout' if split == 'test_template' else 'direct'}_{operation}",
            "problem_key": key, "split": split,
            "distribution": split.removeprefix("test_") if split.startswith("test_") else split,
            "expression": expression, "answer": answer, "generator_version": VERSION,
            "conversations": [{"role": "user", "content": prompt},
                              {"role": "assistant", "content": f"答案：{answer}"}]}


def generate(directory, source="data/processed/easy-addsub-v1", seed=42):
    directory, source = Path(directory), Path(source)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("output is not empty; choose a new version directory")
    original = verify_manifest(source)
    if original["version"] != "easy-addsub-v1":
        raise ValueError("expected the frozen E1 easy-addsub-v1 source")
    train, validation = read_jsonl(source / "train.jsonl"), read_jsonl(source / "validation.jsonl")
    if len(train) != 32 or len(validation) != 32:
        raise ValueError("expected 32 training and 32 validation source records")
    excluded = {r["problem_key"] for r in train + validation}
    rng = random.Random(seed)
    pools, added = {}, []
    for operation in ("+", "-"):
        pool = [e for e in problem_pool(0, 9, operation) if problem_key(e) not in excluded]
        rng.shuffle(pool)
        added.extend(make_row(e, "train") for e in pool[:16])
        pools[operation] = pool[16:]
    expanded = train + added
    rng.shuffle(expanded)
    write_jsonl(directory / "train.jsonl", expanded)
    for split in ("validation", "test_range"):
        shutil.copyfile(source / f"{split}.jsonl", directory / f"{split}.jsonl")
    # The finite addition domain has only seven groups left after train/validation.
    for split in ("test_id", "test_template", "smoke"):
        rows = []
        for operation in ("+", "-"):
            pool = pools[operation]
            if len(pool) < 2:
                raise ValueError("insufficient finite holdout space")
            rows.extend(make_row(e, split) for e in pool[:2])
            del pool[:2]
        rng.shuffle(rows)
        write_jsonl(directory / f"{split}.jsonl", rows)
    write_jsonl(directory / "train_sft.jsonl", ({"conversations": r["conversations"]} for r in expanded))
    shutil.copyfile(source / "validation_sft.jsonl", directory / "validation_sft.jsonl")
    added_keys = {r["problem_key"] for r in added}
    reclassified = {s: [r["id"] for r in read_jsonl(source / f"{s}.jsonl") if r["problem_key"] in added_keys]
                    for s in ("test_id", "test_template", "smoke")}
    manifest = {"version": VERSION, "seed": seed, "generator_sha256": sha256(__file__),
                "pool_generator_sha256": sha256(Path(__file__).with_name("generate_easy_math.py")),
                "source_manifest_sha256": sha256(source / "manifest.json"),
                "source_validation_sha256": sha256(source / "validation.jsonl"),
                "source_training_examples_preserved": 32, "added_training_examples": 32,
                "reclassified_unused_E1_holdout_ids": reclassified,
                "grouping": "canonical AST keys globally disjoint; all original train records retained; validation bytes unchanged",
                "selection_rule": "uniform seeded shuffle by operator, excluding source train and validation; no answer-based selection",
                "holdout_note": "E1 reserve splits are retired for E2 when reclassified; never report them as E2 unseen tests",
                "ranges": original["ranges"], "splits": audit(directory),
                "sft_sha256": {f"{s}_sft.jsonl": sha256(directory / f"{s}_sft.jsonl") for s in ("train", "validation")}}
    if manifest["splits"]["validation"]["sha256"] != original["splits"]["validation"]["sha256"]:
        raise ValueError("validation bytes changed")
    write_json(directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/processed/easy-addsub-v1")
    parser.add_argument("--out", default="data/processed/easy-addsub-coverage-v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = generate(args.out, args.source, args.seed)
    print({s: r["count"] for s, r in result["splits"].items()})


if __name__ == "__main__":
    main()
