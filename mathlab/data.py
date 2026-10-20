"""Generate, audit and freeze synthetic math data, without model dependencies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .answers import evaluate_expression, parse_answer, problem_key
from .io import read_jsonl, sha256, write_json, write_jsonl

VERSION = "math-data-v1"
SUFFIX = "只需给出最终结果，格式为“答案：数值”；非整数请用最简分数，不要带单位。"
SPLITS = ("train", "validation", "test_id", "test_template", "test_range", "smoke")


def candidate(rng, task, heldout=False, large=False):
    low, high = (1000, 9999) if large else (1, 99)
    a, b, c = [rng.randint(low, high) for _ in range(3)]
    op = rng.choice(["+", "-", "*", "/"])
    if task == "arithmetic":
        expr = f"{a} {op} {b}"
        prompt = f"计算 {expr}。" if not heldout else f"表达式 {expr} 的值是多少？"
        template = f"arithmetic_{'heldout' if heldout else 'direct'}_{op}"
    elif task == "mixed":
        kind = rng.randrange(3)
        expr = [f"({a} + {b}) * {c}", f"({a} - {b}) / {c}", f"{a} + {b} * {c}"][kind]
        prompt = f"计算 {expr}。" if not heldout else f"请按照运算顺序求 {expr} 的结果。"
        template = f"mixed_{'heldout' if heldout else 'direct'}_{kind}"
    else:
        if op == "+":
            expr = f"{a} + {b}"
            prompt = (f"盒子里有{a}个球，又放入{b}个球，现在有多少个球？" if not heldout
                      else f"上午售出{a}张票，下午售出{b}张票，两次一共售出多少张？")
        elif op == "-":
            a, b = max(a, b), min(a, b)
            expr = f"{a} - {b}"
            prompt = (f"有{a}支铅笔，用掉{b}支，还剩多少支？" if not heldout
                      else f"水箱原有{a}升水，流出{b}升，现在有多少升？")
        elif op == "*":
            expr = f"{a} * {b}"
            prompt = (f"每盒装{a}个球，{b}盒共装多少个球？" if not heldout
                      else f"每张车票{a}元，买{b}张一共需要多少元？")
        else:
            # Fractional lengths are meaningful, unlike fractional counts of objects.
            expr = f"{a} / {b}"
            prompt = (f"把{a}米绳子平均分成{b}段，每段长多少米？" if not heldout
                      else f"{b}天匀速修完{a}千米道路，每天修多少千米？")
        template = f"word_{'heldout' if heldout else 'direct'}_{op}"
    return expr, prompt + SUFFIX, template


def audit(directory):
    directory = Path(directory)
    seen_ids, seen_keys, seen_prompts = set(), set(), set()
    summaries = {}
    for split in SPLITS:
        rows = read_jsonl(directory / f"{split}.jsonl")
        for row in rows:
            key = problem_key(row["expression"])
            prompt = row["conversations"][0]["content"]
            if row["split"] != split or row["problem_key"] != key:
                raise ValueError(f"incorrect split/key: {row['id']}")
            if row["id"] in seen_ids or key in seen_keys or prompt in seen_prompts:
                raise ValueError(f"duplicate ID/problem/prompt: {row['id']}")
            seen_ids.add(row["id"]); seen_keys.add(key); seen_prompts.add(prompt)
            expected = evaluate_expression(row["expression"])
            if str(expected) != row["answer"] or parse_answer(row["conversations"][1]["content"]) != expected:
                raise ValueError(f"incorrect answer: {row['id']}")
            if row["distribution"] != (split.removeprefix("test_") if split.startswith("test_") else split):
                raise ValueError(f"incorrect distribution: {row['id']}")
            if ("heldout" in row["template_id"]) != (split == "test_template"):
                raise ValueError(f"incorrect template holdout: {row['id']}")
        summaries[split] = {"count": len(rows), "tasks": dict(Counter(r["task"] for r in rows)),
                            "templates": dict(Counter(r["template_id"] for r in rows)),
                            "sha256": sha256(directory / f"{split}.jsonl")}
    training_templates = {r["template_id"] for r in read_jsonl(directory / "train.jsonl")}
    heldout_templates = {r["template_id"] for r in read_jsonl(directory / "test_template.jsonl")}
    if training_templates & heldout_templates:
        raise ValueError("template leakage")
    return summaries


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = audit(directory)
    if actual != manifest["splits"]:
        raise ValueError("data does not match frozen manifest")
    for name, digest in manifest["sft_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"SFT export changed: {name}")
    return manifest


def generate(directory, train=8000, validation=1000, test=1000, ood=300, smoke=240, seed=42):
    directory = Path(directory)
    counts = dict(zip(SPLITS, (train, validation, test, ood, ood, smoke)))
    if any(n <= 0 for n in counts.values()):
        raise ValueError("all split sizes must be positive")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("output is not empty; choose a new version directory")
    rng = random.Random(seed)
    seen, prompts = set(), set()
    for split, count in counts.items():
        rows, attempts = [], 0
        while len(rows) < count:
            attempts += 1
            if attempts > count * 200:
                raise ValueError("requested dataset exceeds unique problem space")
            task = ("arithmetic", "mixed", "word")[len(rows) % 3]
            expr, prompt, template = candidate(rng, task, split == "test_template", split == "test_range")
            key = problem_key(expr)
            if key in seen or prompt in prompts:
                continue
            seen.add(key); prompts.add(prompt)
            answer = str(evaluate_expression(expr))
            rows.append({"id": f"{split}_{hashlib.sha256(key.encode()).hexdigest()[:16]}", "task": task,
                         "template_id": template, "problem_key": key, "split": split,
                         "distribution": split.removeprefix("test_") if split.startswith("test_") else split,
                         "expression": expr, "answer": answer, "generator_version": VERSION,
                         "conversations": [{"role": "user", "content": prompt},
                                           {"role": "assistant", "content": f"答案：{answer}"}]})
        rng.shuffle(rows)
        write_jsonl(directory / f"{split}.jsonl", rows)
        if split in ("train", "validation"):
            write_jsonl(directory / f"{split}_sft.jsonl", ({"conversations": r["conversations"]} for r in rows))
    manifest = {"version": VERSION, "seed": seed, "generator_sha256": sha256(__file__),
                "grouping": "global AST key; flatten/sort associative + and *; no full algebraic equivalence",
                "ranges": {"id_leaf_operands": [1, 99], "range_ood_leaf_operands": [1000, 9999]},
                "splits": audit(directory),
                "sft_sha256": {f"{s}_sft.jsonl": sha256(directory / f"{s}_sft.jsonl") for s in ("train", "validation")}}
    write_json(directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("generate")
    build.add_argument("--out", default="data/processed/v1")
    for arg, default in (("train", 8000), ("validation", 1000), ("test", 1000), ("ood", 300), ("smoke", 240), ("seed", 42)):
        build.add_argument(f"--{arg}", type=int, default=default)
    check = sub.add_parser("validate")
    check.add_argument("directory")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "generate":
        args["directory"] = args.pop("out")
        result = generate(**args)
    else:
        result = verify_manifest(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
