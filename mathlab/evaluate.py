"""Greedy batch=1 exact-answer evaluation, preserving every requested example."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import torch

from .answers import PROTOCOL, parse_answer, rational
from .data import verify_manifest
from .io import read_jsonl, sha256, write_json
from .runtime import (environment, load_adapter, load_model, model_identity, new_run, now,
                      prompt_ids, resolve_model, seed_everything, select_device, sync)


def select_rows(rows, limit):
    if limit is None or limit >= len(rows):
        return rows
    # Stable round-robin across tasks for development subsets, rather than a skewed prefix.
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    result, index = [], 0
    while len(result) < limit:
        for name in sorted(groups):
            if index < len(groups[name]):
                result.append(groups[name][index])
                if len(result) == limit:
                    return result
        index += 1
    return result


def score(row, output, truncated=False, error=None):
    parsed, failure = None, error
    if failure is None and truncated:
        failure = "truncated"
    if failure is None:
        try:
            parsed = str(parse_answer(output))
        except ValueError:
            failure = "parse_failure"
    correct = failure is None and rational(parsed) == rational(row["answer"])
    if failure is None and not correct:
        failure = "wrong_answer"
    return {"parsed_answer": parsed, "correct": bool(correct), "failure_type": failure}


def summarize(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups["all"].append(row)
        groups["task:" + row["task"]].append(row)
        groups["distribution:" + row["distribution"]].append(row)
    result = {}
    for name, rows in groups.items():
        result[name] = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
                        "parse_failure_rate": sum(r["failure_type"] == "parse_failure" for r in rows) / len(rows),
                        "truncation_rate": sum(r["failure_type"] == "truncated" for r in rows) / len(rows),
                        "runtime_error_rate": sum(r["failure_type"] == "runtime_error" for r in rows) / len(rows)}
    result["macro_task_accuracy"] = statistics.mean(v["accuracy"] for k, v in result.items() if k.startswith("task:"))
    return result


@torch.inference_mode()
def evaluate(args):
    seed_everything(42)
    path, device = resolve_model(args.model), select_device(args.device)
    data_path = Path(args.data)
    dataset_manifest = verify_manifest(data_path.parent)
    if data_path.stem not in dataset_manifest["splits"] or data_path.suffix != ".jsonl":
        raise ValueError("evaluation data must be a canonical split listed in the frozen manifest")
    rows = read_jsonl(data_path)
    rows = select_rows(rows, args.limit)
    if not rows or not all("id" in r and "answer" in r for r in rows):
        raise ValueError("evaluation needs nonempty canonical data, not an SFT-only export")
    identity = model_identity(path)
    model, tokenizer = load_model(path, device, args.dtype)
    adapter_meta = load_adapter(model, args.adapter, identity) if args.adapter else None
    model.eval()
    run = new_run(args.out)
    manifest = {"protocol": PROTOCOL, "environment": environment(), "model_files": identity,
                "data_sha256": sha256(data_path), "data_manifest_sha256": sha256(data_path.parent / "manifest.json"),
                "split": data_path.stem, "limit": args.limit, "subset_rule": "stable task round-robin if limited",
                "evaluated_ids": [r["id"] for r in rows],
                "adapter": adapter_meta, "device": str(device), "dtype": args.dtype,
                "decoding": {"do_sample": False, "batch_size": 1, "open_thinking": False,
                             "max_new_tokens": args.max_new_tokens, "repetition_penalty": 1.0,
                             "use_cache": True}, "started_utc": now(),
                "scorer_sha256": sha256(Path(__file__).with_name("answers.py")),
                "evaluator_sha256": sha256(__file__),
                "runtime_sha256": sha256(Path(__file__).with_name("runtime.py"))}
    write_json(run / "run.json", manifest)

    def generate(row):
        ids = prompt_ids(tokenizer, row["conversations"][:1])
        if len(ids) + args.max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("generation exceeds context budget")
        inputs = torch.tensor([ids], device=device)
        generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                   max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                   repetition_penalty=1.0)
        tokens = generated[0, len(ids):].tolist()
        return tokenizer.decode(tokens, skip_special_tokens=True), tokens

    # One full warmup generation, excluded from latency statistics.
    generate(rows[0]); sync(device)
    predictions = []
    with (run / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows):
            sync(device)
            start = time.perf_counter()
            error, tokens, raw, truncated = None, [], "", False
            try:
                raw, tokens = generate(row)
                truncated = len(tokens) >= args.max_new_tokens and tokens[-1] != tokenizer.eos_token_id
            except (RuntimeError, ValueError) as exc:
                error = "runtime_error"
                raw = f"{type(exc).__name__}: {exc}"
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
            sync(device)
            duration = time.perf_counter() - start
            prediction = {"sample_id": row["id"], "problem_key": row["problem_key"], "task": row["task"],
                          "template_id": row["template_id"], "distribution": row["distribution"],
                          "answer": row["answer"], "raw_output": raw, "generation_tokens": len(tokens),
                          "seconds": duration, **score(row, raw, truncated, error)}
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n"); output.flush()
            predictions.append(prediction)
            print(f"[{index + 1}/{len(rows)}] {row['id']} correct={prediction['correct']} {prediction['failure_type']}", flush=True)
    times = sorted(p["seconds"] for p in predictions)
    metrics = summarize(predictions)
    metrics["latency"] = {"p50_seconds": statistics.median(times),
                          "p95_seconds": times[max(0, math_ceil95(len(times)) - 1)],
                          "generation_tokens_per_second": sum(p["generation_tokens"] for p in predictions) / sum(times),
                          "warmup_examples_excluded": 1}
    write_json(run / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return run


def math_ceil95(n):
    return (95 * n + 99) // 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--data", default="data/processed/v1/smoke.jsonl")
    parser.add_argument("--adapter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("token budget and limit must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
