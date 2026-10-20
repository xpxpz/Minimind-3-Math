"""Paired bootstrap of accuracy differences on identical frozen questions."""
import argparse
import json
from pathlib import Path
import random

from .io import read_jsonl, write_json


def compare(base, candidate, repeats=5000, seed=42):
    base, candidate = Path(base), Path(candidate)
    a_meta = json.loads((base / "run.json").read_text())
    b_meta = json.loads((candidate / "run.json").read_text())
    for key in ("protocol", "model_files", "data_sha256", "data_manifest_sha256", "evaluated_ids",
                "device", "dtype", "decoding", "scorer_sha256", "evaluator_sha256", "runtime_sha256"):
        if a_meta[key] != b_meta[key]:
            raise ValueError(f"comparison protocol mismatch: {key}")
    left = read_jsonl(base / "predictions.jsonl")
    right = read_jsonl(candidate / "predictions.jsonl")
    expected_ids = a_meta["evaluated_ids"]
    if [r["sample_id"] for r in left] != expected_ids or [r["sample_id"] for r in right] != expected_ids:
        raise ValueError("incomplete/misaligned predictions")
    if len({r["problem_key"] for r in left}) != len(left):
        raise ValueError("v1 requires one prediction per unique problem group")
    if any(a["answer"] != b["answer"] or a["problem_key"] != b["problem_key"] for a, b in zip(left, right)):
        raise ValueError("prediction reference mismatch")
    differences = [int(b["correct"]) - int(a["correct"]) for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(100 * sum(rng.choices(differences, k=n)) / n for _ in range(repeats))
    return {"n": n, "baseline_accuracy": sum(r["correct"] for r in left) / n,
            "candidate_accuracy": sum(r["correct"] for r in right) / n,
            "difference_percentage_points": 100 * sum(differences) / n,
            "paired_bootstrap_95_percentile_interval_pp": [samples[int(0.025 * (repeats - 1))], samples[int(0.975 * (repeats - 1))]],
            "bootstrap_repeats": repeats, "seed": seed,
            "unit": "one unique canonical problem group per question",
            "limitation": "question sampling uncertainty only; does not measure training-seed variability"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()
    if args.repeats < 100:
        parser.error("use at least 100 bootstrap repeats")
    result = compare(args.baseline, args.candidate, args.repeats)
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
