"""Compare identical validation files across explicitly different data manifests.

Each run is audited against its own frozen dataset before comparison. No metadata
is rewritten, and the original mathlab.compare strict default remains unchanged.
"""
import argparse
import json
from pathlib import Path
import random

from mathlab.io import read_jsonl, sha256, write_json
from mathlab.data import verify_manifest
from scripts.analyze_predictions import analyze, structure


def compare(baseline, candidate, baseline_data, candidate_data, repeats=5000, seed=42):
    baseline, candidate = Path(baseline), Path(candidate)
    baseline_data, candidate_data = Path(baseline_data), Path(candidate_data)
    if repeats < 100:
        raise ValueError("use at least 100 bootstrap repeats")
    if sha256(baseline_data) != sha256(candidate_data):
        raise ValueError("validation bytes differ; shared validation comparison is not allowed")
    verify_manifest(baseline_data.parent)
    verify_manifest(candidate_data.parent)
    # Verify each real manifest, scorer, reference, ID sequence and exact scores.
    analyze(baseline, baseline_data)
    analyze(candidate, candidate_data)
    left_meta = json.loads((baseline / "run.json").read_text())
    right_meta = json.loads((candidate / "run.json").read_text())
    for key in ("protocol", "model_files", "data_sha256", "evaluated_ids", "device", "dtype", "decoding",
                "scorer_sha256", "evaluator_sha256", "runtime_sha256"):
        if left_meta[key] != right_meta[key]:
            raise ValueError(f"comparison protocol mismatch: {key}")
    for key in ("platform", "versions"):
        if left_meta["environment"][key] != right_meta["environment"][key]:
            raise ValueError(f"comparison environment mismatch: {key}")
    left, right = read_jsonl(baseline / "predictions.jsonl"), read_jsonl(candidate / "predictions.jsonl")
    groups = {"all": list(range(len(left)))}
    references = {r["id"]: r for r in read_jsonl(baseline_data)}
    for operation in ("add", "subtract"):
        groups[operation] = [i for i, row in enumerate(left)
                             if structure(references[row["sample_id"]]["expression"])["operation"] == operation]
    rng = random.Random(seed)
    results = {}
    for name, indices in groups.items():
        if not indices:
            continue
        deltas = [int(right[i]["correct"]) - int(left[i]["correct"]) for i in indices]
        n = len(indices)
        samples = sorted(100 * sum(rng.choices(deltas, k=n)) / n for _ in range(repeats))
        results[name] = {"n": n, "baseline_correct": sum(left[i]["correct"] for i in indices),
                         "candidate_correct": sum(right[i]["correct"] for i in indices),
                         "became_correct": sum(d == 1 for d in deltas), "became_incorrect": sum(d == -1 for d in deltas),
                         "difference_pp": 100 * sum(deltas) / n,
                         "paired_bootstrap_95_percentile_interval_pp": [samples[int(.025 * (repeats - 1))], samples[int(.975 * (repeats - 1))]]}
    return {"policy": "identical validation bytes/IDs/protocol; independently validated data manifests may differ",
            "validation_sha256": sha256(baseline_data), "baseline_manifest_sha256": left_meta["data_manifest_sha256"],
            "candidate_manifest_sha256": right_meta["data_manifest_sha256"], "repeats": repeats, "seed": seed,
            "groups": results, "limitation": "exploratory reused validation; one training seed; subgroup intervals not multiplicity-adjusted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--baseline-data", required=True)
    parser.add_argument("--candidate-data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.baseline_data, args.candidate_data)
    write_json(args.out, result)
    print(json.dumps(result["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
