"""Check an official/custom compatibility pair using real training artifacts."""
import argparse
import json
from pathlib import Path

import torch

from mathlab.io import read_jsonl, sha256, write_json


def check(official, custom):
    official, custom = Path(official), Path(custom)
    a, b = [json.loads((p / "run.json").read_text()) for p in (official, custom)]
    if a["config"]["mode"] != "official" or b["config"]["mode"] != "custom":
        raise ValueError("expected official/custom pair")
    for key in ("model_files", "data_manifest_sha256", "sources_sha256", "device", "environment"):
        if a[key] != b[key]:
            raise ValueError(f"compatibility input mismatch: {key}")
    if {k: v for k, v in a["config"].items() if k != "mode"} != {k: v for k, v in b["config"].items() if k != "mode"}:
        raise ValueError("only injection mode may differ")
    if a["lora"]["matched"] != b["lora"]["matched"]:
        raise ValueError("injection module mismatch")
    left, right = [read_jsonl(p / "train.jsonl") for p in (official, custom)]
    if not left or len(left) != len(right):
        raise ValueError("training lengths differ")
    loss_delta = max(abs(x["loss"] - y["loss"]) for x, y in zip(left, right))
    logs_equal = left == right
    tensors = [torch.load(p / "last/adapter.pt", map_location="cpu", weights_only=True) for p in (official, custom)]
    if tensors[0].keys() != tensors[1].keys():
        raise ValueError("adapter names differ")
    tensor_delta = max((tensors[0][k] - tensors[1][k]).abs().max().item() for k in tensors[0])
    equal = all(torch.equal(tensors[0][k], tensors[1][k]) for k in tensors[0])
    result = dict(updates=len(left), matched_modules=len(a["lora"]["matched"]),
                  trainable_parameters=a["trainable_parameters"], training_logs_exact_equal=logs_equal,
                  max_absolute_training_loss_difference=loss_delta, last_tensors_exact_equal=equal,
                  max_absolute_tensor_difference=tensor_delta,
                  official_last_sha256=sha256(official / "last/adapter.pt"),
                  custom_last_sha256=sha256(custom / "last/adapter.pt"),
                  checker_sha256=sha256(__file__),
                  passed=logs_equal and equal)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official")
    parser.add_argument("custom")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = check(args.official, args.custom)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("compatibility failed; inspect evidence before continuing")
