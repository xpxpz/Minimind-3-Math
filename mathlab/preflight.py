"""Inspect the local model, assistant mask and actual LoRA injection shapes."""
import argparse
from pathlib import Path

from .data import verify_manifest
from .io import read_jsonl, write_json
from .lora import inject
from .runtime import environment, load_model, model_identity, resolve_model, select_device
from .sft import encode_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--data", default="data/processed/v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="results/runs/preflight.json")
    args = parser.parse_args()
    path, device = resolve_model(args.model), select_device(args.device)
    model, tokenizer = load_model(path, device)
    manifest = verify_manifest(args.data)
    rows = read_jsonl(Path(args.data) / "train.jsonl")
    encoded = [encode_sample(r, tokenizer, 256) for r in rows]
    base_count = sum(p.numel() for p in model.parameters())
    lora = inject(model, targets="qv")
    report = {"model_files": model_identity(path), "environment": environment(), "device": str(device),
              "base_parameters": base_count, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "train_max_tokens": max(len(r["input_ids"]) for r in encoded),
              "train_mean_tokens": sum(len(r["input_ids"]) for r in encoded) / len(encoded),
              "mask_sample": {"prompt_tokens": sum(i == -100 for i in encoded[0]["labels"]),
                              "supervised_text": tokenizer.decode([i for i in encoded[0]["labels"] if i != -100])},
              "data_version": manifest["version"], "lora": lora}
    write_json(args.out, report)
    print(f"device={device}, base_parameters={base_count:,}, max_train_tokens={report['train_max_tokens']}")
    print(f"report={args.out}")


if __name__ == "__main__":
    main()
