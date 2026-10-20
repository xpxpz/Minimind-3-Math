"""Transparent single-device LoRA-SFT with exact token-weighted accumulation."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from .data import verify_manifest
from .io import read_jsonl, sha256, write_json, write_jsonl
from .lora import adapter_state, inject, load_adapter_state
from .runtime import (environment, load_model, model_identity, new_run, now, resolve_model,
                      save_adapter, seed_everything, select_device, sync)
from .sft import collate, encode_sample, target_tokens

DEFAULTS = {"mode": "custom", "targets": "qv", "rank": 16, "alpha": 16, "dropout": 0.0,
            "seed": 42, "epochs": 3, "batch_size": 4, "accumulation_steps": 4,
            "learning_rate": 0.0001, "weight_decay": 0.01, "grad_clip": 1.0,
            "max_length": 256, "train_limit": None, "validation_limit": None,
            "max_updates": None, "save_every": 100, "dtype": "float32"}


def load_config(path):
    provided = json.loads(Path(path).read_text())
    unknown = provided.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    config = DEFAULTS | provided
    for key in ("rank", "epochs", "batch_size", "accumulation_steps", "max_length", "save_every"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("train_limit", "validation_limit", "max_updates"):
        if config[key] is not None and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError(f"{key} must be null or a positive integer")
    for key in ("learning_rate", "grad_clip", "weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0 or (key != "weight_decay" and config[key] == 0):
            raise ValueError(f"invalid {key}")
    return config


@torch.no_grad()
def validation_loss(model, encoded, batch_size, pad_id, device):
    model.eval()
    loss_sum, count = 0.0, 0
    for start in range(0, len(encoded), batch_size):
        batch = collate(encoded[start:start + batch_size], pad_id)
        n = target_tokens(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("non-finite validation loss")
        loss_sum += loss.item() * n
        count += n
    return loss_sum / count


def train(args):
    config = load_config(args.config)
    seed_everything(config["seed"])
    data_dir = Path(args.data)
    verify_manifest(data_dir)
    data_identity = sha256(data_dir / "manifest.json")
    model_path, device = resolve_model(args.model), select_device(args.device)
    identity = model_identity(model_path)
    model, tokenizer = load_model(model_path, device, config["dtype"])
    lora = inject(model, **{k: config[k] for k in ("mode", "targets", "rank", "alpha", "dropout")})
    print(json.dumps({"device": str(device), "injection": lora}, ensure_ascii=False), flush=True)
    train_rows = read_jsonl(data_dir / "train.jsonl")[:config["train_limit"]]
    val_rows = read_jsonl(data_dir / "validation.jsonl")[:config["validation_limit"]]
    encoded = [encode_sample(r, tokenizer, config["max_length"]) for r in train_rows]
    val_encoded = [encode_sample(r, tokenizer, config["max_length"]) for r in val_rows]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    run = Path(args.out)
    started, updates, token_total, nonpad_total, next_epoch, next_group = now(), 0, 0, 0, 0, 0
    best_loss, accumulated_seconds = float("inf"), 0.0
    source_files = sorted(Path("mathlab").glob("*.py")) + [Path("vendor/minimind/model_lora.py")]
    sources = {str(p): sha256(p) for p in source_files}
    manifest = {"config": config, "model_files": identity, "data_manifest_sha256": data_identity,
                "train_file_sha256": sha256(data_dir / "train.jsonl"),
                "validation_file_sha256": sha256(data_dir / "validation.jsonl"),
                "sources_sha256": sources, "environment": environment(), "device": str(device),
                "train_examples": len(encoded), "validation_examples": len(val_encoded),
                "total_parameters_including_adapters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in params), "lora": lora}
    if args.resume:
        previous = json.loads((run / "run.json").read_text())
        for key in ("config", "model_files", "data_manifest_sha256", "sources_sha256", "device", "environment"):
            if previous[key] != manifest[key]:
                raise ValueError(f"resume mismatch in {key}; preserve model/data/code/environment/device/config")
        state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        load_adapter_state(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        updates, next_epoch, next_group = state["updates"], state["next_epoch"], state["next_group"]
        token_total, nonpad_total = state["target_tokens"], state["nonpad_tokens"]
        best_loss, accumulated_seconds = state["best_validation_loss"], state["training_seconds"]
        # A process may have logged further updates after its last complete checkpoint.
        log_path = run / "train.jsonl"
        if log_path.exists():
            recovered = []
            for line in log_path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                if record["update"] <= updates:
                    recovered.append(record)
            write_jsonl(log_path, recovered)
        started = previous["started_utc"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(state["device_rng"])
        elif device.type == "cuda":
            torch.cuda.set_rng_state(state["device_rng"], device)
    else:
        new_run(run)
        manifest["started_utc"] = started
        write_json(run / "run.json", manifest)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    clock_start = time.perf_counter()

    def seconds():
        sync(device)
        return accumulated_seconds + time.perf_counter() - clock_start

    def checkpoint(epoch, group):
        save_adapter(model, run / "last", identity)
        state = {"adapter": adapter_state(model), "optimizer": optimizer.state_dict(), "updates": updates,
                 "next_epoch": epoch, "next_group": group, "target_tokens": token_total,
                 "nonpad_tokens": nonpad_total, "best_validation_loss": best_loss,
                 "training_seconds": seconds(), "torch_rng": torch.get_rng_state(),
                 "device_rng": torch.mps.get_rng_state() if device.type == "mps" else
                 torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        temporary = run / "checkpoint.tmp"
        torch.save(state, temporary)
        temporary.replace(run / "checkpoint.pt")

    def validate():
        nonlocal best_loss
        loss = validation_loss(model, val_encoded, config["batch_size"], tokenizer.pad_token_id, device)
        if loss < best_loss:
            best_loss = loss
            save_adapter(model, run / "best", identity)
        return loss

    if not args.resume:
        initial_loss = validate()
        write_json(run / "initial_validation.json", {"loss": initial_loss, "updates": 0})
        checkpoint(0, 0)
    stop_at = updates + args.stop_after_updates if args.stop_after_updates else None
    group_size = config["batch_size"] * config["accumulation_steps"]
    groups_per_epoch = math.ceil(len(encoded) / group_size)
    last_val = best_loss
    halted = False
    with (run / "train.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(next_epoch, config["epochs"]):
            generator = torch.Generator().manual_seed(config["seed"] + epoch)
            order = torch.randperm(len(encoded), generator=generator).tolist()
            start_group = next_group if epoch == next_epoch else 0
            for group in range(start_group, groups_per_epoch):
                if config["max_updates"] is not None and updates >= config["max_updates"]:
                    halted = True
                    break
                selected = [encoded[i] for i in order[group * group_size:(group + 1) * group_size]]
                microbatches = [collate(selected[i:i + config["batch_size"]], tokenizer.pad_token_id)
                                for i in range(0, len(selected), config["batch_size"])]
                group_tokens = sum(target_tokens(b) for b in microbatches)
                optimizer.zero_grad(set_to_none=True)
                model.train()
                group_loss = 0.0
                for batch in microbatches:
                    n = target_tokens(batch)
                    nonpad_total += int(batch["attention_mask"].sum())
                    batch = {k: v.to(device) for k, v in batch.items()}
                    raw_loss = model(**batch, use_cache=False).loss
                    if not torch.isfinite(raw_loss):
                        raise ValueError("non-finite training loss")
                    loss = raw_loss * n / group_tokens
                    loss.backward()
                    group_loss += loss.item()
                norm = torch.nn.utils.clip_grad_norm_(params, config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                token_total += group_tokens
                cursor_epoch, cursor_group = (epoch + 1, 0) if group + 1 == groups_per_epoch else (epoch, group + 1)
                record = {"update": updates, "epoch": epoch + 1, "loss": group_loss,
                          "gradient_norm": float(norm), "examples": len(selected), "target_tokens": group_tokens}
                log.write(json.dumps(record) + "\n"); log.flush()
                print(json.dumps(record), flush=True)
                should_stop = ((stop_at is not None and updates >= stop_at) or
                               (config["max_updates"] is not None and updates >= config["max_updates"]))
                if updates % config["save_every"] == 0 or cursor_group == 0 or should_stop:
                    last_val = validate()
                    checkpoint(cursor_epoch, cursor_group)
                if should_stop:
                    halted = True
                    break
            if halted:
                break
    elapsed = seconds()
    write_json(run / "summary.json", {"started_utc": started, "finished_utc": now(), "updates": updates,
                                     "target_tokens": token_total, "nonpadding_tokens": nonpad_total,
                                     "training_and_validation_seconds": elapsed,
                                     "nonpadding_tokens_per_second_including_validation": nonpad_total / max(elapsed, 1e-9),
                                     "best_validation_loss": best_loss, "last_validation_loss": last_val,
                                     "selection_rule": "minimum validation target-token loss (includes step 0)",
                                     "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                                     "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                                     "checkpoint": "checkpoint.pt", "selected_adapter": "best"})
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/processed/v1")
    parser.add_argument("--model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int, help="stop at a checkpoint; resume with the same config")
    args = parser.parse_args()
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        parser.error("--stop-after-updates must be positive")
    train(args)


if __name__ == "__main__":
    main()
