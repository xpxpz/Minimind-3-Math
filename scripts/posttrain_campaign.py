"""Bounded SFT campaign: freeze plan, validate arms, select, train and test.

Run from the repository root. Every subprocess is logged before execution.
Testing is forbidden until a validation-only selection has been frozen.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mathlab.data import verify_manifest
from mathlab.evaluate import select_rows
from mathlab.io import read_jsonl, sha256, write_json
from mathlab.runtime import environment, model_identity, now, resolve_model
from scripts.analyze_predictions import analyze

ROOT = Path("results/evidence/2026-10-22-posttrain")
DATA = Path("data/processed/v1")
ARMS = {
    "official_r16": ("official", "official_square", 16),
    "qv_r16": ("custom", "qv", 16),
    "qkvo_r16": ("custom", "qkvo", 16),
    "ffn_r8": ("custom", "attention_ffn", 8),
    "ffn_r16": ("custom", "attention_ffn", 16),
    "ffn_r32": ("custom", "attention_ffn", 32),
}


def config(mode, targets, rank, seed=42):
    return dict(mode=mode, targets=targets, rank=rank, alpha=rank, dropout=0.0,
                seed=seed, epochs=3, batch_size=4, accumulation_steps=4,
                learning_rate=0.0001, max_length=256, max_updates=1500, save_every=100)


def choose(candidates):
    """Validation accuracy, then fewer parameters, then loss, then name."""
    if not candidates or any(r.get("split") != "validation" for r in candidates):
        raise ValueError("selection requires validation candidates only")
    return min(candidates, key=lambda r: (-r["correct"], r["trainable_parameters"],
                                         r["best_validation_loss"], r["arm"]))


def run(label, module, *args):
    command = [sys.executable, "-m", module, *map(str, args)]
    path = ROOT / "logs" / f"{label}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"log already exists: {path}")
    entry = dict(label=label, command=command, cwd=str(Path.cwd()), started_utc=now(), log=str(path))
    write_json(ROOT / "commands" / f"{label}.json", entry)
    print(f"START {label}", flush=True)
    with path.open("w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    entry.update(returncode=result.returncode, finished_utc=now())
    write_json(ROOT / "commands" / f"{label}.json", entry)
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {path}")
    print(f"DONE {label}", flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def archive(source, name):
    """Archive small manifests, predictions and logs, never tensors/checkpoints."""
    source = Path(source)
    destination = ROOT / "runs" / name
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.suffix in (".json", ".jsonl") and file.is_file():
            shutil.copy2(file, destination / file.name)
    for tag in ("best", "last"):
        if (source / tag / "adapter.json").exists():
            shutil.copy2(source / tag / "adapter.json", destination / f"{tag}_adapter.json")


def freeze():
    if (ROOT / "plan.json").exists():
        raise ValueError("campaign plan already exists; never overwrite a registered plan")
    verify_manifest(DATA)
    for name, values in ARMS.items():
        write_json(Path("configs/campaign") / f"{name}.json", config(*values))
    plan = dict(created_utc=now(), environment=environment(), model_files=model_identity(resolve_model()),
                dataset_manifest_sha256=sha256(DATA / "manifest.json"),
                source_sha256={str(p): sha256(p) for folder in ("mathlab", "scripts")
                               for p in sorted(Path(folder).glob("*.py"))},
                arms={name: dict(config=f"configs/campaign/{name}.json",
                                 config_sha256=sha256(Path("configs/campaign") / f"{name}.json")) for name in ARMS},
                screening_updates=500, final_total_updates=1500, final_seeds=[42, 123, 2026],
                adapter_selection="best validation target-token loss, including step zero",
                arm_selection="validation120 correct descending; parameters ascending; best loss ascending; arm name ascending",
                release_seed=42, validation_ids=[r["id"] for r in select_rows(read_jsonl(DATA / "validation.jsonl"), 120)],
                test_ids={s: [r["id"] for r in select_rows(read_jsonl(DATA / f"{s}.jsonl"), 90)]
                          for s in ("test_id", "test_template", "test_range")},
                decoding=dict(max_new_tokens=256, do_sample=False, open_thinking=False, batch_size=1),
                limitation="bounded first SFT release; screening one seed/one epoch, 270 test questions, not all 1600; no GRPO")
    write_json(ROOT / "plan.json", plan)
    print("Frozen campaign plan", flush=True)


def check_plan():
    plan = read(ROOT / "plan.json")
    if sha256(DATA / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("campaign dataset changed")
    if model_identity(resolve_model()) != plan["model_files"]:
        raise ValueError("campaign base model changed")
    for name, digest in plan["source_sha256"].items():
        if sha256(name) != digest:
            raise ValueError(f"campaign source changed: {name}")
    for arm in plan["arms"].values():
        if sha256(arm["config"]) != arm["config_sha256"]:
            raise ValueError("campaign config changed")
    return plan


def screening():
    plan = check_plan()
    candidates = []
    for arm, spec in plan["arms"].items():
        train_dir = Path("artifacts") / f"SFT-{arm}"
        eval_dir = Path("results/runs") / f"SFT-screen-{arm}"
        run(f"screen-train-{arm}", "mathlab.train", "--config", spec["config"], "--data", DATA,
            "--out", train_dir, "--stop-after-updates", plan["screening_updates"])
        run(f"screen-eval-{arm}", "mathlab.evaluate", "--data", DATA / "validation.jsonl",
            "--adapter", train_dir / "best", "--limit", 120, "--out", eval_dir)
        meta, summary = read(train_dir / "run.json"), read(train_dir / "summary.json")
        analysis = analyze(eval_dir, DATA / "validation.jsonl")
        if analysis["evaluated_ids"] != plan["validation_ids"]:
            raise ValueError("selection IDs differ from preregistered validation subset")
        write_json(eval_dir / "analysis.json", analysis)
        candidates.append(dict(arm=arm, split="validation", correct=analysis["groups"]["all"]["correct"], n=120,
                               trainable_parameters=meta["trainable_parameters"],
                               best_validation_loss=summary["best_validation_loss"],
                               seconds=summary["training_and_validation_seconds"],
                               updates=summary["updates"], adapter=str(train_dir / "best")))
        archive(train_dir, f"screen-train-{arm}")
        archive(eval_dir, f"screen-eval-{arm}")
        write_json(ROOT / "screening.json", candidates)
    selection = dict(selected_utc=now(), plan_sha256=sha256(ROOT / "plan.json"),
                     rule=plan["arm_selection"], candidates=candidates, selected=choose(candidates),
                     test_results_used=False)
    write_json(ROOT / "selection.json", selection)
    print(json.dumps(selection["selected"], indent=2), flush=True)


def final():
    plan = check_plan()
    selection = read(ROOT / "selection.json")
    if selection["plan_sha256"] != sha256(ROOT / "plan.json") or selection["selected"] != choose(selection["candidates"]):
        raise ValueError("invalid validation-only selection")
    arm = selection["selected"]["arm"]
    for seed in plan["final_seeds"]:
        train_dir = Path("artifacts") / (f"SFT-{arm}" if seed == 42 else f"SFT-final-seed{seed}")
        config_path = Path(plan["arms"][arm]["config"])
        args = []
        if seed == 42:
            args = ["--resume", "--stop-after-updates", 1000]
        else:
            configuration = read(config_path)
            configuration["seed"] = seed
            config_path = Path("configs/campaign") / f"final_seed{seed}.json"
            write_json(config_path, configuration)
        run(f"final-train-seed{seed}", "mathlab.train", "--config", config_path, "--data", DATA,
            "--out", train_dir, *args)
        if read(train_dir / "summary.json")["updates"] != 1500:
            raise ValueError("final training budget was not completed")
        archive(train_dir, f"final-train-seed{seed}")
    # All candidate choices and training budgets are fixed before any test inference.
    for split, ids in plan["test_ids"].items():
        runs = [("base", None)] + [(f"seed{seed}",
                 Path("artifacts") / (f"SFT-{arm}" if seed == 42 else f"SFT-final-seed{seed}") / "best")
                 for seed in plan["final_seeds"]]
        for name, adapter in runs:
            eval_dir = Path("results/runs") / f"SFT-final-{name}-{split}"
            args = ["--adapter", adapter] if adapter else []
            run(f"test-{name}-{split}", "mathlab.evaluate", "--data", DATA / f"{split}.jsonl",
                "--limit", len(ids), "--out", eval_dir, *args)
            analysis = analyze(eval_dir, DATA / f"{split}.jsonl")
            if analysis["evaluated_ids"] != ids:
                raise ValueError("final IDs differ from preregistered test subset")
            write_json(eval_dir / "analysis.json", analysis)
            archive(eval_dir, f"test-{name}-{split}")
            if adapter:
                run(f"compare-{name}-{split}", "mathlab.compare",
                    Path("results/runs") / f"SFT-final-base-{split}", eval_dir,
                    "--out", ROOT / f"comparison-{name}-{split}.json")
    write_json(ROOT / "completed.json", dict(finished_utc=now(), selection_sha256=sha256(ROOT / "selection.json"),
                                             release_arm=arm, release_seed=42, updates_per_seed=1500))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "screening", "final"])
    options = parser.parse_args()
    {"freeze": freeze, "screening": screening, "final": final}[options.stage]()
