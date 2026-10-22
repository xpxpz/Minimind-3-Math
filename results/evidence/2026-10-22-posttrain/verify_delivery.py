"""Saved integration audit; run after summary, release and reports exist."""
import json
import math
from pathlib import Path
import re
import subprocess

from mathlab.io import read_jsonl, sha256, write_json
from scripts.posttrain_campaign import ROOT, check_plan

plan = check_plan()
summary = json.loads((ROOT / "final_summary.json").read_text())
release = json.loads((ROOT / "release.json").read_text())
config_reference = None
training = {}
for seed in plan["final_seeds"]:
    label = f"final-train-seed{seed}"
    root = ROOT / "runs" / label
    meta = json.loads((root / "run.json").read_text())
    stats = json.loads((root / "summary.json").read_text())
    logs = read_jsonl(root / "train.jsonl")
    assert [r["update"] for r in logs] == list(range(1, 1501))
    assert sum(r["examples"] for r in logs) == 24000
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in logs)
    assert stats["updates"] == 1500 and stats["target_tokens"] == 162435
    assert stats["nonpadding_tokens"] == 1512606
    assert meta["config"]["seed"] == seed and meta["trainable_parameters"] == 344064
    cfg = {k: v for k, v in meta["config"].items() if k != "seed"}
    if config_reference is None:
        config_reference = cfg
    assert cfg == config_reference
    best = json.loads((root / "best_adapter.json").read_text())
    artifact = Path("artifacts") / ("SFT-qv_r16" if seed == 42 else f"SFT-final-seed{seed}")
    assert sha256(artifact / "best/adapter.pt") == best["weights_sha256"]
    assert meta["model_files"] == plan["model_files"]
    assert summary["records"][f"seed{seed}"]["total_questions"] == 270
    training[str(seed)] = dict(updates=1500, presentations=24000,
                               best_sha256=best["weights_sha256"],
                               best_equals_last=sha256(artifact / "last/adapter.pt") == best["weights_sha256"])

commands = [json.loads(p.read_text()) for p in (ROOT / "commands").glob("*.json")]
assert len(commands) == 36 and all(r.get("returncode") == 0 for r in commands)
assert sha256("artifacts/release/adapter/adapter.pt") == release["adapter_sha256"] == training["42"]["best_sha256"]
export = json.loads((ROOT / "export_verification.json").read_text())
assert export["adapter_sha256"] == release["adapter_sha256"]
assert export["reloaded_logits_exact_equal"]
for filename, digest in export["merged_files"].items():
    assert sha256(Path("artifacts/release/merged") / filename) == digest

documents = [Path("README.md"), Path("docs/model_card.md"), Path("docs/posttraining_usage.md"),
             Path("docs/reports/README.md")] + [Path("docs/reports") / name for name in [
                 "2026-10-22_E3-official-custom-compatibility.md",
                 "2026-10-22_E4-module-rank-screening.md", "2026-10-22_E4b-ffn-rank-ablation.md",
                 "2026-10-22_E5-final-sft-release.md"]]
write_json(ROOT / "delivery_verification.json", {"status": "running", "purpose": "self-linked report audit"})
links_checked = 0
for document in documents:
    content = document.read_text()
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
        if "://" in target or target.startswith("#"):
            continue
        target = target.split("#", 1)[0].strip("<>")
        target = re.sub(r":\d+$", "", target)
        if target:
            path = Path(target) if target.startswith("/") else document.parent / target
            assert path.exists(), (document, target)
            links_checked += 1
for path in ["artifacts/release/adapter/adapter.pt", "artifacts/release/merged/model.safetensors"]:
    subprocess.run(["git", "check-ignore", "-q", path], check=True)
assert not list(ROOT.rglob("*.pt")) and not list(ROOT.rglob("*.safetensors"))
tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
assert not any(p.endswith((".pt", ".safetensors", ".pth", ".ckpt", ".gguf")) for p in tracked)
write_json(ROOT / "delivery_verification.json", dict(
    status="passed",
    training=training, successful_subprocesses=36, final_question_count=270, final_training_seeds=3,
    report_links_checked=links_checked, release_adapter_hash_verified=True,
    merged_files_hash_verified=True, exported_reload_logits_exact_equal=True, weights_git_ignored=True,
    final_summary_sha256=sha256(ROOT / "final_summary.json"),
    source_sha256={str(p): sha256(p) for p in sorted(Path("scripts").glob("*.py"))},
    audit_script_sha256=sha256(__file__)))
print("Delivery audit passed: budgets, predictions, selection, weights, export, commands and report links")
