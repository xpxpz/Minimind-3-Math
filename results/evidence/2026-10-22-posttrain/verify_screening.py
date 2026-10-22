"""One-off saved audit of completed E4 screening artifacts."""
import json
from pathlib import Path
import math

from mathlab.io import read_jsonl, sha256, write_json
from scripts.posttrain_campaign import ROOT, check_plan, choose

plan = check_plan()
selection = json.loads((ROOT / "selection.json").read_text())
assert choose(selection["candidates"]) == selection["selected"]
results = {}
for arm in plan["arms"]:
    root = ROOT / "runs" / f"screen-train-{arm}"
    logs = read_jsonl(root / "train.jsonl")
    summary = json.loads((root / "summary.json").read_text())
    assert [r["update"] for r in logs] == list(range(1, 501))
    assert sum(r["examples"] for r in logs) == 8000
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in logs)
    assert summary["target_tokens"] == 54145
    assert summary["nonpadding_tokens"] == 504202
    artifact = Path("artifacts") / (f"SFT-screen-snapshot-{arm}" if arm == selection["selected"]["arm"] else f"SFT-{arm}")
    for tag in ["best", "last"]:
        metadata = json.loads((root / f"{tag}_adapter.json").read_text())
        assert sha256(artifact / tag / "adapter.pt") == metadata["weights_sha256"]
    prediction_root = ROOT / "runs" / f"screen-eval-{arm}"
    meta = json.loads((prediction_root / "run.json").read_text())
    predictions = read_jsonl(prediction_root / "predictions.jsonl")
    assert [r["sample_id"] for r in predictions] == plan["validation_ids"] == meta["evaluated_ids"]
    results[arm] = dict(updates=500, presentations=8000, correct=sum(r["correct"] for r in predictions),
                        target_tokens=summary["target_tokens"], nonpadding_tokens=summary["nonpadding_tokens"],
                        local_adapter_hashes_verified=True)
assert len(results) == 6
assert not list(ROOT.rglob("*.pt")) and not list(ROOT.rglob("*.safetensors"))
write_json(ROOT / "screening_verification.json", dict(arms=results, selected=selection["selected"]["arm"],
           plan_sha256=sha256(ROOT / "plan.json"), selection_sha256=sha256(ROOT / "selection.json"),
           test_results_used=False, audit_script_sha256=sha256(__file__)))
print("Six arms, training budgets, frozen validation IDs and local adapter hashes verified")
