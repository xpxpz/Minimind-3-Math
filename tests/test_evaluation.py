import argparse
import json

import pytest

from mathlab.compare import compare
from mathlab.data import generate
from mathlab.evaluate import evaluate
from mathlab.io import read_jsonl, write_json


def test_offline_evaluation_and_protocol_guard(tiny_hf_model, tmp_path):
    path, _ = tiny_hf_model
    data = tmp_path / "data"
    generate(data, train=6, validation=3, test=3, ood=3, smoke=6)
    a, b = tmp_path / "base", tmp_path / "candidate"
    def args(out):
        return argparse.Namespace(model=str(path), device="cpu", data=str(data / "smoke.jsonl"),
                                  limit=3, dtype="float32", adapter=None, max_new_tokens=2, out=str(out))
    evaluate(args(a)); evaluate(args(b))
    rows = read_jsonl(a / "predictions.jsonl")
    assert len(rows) == 3
    assert {r["task"] for r in rows} == {"arithmetic", "mixed", "word"}
    result = compare(a, b, repeats=100)
    assert result["difference_percentage_points"] == 0
    meta = json.loads((b / "run.json").read_text())
    meta["decoding"]["max_new_tokens"] += 1
    write_json(b / "run.json", meta)
    with pytest.raises(ValueError, match="decoding"):
        compare(a, b, repeats=100)


def test_incomplete_predictions_rejected(tiny_hf_model, tmp_path):
    # Reuse the minimum metadata contract to isolate missing-row detection.
    metadata = {key: "same" for key in ("protocol", "model_files", "data_sha256", "data_manifest_sha256",
                "device", "dtype", "decoding", "scorer_sha256", "evaluator_sha256", "runtime_sha256")}
    metadata["evaluated_ids"] = ["one"]
    for name in ("a", "b"):
        write_json(tmp_path / name / "run.json", metadata)
        (tmp_path / name / "predictions.jsonl").write_text("")
    with pytest.raises(ValueError, match="incomplete"):
        compare(tmp_path / "a", tmp_path / "b", repeats=100)
