import json

import pytest

from mathlab import answers
from mathlab.compare import compare as strict_compare
from mathlab.data import verify_manifest
from mathlab.io import read_jsonl, sha256, write_json, write_jsonl
from scripts.generate_easy_math import generate as generate_source
from scripts.expand_easy_coverage import generate as expand
from scripts.compare_shared_validation import compare


def datasets(tmp_path):
    source, expanded = tmp_path / "e1", tmp_path / "e2"
    generate_source(source)
    expand(expanded, source)
    return source, expanded


def test_expansion_retains_train_and_identical_validation(tmp_path):
    source, expanded = datasets(tmp_path)
    assert (source / "validation.jsonl").read_bytes() == (expanded / "validation.jsonl").read_bytes()
    old = read_jsonl(source / "train.jsonl"); new = read_jsonl(expanded / "train.jsonl")
    assert len(new) == 64
    by_id = {r["id"]: r for r in new}
    assert all(by_id[r["id"]] == r for r in old)
    assert sum("+" in r["expression"] for r in new) == 32
    assert sum("-" in r["expression"] for r in new) == 32
    manifest = verify_manifest(expanded)
    again = tmp_path / "again"
    assert expand(again, source) == manifest
    assert manifest["source_validation_sha256"] == sha256(source / "validation.jsonl")
    old_reserve = {r["problem_key"] for split in ("test_id", "test_template", "smoke")
                   for r in read_jsonl(source / f"{split}.jsonl")}
    added = {r["problem_key"] for r in new} - {r["problem_key"] for r in old}
    assert sum(len(ids) for ids in manifest["reclassified_unused_E1_holdout_ids"].values()) == len(added & old_reserve)
    with pytest.raises(ValueError, match="not empty"):
        expand(expanded, source)


def make_run(path, data, correct_count):
    refs = read_jsonl(data)
    preds = []
    for i, row in enumerate(refs):
        value = answers.rational(row["answer"]) + (0 if i < correct_count else 1)
        preds.append({"sample_id": row["id"], "answer": row["answer"], "task": row["task"],
                      "problem_key": row["problem_key"], "raw_output": f"答案：{value}",
                      "parsed_answer": str(value), "correct": i < correct_count,
                      "failure_type": None if i < correct_count else "wrong_answer"})
    write_jsonl(path / "predictions.jsonl", preds)
    write_json(path / "run.json", {"protocol": answers.PROTOCOL, "model_files": {"fixture": "same"},
               "data_sha256": sha256(data), "data_manifest_sha256": sha256(data.parent / "manifest.json"),
               "evaluated_ids": [r["id"] for r in refs], "device": "cpu", "dtype": "float32",
               "decoding": {"max_new_tokens": 256}, "scorer_sha256": sha256(answers.__file__),
               "evaluator_sha256": "fixture", "runtime_sha256": "fixture",
               "environment": {"platform": "fixture", "versions": {"fixture": "1"}}})


def test_explicit_shared_validation_policy_keeps_strict_default(tmp_path):
    source, expanded = datasets(tmp_path)
    a, b = tmp_path / "run-a", tmp_path / "run-b"
    make_run(a, source / "validation.jsonl", 6)
    make_run(b, expanded / "validation.jsonl", 10)
    with pytest.raises(ValueError, match="data_manifest_sha256"):
        strict_compare(a, b, repeats=100)
    result = compare(a, b, source / "validation.jsonl", expanded / "validation.jsonl", repeats=100)
    assert result["groups"]["all"]["difference_pp"] == 12.5
    assert result["baseline_manifest_sha256"] != result["candidate_manifest_sha256"]
    assert result["groups"]["all"]["became_correct"] == 4


@pytest.mark.parametrize("tamper", ["validation", "decoding", "manifest", "missing_row"])
def test_shared_policy_still_rejects_mismatch(tmp_path, tamper):
    source, expanded = datasets(tmp_path)
    a, b = tmp_path / "run-a", tmp_path / "run-b"
    make_run(a, source / "validation.jsonl", 6)
    make_run(b, expanded / "validation.jsonl", 10)
    if tamper == "validation":
        path = expanded / "validation.jsonl"
        path.write_text(path.read_text() + "\n")
    elif tamper == "missing_row":
        path = b / "predictions.jsonl"
        write_jsonl(path, read_jsonl(path)[:-1])
    else:
        path = b / "run.json" if tamper == "decoding" else expanded / "manifest.json"
        meta = json.loads(path.read_text())
        if tamper == "decoding":
            meta["decoding"]["max_new_tokens"] = 128
        else:
            meta["seed"] = 99
        write_json(path, meta)
    with pytest.raises(ValueError):
        compare(a, b, source / "validation.jsonl", expanded / "validation.jsonl", repeats=100)
