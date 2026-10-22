import json
from pathlib import Path

import pytest

from mathlab import answers
from mathlab.io import sha256, write_json, write_jsonl
from scripts.analyze_predictions import analyze, literal_fraction_copy, structure


def make_run(tmp_path):
    data_dir, run = tmp_path / "data", tmp_path / "run"
    expressions = ["53 / 31", "87 / 58", "94 + 55", "(3 - 1) / 2", "3 * 4"]
    outputs = ["答案：53/31", "答案：3/2", "答案：-49", "答案：1", "unfinished"]
    records, predictions = [], []
    for i, (expression, output) in enumerate(zip(expressions, outputs)):
        record = {"id": str(i), "expression": expression, "answer": str(answers.evaluate_expression(expression)),
                  "task": "mixed" if i == 3 else "arithmetic", "problem_key": answers.problem_key(expression)}
        records.append(record)
        parsed = str(answers.parse_answer(output)) if i != 4 else None
        correct = parsed == record["answer"]
        predictions.append({"sample_id": str(i), "answer": record["answer"], "task": record["task"],
                            "problem_key": record["problem_key"], "raw_output": output, "parsed_answer": parsed,
                            "correct": correct, "failure_type": "truncated" if i == 4 else None if correct else "wrong_answer"})
    data = data_dir / "validation.jsonl"
    write_jsonl(data, records)
    write_json(data_dir / "manifest.json", {"version": "unit-test"})
    write_jsonl(run / "predictions.jsonl", predictions)
    write_json(run / "run.json", {"data_sha256": sha256(data), "data_manifest_sha256": sha256(data_dir / "manifest.json"),
                                    "scorer_sha256": sha256(answers.__file__), "evaluated_ids": [r["id"] for r in records]})
    return run, data


def test_non_division_and_copy_diagnostics_keep_failures(tmp_path):
    run, data = make_run(tmp_path)
    result = analyze(run, data)
    assert result["groups"]["all"]["n"] == 5
    assert result["groups"]["all"]["correct"] == 3
    assert result["groups"]["all"]["correct_literal_fraction_copies"] == 1
    assert result["groups"]["non_division"]["n"] == 2
    assert result["groups"]["non_division"]["accuracy"] == 0
    assert result["groups"]["single_division"]["n"] == 2
    assert result["groups"]["operation:mixed"]["correct"] == 1


def test_nested_division_and_reduced_fraction_are_not_literal_copies():
    assert structure("(3 - 1) / 2")["operation"] == "mixed"
    assert structure("(3 - 1) / 2")["contains_division"]
    assert literal_fraction_copy("53 / 31", "答案：53/31")
    assert literal_fraction_copy("-53 / 31", "答案：-53/31")
    assert not literal_fraction_copy("87 / 58", "答案：3/2")
    assert not literal_fraction_copy("(3 - 1) / 2", "答案：2/2")


@pytest.mark.parametrize("tamper", ["drop_row", "fake_correct", "reference", "data", "scorer"])
def test_bad_provenance_or_scores_rejected(tmp_path, tamper):
    run, data = make_run(tmp_path)
    path = run / "predictions.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if tamper == "drop_row":
        rows.pop()
    elif tamper == "fake_correct":
        rows[2]["correct"] = True
    elif tamper == "reference":
        rows[0]["answer"] = "999"
    elif tamper == "data":
        data.write_text(data.read_text() + "\n")
    else:
        meta = json.loads((run / "run.json").read_text())
        meta["scorer_sha256"] = "changed"
        write_json(run / "run.json", meta)
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        analyze(run, data)
