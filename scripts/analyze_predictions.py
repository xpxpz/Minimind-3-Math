"""Audit frozen predictions and separate arithmetic from literal fraction copying.

This is supplementary analysis. It never changes the generation or answer scorer.
"""
import argparse
import ast
from collections import Counter, defaultdict
import re
from pathlib import Path

from mathlab import answers
from mathlab.io import read_jsonl, sha256, write_json
import json

VERSION = "math-prediction-analysis-v1"


def structure(expression):
    answers.evaluate_expression(expression)
    node = ast.parse(expression, mode="eval").body
    operations = [n for n in ast.walk(node) if isinstance(n, ast.BinOp)]
    names = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}
    operation = "mixed" if len(operations) > 1 else names[type(operations[0].op)] if operations else "constant"
    single_division = len(operations) == 1 and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
    return {"operation": operation, "contains_division": any(isinstance(n.op, ast.Div) for n in operations),
            "single_division": single_division}


def literal_fraction_copy(expression, raw_output):
    if not structure(expression)["single_division"]:
        return False
    node = ast.parse(expression, mode="eval").body
    left = answers.evaluate_expression(ast.unparse(node.left))
    right = answers.evaluate_expression(ast.unparse(node.right))
    if left.denominator != 1 or right.denominator != 1:
        return False
    last = raw_output.rsplit("</think>", 1)[-1].strip().splitlines()[-1:]
    match = re.fullmatch(r"\s*答案[：:]\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*", last[0]) if last else None
    return bool(match and int(match.group(1)) == left and int(match.group(2)) == right)


def group_metrics(rows):
    n = len(rows)
    return {"n": n, "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / n if n else None,
            "valid_completed_answers": sum(r["failure_type"] in (None, "wrong_answer") for r in rows),
            "failures": dict(Counter(r["failure_type"] or "correct" for r in rows)),
            "correct_literal_fraction_copies": sum(r["correct"] and r["literal_fraction_copy"] for r in rows)}


def analyze(run_directory, data_path):
    run_directory, data_path = Path(run_directory), Path(data_path)
    meta = json.loads((run_directory / "run.json").read_text())
    if sha256(data_path) != meta["data_sha256"]:
        raise ValueError("data checksum mismatch")
    if sha256(data_path.parent / "manifest.json") != meta["data_manifest_sha256"]:
        raise ValueError("data manifest checksum mismatch")
    if sha256(answers.__file__) != meta["scorer_sha256"]:
        raise ValueError("scorer code changed; use the recorded scorer before analyzing")
    data_rows = read_jsonl(data_path)
    reference = {r["id"]: r for r in data_rows}
    if len(reference) != len(data_rows):
        raise ValueError("duplicate reference IDs")
    rows = read_jsonl(run_directory / "predictions.jsonl")
    ids = [r["sample_id"] for r in rows]
    if not ids or ids != meta["evaluated_ids"] or len(set(ids)) != len(ids):
        raise ValueError("incomplete, duplicate or misaligned predictions")
    groups = defaultdict(list)
    annotated = []
    for prediction in rows:
        row = reference.get(prediction["sample_id"])
        if row is None or any(prediction[k] != row[k] for k in ("answer", "task", "problem_key")):
            raise ValueError("prediction/reference identity mismatch")
        failure = prediction["failure_type"]
        if failure not in (None, "wrong_answer", "parse_failure", "truncated", "runtime_error"):
            raise ValueError("unknown failure type")
        expected_correct = False
        if failure in (None, "wrong_answer"):
            parsed = answers.parse_answer(prediction["raw_output"])
            if str(parsed) != prediction["parsed_answer"]:
                raise ValueError("parsed answer does not match raw output")
            expected_correct = parsed == answers.rational(row["answer"])
            if (failure is None) != expected_correct:
                raise ValueError("failure label inconsistent with exact scorer")
        if type(prediction["correct"]) is not bool or prediction["correct"] != expected_correct:
            raise ValueError("stored correctness disagrees with exact scorer")
        item = {**prediction, "expression": row["expression"], **structure(row["expression"]),
                "literal_fraction_copy": literal_fraction_copy(row["expression"], prediction["raw_output"])}
        annotated.append(item)
        for key in ("all", "task:" + item["task"], "operation:" + item["operation"],
                    "contains_division" if item["contains_division"] else "non_division"):
            groups[key].append(item)
        if item["single_division"]:
            groups["single_division"].append(item)
    # Keep empty key groups explicit to prevent omission from being read as success.
    for name in ("non_division", "single_division", "contains_division"):
        groups.setdefault(name, [])
    examples = {}
    for name, grouped in groups.items():
        if not name.startswith("operation:"):
            continue
        examples[name] = {}
        for label, correct in (("first_correct", True), ("first_incorrect", False)):
            chosen = next((r for r in grouped if r["correct"] == correct), None)
            examples[name][label] = ({k: chosen[k] for k in ("sample_id", "expression", "answer", "raw_output",
                                                                  "parsed_answer", "failure_type", "literal_fraction_copy")}
                                      if chosen else None)
    return {"version": VERSION, "analysis_sha256": sha256(__file__), "data_sha256": meta["data_sha256"],
            "predictions_sha256": sha256(run_directory / "predictions.jsonl"),
            "scorer_sha256": meta["scorer_sha256"], "evaluated_ids": ids,
            "groups": {k: group_metrics(v) for k, v in sorted(groups.items())}, "examples": examples,
            "example_rule": "first correct and first incorrect per operation in frozen prediction order",
            "interpretation": "Literal fraction copying is a diagnostic, not proof of a model's internal algorithm."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(args.run_directory, args.data)
    write_json(args.out, result)
    print(json.dumps(result["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
