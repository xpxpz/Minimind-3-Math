"""Audit and aggregate the registered 270-question, three-seed final evaluation."""
import json
from pathlib import Path
import statistics

from mathlab.compare import compare
from mathlab.io import read_jsonl, sha256, write_json
from scripts.analyze_predictions import analyze
from scripts.posttrain_campaign import DATA, ROOT, check_plan


def main():
    plan = check_plan()
    completed = json.loads((ROOT / "completed.json").read_text())
    records = {}
    tests = plan["test_ids"]
    names = ["base"] + [f"seed{s}" for s in plan["final_seeds"]]
    for name in names:
        combined = {}
        groups = {}
        per_split = {}
        for split, ids in tests.items():
            path = Path("results/runs") / f"SFT-final-{name}-{split}"
            meta = json.loads((path / "run.json").read_text())
            if meta["evaluated_ids"] != ids:
                raise ValueError("test subset differs from registered IDs")
            report = analyze(path, DATA / f"{split}.jsonl")
            per_split[split] = report["groups"]
            rows = read_jsonl(path / "predictions.jsonl")
            for row in rows:
                if row["problem_key"] in combined:
                    raise ValueError("duplicate problem across final test splits")
                combined[row["problem_key"]] = row
            for group, metrics in report["groups"].items():
                total = groups.setdefault(group, dict(n=0, correct=0, valid_completed_answers=0,
                                                       correct_literal_fraction_copies=0))
                for key in total:
                    total[key] += metrics[key]
        for group in groups.values():
            group["accuracy"] = group["correct"] / group["n"] if group["n"] else None
        records[name] = dict(total_questions=len(combined), groups=groups, splits=per_split)
    comparisons = {}
    for seed in plan["final_seeds"]:
        comparisons[str(seed)] = {}
        for split in tests:
            result = compare(Path("results/runs") / f"SFT-final-base-{split}",
                             Path("results/runs") / f"SFT-final-seed{seed}-{split}")
            saved = json.loads((ROOT / f"comparison-seed{seed}-{split}.json").read_text())
            if result != saved:
                raise ValueError("saved paired comparison differs from recomputed result")
            comparisons[str(seed)][split] = result
    seed_values = [records[f"seed{s}"]["groups"]["all"]["accuracy"] for s in plan["final_seeds"]]
    write_json(ROOT / "final_summary.json", dict(
        completed=completed, preregistration_sha256=sha256(ROOT / "plan.json"),
        selection_sha256=sha256(ROOT / "selection.json"), records=records, comparisons=comparisons,
        seed_accuracy_mean=statistics.mean(seed_values), seed_accuracy_sample_std=statistics.stdev(seed_values),
        aggregation="270 distinct problems, 90 per split; pooled accuracy equals equally weighted split mean",
        limitations=["three training seeds; sample std is not a confidence interval",
                     "paired bootstrap intervals describe question sampling within each split",
                     "same 270 questions for all seeds, not 810 independent questions",
                     "validation used for selection; test results never used for release seed choice"],
        summarizer_sha256=sha256(__file__)))
    print(json.dumps({name: r["groups"]["all"] for name, r in records.items()}, indent=2))


if __name__ == "__main__":
    main()
