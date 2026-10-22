"""Package the preregistered seed-42 adapter after all final evaluations finish."""
import json
from pathlib import Path
import shutil

from mathlab.io import sha256, write_json
from mathlab.runtime import now
from scripts.posttrain_campaign import ROOT, check_plan, choose


def main():
    plan = check_plan()
    if plan["release_seed"] != 42:
        raise ValueError("this campaign release is fixed to seed 42")
    selection = json.loads((ROOT / "selection.json").read_text())
    completed = json.loads((ROOT / "completed.json").read_text())
    summary = json.loads((ROOT / "final_summary.json").read_text())
    arm = choose(selection["candidates"])["arm"]
    if selection["selected"]["arm"] != arm or completed["release_arm"] != arm:
        raise ValueError("release does not match validation selection")
    source = Path("artifacts") / f"SFT-{arm}"
    training = json.loads((source / "summary.json").read_text())
    metadata = json.loads((source / "best/adapter.json").read_text())
    if training["updates"] != 1500 or sha256(source / "best/adapter.pt") != metadata["weights_sha256"]:
        raise ValueError("release training or checksum check failed")
    for split in plan["test_ids"]:
        run = json.loads((Path("results/runs") / f"SFT-final-seed42-{split}/run.json").read_text())
        if run["adapter"] != metadata:
            raise ValueError("release differs from evaluated adapter")
    destination = Path("artifacts/release")
    destination.mkdir(exist_ok=False)
    shutil.copytree(source / "best", destination / "adapter")
    info = dict(created_utc=now(), arm=arm, seed=42, source=str(source), updates=1500,
                base_files=metadata["base_files"], adapter_sha256=metadata["weights_sha256"],
                artifact_path=str(destination.resolve()),
                selection_sha256=sha256(ROOT / "selection.json"), plan_sha256=sha256(ROOT / "plan.json"),
                summary_sha256=sha256(ROOT / "final_summary.json"),
                final_test=summary["records"]["seed42"],
                limitations=summary["limitations"], script_sha256=sha256(__file__))
    write_json(destination / "release.json", info)
    write_json(ROOT / "release.json", info)
    print(f"Packaged evaluated adapter: {destination.resolve()}")


if __name__ == "__main__":
    main()
