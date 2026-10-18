"""Train one fold model, then write dihedral TTA candidates for its held-out fold and the test set.

Every stage is resumable: training resumes from last.pt unless result.json is
complete, and TTA skips images whose candidate file already exists.

    python infra/train-and-tta.py --family maskrcnn --config configs/x.yaml --output RUN
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def run(arguments: list[str]):
    print("+", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], check=True, cwd=HERE.parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("maskrcnn", "yolo", "dense"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Accepted for worker compatibility; resume is automatic")
    args = parser.parse_args()
    output = args.output
    result = output / "result.json"
    complete = result.exists() and json.loads(result.read_text(encoding="utf-8")).get("status") == "complete"
    if not complete:
        if args.family == "maskrcnn":
            command = ["-m", "solar_filament.cli", "train-detector", "--config", args.config, "--output", str(output)]
        elif args.family == "yolo":
            command = [str(HERE / "train-yolo.py"), "--config", args.config, "--output", str(output)]
        else:
            command = [str(HERE / "train-dense.py"), "--config", args.config, "--output", str(output)]
        if (output / "last.pt").exists():
            command.append("--resume")
        run(command)
        if json.loads(result.read_text(encoding="utf-8")).get("status") != "complete":
            raise SystemExit("Training stopped before completion; relaunch to resume")
    fold = yaml.safe_load((HERE.parent / args.config).read_text(encoding="utf-8"))["fold"]
    subsets = (["validation"] if fold >= 0 else []) + ([] if args.skip_test else ["test"])
    for subset in subsets:
        target = output / f"tta-{subset}"
        if (target / "metadata.json").exists():
            continue
        run([str(HERE / "tta-inference.py"), "--family", args.family, "--run", str(output), "--subset", subset,
             "--views", *map(str, args.views), "--output", str(target)])


if __name__ == "__main__":
    main()
