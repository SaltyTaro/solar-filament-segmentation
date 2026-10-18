"""Release check: export the final weights, then rebuild the submission from that export alone.

Runs ``infra/export-weights.py`` on the training checkpoints and then
``infra/reproduce-submission.py`` on the exported folder (no training run is
read in the second step), writing both under OUTPUT.

    python infra/release-check.py --lineage METADATA.json --runs RUNS_DIR --refiner REFINER_RUN \
        --rescorer RESCORER.pkl --reference SUBMITTED.csv --output OUTPUT
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--lineage", "--runs", "--refiner", "--rescorer", "--reference", "--output"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Accepted for worker compatibility")
    args = parser.parse_args()
    weights = args.output / "weights"
    if not (weights / "manifest.json").exists():
        subprocess.run([sys.executable, str(HERE / "export-weights.py"), "--lineage", str(args.lineage), "--runs", str(args.runs),
                        "--refiner", str(args.refiner), "--rescorer", str(args.rescorer), "--output", str(weights)], check=True)
    subprocess.run([sys.executable, str(HERE / "reproduce-submission.py"), "--weights", str(weights),
                    "--output", str(args.output / "reproduction"), "--reference", str(args.reference),
                    "--workers", str(args.workers)], check=True)


if __name__ == "__main__":
    main()
