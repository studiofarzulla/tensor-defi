#!/usr/bin/env python3
"""
TENSOR-DEFI: Full Pipeline Orchestrator

Runs the complete narrative-market alignment analysis:
1. Data collection (whitepapers + market data)
2. NLP pipeline (claims matrix)
3. Market representations (stats + tensor factors)
4. Alignment testing (Procrustes + Tucker's φ)
5. Extended analysis
6. Figure generation
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    ("Data Collection", "scripts/run_data_collection.py"),
    ("NLP Pipeline", "scripts/run_nlp.py"),
    ("Market Statistics", "scripts/run_market.py"),
    ("Tensor Decomposition", "scripts/run_tensor.py"),
    ("Alignment Testing", "scripts/run_alignment.py"),
    ("Extended Analysis", "scripts/run_analysis.py"),
    ("Figure Generation", "scripts/run_figures.py"),
]


def run_script(name: str, script_path: str, dry_run: bool = False) -> bool:
    """Run a pipeline script."""
    print(f"\n{'='*60}")
    print(f"PHASE: {name}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[DRY RUN] Would run: {script_path}")
        return True

    path = Path(script_path)
    if not path.exists():
        print(f"[SKIP] Script not yet implemented: {script_path}")
        return True

    result = subprocess.run([sys.executable, script_path], check=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run TENSOR-DEFI pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    parser.add_argument("--start-from", type=int, default=0, help="Start from phase N")
    args = parser.parse_args()

    print("="*60)
    print("TENSOR-DEFI: Unified Narrative-Market Alignment Framework")
    print("="*60)

    for i, (name, script) in enumerate(SCRIPTS):
        if i < args.start_from:
            print(f"[SKIP] Phase {i}: {name}")
            continue

        success = run_script(name, script, args.dry_run)
        if not success:
            print(f"\n[FAILED] Pipeline stopped at phase: {name}")
            sys.exit(1)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
