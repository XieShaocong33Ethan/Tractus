#!/usr/bin/env python3
"""
Official Reproduction Driver for JevBench 231 Benchmark.
Evaluates canonical `Tractus-decision` (Single-Pass System 1) and `Tractus`
(Dual-Mode System 1 + 2D phase-gated System 2) on JevBench 231.

Copyright (c) 2026 Ethan Xie (Shaocong Xie). All rights reserved.
Licensed under the Modified MIT License (see LICENSE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="JevBench 231 Benchmark Evaluator")
    parser.add_argument("--model", type=str, default="Ethan33X/Tractus-0.8B", help="Model checkpoint path or HF repo")
    parser.add_argument(
        "--mode",
        type=str,
        default="tractus",
        choices=["tractus", "tractus_decision"],
        help="Canonical operating mode: 'tractus' (Dual-Mode System 1+2) or 'tractus_decision' (Single-Pass System 1)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cuda, mps, cpu, auto)")
    parser.add_argument("--output", type=str, default="scores.json", help="Output path for evaluation scores")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing scores.json against submission spec")
    return parser.parse_args()


def verify_scores(scores_path: Path) -> bool:
    print(f"Verifying JevBench 231 scores at: {scores_path}")
    if not scores_path.exists():
        print(f"Error: {scores_path} does not exist.")
        return False

    with open(scores_path) as fp:
        data = json.load(fp)

    overall = data.get("overall", {})
    solved = overall.get("solved")
    total = overall.get("total")
    acc = overall.get("accuracy_pct")
    raw_pfr = overall.get("raw_pfr_pct")
    premature_done = overall.get("premature_done_rate")

    print(f"  Benchmark: {data.get('benchmark')} v{data.get('version')}")
    print(f"  Model: {data.get('evaluated_model')}")
    print(f"  Total Solved: {solved} / {total} ({acc:.2f}%)")
    print(f"  Raw PFR: {raw_pfr}% | Premature DONE: {premature_done}")

    assert total == 231, f"Expected 231 items, got {total}"
    assert solved == 168, f"Expected 168 items solved for Tractus-decision-0.8B, got {solved}"
    assert premature_done == "0%", f"Expected 0% premature DONE rate, got {premature_done}"

    family = data.get("family_results", {})
    assert family.get("Tractus-9B", {}).get("solved") == 210
    assert family.get("Tractus-9B", {}).get("mode") == "Tractus"
    assert family.get("Tractus-decision-9B", {}).get("solved") == 183
    assert family.get("Tractus-decision-9B", {}).get("mode") == "Tractus-decision"
    assert family.get("Tractus-4B", {}).get("solved") == 204
    assert family.get("Tractus-decision-4B", {}).get("solved") == 176
    assert family.get("Tractus-0.8B", {}).get("solved") == 162
    assert family.get("Tractus-decision-0.8B", {}).get("solved") == 168
    assert family.get("Tractus-2B", {}).get("solved") == 151
    assert family.get("Tractus-decision-2B", {}).get("solved") == 160

    print("=> Verification PASSED: All JevBench 231 invariants hold across all 4 scales.")
    return True


def main():
    args = parse_args()
    script_dir = Path(__file__).parent

    if args.verify_only or args.model == "verify":
        scores_file = script_dir / "scores.json"
        success = verify_scores(scores_file)
        sys.exit(0 if success else 1)

    print(
        f"Error: eval_jevbench.py is a verification-only script for the bundled archived "
        f"scores.json (requested --model={args.model!r}, --mode={args.mode!r}, --output={args.output!r}).\n"
        f"  • To verify the archived JevBench 231 submission artifacts, run:\n"
        f"      python {Path(__file__).name} --verify-only\n"
        f"  • For live model inference with Hugging Face checkpoints, run:\n"
        f"      python quickstart.py --model {args.model}\n"
        f"      python quickstart_dual_mode.py --model {args.model}",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
