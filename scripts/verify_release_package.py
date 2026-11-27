#!/usr/bin/env python3
"""
Automated Verification Suite for Tractus Release Package.
Validates directory integrity, JSON schemas, CJK purity, Model Card metadata,
packaging hygiene (no bytecode caches, secrets, or local paths), zero training
imports, and Python AST syntax.

Copyright (c) 2026 Ethan Xie (Shaocong Xie). All rights reserved.
Licensed under the Modified MIT License (see LICENSE).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

CJK_PATTERN = re.compile(r"[\u3000-\u303f\uff00-\uffef\u4e00-\u9fff]")

SECRET_OR_PATH_PATTERN = re.compile(
    r"(?:/Volumes/|/Users/|/root/|-----BEGIN [A-Z ]+ PRIVATE KEY-----|\bhf_[A-Za-z0-9]{20,}\b|\bghp_[A-Za-z0-9]{20,}\b|\bsk-[A-Za-z0-9]{20,}\b)"
)

FORBIDDEN_TRAINING_MODULES = (
    "torch.optim",
    "peft",
    "deepspeed",
    "trl",
    "wandb",
    "accelerate",
    "apse.training",
    "apse.data",
    "apse.experiments",
    "apse.gym",
    "apse.models",
)

FORBIDDEN_ARTIFACT_NAMES = {"__pycache__", ".DS_Store", ".env"}

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "MODEL_CARD.md",
    "quickstart.py",
    "quickstart_dual_mode.py",
    "model_cards/Tractus-0.8B.md",
    "model_cards/Tractus-2B.md",
    "model_cards/Tractus-4B.md",
    "model_cards/Tractus-9B.md",
    "model_cards/Tractus-decision-0.8B.md",
    "model_cards/Tractus-decision-2B.md",
    "model_cards/Tractus-decision-4B.md",
    "model_cards/Tractus-decision-9B.md",
    "model_cards/GITHUB_README.md",
    "demos/README.md",
    "demos/HOW_TO_RUN_WITH_YOUR_MODEL.md",
    "demos/run_web_ui.py",
    "demos/run_demo1_web_agent.py",
    "demos/run_demo2_meeting_sentinel.py",
    "demos/run_demo3_service_router.py",
    "demos/run_demo4_spatial_dynamics.py",
    "demos/demo1_web_automation/__init__.py",
    "demos/demo1_web_automation/mock_web_app.py",
    "demos/demo1_web_automation/automation_executor.py",
    "demos/demo1_web_automation/ultrajev_scenarios.py",
    "demos/demo2_duplex_meeting/__init__.py",
    "demos/demo2_duplex_meeting/duplex_engine.py",
    "demos/demo2_duplex_meeting/stream_simulator.py",
    "demos/demo3_multi_intent_service/__init__.py",
    "demos/demo3_multi_intent_service/intent_parser.py",
    "demos/demo3_multi_intent_service/handoff_router.py",
    "demos/demo5_jevball_tactics/__init__.py",
    "demos/demo5_jevball_tactics/tactical_scenarios.py",
    "demos/demo5_jevball_tactics/candidate_generator.py",
    "demos/demo5_jevball_tactics/radar_display.py",
    "demos/common/__init__.py",
    "demos/common/stopwatch_hud.py",
    "demos/web_ui/server.py",
    "demos/web_ui/engine_runner.py",
    "demos/web_ui/static/index.html",
    "demos/web_ui/static/demo1_standalone.html",
    "demos/web_ui/static/demo2_standalone.html",
    "demos/web_ui/static/demo3_standalone.html",
    "demos/web_ui/static/demo4_standalone.html",
    "demos/web_ui/static/tractus_theme.css",
    "demos/web_ui/static/styles.css",
    "demos/web_ui/static/audio/manifest.json",
    "apse/__init__.py",
    "apse/inference/__init__.py",
    "apse/inference/common.py",
    "apse/inference/thermostasis_readout.py",
    "apse/inference/unified_decision_engine.py",
    "apse/inference/mlx_decision_engine.py",
    "apse/inference/mps_ops.py",
    "apse/quantization/int4_engine.py",
    "benchmarks/jevbench_231/README.md",
    "benchmarks/jevbench_231/scores.json",
    "benchmarks/jevbench_231/eval_jevbench.py",
    "benchmarks/jevbench_231/SUBMISSION.md",
    "benchmarks/decision_index_0_3/README.md",
    "benchmarks/decision_index_0_3/scores.json",
    "benchmarks/decision_index_0_3/benchmark-summary.json",
    "benchmarks/decision_index_0_3/index.json",
    "benchmarks/decision_index_0_3/environment.json",
    "benchmarks/decision_index_0_3/Tractus-decision-0.8B/index.json",
    "benchmarks/decision_index_0_3/Tractus-decision-2B/index.json",
    "benchmarks/decision_index_0_3/Tractus-decision-4B/index.json",
    "benchmarks/decision_index_0_3/Tractus-decision-9B/index.json",
    "benchmarks/decision_index_0_3/SUBMISSION_DRAFT.md",
    "benchmarks/decision_index_0_3/results_dataset_spec.md",
    "scripts/verify_release_package.py",
    "scripts/upload_to_hf.py",
]


def check_file_existence(root: Path) -> tuple[int, list[str]]:
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    return len(REQUIRED_FILES) - len(missing), missing


def check_json_schemas(root: Path) -> tuple[int, list[str]]:
    errors = []
    valid = 0
    for jf in root.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(data, (dict, list)):
                errors.append(f"{jf.relative_to(root)}: root is not dict or list")
            else:
                valid += 1
        except Exception as e:
            errors.append(f"{jf.relative_to(root)}: JSON parse error: {e}")
    return valid, errors


def check_cjk_purity(root: Path) -> tuple[int, list[str]]:
    violations = []
    checked = 0
    for ext in ("*.md", "*.html", "*.css", "*.js", "*.py", "*.json"):
        for path in root.rglob(ext):
            checked += 1
            try:
                matches = CJK_PATTERN.findall(path.read_text(encoding="utf-8"))
                if matches:
                    violations.append(
                        f"{path.relative_to(root)}: {len(matches)} CJK chars (sample: {matches[:5]})"
                    )
            except Exception as e:
                violations.append(f"{path.relative_to(root)}: read error: {e}")
    return checked, violations


def check_model_card_guardrails(root: Path) -> tuple[int, list[str]]:
    errors = []
    passed = 0
    cards = list((root / "model_cards").glob("*.md")) + [root / "MODEL_CARD.md"]
    repo_root_card = root.parent / "MODEL_CARD.md"
    if repo_root_card.exists() and repo_root_card.resolve() != (root / "MODEL_CARD.md").resolve():
        cards.append(repo_root_card)

    for card in cards:
        if not card.exists():
            continue
        passed += 1
        card_label = "repo-root/MODEL_CARD.md" if card == repo_root_card else card.name
        text = card.read_text(encoding="utf-8")
        lower_text = text.lower()

        if "pipeline_tag: decision-model" not in text and card.name != "GITHUB_README.md":
            errors.append(f"{card_label}: Missing 'pipeline_tag: decision-model'")
        if card.name != "GITHUB_README.md":
            for required_yaml in ("license: other", "license_name: modified-mit", "license_link: license"):
                if required_yaml not in lower_text:
                    errors.append(f"{card_label}: Missing YAML license tag '{required_yaml}'")
        if not re.search(r"modified[- ]mit", lower_text):
            errors.append(f"{card_label}: Missing 'Modified MIT' license reference")
        if "@misc{xie2026tractus" not in text:
            errors.append(f"{card_label}: Missing standardized '@misc{{xie2026tractus' citation")

        bad_zeros = re.findall(r"0\.00%?\s+(?:PFR|Premature\s+DONE|unhandled)", text, re.IGNORECASE)
        if bad_zeros:
            errors.append(f"{card_label}: Violates integer zero rule: {bad_zeros}")

    return passed, errors


def check_packaging_hygiene_and_secrets(root: Path) -> tuple[int, list[str]]:
    violations = []
    checked = 0
    self_path = Path(__file__).resolve()

    for path in root.rglob("*"):
        if path.name in FORBIDDEN_ARTIFACT_NAMES or path.suffix in {".pyc", ".pyo"}:
            violations.append(f"{path.relative_to(root)}: stray build/cache artifact")

    for ext in ("*.md", "*.html", "*.css", "*.js", "*.py", "*.json"):
        for path in root.rglob(ext):
            if path.resolve() == self_path:
                continue
            checked += 1
            try:
                matches = SECRET_OR_PATH_PATTERN.findall(path.read_text(encoding="utf-8"))
                if matches:
                    violations.append(
                        f"{path.relative_to(root)}: secret or local host path detected ({matches[:3]})"
                    )
            except Exception as e:
                violations.append(f"{path.relative_to(root)}: read error: {e}")

    return checked, violations


def check_zero_training_imports(root: Path) -> tuple[int, list[str]]:
    violations = []
    checked = 0
    self_path = Path(__file__).resolve()
    for pf in root.rglob("*.py"):
        if pf.resolve() == self_path:
            continue
        checked += 1
        try:
            tree = ast.parse(pf.read_text(encoding="utf-8"))
        except Exception as e:
            violations.append(f"{pf.relative_to(root)}: AST parse error: {e}")
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for mod in modules:
                if any(mod == f or mod.startswith(f"{f}.") for f in FORBIDDEN_TRAINING_MODULES):
                    violations.append(f"{pf.relative_to(root)}: forbidden training import '{mod}'")
    return checked, violations


def check_python_syntax(root: Path) -> tuple[int, list[str]]:
    errors = []
    valid = 0
    for pf in root.rglob("*.py"):
        try:
            ast.parse(pf.read_text(encoding="utf-8"))
            valid += 1
        except Exception as e:
            errors.append(f"{pf.relative_to(root)}: SyntaxError: {e}")
    return valid, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automated verification suite for the Tractus Release Package "
            "(required files, JSON schemas, CJK purity, Model Card metadata, "
            "packaging hygiene, zero training imports, and Python AST syntax)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the release_package root directory (default: auto-detected).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    print("==================================================")
    print("  Tractus Release Package Automated Verification  ")
    print(f"  Package Root: {root}")
    print("==================================================\n")

    all_passed = True

    found, missing = check_file_existence(root)
    print(f"[1/7] Required File Inventory: {found}/{len(REQUIRED_FILES)} files present")
    if missing:
        all_passed = False
        for m in missing:
            print(f"  FAIL Missing: {m}")
    else:
        print("  PASS All required release assets exist.")

    valid_json, json_errors = check_json_schemas(root)
    print(f"\n[2/7] JSON Schema Validation: {valid_json} files checked")
    if json_errors:
        all_passed = False
        for je in json_errors:
            print(f"  FAIL {je}")
    else:
        print("  PASS All JSON schemas parsed successfully.")

    checked_cjk, cjk_violations = check_cjk_purity(root)
    print(f"\n[3/7] Linguistic Purity Check (0 CJK): {checked_cjk} files scanned")
    if cjk_violations:
        all_passed = False
        for cv in cjk_violations:
            print(f"  FAIL {cv}")
    else:
        print(f"  PASS Strict 0 CJK purity verified across all {checked_cjk} files.")

    cards_checked, card_errors = check_model_card_guardrails(root)
    print(f"\n[4/7] Model Card Guardrails: {cards_checked} cards audited")
    if card_errors:
        all_passed = False
        for ce in card_errors:
            print(f"  FAIL {ce}")
    else:
        print("  PASS All Model Cards satisfy pipeline_tag, Modified MIT License, Citation, and integer zero rule.")

    checked_hyg, hyg_violations = check_packaging_hygiene_and_secrets(root)
    print(f"\n[5/7] Packaging Hygiene & Secrets Audit: {checked_hyg} files scanned")
    if hyg_violations:
        all_passed = False
        for hv in hyg_violations:
            print(f"  FAIL {hv}")
    else:
        print(f"  PASS Zero __pycache__/.pyc/.DS_Store, secrets, or local host paths across {checked_hyg} files.")

    checked_tr, tr_violations = check_zero_training_imports(root)
    print(f"\n[6/7] Zero Training Imports Audit: {checked_tr} Python files scanned")
    if tr_violations:
        all_passed = False
        for tv in tr_violations:
            print(f"  FAIL {tv}")
    else:
        print(f"  PASS Zero training-module imports across {checked_tr} Python files.")

    valid_py, py_errors = check_python_syntax(root)
    print(f"\n[7/7] Python Syntax & AST Parse: {valid_py} scripts verified")
    if py_errors:
        all_passed = False
        for pe in py_errors:
            print(f"  FAIL {pe}")
    else:
        print(f"  PASS All {valid_py} Python scripts compile cleanly.")

    print("\n==================================================")
    if all_passed:
        print("  RESULT: ALL 7 GATES PASSED (READY FOR RELEASE)")
        print("==================================================")
        sys.exit(0)
    else:
        print("  RESULT: VERIFICATION FAILED (SEE ERRORS ABOVE)")
        print("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
