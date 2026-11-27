#!/usr/bin/env python3
"""
Tractus-decision Single-Pass Quickstart (~30 seconds)
-----------------------------------------------------
Demonstrates zero-token System 1 inference with Tractus-decision models using:
  1. Single causal forward pass (logits_to_keep=1, 0 generated tokens)
  2. Optional Binary Prior Shift (asymmetric K = 2 in Decision Index 0.3) -> Logit Softcapping (C = 30.0) -> Zero-Sum Centering
  3. 2D Analytic e-Thermostat probability calibration (zero validation-fitted temperature scalars)

Usage:
    python quickstart.py
    python quickstart.py --model Ethan33X/Tractus-decision-0.8B
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apse.inference.thermostasis_readout import thermostasis_calibrate_logits  # noqa: E402


def _apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    enable_thinking: bool = False,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _resolve_option_token_ids(tokenizer: AutoTokenizer, option_keys: list[str]) -> list[int]:
    token_ids: list[int] = []
    seen: dict[int, str] = {}
    for key in option_keys:
        if not key or not key.strip():
            raise ValueError("Option keys must be non-empty strings.")
        ids = tokenizer.encode(key, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Option key {key!r} tokenized to {len(ids)} tokens ({ids}); "
                "single-pass readout requires single-token option keys."
            )
        tok_id = ids[0]
        if tok_id in seen:
            raise ValueError(f"Option keys {seen[tok_id]!r} and {key!r} collide on token ID {tok_id}.")
        seen[tok_id] = key
        token_ids.append(tok_id)
    return token_ids


def run_single_pass(model_id: str, state: str, question: str, options: dict[str, str]) -> None:
    print(f"Loading Tractus-decision checkpoint: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
    ).eval()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a decision engine. Read the state, then answer the question "
                "by replying with exactly one option key from the list. Do not explain."
            ),
        },
        {
            "role": "user",
            "content": (
                f"State:\n{state}\n\n"
                f"Question:\n{question}\n\n"
                "Options:\n"
                + "\n".join(f"{k}: {v}" for k, v in options.items())
                + "\n\nReply with exactly one option key."
            ),
        },
    ]

    prompt = _apply_chat_template(tokenizer, messages, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    option_keys = list(options.keys())
    option_texts = [options[k] for k in option_keys]
    option_token_ids = _resolve_option_token_ids(tokenizer, option_keys)

    t0 = time.perf_counter()
    with torch.inference_mode():
        terminal_logits = model(input_ids=inputs["input_ids"], logits_to_keep=1).logits[0, -1]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    raw_scores = [float(terminal_logits[tok_id].item()) for tok_id in option_token_ids]
    probs, centered_scores, phase_zone, delta_z, delta_f, thermo_t = thermostasis_calibrate_logits(
        raw_scores, option_texts=option_texts
    )
    prob_map = {k: round(p, 4) for k, p in zip(option_keys, probs)}
    best_idx = max(range(len(centered_scores)), key=lambda i: centered_scores[i])
    best_key = option_keys[best_idx]

    print("-" * 64)
    print(f"Mode        : Tractus-decision (Single-Pass System 1, 0 generated tokens)")
    print(f"Prediction  : {best_key} -> {options[best_key]}")
    print(f"Confidence  : {prob_map[best_key]:.4f} (e-Thermostat T={thermo_t:.3f})")
    print(f"Phase Plane : {phase_zone} | delta_z={delta_z:.3f} | delta_F={delta_f:.3f}")
    print(f"Probs       : {prob_map}")
    print(f"Latency     : {latency_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tractus-decision Single-Pass Quickstart")
    parser.add_argument(
        "--model",
        type=str,
        default="Ethan33X/Tractus-decision-0.8B",
        help="HF repo or local path (e.g. Ethan33X/Tractus-decision-0.8B)",
    )
    args = parser.parse_args()

    sample_state = "My integration keeps failing with HTTP 401 since we rotated our API key this morning."
    sample_question = "Which support queue should handle this request?"
    sample_options = {
        "A": "Billing and subscription invoices",
        "B": "Technical API authentication support",
        "C": "New enterprise sales inquiries",
        "D": "Account closure and retention",
    }
    run_single_pass(args.model, sample_state, sample_question, sample_options)


if __name__ == "__main__":
    main()
