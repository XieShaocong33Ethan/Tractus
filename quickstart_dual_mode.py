#!/usr/bin/env python3
"""
Tractus Dual-Mode (System 1 + System 2) Quickstart
--------------------------------------------------
Demonstrates canonical `Tractus` inference (>=4B recommended for System 2):
  1. Single-pass calibrated System 1 readout (Optional Binary Prior Shift -> Softcap C=30 -> Zero-Sum Centering -> 2D e-Thermostat)
  2. Joint 2D Confidence-Entropy Phase Gate (phase_zone == SYSTEM_2 and conf < 0.80)
  3. Gated System 2 deliberation + Defensive Safety Ratchet (accepting y^{S2} iff p_{y^{S2}}^{S2} >= max(0.40, 1/K))

Usage:
    python quickstart_dual_mode.py
    python quickstart_dual_mode.py --model Ethan33X/Tractus-4B --max-new-tokens 8192
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

from apse.inference.thermostasis_readout import (  # noqa: E402
    thermostasis_calibrate_logits,
    check_safety_ratchet,
    extract_answer_key,
    should_escalate_system2,
)


def _apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    enable_thinking: bool,
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


def _resolve_option_token_ids(tokenizer: AutoTokenizer, keys: list[str]) -> list[int]:
    token_ids: list[int] = []
    seen: dict[int, str] = {}
    for key in keys:
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


def _score_options(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    options: dict[str, str],
):
    keys = list(options.keys())
    texts = [options[k] for k in keys]
    option_token_ids = _resolve_option_token_ids(tokenizer, keys)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        terminal_logits = model(input_ids=enc["input_ids"], logits_to_keep=1).logits[0, -1]
    raw_scores = [float(terminal_logits[tok_id].item()) for tok_id in option_token_ids]
    probs, centered, zone, dz, df, temp = thermostasis_calibrate_logits(raw_scores, option_texts=texts)
    best_idx = max(range(len(centered)), key=lambda i: centered[i])
    return keys, probs, zone, dz, df, temp, best_idx, enc


def run_dual_mode(
    model_id: str,
    state: str,
    question: str,
    options: dict[str, str],
    max_new_tokens: int = 8192,
    ratchet_off: bool = False,
) -> None:
    print(f"Loading Tractus Dual-Mode checkpoint: {model_id}")
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
            "content": "You are a decision engine. Analyze the state carefully and choose the best option key.",
        },
        {
            "role": "user",
            "content": (
                f"State:\n{state}\n\n"
                f"Question:\n{question}\n\n"
                "Options:\n" + "\n".join(f"{k}: {v}" for k, v in options.items())
            ),
        },
    ]
    prompt_s1 = _apply_chat_template(tokenizer, messages, enable_thinking=False)

    t0 = time.perf_counter()
    keys, s1_probs, zone, dz, df, temp, s1_idx, _ = _score_options(model, tokenizer, prompt_s1, options)
    s1_key = keys[s1_idx]
    s1_conf = s1_probs[s1_idx]

    escalate = should_escalate_system2(zone, s1_conf, conf_threshold=0.80)
    print("-" * 64)
    print(f"Mode        : Tractus (Dual-Mode System 1 + System 2)")
    print(f"S1 Pass     : {s1_key} (conf={s1_conf:.4f}, zone={zone}, delta_z={dz:.3f}, delta_F={df:.3f}, T={temp:.3f})")

    if not escalate:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"Gate Action : SYSTEM_1 Fast-Pass (0 generated reasoning tokens)")
        print(f"Final Answer: {s1_key} -> {options[s1_key]} | Latency: {elapsed_ms:.2f} ms")
        return

    print("Gate Action : SYSTEM_2 Escalation (Joint 2D Gate triggered)")
    prompt_s2 = _apply_chat_template(tokenizer, messages, enable_thinking=True)
    enc_s2 = tokenizer(prompt_s2, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc_s2,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_tokens = out[0][enc_s2["input_ids"].shape[1]:]
    thought = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    thought_clean = thought.rstrip()
    if "<think>" in prompt_s2 and "</think>" not in thought_clean:
        thought_clean = f"{thought_clean}\n</think>"
    followup = f"{prompt_s2}{thought_clean}\nFinal Decision: "
    _, s2_probs, _, _, _, _, s2_idx, _ = _score_options(model, tokenizer, followup, options)
    parsed_key = extract_answer_key(thought, keys)
    if parsed_key in keys:
        s2_idx = keys.index(parsed_key)
    s2_key = keys[s2_idx]
    s2_conf = s2_probs[s2_idx]

    use_s2, threshold = check_safety_ratchet(s2_idx, s2_conf, len(keys), ratchet_off=ratchet_off)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    final_key = s2_key if use_s2 else s1_key
    final_conf = s2_conf if use_s2 else s1_conf

    print(f"S2 Rollout  : {len(gen_tokens)} tokens | S2 choice={s2_key} (conf={s2_conf:.4f})")
    print(f"Ratchet     : {'ADOPTED S2' if use_s2 else f'FALLBACK S1 (conf < {threshold:.2f})'}")
    print(f"Final Answer: {final_key} -> {options[final_key]} | Conf: {final_conf:.4f} | Latency: {elapsed_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tractus Dual-Mode (System 1 + System 2) Quickstart")
    parser.add_argument("--model", type=str, default="Ethan33X/Tractus-4B", help="HF model repo or local path")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="System 2 generation budget")
    parser.add_argument("--ratchet-off", action="store_true", help="Disable Defensive Safety Ratchet")
    args = parser.parse_args()

    state = (
        "Ledger audit: Rule R1 denies travel reimbursements over $250/night unless "
        "approved by VP Finance or accompanied by an emergency airport-closure notice. "
        "Receipt shows $310/night at Airport Inn on Tuesday; airline cancellation notice "
        "confirms storm closure at 23:10, and VP Finance is out of office."
    )
    question = "Should the expense claim be reimbursed under policy?"
    options = {
        "A": "Approve reimbursement under the emergency airport-closure exception",
        "B": "Deny because nightly rate exceeds $250",
        "C": "Escalate for missing VP Finance signature only",
    }
    run_dual_mode(
        args.model,
        state,
        question,
        options,
        max_new_tokens=args.max_new_tokens,
        ratchet_off=args.ratchet_off,
    )


if __name__ == "__main__":
    main()
