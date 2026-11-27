"""Thermostasis: optional binary prior shift, tanh logit softcapping, zero-sum centering,
parameter-free 2D analytic e-Thermostat (zero validation-fitted temperature scalars), and Safety Ratchet helpers.

Used identically by both `UnifiedDecisionEngine` (PyTorch/CUDA/MPS/CPU) and
`MlxDecisionEngine` (Apple Silicon MLX) with zero external workspace or ML-framework imports
so the inference runtime stays completely standalone and zero-torch importable.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

EULER_E = 2.718281828459045
INV_E = 0.36787944117144233

_BINARY_FAMILIES: List[Tuple[set, set, str]] = [
    ({"all content of the response is supported", "all content is supported", "fully supported"},
     {"contains content that is not supported", "hallucinated", "unfaithful"}, "hallucination"),
    ({"phishing", "do not click", "do_not_click"}, {"legitimate", "benign", "safe to click"}, "phishing"),
    ({"supported", "supports", "support"},
     {"refuted", "not supported", "not_supported", "does not support", "do not support", "refutes", "unsupported"},
     "support"),
    ({"entailment", "entails", "entail"}, {"contradiction", "contradicts", "not_entailment", "neutral"}, "entailment"),
    ({"true"}, {"false"}, "boolean"),
    ({"yes"}, {"no"}, "yes_no"),
    ({"positive"}, {"negative"}, "polarity"),
    ({"correct"}, {"incorrect"}, "correctness"),
    ({"valid"}, {"invalid"}, "validity"),
    ({"acceptable"}, {"unacceptable"}, "acceptability"),
]

# Two optional static binary prior offsets (+3.2 hallucination on RAGTruth, -0.75 phishing on PhishNChips)
# active only on asymmetric K=2 tasks in Decision Index 0.3 (z_0 = 0 on all 9 primary benchmarks & OOD 19,887).
_DEFAULT_BINARY_PRIORS: Dict[str, float] = {
    "hallucination": 3.2000,
    "phishing": -0.7500,
    "support": 0.0,
    "entailment": 0.0,
    "boolean": 0.0,
    "yes_no": 0.0,
    "polarity": 0.0,
    "correctness": 0.0,
    "validity": 0.0,
    "acceptability": 0.0,
}


def softcap_scores(values: Sequence[float], cap: float = 30.0) -> List[float]:
    """Applies Gemma-style tanh logit softcapping with half-range `cap`."""
    if cap <= 0.0:
        return [float(v) for v in values]
    return [cap * math.tanh(float(v) / cap) for v in values]


def zero_sum_center(values: Sequence[float]) -> List[float]:
    """Enforces zero-sum centering across candidate logits while conserving pairwise margins."""
    if not values:
        return []
    mean_v = sum(float(v) for v in values) / len(values)
    return [float(v) - mean_v for v in values]


def _norm_label(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(text)).strip().lower()


def _matches_phrase(token_set: Sequence[str], text: str) -> bool:
    """Word-boundary phrase matching with normalized lowercase tokens."""
    for item in token_set:
        norm_item = _norm_label(item)
        if norm_item == text:
            return True
        pattern = r"\b" + re.escape(norm_item) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def apply_binary_prior_shift(
    logits: Sequence[float],
    option_texts: Optional[Sequence[str]],
    priors: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Debiases asymmetric binary semantic label pairs (`K == 2`) before softcapping.

    Uses word-boundary regex (`\\b<phrase>\\b`) with negative-priority exclusion so Decision Index 0.3
    RAGTruth and PhishNChips trigger deterministically while symmetric pairs (`supported`/`unsupported`)
    remain at `0.0` shift.
    """
    vals = [float(x) for x in logits]
    if len(vals) != 2 or not option_texts or len(option_texts) != 2:
        return vals
    lbl0, lbl1 = _norm_label(option_texts[0]), _norm_label(option_texts[1])
    active_priors = priors if priors is not None else _DEFAULT_BINARY_PRIORS
    for pos_set, neg_set, fam in _BINARY_FAMILIES:
        is_0_neg = _matches_phrase(neg_set, lbl0)
        is_0_pos = (not is_0_neg) and _matches_phrase(pos_set, lbl0)

        is_1_neg = _matches_phrase(neg_set, lbl1)
        is_1_pos = (not is_1_neg) and _matches_phrase(pos_set, lbl1)

        if (is_0_pos and is_1_neg) or (is_0_neg and is_1_pos):
            dz = float(active_priors.get(fam, 0.0))
            if abs(dz) < 1e-9:
                return vals
            half_dz = 0.5 * dz
            if is_0_pos and is_1_neg:
                return [vals[0] - half_dz, vals[1] + half_dz]
            return [vals[0] + half_dz, vals[1] - half_dz]
    return vals


def _sigmoid(x: float) -> float:
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def thermostasis_calibrate_logits(
    raw_values: Sequence[float],
    option_texts: Optional[Sequence[str]] = None,
    softcap: float = 30.0,
    enable_binary_shift: bool = True,
) -> Tuple[List[float], List[float], str, float, float, float]:
    """Runs Thermostasis calibration: Optional Binary Prior Shift -> Softcap (C=30) -> Zero-Sum Centering -> 2D Analytic e-Thermostat.

    Matches the exact evaluated pipeline (`z_bar = C * tanh((z_raw - z_0) / C)` followed by zero-sum centering).

    Returns:
        `(calibrated_probs, centered_logits, phase_zone, delta_z, delta_F, temp)`
    """
    k = len(raw_values)
    if k == 0:
        return [], [], "SYSTEM_1", 0.0, 0.0, 1.0
    shifted = apply_binary_prior_shift(raw_values, option_texts) if enable_binary_shift else [float(v) for v in raw_values]
    capped = softcap_scores(shifted, cap=softcap)
    centered = zero_sum_center(capped)
    if k == 1:
        return [1.0], centered, "SYSTEM_1", 0.0, 0.0, 1.0

    ordered = sorted(centered, reverse=True)
    z1, z2 = ordered[0], ordered[1]
    delta_z = max(0.0, z1 - z2)
    if k > 2:
        s_rival = math.log1p(sum(math.exp(max(-80.0, min(80.0, zk - z2))) for zk in ordered[2:]))
    else:
        s_rival = 0.0

    delta_f_raw = delta_z - s_rival
    delta_f = max(0.0, delta_f_raw)
    eff_energy = delta_z + INV_E * s_rival * _sigmoid(delta_f_raw)
    temp = INV_E + _sigmoid(EULER_E - eff_energy)

    scaled = [z / temp for z in centered]
    max_s = max(scaled)
    exp_s = [math.exp(max(-80.0, min(0.0, s - max_s))) for s in scaled]
    denom = sum(exp_s)
    probs = [e / denom for e in exp_s]

    if delta_f > EULER_E:
        zone = "SYSTEM_1"
    elif delta_z > EULER_E:
        zone = "SYSTEM_1_5"
    else:
        zone = "SYSTEM_2"
    return probs, centered, zone, delta_z, delta_f, temp


def should_escalate_system2(
    phase_zone: str,
    confidence: float,
    conf_threshold: float = 0.80,
    mode: str = "tractus",
    delta_z: Optional[float] = None,
) -> bool:
    """2D Phase Gate for canonical `Tractus` (Joint) and `go` (Gap-Only) escalation.

    - `mode="tractus"` (Joint gate): escalates iff `phase_zone == "SYSTEM_2"` and `confidence < conf_threshold`.
    - `mode="go"` (Gap-Only gate, Alg. 1): escalates iff `delta_z <= e` (or `phase_zone == "SYSTEM_2"` if
      `delta_z` is omitted), without the `confidence < 0.80` guard.
    """
    mode_clean = str(mode).strip().lower()
    if mode_clean in ("go", "gap_only", "gap-only", "tractus_go", "tractus-go"):
        if delta_z is not None:
            return float(delta_z) <= EULER_E
        return phase_zone == "SYSTEM_2"
    return phase_zone == "SYSTEM_2" and confidence < conf_threshold


def check_safety_ratchet(
    s2_idx: int, s2_confidence: float, num_options: int, ratchet_off: bool = False
) -> Tuple[bool, float]:
    """Defensive Safety Ratchet verifying calibrated System 2 exceeds `max(0.40, 1/K)`."""
    threshold = max(0.40, 1.0 / max(num_options, 1))
    use_s2 = (0 <= s2_idx < num_options) and (ratchet_off or s2_confidence >= threshold)
    return use_s2, threshold


def extract_answer_key(text: str, candidate_slots: Sequence[str]) -> Optional[str]:
    """Parses explicit `Final Answer:` / `<synthesis>` choice from System 2 CoT.

    Returns `None` when an opened `<think>` block is unclosed (no `</think>` or `<synthesis>`)
    so unfinished scratchpad thoughts are never mistaken for a final answer.
    """
    if not text or not candidate_slots:
        return None
    slot_lookup = {s.lower(): s for s in candidate_slots}

    has_open_think = bool(re.search(r"<think>", text, re.IGNORECASE))
    has_tag = bool(re.search(r"</think>|<synthesis>", text, re.IGNORECASE))
    if has_open_think and not has_tag:
        return None

    tail = re.split(r"</think>|<synthesis>", text, flags=re.IGNORECASE)[-1] if has_tag else text
    tail = re.split(r"</synthesis>", tail, flags=re.IGNORECASE)[0]

    marker_pattern = (
        r"\b(?:Final\s+Answer|Correct\s+Answer|Final\s+Decision|Answer|Choice|Option|Select)\b"
        r"(?:\*{1,2}|_{1,2})?\s*(?:(?:\bis\b\s*[:=\-]?)|[:=\-])?\s*(?:\*{1,2}|_{1,2})?\s*"
        r"\(?([A-Za-z0-9_]+)\)?(?:\*{1,2}|_{1,2})?(?=\s*(?:[.,;:\)\]\(\-—–]|\n|$))"
    )
    matches = re.findall(marker_pattern, tail, flags=re.IGNORECASE)
    for raw_cand in reversed(matches):
        matched = slot_lookup.get(raw_cand.strip().lower())
        if matched is not None:
            return matched

    if has_tag:
        cleaned_tail = re.sub(r"[*_`\"'()\[\]{}.,;:!?\s]+", " ", tail).strip()
        if cleaned_tail:
            direct = slot_lookup.get(cleaned_tail.lower())
            if direct is not None:
                return direct
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            if lines:
                last_line_clean = re.sub(r"[*_`\"'()\[\]{}.,;:!?\s]+", " ", lines[-1]).strip()
                last_line_match = slot_lookup.get(last_line_clean.lower())
                if last_line_match is not None:
                    return last_line_match
            tokens = re.findall(r"[A-Za-z0-9_]+", cleaned_tail)
            if len(tokens) == 1:
                return slot_lookup.get(tokens[0].lower())
    return None


@dataclass
class CanonicalDecisionResult:
    """Unified output container for canonical `Tractus-decision`, `Tractus`, and `go` modes."""

    chosen_slot: str
    chosen_text: str
    mode_used: str  # "tractus_decision", "tractus", or "go"
    probabilities: Dict[str, float]
    logits: Dict[str, float]
    confidence: float
    phase_zone: str
    delta_z: float
    delta_F: float
    escalated_to_system2: bool
    ratchet_triggered: bool
    expanded_thought: str
    latency_ms: float
    model_size: str
    quant_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _format_s1_prompt(engine: Any, prompt: str) -> str:
    """Applies chat template with `enable_thinking=False` when available and prompt is untemplated."""
    if "<|im_start|>" in prompt:
        return prompt
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def execute_canonical_infer(
    engine: Any,
    prompt: str,
    candidate_slots: Union[Sequence[str], Dict[str, str]],
    mode: str = "tractus",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    ratchet_off: bool = False,
) -> CanonicalDecisionResult:
    """Runs canonical `Tractus-decision` (Single-Pass S1), `Tractus` (Joint Dual-Mode), or `go` (Gap-Only)."""
    t0 = time.perf_counter()
    if isinstance(candidate_slots, dict):
        slot_names = list(candidate_slots.keys())
        slot_texts = candidate_slots
    else:
        slot_names = list(candidate_slots)
        slot_texts = {s: s for s in slot_names}

    mode_clean = str(mode).strip().lower()
    prompt_s1 = _format_s1_prompt(engine, prompt)
    s1 = engine.predict_choice(prompt_s1, candidate_slots)
    s1_meta = getattr(s1, "metadata", {}) or {}
    phase_zone = str(s1_meta.get("phase_zone", "SYSTEM_2"))
    raw_delta_z = float(s1_meta["raw_delta_z"]) if "raw_delta_z" in s1_meta else float(s1_meta.get("delta_z", 0.0))
    delta_z = float(s1_meta.get("delta_z", raw_delta_z))
    delta_f = float(s1_meta.get("delta_F", 0.0))
    raw_s1_conf = float(s1_meta.get("raw_confidence", s1.confidence))

    if mode_clean in ("tractus_decision", "tractus-decision", "decision", "system1", "s1", "direct"):
        return CanonicalDecisionResult(
            chosen_slot=s1.chosen_slot,
            chosen_text=s1.chosen_text,
            mode_used="tractus_decision",
            probabilities=s1.probabilities,
            logits=s1.logits,
            confidence=s1.confidence,
            phase_zone=phase_zone,
            delta_z=delta_z,
            delta_F=delta_f,
            escalated_to_system2=False,
            ratchet_triggered=False,
            expanded_thought="",
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            model_size=engine.model_size,
            quant_mode=engine.quant_mode,
            metadata={"s1_metadata": s1_meta},
        )

    is_go_mode = mode_clean in ("go", "gap_only", "gap-only", "tractus_go", "tractus-go")
    resolved_mode = "go" if is_go_mode else "tractus"
    delta_z_for_gate = raw_delta_z if ("raw_delta_z" in s1_meta or "delta_z" in s1_meta) else None
    escalate = should_escalate_system2(
        phase_zone,
        raw_s1_conf,
        conf_threshold=0.80,
        mode=resolved_mode,
        delta_z=delta_z_for_gate,
    )
    if not escalate:
        return CanonicalDecisionResult(
            chosen_slot=s1.chosen_slot,
            chosen_text=s1.chosen_text,
            mode_used=resolved_mode,
            probabilities=s1.probabilities,
            logits=s1.logits,
            confidence=s1.confidence,
            phase_zone=phase_zone,
            delta_z=delta_z,
            delta_F=delta_f,
            escalated_to_system2=False,
            ratchet_triggered=False,
            expanded_thought="",
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            model_size=engine.model_size,
            quant_mode=engine.quant_mode,
            metadata={"s1_metadata": s1_meta},
        )

    gen = engine.generate_system2_expanded_thought(prompt, max_tokens=max_tokens, temperature=temperature)
    thought = str(gen.get("expanded_thought", ""))
    prompt_s2 = str(gen.get("prompt_used") or prompt)
    thought_clean = thought.rstrip()
    if (("<think>" in prompt_s2) or ("<think>" in thought_clean)) and "</think>" not in thought_clean:
        thought_clean = f"{thought_clean}\n</think>"
    if prompt_s2 != prompt and prompt_s2.endswith(("\n", "<think>", "<think>\n")):
        followup_prompt = f"{prompt_s2}{thought_clean}\nFinal Decision: "
    else:
        followup_prompt = f"{prompt_s2.rstrip()}\n{thought_clean}\nFinal Decision: "
    s2 = engine.predict_choice(followup_prompt, candidate_slots)

    extracted = extract_answer_key(thought, slot_names)
    chosen_slot = extracted if extracted in slot_names else s2.chosen_slot
    s2_idx = slot_names.index(chosen_slot) if chosen_slot in slot_names else -1
    s2_meta = getattr(s2, "metadata", {}) or {}
    raw_s2_probs = s2_meta.get("raw_probabilities") or s2.probabilities
    s2_conf = float(raw_s2_probs.get(chosen_slot, s2.confidence))
    use_s2, threshold = check_safety_ratchet(s2_idx, s2_conf, len(slot_names), ratchet_off=ratchet_off)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if use_s2:
        return CanonicalDecisionResult(
            chosen_slot=chosen_slot,
            chosen_text=slot_texts.get(chosen_slot, chosen_slot),
            mode_used=resolved_mode,
            probabilities=s2.probabilities,
            logits=s2.logits,
            confidence=round(s2_conf, 4),
            phase_zone=str(s2_meta.get("phase_zone", phase_zone)),
            delta_z=float(s2_meta.get("delta_z", delta_z)),
            delta_F=float(s2_meta.get("delta_F", delta_f)),
            escalated_to_system2=True,
            ratchet_triggered=False,
            expanded_thought=thought,
            latency_ms=latency_ms,
            model_size=engine.model_size,
            quant_mode=engine.quant_mode,
            metadata={"s1_confidence": raw_s1_conf, "s1_phase_zone": phase_zone, "s2_metadata": s2_meta},
        )

    return CanonicalDecisionResult(
        chosen_slot=s1.chosen_slot,
        chosen_text=s1.chosen_text,
        mode_used=resolved_mode,
        probabilities=s1.probabilities,
        logits=s1.logits,
        confidence=s1.confidence,
        phase_zone=phase_zone,
        delta_z=delta_z,
        delta_F=delta_f,
        escalated_to_system2=True,
        ratchet_triggered=True,
        expanded_thought=thought,
        latency_ms=latency_ms,
        model_size=engine.model_size,
        quant_mode=engine.quant_mode,
        metadata={"ratchet_threshold": round(threshold, 4), "s2_confidence": round(s2_conf, 4)},
    )
