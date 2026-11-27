"""Shared pure-Python types, checkpoint registries, and tokenization helpers for Tractus inference.

Kept strictly dependency-free (zero `torch`, zero `transformers`, zero `mlx`) so standalone
`apse.inference.thermostasis_readout` and Apple Silicon `MlxDecisionEngine` run cleanly without
PyTorch or Transformers installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("TractusInferenceCommon")

# Physical Checkpoint Registry (Default Canonical Models)
CHECKPOINT_REGISTRY: Dict[str, str] = {
    "0.8b": "Ethan33X/Tractus-0.8B",
    "2b": "Ethan33X/Tractus-2B",
    "4b": "Ethan33X/Tractus-4B",
    "9b": "Ethan33X/Tractus-9B",
}

CHECKPOINT_REGISTRY_DECISION: Dict[str, str] = {
    "0.8b": "Ethan33X/Tractus-decision-0.8B",
    "2b": "Ethan33X/Tractus-decision-2B",
    "4b": "Ethan33X/Tractus-decision-4B",
    "9b": "Ethan33X/Tractus-decision-9B",
}

# Backward-compatible alias resolving to canonical Hugging Face repositories
# (PyTorch quantizes on the fly when quant_mode="int4"; MLX loads local INT4 dirs if present or HF Hub weights).
CHECKPOINT_REGISTRY_INT4: Dict[str, str] = dict(CHECKPOINT_REGISTRY)

# Standard Uppercase Latin Choice Tokens [A..H]
LATIN_CHOICE_MAP: Dict[str, int] = {
    "A": 32,
    "B": 33,
    "C": 34,
    "D": 35,
    "E": 36,
    "F": 37,
    "G": 38,
    "H": 39,
}

# Frozen 5 Greek Meta-Tokens
GREEK_META_TOKENS: Dict[str, int] = {
    "α": 18374,  # System 1 Direct Fast Action
    "β": 49531,  # Fast Structured Text
    "γ": 57369,  # Interactive Clarification
    "δ": 82435,  # Structured Refusal / Out of Bounds
    "ε": 29159,  # System 2 Expanded Metacognitive Thought
}


def _is_hf_repo_id(path_str: str) -> bool:
    """Returns True if path_str looks like a Hugging Face Hub repo ID ('owner/repo')."""
    s = path_str.strip()
    if not s or s.startswith((".", "/", "~")) or "\\" in s:
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts)


def resolve_checkpoint_path(
    model_size: str = "0.8b",
    quant_mode: str = "int4",
    custom_path: Optional[Union[str, Path]] = None,
    variant: str = "tractus",
) -> Path:
    """Resolves model checkpoint directory or Hugging Face Hub repo ID with multi-tier priority.

    Allows external users to seamlessly connect their own checkpoints via:
    1. Direct explicit argument: custom_path (local directory or Hugging Face repo ID)
    2. Environment variable: TRACTUS_CHECKPOINT
    3. Project local directory: ./checkpoints/
    4. Canonical Hugging Face Hub registry lookup (Ethan33X/Tractus-*)
    """
    # Tier 1: Explicit custom path (local path or HF repo ID)
    if custom_path:
        raw_str = str(custom_path).strip()
        p = Path(raw_str).expanduser().resolve()
        if p.exists():
            return p
        if _is_hf_repo_id(raw_str):
            return Path(raw_str)
        raise FileNotFoundError(f"Specified custom checkpoint path does not exist: {custom_path}")

    # Tier 2: Environment variables
    for env_var in ("TRACTUS_CHECKPOINT", f"TRACTUS_{model_size.upper().replace('.', '_')}_CHECKPOINT"):
        env_val = os.environ.get(env_var)
        if env_val:
            p = Path(env_val).expanduser().resolve()
            if p.exists():
                return p
            if _is_hf_repo_id(env_val):
                return Path(env_val.strip())
            logger.warning(f"Environment variable {env_var}={env_val} does not exist, falling back...")

    # Tier 3: Local workspace ./checkpoints directory
    proj_root = Path(__file__).resolve().parents[2]
    clean_size = model_size.lower().strip()
    underscored_size = clean_size.replace(".", "_")
    is_int4 = quant_mode.lower().strip() in ("int4", "mlx_int4")

    local_candidates = [
        proj_root / "checkpoints" / (f"tractus-{clean_size}-mlx-int4" if is_int4 else f"tractus-{clean_size}"),
        proj_root / "checkpoints" / f"tractus-{clean_size}",
        proj_root / "checkpoints" / (f"tractus_{clean_size}-mlx-int4" if is_int4 else f"tractus_{clean_size}"),
        proj_root / "checkpoints" / f"tractus_{clean_size}",
        proj_root / "checkpoints" / (f"tractus-{underscored_size}-mlx-int4" if is_int4 else f"tractus-{underscored_size}"),
        proj_root / "checkpoints" / f"tractus_{underscored_size}",
        proj_root / "checkpoints" / clean_size,
    ]
    for cand in local_candidates:
        if cand.exists():
            return cand

    # Tier 4: Environment DATA_DIR or TRACTUS_DATA_DIR fallback
    data_dir = os.environ.get("TRACTUS_DATA_DIR", os.environ.get("DATA_DIR", ""))
    if data_dir and Path(data_dir).exists():
        fallback_candidates = [
            Path(data_dir) / "checkpoints" / (f"tractus-{clean_size}-mlx-int4" if is_int4 else f"tractus-{clean_size}"),
            Path(data_dir) / "checkpoints" / f"tractus-{clean_size}",
            Path(data_dir) / "checkpoints" / f"tractus_{clean_size}",
        ]
        for cand in fallback_candidates:
            if cand.exists():
                return cand

    # Tier 5: Canonical Hugging Face Hub registry resolution
    # Always resolve to canonical HF repositories (Ethan33X/Tractus-* or Ethan33X/Tractus-decision-*);
    # PyTorch quantizes on the fly when quant_mode="int4".
    is_decision = str(variant).strip().lower() in ("tractus_decision", "tractus-decision", "decision")
    registry = CHECKPOINT_REGISTRY_DECISION if is_decision else CHECKPOINT_REGISTRY
    if clean_size in registry:
        reg_id = registry[clean_size]
        reg_path = Path(reg_id)
        if reg_path.exists():
            return reg_path
        return reg_path

    raise FileNotFoundError(
        f"Unable to find checkpoint for size '{model_size}' (quant='{quant_mode}').\n"
        f"Please provide your model path via --checkpoint <path> or export TRACTUS_CHECKPOINT=<path>."
    )


def resolve_slot_token_ids(
    tokenizer: Any,
    slot_names: Sequence[str],
    log: Optional[logging.Logger] = None,
) -> List[int]:
    """Resolves candidate slot keys to single token IDs with exact case consistency and collision checks.

    - Exact uppercase keys in `LATIN_CHOICE_MAP` (`"A"`..`"H"`) use the canonical Qwen3.5 ID map;
      all other keys (including lowercase `"a"`..`"z"` and `"I"`..`"Z"`) are encoded via
      `tokenizer.encode(slot, add_special_tokens=False)` so case handling is uniform.
    - Emits a warning if a key tokenizes into multiple tokens and raises `ValueError` if two
      distinct slots collide on the same leading token ID.
    """
    active_logger = log or logger
    token_ids: List[int] = []
    seen: Dict[int, str] = {}
    for s in slot_names:
        clean_s = str(s).strip()
        if not clean_s:
            raise ValueError("Candidate slot keys must be non-empty strings.")
        if clean_s in LATIN_CHOICE_MAP:
            tok_id = LATIN_CHOICE_MAP[clean_s]
        else:
            try:
                encoded = tokenizer.encode(s, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(s)
            if not encoded:
                raise ValueError(f"Candidate slot key {s!r} tokenized to 0 tokens.")
            if len(encoded) > 1:
                active_logger.warning(
                    "Candidate slot key %r tokenized to %d tokens (%s); using leading token ID %d. "
                    "Prefer distinct single-token keys (e.g., A..Z) for single-pass readout.",
                    s,
                    len(encoded),
                    encoded,
                    encoded[0],
                )
            tok_id = int(encoded[0])
        if tok_id in seen:
            raise ValueError(
                f"Candidate slot keys {seen[tok_id]!r} and {s!r} collide on token ID {tok_id}. "
                "Single-pass readout requires distinct single-token option keys."
            )
        seen[tok_id] = s
        token_ids.append(tok_id)
    return token_ids


def format_chat_prompt(
    tokenizer: Any,
    prompt: str,
    enable_thinking: bool = False,
    fallback_chatml: bool = False,
) -> str:
    """Formats a raw prompt with the tokenizer's chat template (safe `enable_thinking` fallback)."""
    if "<|im_start|>" in prompt:
        return prompt
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
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
    if fallback_chatml:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


@dataclass
class DecisionChoiceResult:
    """Standardized discrete decision output container."""
    chosen_slot: str
    chosen_text: str
    probabilities: Dict[str, float]
    logits: Dict[str, float]
    confidence: float
    latency_ms: float
    model_size: str
    quant_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DualModeRouteResult:
    """Tractus System 1 / 2 phase transition output container."""
    route_symbol: str  # 'α' or 'ε'
    route_name: str    # 'System1_Direct' or 'System2_Expanded'
    delta_z: float     # z_alpha - z_epsilon
    probabilities: Dict[str, float]
    logits: Dict[str, float]
    latency_ms: float
    is_system1: bool
    model_size: str
    quant_mode: str


@dataclass
class WebStepResult:
    """Real browser navigation step output container."""
    chosen_action: str
    action_slot: str
    confidence: float
    is_done: bool
    latency_ms: float
    step_index: int


@dataclass
class MeetingUtteranceResult:
    """Real duplex meeting utterance classification container."""
    category: str        # 'FILLER', 'QUESTION', 'COMMAND', 'STATEMENT'
    is_filler: bool
    confidence: float
    latency_ms: float
    probabilities: Dict[str, float]


@dataclass
class ServiceComplaintResult:
    """Real customer service multi-intent evaluation container."""
    billing_dispute: bool
    handoff_demanded: bool
    regulatory_threat: bool
    risk_level: str      # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    needs_human: bool
    composite_risk_score: float
    latency_ms: float
    intent_confidences: Dict[str, float]
    emotional_distress: bool = False
    emotional_score: float = 0.0


@dataclass
class IvrRoutingResult:
    """Real banking/credit card IVR conversational menu routing container."""
    chosen_slot: str        # 'A', 'B', 'C', 'D', 'E'
    menu_key: str           # 'MENU_1_BILLING_LIMIT', etc.
    menu_name: str          # e.g. 'Menu 1: Account Balance & Credit Limit Inquiry'
    confidence: float       # 0.0 .. 1.0
    is_human_handoff: bool  # True if Menu 0 (Human Specialist)
    probabilities: Dict[str, float]
    latency_ms: float
