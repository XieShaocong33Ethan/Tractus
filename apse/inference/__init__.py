# apse/inference/__init__.py
"""Unified Inference and Real Model Execution Package for Tractus.

Uses lazy `__getattr__` for PyTorch (`UnifiedDecisionEngine`) and Apple Silicon MLX
(`MlxDecisionEngine`) backends so pure-Python modules (`thermostasis_readout`, `common`)
and MLX-only environments import cleanly without `torch` or `transformers` installed.
"""

from __future__ import annotations

from typing import Any

from apse.inference.common import (
    CHECKPOINT_REGISTRY,
    CHECKPOINT_REGISTRY_DECISION,
    CHECKPOINT_REGISTRY_INT4,
    GREEK_META_TOKENS,
    LATIN_CHOICE_MAP,
    DecisionChoiceResult,
    DualModeRouteResult,
    IvrRoutingResult,
    MeetingUtteranceResult,
    ServiceComplaintResult,
    WebStepResult,
    _is_hf_repo_id,
    resolve_checkpoint_path,
)
from apse.inference.thermostasis_readout import CanonicalDecisionResult

__all__ = [
    "UnifiedDecisionEngine",
    "MlxDecisionEngine",
    "CanonicalDecisionResult",
    "DecisionChoiceResult",
    "DualModeRouteResult",
    "WebStepResult",
    "MeetingUtteranceResult",
    "ServiceComplaintResult",
    "IvrRoutingResult",
    "CHECKPOINT_REGISTRY",
    "CHECKPOINT_REGISTRY_DECISION",
    "CHECKPOINT_REGISTRY_INT4",
    "LATIN_CHOICE_MAP",
    "GREEK_META_TOKENS",
    "resolve_checkpoint_path",
]


def __getattr__(name: str) -> Any:
    if name == "UnifiedDecisionEngine":
        from apse.inference.unified_decision_engine import UnifiedDecisionEngine
        return UnifiedDecisionEngine
    if name == "MlxDecisionEngine":
        try:
            from apse.inference.mlx_decision_engine import MlxDecisionEngine
            return MlxDecisionEngine
        except ImportError:
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
