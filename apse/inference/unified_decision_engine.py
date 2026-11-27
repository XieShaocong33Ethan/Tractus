#!/usr/bin/env python3
"""apse/inference/unified_decision_engine.py

Unified Real-Model Decision Inference Engine for Tractus Foundation Models.
Adheres to Physical Truth Invariants & Multi-Scale INT4 Execution:
- Real PyTorch forward pass without mock or simulated outputs.
- Supports 0.8B, 2B, 4B, and 9B models in INT4 and BF16 modes.
- Single unified engine backing all four launch demos:
  1. Web/OS Navigation (discrete DOM candidate selection)
  2. Duplex Meeting Copilot (utterance intent and filler classification)
  3. Multi-Intent Customer Service (multi-head risk & handoff evaluation)
  4. Spatial Dynamics & System 1/2 Neural Throttle
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from apse.quantization.int4_engine import QuantizationStats, quantize_model_int4

logger = logging.getLogger("UnifiedDecisionEngine")

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
    format_chat_prompt,
    resolve_checkpoint_path,
    resolve_slot_token_ids,
)
from apse.inference.thermostasis_readout import (
    CanonicalDecisionResult,
    thermostasis_calibrate_logits,
    execute_canonical_infer,
)

# Import MPS acceleration ops
try:
    from apse.inference.mps_ops import patch_mps
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    patch_mps = None
    MPS_AVAILABLE = False


class UnifiedDecisionEngine:
    """
    Unified Real-Model Decision Inference Engine.
    Handles model instantiation, INT4 quantization, caching, and discrete forward passes.
    """

    _instances: Dict[Tuple[str, str, Optional[str]], "UnifiedDecisionEngine"] = {}

    def __init__(
        self,
        model_size: str = "0.8b",
        quant_mode: str = "int4",
        device: Optional[str] = None,
        group_size: int = 64,
        enable_weight_cache: bool = True,
        enable_mps_acceleration: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        model_size_clean = model_size.lower().strip()
        self.model_size = model_size_clean
        self.quant_mode = quant_mode.lower().strip()
        self.checkpoint_path = resolve_checkpoint_path(
            model_size=model_size_clean,
            quant_mode=self.quant_mode,
            custom_path=checkpoint_path,
        )
        self.group_size = group_size
        self.enable_weight_cache = enable_weight_cache

        # Device selection: auto-detect MPS on macOS (experimental)
        # Note: MPS may crash on Qwen3.5's gated-delta-rule 5D tensors
        # Fallback to optimized CPU with sliced projection for stability
        if device is None:
            # Prefer CPU for stability unless explicitly requested MPS
            self.device = "cpu"
            torch.set_num_threads(8)  # Optimize CPU threading
        elif device == "mps" and not MPS_AVAILABLE:
            logger.warning("MPS requested but not available, falling back to CPU")
            self.device = "cpu"
            torch.set_num_threads(8)
        else:
            self.device = device

        logger.info(f"🚀 Initializing UnifiedDecisionEngine [{self.model_size.upper()} | {self.quant_mode.upper()}] on {self.device}")
        logger.info(f"🔧 Optimization: INT4 quantization + sliced projection (avoid 248k vocab matmul)")
        load_start = time.perf_counter()

        # 1. Install MPS acceleration patch (experimental, only if explicitly using MPS)
        self.mps_patched = False
        if self.device == "mps" and patch_mps is not None and enable_mps_acceleration:
            logger.warning("⚠️  MPS acceleration is experimental for Qwen3.5 (may crash on 5D tensors)")
            self.mps_patched = patch_mps()
            if self.mps_patched:
                logger.info("✅ MPS acceleration patch installed successfully")

        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_path),
            trust_remote_code=True,
        )

        # 3. Load Model Base
        config = AutoConfig.from_pretrained(
            str(self.checkpoint_path),
            trust_remote_code=True,
        )

        # Critical: Load model on CPU first, then move to MPS after quantization
        # This avoids triggering MPS operators during model initialization
        load_device = "cpu" if self.device == "mps" else self.device

        self.raw_model = AutoModelForCausalLM.from_pretrained(
            str(self.checkpoint_path),
            config=config,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(load_device)
        self.raw_model.eval()

        # 4. Apply INT4 Block-wise Quantization if requested
        self.quant_stats: Optional[QuantizationStats] = None
        if self.quant_mode == "int4":
            logger.info(f"Applying block-wise INT4 quantization (group_size={group_size})...")
            self.model, self.quant_stats = quantize_model_int4(
                self.raw_model,
                group_size=group_size,
                exempt_modules=("lm_head", "embed_tokens", "norm", "layernorm", "mva_scorer", "decision_head"),
                enable_weight_cache=enable_weight_cache,
            )
            logger.info(
                f"INT4 Quantized: {self.quant_stats.original_size_mb:.1f}MB -> {self.quant_stats.quantized_size_mb:.1f}MB "
                f"({self.quant_stats.compression_ratio:.2f}x compression)"
            )
        else:
            self.model = self.raw_model

        # 5. Move to MPS if needed (after quantization on CPU)
        if self.device == "mps" and load_device == "cpu":
            logger.info("Moving quantized model to MPS device...")
            self.model = self.model.to("mps")
            torch.mps.synchronize()

        # 6. Extract lm_head weight for candidate token projection
        # Slice only candidate tokens instead of 248,320 full vocab
        self.lm_head_weight = self.model.lm_head.weight.detach()  # [vocab_size, hidden_dim]

        # Pre-build tensor buffers for candidate tokens
        self.greek_tensor = torch.tensor(list(GREEK_META_TOKENS.values()), dtype=torch.long, device=self.device)
        self.latin_tensor = torch.tensor(list(LATIN_CHOICE_MAP.values()), dtype=torch.long, device=self.device)

        self.load_latency_s = time.perf_counter() - load_start
        logger.info(f"✅ Model ready in {self.load_latency_s:.2f}s (MPS: {self.mps_patched})")

    @classmethod
    def get_instance(
        cls,
        model_size: str = "0.8b",
        quant_mode: str = "int4",
        device: Optional[str] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> "UnifiedDecisionEngine":
        """Singleton factory avoiding redundant checkpoint loads."""
        key = (model_size.lower().strip(), quant_mode.lower().strip(), str(checkpoint_path) if checkpoint_path else None)
        if key not in cls._instances:
            cls._instances[key] = cls(
                model_size=model_size.lower().strip(),
                quant_mode=quant_mode.lower().strip(),
                device=device,
                checkpoint_path=checkpoint_path,
            )
        return cls._instances[key]

    # =========================================================================
    # Task 1: Generic Discrete Choice Prediction
    # =========================================================================
    def predict_choice(
        self,
        prompt: str,
        candidate_slots: Union[Sequence[str], Dict[str, str]],
        use_sliced_projection: bool = True,
    ) -> DecisionChoiceResult:
        """
        Executes real PyTorch forward pass to select among candidate options.

        Args:
            prompt: Input prompt string
            candidate_slots: Either list of slot names or dict mapping slot -> text
            use_sliced_projection: If True, only project candidate tokens (avoid 248k vocab matmul)
        """
        t0 = time.perf_counter()

        if isinstance(candidate_slots, dict):
            slot_names = list(candidate_slots.keys())
            slot_texts = candidate_slots
        else:
            slot_names = list(candidate_slots)
            slot_texts = {s: s for s in slot_names}

        # Resolve token IDs for candidate slots (exact case consistency & collision check)
        token_ids = resolve_slot_token_ids(self.tokenizer, slot_names, log=logger)

        cand_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Forward through transformer backbone (without lm_head)
            hidden_states = self.model.model(**inputs).last_hidden_state  # [1, seq_len, hidden_dim]
            last_hidden = hidden_states[0, -1, :]  # [hidden_dim]

            if use_sliced_projection:
                # Sliced projection: only compute logits for candidate tokens
                cand_weights = self.lm_head_weight[cand_tensor]  # [num_cands, hidden_dim]
                cand_logits = F.linear(last_hidden, cand_weights).float()  # [num_cands]
            else:
                # Full projection (fallback)
                full_logits = self.model.lm_head(last_hidden)  # [vocab_size]
                cand_logits = full_logits[cand_tensor].float()

        cand_logits_list = [float(cand_logits[i].item()) for i in range(len(slot_names))]
        slot_label_texts = [slot_texts[s] for s in slot_names]
        cand_probs, centered_list, phase_zone, delta_z, delta_f, thermo_t = thermostasis_calibrate_logits(
            cand_logits_list, option_texts=slot_label_texts
        )

        raw_logits_dict = {slot_names[i]: cand_logits_list[i] for i in range(len(slot_names))}
        raw_probs_dict = {slot_names[i]: float(cand_probs[i]) for i in range(len(slot_names))}
        probs_dict = {slot_names[i]: round(cand_probs[i], 4) for i in range(len(slot_names))}

        best_idx = max(range(len(centered_list)), key=lambda i: centered_list[i])
        best_slot = slot_names[best_idx]
        best_text = slot_texts[best_slot]
        raw_confidence = float(cand_probs[best_idx])

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DecisionChoiceResult(
            chosen_slot=best_slot,
            chosen_text=best_text,
            probabilities=probs_dict,
            logits=raw_logits_dict,
            confidence=raw_confidence,
            latency_ms=round(latency_ms, 2),
            model_size=self.model_size,
            quant_mode=self.quant_mode,
            metadata={
                "phase_zone": phase_zone,
                "delta_z": round(delta_z, 4),
                "raw_delta_z": float(delta_z),
                "delta_F": round(delta_f, 4),
                "thermostat_temp": round(thermo_t, 4),
                "raw_confidence": raw_confidence,
                "raw_probabilities": raw_probs_dict,
                "centered_logits": {slot_names[i]: round(centered_list[i], 6) for i in range(len(slot_names))},
            },
        )

    # =========================================================================
    # Task 2: System 1 / 2 Meta-Routing Phase Governor
    # =========================================================================
    def predict_dual_mode_route(
        self,
        prompt: str,
        threshold_delta_z: float = 0.5,
        use_sliced_projection: bool = True,
    ) -> DualModeRouteResult:
        """
        Projects terminal logits onto Greek Meta-Tokens to dynamically route System 1 vs 2.

        Args:
            prompt: Input prompt
            threshold_delta_z: Decision threshold for z_alpha - z_epsilon
            use_sliced_projection: If True, only project 5 Greek tokens
        """
        t0 = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden_states = self.model.model(**inputs).last_hidden_state
            last_hidden = hidden_states[0, -1, :]

            if use_sliced_projection:
                # Sliced projection: only 5 Greek meta-tokens
                greek_weights = self.lm_head_weight[self.greek_tensor]  # [5, hidden_dim]
                greek_logits = F.linear(last_hidden, greek_weights).float()  # [5]
            else:
                full_logits = self.model.lm_head(last_hidden)
                greek_logits = full_logits[self.greek_tensor]

            greek_probs = F.softmax(greek_logits, dim=-1).cpu().tolist()

        symbols = list(GREEK_META_TOKENS.keys())
        logits_dict = {symbols[i]: float(greek_logits[i].item()) for i in range(len(symbols))}
        probs_dict = {symbols[i]: round(greek_probs[i], 4) for i in range(len(symbols))}

        # Phase transition delta_z: z_alpha (System 1) - z_epsilon (System 2)
        z_alpha = logits_dict.get("α", 0.0)
        z_epsilon = logits_dict.get("ε", 0.0)
        delta_z = z_alpha - z_epsilon

        # Route decision: if delta_z >= threshold, resolve instantly via System 1
        is_system1 = delta_z >= threshold_delta_z
        route_symbol = "α" if is_system1 else "ε"
        route_name = "System1_Direct" if is_system1 else "System2_Expanded"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DualModeRouteResult(
            route_symbol=route_symbol,
            route_name=route_name,
            delta_z=round(delta_z, 4),
            probabilities=probs_dict,
            logits=logits_dict,
            latency_ms=round(latency_ms, 2),
            is_system1=is_system1,
            model_size=self.model_size,
            quant_mode=self.quant_mode,
        )

    # =========================================================================
    # Task 3: Real Browser Navigation Step (Demo 1)
    # =========================================================================
    def step_web_navigation(
        self,
        dom_state: str,
        valid_actions: Sequence[str],
        step_index: int = 1,
    ) -> WebStepResult:
        """Executes a real model forward pass to select the next navigation action from the DOM."""
        slots = [chr(ord("A") + i) for i in range(len(valid_actions))]
        action_mapping = {slots[i]: valid_actions[i] for i in range(len(valid_actions))}
        options_rendered = "\n".join([f"{slots[i]}) {valid_actions[i]}" for i in range(len(valid_actions))])

        prompt = (
            f"Evidence: Web Application DOM State: {dom_state}\n"
            f"Step: {step_index}\n"
            f"Question: Which navigation action should the autonomous agent execute next to progress through the form?\n"
            f"Options:\n{options_rendered}\n"
            f"Answer:\n"
        )

        choice_res = self.predict_choice(prompt, action_mapping)
        is_done = "done" in choice_res.chosen_text.lower() or "submit" in choice_res.chosen_text.lower()

        return WebStepResult(
            chosen_action=choice_res.chosen_text,
            action_slot=choice_res.chosen_slot,
            confidence=choice_res.confidence,
            is_done=is_done,
            latency_ms=choice_res.latency_ms,
            step_index=step_index,
        )

    # =========================================================================
    # Task 4: Duplex Meeting Utterance Intent Classification (Demo 2)
    # =========================================================================
    def classify_meeting_utterance(self, utterance: str) -> MeetingUtteranceResult:
        """Classifies real-time speech slice into Filler vs Actionable Command/Question."""
        meeting_options = [
            'Conversational filler or casual agreement with no action (e.g., yeah, okay, mm-hmm, yeah true, right, sure)',
            'Substantive question seeking information, data, or code (e.g., what is the retention curve, what do we talk about, is the code accessible)',
            'Actionable task command or layout change instruction (e.g., check conversion trend, change the window contents, switch to summary)',
            'Substantive business statement or status update (e.g., sounds like you done some stuff)',
        ]
        prompt = (
            f"Task: Speech intent classification.\n"
            f"Utterance: \"{utterance}\"\n"
            f"Options:\n"
            f"A) {meeting_options[0]}\n"
            f"B) {meeting_options[1]}\n"
            f"C) {meeting_options[2]}\n"
            f"D) {meeting_options[3]}\n"
            f"Answer:\n"
        )
        slot_categories = {
            "A": "FILLER",
            "B": "QUESTION",
            "C": "COMMAND",
            "D": "STATEMENT",
        }
        res = self.predict_choice(prompt, slot_categories)
        cat = slot_categories.get(res.chosen_slot, "STATEMENT")

        return MeetingUtteranceResult(
            category=cat,
            is_filler=(cat == "FILLER"),
            confidence=res.confidence,
            latency_ms=res.latency_ms,
            probabilities=res.probabilities,
        )

    # =========================================================================
    # Task 5: Multi-Intent Customer Service & Escalation (Demo 3)
    # =========================================================================
    def analyze_service_complaint(self, complaint_text: str) -> ServiceComplaintResult:
        """Evaluates customer complaint across billing dispute, regulatory threat, handoff, and emotion."""
        t0 = time.perf_counter()

        # Dimension 1: Billing dispute
        p1 = (
            f"Evidence: Customer service conversation message: \"{complaint_text}\"\n"
            f"Question: Does this customer message contain an unauthorized fee, surprise billing charge, or unknown debit dispute?\n"
            f"Options:\n"
            f"A) Yes, explicit billing or unauthorized fee dispute\n"
            f"B) No billing dispute\n"
            f"Answer:\n"
        )
        r1 = self.predict_choice(p1, ["A", "B"])
        is_billing = (r1.chosen_slot == "A")

        # Dimension 2: Handoff demand
        p2 = (
            f"Evidence: Customer service conversation message: \"{complaint_text}\"\n"
            f"Question: Does the customer explicitly demand immediate transfer to a human customer representative / live specialist?\n"
            f"Options:\n"
            f"A) Yes, demands human representative transfer\n"
            f"B) No human transfer requested\n"
            f"Answer:\n"
        )
        r2 = self.predict_choice(p2, ["A", "B"])
        is_handoff = (r2.chosen_slot == "A")

        # Dimension 3: Regulatory / legal threat (zero specific agency names)
        p3 = (
            f"Evidence: Customer service conversation message: \"{complaint_text}\"\n"
            f"Question: Does the customer threaten formal regulatory agency complaint, legal action, or external compliance escalation?\n"
            f"Options:\n"
            f"A) Yes, threatens regulatory complaint or legal action\n"
            f"B) No regulatory threat\n"
            f"Answer:\n"
        )
        r3 = self.predict_choice(p3, ["A", "B"])
        is_threat = (r3.chosen_slot == "A")

        # Dimension 4: Emotional distress / agitation
        p4 = (
            f"Evidence: Customer service conversation message: \"{complaint_text}\"\n"
            f"Question: Does this customer message express intense anger, severe dissatisfaction, or emotional frustration?\n"
            f"Options:\n"
            f"A) Yes, high customer anger and severe frustration\n"
            f"B) Normal or mild tone\n"
            f"Answer:\n"
        )
        r4 = self.predict_choice(p4, ["A", "B"])
        is_emotion = (r4.chosen_slot == "A")

        # Compute composite risk score from positive-class ("A") risk probabilities
        billing_score = r1.probabilities.get("A", 0.0)
        handoff_score = r2.probabilities.get("A", 0.0)
        threat_score = r3.probabilities.get("A", 0.0)
        emotion_score = r4.probabilities.get("A", 0.0)
        risk_score = (
            0.35 * billing_score
            + 0.30 * handoff_score
            + 0.25 * threat_score
            + 0.10 * emotion_score
        )

        if risk_score >= 0.65 or is_threat:
            risk_level = "CRITICAL"
            needs_human = True
        elif risk_score >= 0.35 or is_handoff:
            risk_level = "HIGH"
            needs_human = True
        elif risk_score >= 0.20:
            risk_level = "MEDIUM"
            needs_human = False
        else:
            risk_level = "LOW"
            needs_human = False

        total_lat = (time.perf_counter() - t0) * 1000.0

        return ServiceComplaintResult(
            billing_dispute=is_billing,
            handoff_demanded=is_handoff,
            regulatory_threat=is_threat,
            risk_level=risk_level,
            needs_human=needs_human,
            composite_risk_score=round(risk_score, 4),
            latency_ms=round(total_lat, 2),
            intent_confidences={
                "billing_dispute": billing_score,
                "handoff_demanded": handoff_score,
                "regulatory_threat": threat_score,
                "emotional_distress": emotion_score,
            },
            emotional_distress=is_emotion,
            emotional_score=emotion_score,
        )

    def route_ivr_intent(self, text: str) -> IvrRoutingResult:
        """Evaluates customer utterance and maps to standard banking IVR functional menus."""
        prompt = (
            f"Evidence: Customer telephony voice request: \"{text}\"\n"
            f"Question: Which automated IVR functional menu should this customer intent route to?\n"
            f"Options:\n"
            f"A) Billing and Account Balance Inquiry (Menu 1)\n"
            f"B) Recent Transactions and Statement Details (Menu 3)\n"
            f"C) Installment Plans and Consumer Credit (Menu 2)\n"
            f"D) Emergency Card Lock and Fraud Protection (Menu 5)\n"
            f"E) Human Customer Support Specialist (Menu 0)\n"
            f"Answer:\n"
        )
        res = self.predict_choice(prompt, ["A", "B", "C", "D", "E"])
        mapping = {
            "A": ("MENU_1_BILLING_LIMIT", "Menu 1: Billing & Balance Inquiry", False),
            "B": ("MENU_3_TRANSACTIONS", "Menu 3: Recent Transactions & Statement", False),
            "C": ("MENU_2_INSTALLMENTS", "Menu 2: Installments & Consumer Credit", False),
            "D": ("MENU_5_EMERGENCY_LOCK", "Menu 5: Emergency Lock & Fraud Guard", False),
            "E": ("MENU_0_HUMAN_SPECIALIST", "Menu 0: Human Support Specialist", True),
        }
        key, name, is_human = mapping.get(res.chosen_slot, ("MENU_1_BILLING_LIMIT", "Menu 1", False))
        return IvrRoutingResult(
            chosen_slot=res.chosen_slot,
            menu_key=key,
            menu_name=name,
            confidence=res.confidence,
            is_human_handoff=is_human,
            probabilities=res.probabilities,
            latency_ms=res.latency_ms,
        )

    # =========================================================================
    # System 2: Full Vocabulary Expanded Generative Deliberation (PyTorch)
    # =========================================================================
    def generate_system2_expanded_thought(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Runs full-vocabulary autoregressive generative reasoning for System 2 via PyTorch."""
        t0 = time.perf_counter()
        prompt_to_use = format_chat_prompt(
            self.tokenizer,
            prompt,
            enable_thinking=True,
            fallback_chatml=True,
        )

        inputs = self.tokenizer(prompt_to_use, return_tensors="pt").to(self.device)
        do_sample = temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "expanded_thought": generated_text,
            "prompt_used": prompt_to_use,
            "latency_ms": round(latency_ms, 2),
            "max_tokens": max_tokens,
            "backend": "pytorch",
        }

    def infer(
        self,
        prompt: str,
        candidate_slots: Union[Sequence[str], Dict[str, str]],
        mode: str = "tractus",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        ratchet_off: bool = False,
    ) -> CanonicalDecisionResult:
        """Executes canonical `Tractus-decision` (Single-Pass), `Tractus` (Joint Dual-Mode), or `go` (Gap-Only)."""
        return execute_canonical_infer(
            self,
            prompt=prompt,
            candidate_slots=candidate_slots,
            mode=mode,
            max_tokens=max_tokens,
            temperature=temperature,
            ratchet_off=ratchet_off,
        )
