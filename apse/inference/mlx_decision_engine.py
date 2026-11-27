"""apse/inference/mlx_decision_engine.py

High-Performance Native Apple Silicon (Metal GPU) Inference Runtime via MLX.
Eliminates autoregressive decoding latency in System 1 via single-pass prefill (0 generated tokens, 99.9% readout cut).
Eliminates PyTorch MPS Graph 5D tensor compiler crash by running natively in MLX Metal shaders.
Supports 255-slot decision manifold (MAX_OPTIONS = 255) for System 1 and full vocabulary for System 2.
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import mlx.core as mx
    import mlx_lm
    import mlx_lm.models.qwen3_5
    from mlx_lm.utils import load_model, load_tokenizer

    # Alias Qwen3.5 text architecture in MLX registry
    sys.modules["mlx_lm.models.qwen3_5_text"] = mlx_lm.models.qwen3_5
    _HAS_MLX = True
except ImportError:
    mx = None
    mlx_lm = None
    load_model = None
    load_tokenizer = None
    _HAS_MLX = False

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
    format_chat_prompt,
    resolve_checkpoint_path,
    resolve_slot_token_ids,
)
from apse.inference.thermostasis_readout import (
    CanonicalDecisionResult,
    thermostasis_calibrate_logits,
    execute_canonical_infer,
)

logger = logging.getLogger("MlxDecisionEngine")

MAX_OPTIONS: int = 255


class MlxDecisionEngine:
    """High-Performance Native Apple Silicon GPU Decision Engine powered by MLX."""

    _instances: Dict[str, "MlxDecisionEngine"] = {}

    def __init__(
        self,
        model_size: str = "0.8b",
        quant_mode: str = "int4",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        if not _HAS_MLX:
            raise ImportError(
                "MlxDecisionEngine requires Apple Silicon Metal GPU and the 'mlx' / 'mlx_lm' packages. "
                "For Linux, Windows, or NVIDIA CUDA systems, use UnifiedDecisionEngine with PyTorch."
            )

        model_size_clean = model_size.lower().strip()
        quant_clean = quant_mode.lower().strip()
        self.model_size = model_size_clean

        self.checkpoint_path = resolve_checkpoint_path(
            model_size=model_size_clean,
            quant_mode=quant_clean,
            custom_path=checkpoint_path,
        )

        logger.info(f"🚀 Initializing MlxDecisionEngine [{self.model_size.upper()} | {quant_clean.upper()}] on Apple Silicon GPU (Metal)...")
        logger.info(f"📁 Checkpoint path: {self.checkpoint_path}")
        load_start = time.perf_counter()

        if self.checkpoint_path.exists():
            self.model, self.config = load_model(self.checkpoint_path, strict=False)
            self.tokenizer = load_tokenizer(self.checkpoint_path)
        else:
            repo_id = str(self.checkpoint_path)
            logger.info(f"🌐 Resolving Hugging Face Hub checkpoint for MLX: {repo_id}")
            try:
                from huggingface_hub import snapshot_download

                local_dir = Path(snapshot_download(repo_id=repo_id))
                self.model, self.config = load_model(local_dir, strict=False)
                self.tokenizer = load_tokenizer(local_dir)
            except ImportError:
                loaded = mlx_lm.load(repo_id, return_config=True)
                self.model, self.tokenizer, self.config = loaded

        # Extract backbone and lm_head weight for sliced linear projection
        self.text_model = self.model.language_model if hasattr(self.model, "language_model") else self.model
        if hasattr(self.text_model, "lm_head"):
            head_module = self.text_model.lm_head
        else:
            head_module = self.text_model.model.embed_tokens

        # Check if head is quantized (e.g. QuantizedEmbedding or QuantizedLinear)
        if hasattr(head_module, "scales") and hasattr(head_module, "weight"):
            logger.info("Dequantizing head weight for high-speed candidate-sliced readout...")
            self.lm_head_weight = mx.dequantize(
                head_module.weight,
                head_module.scales,
                getattr(head_module, "biases", None),
                group_size=head_module.group_size,
                bits=head_module.bits,
            )
        else:
            self.lm_head_weight = head_module.weight

        # Detect quantization mode
        is_quant_cfg = (
            getattr(self.config, "quantization", None)
            or getattr(self.config, "quantization_config", None)
            or (
                isinstance(self.config, dict)
                and bool(self.config.get("quantization") or self.config.get("quantization_config"))
            )
        )
        self.quant_mode = (
            "mlx_int4"
            if is_quant_cfg or "INT4" in str(self.checkpoint_path)
            else "mlx_fp16"
        )

        # Build 255-Slot Decision Table Tokens & Pre-cache Sliced Manifold Weights W_255 in L2 Cache
        self.slot_tokens = self._build_255_slot_tokens()
        self.manifold_token_indices = mx.array(self.slot_tokens, dtype=mx.int32)
        self.manifold_weights = self.lm_head_weight[self.manifold_token_indices]  # [255, H]

        self.greek_token_ids = mx.array(list(GREEK_META_TOKENS.values()), dtype=mx.int32)
        self.greek_symbols = list(GREEK_META_TOKENS.keys())
        self.greek_weights = self.lm_head_weight[self.greek_token_ids]  # [5, H]

        # Pre-warm GPU with a dummy forward pass through backbone and sliced projection
        dummy_ids = mx.array([[self.slot_tokens[0]]])
        hidden = self.text_model.model(dummy_ids)
        out = hidden[:, -1, :] @ self.manifold_weights[:4].T
        mx.eval(out)

        self.load_latency_s = time.perf_counter() - load_start
        logger.info(f"✅ MlxDecisionEngine [{self.quant_mode.upper()}] ready in {self.load_latency_s:.2f}s on Metal GPU.")

    def _build_255_slot_tokens(self) -> List[int]:
        """Constructs canonical 255-slot token table (A..Z + single-token 2-letter combos)."""
        import string
        letters_list = list(string.ascii_uppercase)
        combos = [a + b for a in letters_list for b in letters_list]
        all_candidates = letters_list + combos
        slot_tokens: List[int] = []
        for name in all_candidates:
            try:
                encoded = self.tokenizer.encode(name, add_special_tokens=False)
            except TypeError:
                encoded = self.tokenizer.encode(name)
            if len(encoded) == 1:
                slot_tokens.append(encoded[0])
            if len(slot_tokens) == MAX_OPTIONS:
                break
        return slot_tokens

    @classmethod
    def get_instance(
        cls,
        model_size: str = "0.8b",
        quant_mode: str = "int4",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> "MlxDecisionEngine":
        m_clean = model_size.lower().strip()
        q_clean = quant_mode.lower().strip()
        key = f"{m_clean}_{q_clean}_{checkpoint_path}" if checkpoint_path else f"{m_clean}_{q_clean}"
        if key not in cls._instances:
            cls._instances[key] = cls(model_size=m_clean, quant_mode=q_clean, checkpoint_path=checkpoint_path)
        return cls._instances[key]

    # =========================================================================
    # Task 1: Discrete Choice (System 1 Single-Pass Decision)
    # =========================================================================
    def predict_choice(
        self,
        prompt: str,
        candidate_slots: Union[Sequence[str], Dict[str, str]],
    ) -> DecisionChoiceResult:
        """Executes native Metal GPU forward pass to select among candidate options."""
        t0 = time.perf_counter()

        if isinstance(candidate_slots, dict):
            slot_names = list(candidate_slots.keys())
            slot_texts = candidate_slots
        else:
            slot_names = list(candidate_slots)
            slot_texts = {s: s for s in slot_names}

        # Map slot names to token IDs (add_special_tokens=False, exact case consistency & collision checks)
        token_ids = resolve_slot_token_ids(self.tokenizer, slot_names, log=logger)

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        hidden = self.text_model.model(input_ids)
        last_hidden = hidden[:, -1, :]  # [1, H]

        cand_tensor = mx.array(token_ids, dtype=mx.int32)
        cand_weights = self.lm_head_weight[cand_tensor]  # [K, H] sliced linear projection
        cand_logits_mx = (last_hidden @ cand_weights.T)[0]  # [K]
        mx.eval(cand_logits_mx)

        cand_logits_list = [float(cand_logits_mx[i].item()) for i in range(len(token_ids))]
        slot_label_texts = [slot_texts[s] for s in slot_names]
        cand_probs_list, centered_list, phase_zone, delta_z, delta_f, thermo_t = thermostasis_calibrate_logits(
            cand_logits_list, option_texts=slot_label_texts
        )

        raw_logits_dict = {slot_names[i]: cand_logits_list[i] for i in range(len(slot_names))}
        raw_probs_dict = {slot_names[i]: float(cand_probs_list[i]) for i in range(len(slot_names))}
        probs_dict = {slot_names[i]: round(cand_probs_list[i], 4) for i in range(len(slot_names))}

        best_idx = max(range(len(centered_list)), key=lambda i: centered_list[i])
        best_slot = slot_names[best_idx]
        best_text = slot_texts[best_slot]
        raw_confidence = float(cand_probs_list[best_idx])

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
    # Task 2: System 1 / 2 Meta-Routing (Greek Tokens α vs ε)
    # =========================================================================
    def predict_dual_mode_route(
        self,
        prompt: str,
        threshold_delta_z: float = 0.5,
    ) -> DualModeRouteResult:
        """Projects terminal logits onto Greek Meta-Tokens to dynamically route System 1 vs 2."""
        t0 = time.perf_counter()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        hidden = self.text_model.model(input_ids)
        last_hidden = hidden[:, -1, :]  # [1, H]

        # Sliced linear projection directly onto pre-cached Greek meta-tokens
        greek_logits_mx = (last_hidden @ self.greek_weights.T)[0]  # [5]
        greek_probs_mx = mx.softmax(greek_logits_mx)
        mx.eval(greek_logits_mx, greek_probs_mx)

        logits_dict = {self.greek_symbols[i]: float(greek_logits_mx[i].item()) for i in range(len(self.greek_symbols))}
        probs_dict = {self.greek_symbols[i]: round(float(greek_probs_mx[i].item()), 4) for i in range(len(self.greek_symbols))}

        z_alpha = logits_dict.get("α", 0.0)
        z_epsilon = logits_dict.get("ε", 0.0)
        delta_z = z_alpha - z_epsilon

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
    # =========================================================================
    # Task 3: Real Browser Navigation Step (Demo 1)
    # =========================================================================
    def step_web_navigation(
        self,
        dom_state: str,
        valid_actions: List[str],
        step_index: int = 1,
    ) -> WebStepResult:
        """Executes one real browser navigation step on Metal GPU."""
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

        res = self.predict_choice(prompt=prompt, candidate_slots=action_mapping)
        is_done = "done" in res.chosen_text.lower() or "submit" in res.chosen_text.lower()
        return WebStepResult(
            chosen_action=res.chosen_text,
            action_slot=res.chosen_slot,
            confidence=res.confidence,
            is_done=is_done,
            latency_ms=res.latency_ms,
            step_index=step_index,
        )

    # =========================================================================
    # Task 4: Real Duplex Meeting Utterance Classification (Demo 2)
    # =========================================================================
    def classify_meeting_utterance(self, utterance: str) -> MeetingUtteranceResult:
        """Classifies incoming audio transcript chunk into speech categories."""
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
        slots = {
            "A": "FILLER",
            "B": "QUESTION",
            "C": "COMMAND",
            "D": "STATEMENT",
        }

        res = self.predict_choice(prompt=prompt, candidate_slots=slots)
        category = res.chosen_text
        is_filler = category == "FILLER"

        return MeetingUtteranceResult(
            category=category,
            is_filler=is_filler,
            confidence=res.confidence,
            latency_ms=res.latency_ms,
            probabilities=res.probabilities,
        )

    # =========================================================================
    # Task 5: Real Multi-Intent Customer Service Complaint (Demo 3)
    # =========================================================================
    def analyze_service_complaint(self, complaint_text: str) -> ServiceComplaintResult:
        """Performs multi-head risk & escalation analysis across 4 dimensions."""
        t0 = time.perf_counter()

        # Dimension 1: Billing Dispute
        d1_prompt = (
            f"Evidence: Customer service conversation message: \"{complaint_text}\"\n"
            f"Question: Does this customer message contain an unauthorized fee, surprise billing charge, or unknown debit dispute?\n"
            f"Options:\n"
            f"A) Yes, explicit billing or unauthorized fee dispute\n"
            f"B) No billing dispute\n"
            f"Answer:\n"
        )
        res_d1 = self.predict_choice(d1_prompt, ["A", "B"])
        is_billing_dispute = res_d1.chosen_slot == "A"
        billing_score = res_d1.probabilities.get("A", 0.0)

        # Dimension 2: Explicit Human Escalation
        d2_prompt = (
            f"Evidence: Customer service conversation message: \"{complaint_text}\"\n"
            f"Question: Does the customer demand an immediate transfer to a human specialist / live representative?\n"
            f"Options:\n"
            f"A) Yes, demands human representative transfer\n"
            f"B) No human transfer requested\n"
            f"Answer:\n"
        )
        res_d2 = self.predict_choice(d2_prompt, ["A", "B"])
        demands_human = res_d2.chosen_slot == "A"
        human_demand_score = res_d2.probabilities.get("A", 0.0)

        # Dimension 3: Regulatory Threat (zero specific agency names)
        d3_prompt = (
            f"Evidence: Customer service conversation message: \"{complaint_text}\"\n"
            f"Question: Does the customer threaten formal regulatory agency complaint, legal action, or external compliance escalation?\n"
            f"Options:\n"
            f"A) Yes, threatens regulatory complaint or legal action\n"
            f"B) No regulatory threat\n"
            f"Answer:\n"
        )
        res_d3 = self.predict_choice(d3_prompt, ["A", "B"])
        regulatory_threat = res_d3.chosen_slot == "A"
        regulatory_score = res_d3.probabilities.get("A", 0.0)

        # Dimension 4: Emotional Distress / Agitation
        d4_prompt = (
            f"Evidence: Customer service conversation message: \"{complaint_text}\"\n"
            f"Question: Does this customer message express intense anger, severe dissatisfaction, or emotional frustration?\n"
            f"Options:\n"
            f"A) Yes, high customer anger and severe frustration\n"
            f"B) Normal or mild tone\n"
            f"Answer:\n"
        )
        res_d4 = self.predict_choice(d4_prompt, ["A", "B"])
        emotional_distress = res_d4.chosen_slot == "A"
        emotional_score = res_d4.probabilities.get("A", 0.0)

        composite_risk_score = (
            0.35 * billing_score +
            0.30 * human_demand_score +
            0.25 * regulatory_score +
            0.10 * emotional_score
        )

        if composite_risk_score >= 0.65 or regulatory_threat:
            risk_level = "CRITICAL"
            needs_human = True
        elif composite_risk_score >= 0.35 or demands_human:
            risk_level = "HIGH"
            needs_human = True
        elif composite_risk_score >= 0.20:
            risk_level = "MEDIUM"
            needs_human = False
        else:
            risk_level = "LOW"
            needs_human = False

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return ServiceComplaintResult(
            billing_dispute=is_billing_dispute,
            handoff_demanded=demands_human,
            regulatory_threat=regulatory_threat,
            risk_level=risk_level,
            needs_human=needs_human,
            composite_risk_score=round(composite_risk_score, 4),
            latency_ms=round(total_latency_ms, 2),
            intent_confidences={
                "billing_dispute": round(billing_score, 4),
                "handoff_demanded": round(human_demand_score, 4),
                "regulatory_threat": round(regulatory_score, 4),
                "emotional_distress": round(emotional_score, 4),
            },
            emotional_distress=emotional_distress,
            emotional_score=round(emotional_score, 4),
        )

    # =========================================================================
    # Task 6: Banking & Credit Card IVR Menu Streaming Routing (Demo 3 Upgrade)
    # =========================================================================
    def route_ivr_intent(self, text: str) -> IvrRoutingResult:
        """Evaluates customer speech in real time and maps to standard banking IVR functional menus."""
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
    # System 2: Full Vocabulary Expanded Generative Deliberation
    # =========================================================================
    def generate_system2_expanded_thought(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Runs full-vocabulary autoregressive generative reasoning for System 2."""
        t0 = time.perf_counter()
        prompt_to_use = format_chat_prompt(
            self.tokenizer,
            prompt,
            enable_thinking=True,
            fallback_chatml=True,
        )

        gen_kwargs: Dict[str, Any] = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt_to_use,
            "max_tokens": max_tokens,
            "verbose": False,
        }
        try:
            from mlx_lm.sample_utils import make_sampler
            gen_kwargs["sampler"] = make_sampler(temp=float(temperature))
        except Exception:
            gen_kwargs["temp"] = float(temperature)
        try:
            generated_text = mlx_lm.generate(**gen_kwargs)
        except TypeError:
            gen_kwargs.pop("sampler", None)
            gen_kwargs.pop("temp", None)
            generated_text = mlx_lm.generate(**gen_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "expanded_thought": generated_text,
            "prompt_used": prompt_to_use,
            "latency_ms": round(latency_ms, 2),
            "max_tokens": max_tokens,
            "backend": "mlx_metal_gpu",
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
