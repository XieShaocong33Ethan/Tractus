# How to Run Tractus Demos with Your Model Checkpoint

[![Hardware: Apple Silicon & PyTorch](https://img.shields.io/badge/Hardware-Apple%20Silicon%20Metal%20%7C%20CUDA%20%7C%20MPS-green.svg)]()
[![Inference: 100% Physical Neural](https://img.shields.io/badge/Inference-100%25%20Physical%20Neural%20Tensors-brightgreen.svg)]()
[![Zero-Mock Standard](https://img.shields.io/badge/Standard-Zero%20Mock%20Cheats%20%7C%20Zero%20Fake%20Delays-red.svg)]()

This guide explains how to connect your own model weights (fine-tuned checkpoints, quantized MLX weights, or PyTorch models) to the **Tractus Production Demo Suite**.

---

## 1. Zero-Mock Physical Guarantee

Every demo in this repository executes **100% authentic neural forward passes** on physical hardware (Metal GPU on Apple Silicon, PyTorch MPS, or CUDA on Linux). 

- ❌ **Zero Mock Engines**: No hardcoded decision trees, no random probability generators.
- ❌ **Zero Synthetic Delays**: No artificial `time.sleep()` delays added to inflate or smooth latencies.
- ❌ **Zero Keyword Cheats**: No substring/regex matching to bypass neural classification.
- ✅ **Authentic Sliced Linear Projections**: Decision probabilities are computed directly from the model's vocabulary logits ($\mathbf{W}_{\text{head}} \cdot \mathbf{h}_{\text{last}}$) across the active candidate action manifold.

---

## 2. Model Parameter Specifications & Portfolio

All 4 production release demos are physically executed by our **Tractus** model family on-device:

| Model Tier | Total Parameters | Non-Embedding Backbone | Precision | Memory (INT4 / BF16) | Primary Role | Used in Demos |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tractus-0.8B** | **0.82B** | **0.58B** | **INT4 / BF16** | **909 MB / 1.8 GB** | **System 1 Fast-Path Decision Backbone** | **Demo 1, Demo 2, Demo 3, Demo 4** |
| **Tractus-2B** | **1.88B** | **1.48B** | **INT4 / BF16** | **~1.0 GB / ~3.8 GB** | **Balanced Multi-Domain Single-Pass Decision (System 1 Recommended)** | **Demo 1, Demo 2, Demo 3, Demo 4** |
| **Tractus-4B** | **4.21B** | **3.57B** | **INT4 / BF16** | **~2.2 GB / ~8.4 GB** | **High-Capacity Causal Reasoning & Tool Orchestration** | **Demo 1, Demo 2, Demo 3, Demo 4** |
| **Tractus-9B** | **9.24B** | **8.12B** | **INT4 / BF16** | **~6.1 GB / ~18.5 GB** | **Flagship Multi-Domain Arbitration & Long-Horizon Planning** | **Demo 1, Demo 2, Demo 3, Demo 4** |

### Why Tractus-0.8B Powers Default Edge Demos?
- **Zero Autoregressive Decoding Latency (`0` Generated Tokens)**: By scoring only the active `2–255` candidate rows in a single prefill pass (cutting readout projection compute by `99.9%` versus the `248,320`-way vocabulary), Tractus-0.8B eliminates multi-token decoding bottlenecks completely.
- **Zero Cloud Dependency & Zero Privacy Leak**: Weighing ~0.9 GB in INT4 memory, the model runs 100% locally on laptops, edge gateways, and in-car systems.
- **Proven Cognitive Efficiency**: Proves that enterprise web automation, meeting noise filtering, and banking switchboards do NOT require heavy 70B+ LLMs.

---

## 3. Checkpoint Resolution Priority

When launching any demo, the runtime automatically searches for model weights in the following priority order:

1. **Explicit CLI Argument**: `--checkpoint <path_to_weights_dir>`
2. **Environment Variable**: `export TRACTUS_CHECKPOINT=<path_to_weights_dir>`
3. **Local Project Checkpoints**: `./checkpoints/tractus-<size>/` (e.g., `./checkpoints/tractus-0.8b-mlx-int4`)
4. **Hugging Face Hub Registry**: `Ethan33X/Tractus-0.8B`, `Ethan33X/Tractus-2B`, `Ethan33X/Tractus-4B`, `Ethan33X/Tractus-9B`

---

## 4. Quick Start: Launching the Web Cockpit

### Step 1: Install Dependencies

For Apple Silicon (Recommended for low-latency MLX INT4 execution):
```bash
pip install mlx mlx-lm aiohttp numpy
```

For PyTorch MPS / CUDA:
```bash
pip install torch transformers accelerate aiohttp
```

### Step 2: Launch the Cockpit with Your Checkpoint

Point `--checkpoint` to your model weights folder (must contain `config.json`, tokenizer files, and model weights `.safetensors` or `.npz`):

```bash
# Launch with MLX backend on Apple Silicon Metal GPU (Default: port 8080)
python demos/run_web_ui.py --engine mlx --checkpoint /path/to/your/checkpoint

# Or on a custom port
python demos/run_web_ui.py --port 8081 --checkpoint /path/to/your/checkpoint

# Launch with PyTorch MPS / CUDA backend
python demos/run_web_ui.py --engine pytorch --checkpoint /path/to/your/checkpoint
```

Once running on the default port (`8080`, or your chosen `--port`), navigate to:
- **Mission Control Flight Deck**: `http://127.0.0.1:8080`
- **Standalone Demo 1 (Autonomous Web Agent)**: `http://127.0.0.1:8080/demo1`
- **Standalone Demo 2 (Duplex Meeting Sentinel)**: `http://127.0.0.1:8080/demo2`
- **Standalone Demo 3 (High-Frequency Service Switchboard)**: `http://127.0.0.1:8080/demo3`
- **Standalone Demo 4 (Autonomous Spatial Dynamics & Football)**: `http://127.0.0.1:8080/demo4`

---

## 5. Running the Standalone Demos via CLI

You can run each demo independently via Python CLI with real-time terminal stopwatch telemetry (all Demos 1–4 accept `--engine mlx` or `--engine pytorch` as well as `--checkpoint <path>`; Demo 4 also accepts `--engine unified` as a PyTorch alias):

### Demo 1: Autonomous Web Agent (`MockWebApp` Closed-Loop CLI)
Runs the closed-loop multi-step enterprise travel form using the 255-slot action manifold (use the Web UI `/demo1` for Google Flights, Casa Flora, MegaStore, Global Service, and live Trip.com):
```bash
python demos/run_demo1_web_agent.py --engine mlx --model-size 0.8b --max-steps 25
```

### Demo 2: Duplex Meeting Sentinel (Streamed ASR Transcripts)
Monitors streamed ASR text transcripts from the AMI meeting corpus, classifies each utterance over `[FILLER, QUESTION, COMMAND, STATEMENT]`, and triggers stateful duplex actions (`[SILENT_OBSERVE, DISPATCH_TASK, SUPPLEMENT_TASK]`):
```bash
python demos/run_demo2_meeting_sentinel.py --engine mlx --model-size 0.8b --num-chunks 12
```

### Demo 3: High-Frequency Banking & Service Switchboard (Streamed ASR Transcripts)
Processes streamed ASR text transcripts from customer complaints (audio playback is for demo monitoring; Tractus operates on text transcripts) and evaluates 4 binary risk/escalation heads plus banking-intent routing:
```bash
python demos/run_demo3_service_router.py --engine mlx --model-size 0.8b
```

### Demo 4: Autonomous Spatial Dynamics & JevBall Football Pitch
Executes 22 continuous multi-agent decision policies with zero scripts, generating a 30s tactical build-up to a 30.0s goal + 5s celebration (`--mode match`, `scenarios`, or `latency-benchmark`; `--engine mlx`, `pytorch`, or `unified`):
```bash
python demos/run_demo4_spatial_dynamics.py --mode match --engine mlx --model-size 0.8b
# Or inspect discrete tactical scenarios:
python demos/run_demo4_spatial_dynamics.py --mode scenarios --engine mlx --model-size 0.8b
```

---

## 6. Python API Integration Guide

If you are developing your own agent application and want to interface directly with Tractus engines, use the clean Python API:

```python
# For Linux / NVIDIA CUDA / Cross-Platform Systems:
from apse.inference.unified_decision_engine import UnifiedDecisionEngine
engine = UnifiedDecisionEngine(model_size="0.8b", checkpoint_path="/path/to/your/checkpoint")

# For Apple Silicon Metal GPU (Optional High-Speed MLX Backend):
# from apse.inference.mlx_decision_engine import MlxDecisionEngine
# engine = MlxDecisionEngine(model_size="0.8b", quant_mode="int4", checkpoint_path="/path/to/your/checkpoint")

# -------------------------------------------------------------
# Task 1: Autonomous Web Navigation Decision (Demo 1)
# -------------------------------------------------------------
step = engine.step_web_navigation(
    dom_state="URL: https://flights.google.com | Title: Google Flights | Goal: Book a non-stop flight from SFO to JFK on Nov 15",
    valid_actions=[
        "CLICK: Departure airport input field",
        "TYPE: SFO into origin input",
        "CLICK: Destination airport input field",
        "TYPE: JFK into destination input",
        "CLICK: Search Flights button",
    ],
    step_index=1,
)
print(f"Action: {step.chosen_action} | Confidence: {step.confidence:.1%} | Latency: {step.latency_ms:.2f}ms")

# -------------------------------------------------------------
# Task 2: Meeting Utterance & Intent Classification (Demo 2)
# -------------------------------------------------------------
intent = engine.classify_meeting_utterance("Yeah, sure, mm-hmm, I agree with that.")
print(f"Is Filler: {intent.is_filler} | Category: {intent.category} | Latency: {intent.latency_ms:.2f}ms")

action_intent = engine.classify_meeting_utterance("Please check the BOM silicon cost projection for Q3.")
print(f"Is Filler: {action_intent.is_filler} | Category: {action_intent.category} | Actionable: {not action_intent.is_filler}")

# -------------------------------------------------------------
# Task 3: Customer Service 4-Head Risk Intent Evaluation (Demo 3)
# -------------------------------------------------------------
analysis = engine.analyze_service_complaint(
    "I saw an unauthorized fee of $45 on my statement and I want to speak to a supervisor immediately!"
)
print(f"Risk Level: {analysis.risk_level}")
print(f"Needs Human Escalation: {analysis.needs_human}")
print(f"Billing Dispute Score: {analysis.intent_confidences['billing_dispute']:.2f}")
print(f"Handoff Demanded Score: {analysis.intent_confidences['handoff_demanded']:.2f}")
print(f"Composite Latency (4 heads): {analysis.latency_ms:.2f}ms")

# -------------------------------------------------------------
# Task 4: Real-Time IVR Menu Routing (Demo 3)
# -------------------------------------------------------------
ivr = engine.route_ivr_intent("I want to know my current credit limit and available balance.")
print(f"Routed Menu: {ivr.menu_name} | Human Handoff: {ivr.is_human_handoff} | Latency: {ivr.latency_ms:.2f}ms")
```

---

## 7. How the 255-Slot Action Manifold Works

Traditional LLM agent frameworks produce free-form text or JSON (e.g. `{"action": "click", "target": "#btn"}`), requiring dozens to hundreds of autoregressive decoding steps and risking invalid syntax.

Tractus solves this via **Single-Pass Sliced Linear Projection**:

$$\mathbf{z}_{\text{candidates}} = \mathbf{h}_{\text{last}} \cdot \mathbf{W}_{\text{head}}[\mathcal{T}_{\text{slots}}]^T$$

1. The environment formats $2 \le K \le 255$ available candidate actions into discrete slots ($A, B, C, \dots$).
2. The model runs a **single forward pass** over the context prompt.
3. Only the weight vectors corresponding to the candidate action tokens are gathered from $\mathbf{W}_{\text{head}}$ (`99.9%` readout projection reduction).
4. A single dot product followed by **Thermostasis** calibration (optional Binary Prior Shift $\rightarrow$ $\tanh$ softcap `C = 30.0` $\rightarrow$ zero-sum centering $\rightarrow$ 2D analytic $e$-thermostat) computes calibrated option probabilities across all candidates with **`0` generated tokens**.
5. The action is deterministic, verified syntax-free, and directly executable by Playwright or browser DOM.

---

## 8. Verifying Release Package & Execution

Run our automated verification suite to prove package integrity, zero mock cheats, and schema correctness:

```bash
# Verify release package integrity, schemas, and 100% CJK-free purity
python scripts/verify_release_package.py
```

If you encounter any issues connecting your checkpoint, ensure your checkpoint includes the standard HuggingFace tokenizer files (`tokenizer.json`, `tokenizer_config.json`) and vocabulary configuration matching the base model architecture.
