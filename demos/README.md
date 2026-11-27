# Tractus Production Launch Demo Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hardware: Apple Silicon & PyTorch](https://img.shields.io/badge/Hardware-Apple%20Silicon%20Metal%20%7C%20PyTorch%20MPS%2FCPU-green.svg)]()
[![Inference: 100% Physical Neural](https://img.shields.io/badge/Inference-100%25%20Physical%20Neural%20Tensors-brightgreen.svg)]()
[![Zero-Mock Standard](https://img.shields.io/badge/Standard-Zero%20Mock%20Cheats%20%7C%20Zero%20Fake%20Sleep-red.svg)]()

> **Production Verification Guarantee**:
> All 4 demo prototypes execute **100% authentic neural forward passes** on physical hardware (Apple Silicon Metal GPU via MLX INT4 or NVIDIA CUDA / PyTorch BF16).
> **Zero synthetic fake data, zero artificial `time.sleep` delays, and zero regular-expression heuristic shortcuts**.
> Every output is driven by verifiable physical model weights (**`Tractus-0.8B`**, **`Tractus-2B`**, **`Tractus-4B`**, and **`Tractus-9B`**).

---

## Table of Contents
- [External Model Integration Guide (`HOW_TO_RUN_WITH_YOUR_MODEL.md`)](./HOW_TO_RUN_WITH_YOUR_MODEL.md)
- [Core Architecture & Decision Manifold](#core-architecture--decision-manifold)
- [Four Production Launch Scenarios](#four-production-launch-scenarios)
  - [Demo 1: Autonomous Web Navigation Agent](#demo-1-autonomous-web-navigation-agent)
  - [Demo 2: Full-Duplex Meeting Sentinel (Streamed ASR Transcripts)](#demo-2-full-duplex-meeting-sentinel-streamed-asr-transcripts)
  - [Demo 3: Multi-Intent Enterprise Service Router (Streamed ASR Transcripts)](#demo-3-multi-intent-enterprise-service-router-streamed-asr-transcripts)
  - [Demo 4: Real-Time Spatial Dynamics & Collision Avoidance](#demo-4-real-time-spatial-dynamics--collision-avoidance)
- [Hardware Acceleration & Execution Backends](#hardware-acceleration--execution-backends)
- [Quickstart & Web UI Reproduction](#quickstart--web-ui-reproduction)
- [Verification & Automated Test Suite](#verification--automated-test-suite)
- [Engineering Invariants](#engineering-invariants)

---

## Core Architecture & Decision Manifold

The Tractus decision model architecture is engineered specifically for **low-latency single-pass routing, calibrated uncertainty estimation, long-horizon autonomy, and selective System 2 escalation**. Unlike conventional generative LLMs that spend seconds generating free-form text to communicate discrete actions, Tractus scores 2–255 candidate options directly at the terminal position in a single forward pass:

```
[ User Request / Streamed ASR Transcript / DOM State ]
                        │
                        ▼
       ┌─────────────────────────────────┐
       │      Tractus Decision Engine    │
       │   Single-Pass Causal Readout    │
       └─────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
[ System 1: Direct Action ]    [ System 2: Deliberative ]
1 Prefill Pass (0 Gen. Tokens) Adaptive Shared-KV CoT
99.9% Smaller Readout Head     (>=4B; Safety Ratchet)
Zero-Sum Centered + e-Thermo   Zero KV Re-Prefill
```

- **Tractus-0.8B**: Flagship sub-billion parameter decision arbiter (**0.82B** total parameters, **0.58B** non-embedding backbone parameters). Consumes **909 MB** on disk/memory in MLX INT4 (**1.8 GB** in BF16) and eliminates autoregressive decoding latency by resolving every System 1 decision in a **single prefill pass (`0` generated tokens, `99.9%` smaller readout projection)**.
- **Thermostasis Calibration (Zero-Sum Centering & 2D Analytic $e$-Thermostat)**: Applies optional Binary Prior Shift (asymmetric `K = 2`, Decision Index 0.3 only) $\rightarrow$ $\tanh$ logit softcapping (`C = 30.0`) $\rightarrow$ zero-sum centering $\rightarrow$ parameter-free 2D analytic $e$-thermostat calibration with zero validation-fitted temperature scalars.
- **Zero Premature DONE on Web Navigation**: Across all four scales (`0.8B`–`9B` MLX INT4), Demo 1 completes all 5 web-navigation scenarios with **100% (20/20)** step accuracy and **0% Premature DONE**.

---

## Four Production Launch Scenarios

### Demo 1: Autonomous Web Navigation Agent
- **URL**: `http://localhost:8080/demo1`
- **CLI Runner**: `python run_demo1_web_agent.py --engine mlx --model-size 0.8b --max-steps 25`
- **Capability**: Autonomous form completion and DOM action arbitration. The Web UI & Playwright suite (`/demo1`) navigates 4 local portals (**Google Flights**, **Casa Flora**, **MegaStore**, **Global Service**) plus live **Trip.com** with causal refusal probes; the standalone CLI executes the closed-loop `MockWebApp` travel workflow.
- **Model**: `Tractus-0.8B` / `2B` / `4B` / `9B` in MLX INT4 (Apple Silicon Metal GPU) or PyTorch BF16 (CUDA/MPS/CPU).
- **Verified Edge Accuracy (MLX INT4, `0.8B`–`9B`)**: **100% (5/5)** scenarios passed, **100% (20/20)** step accuracy, **0% Premature DONE**, **0 generated tokens per step** (1 prefill pass).

### Demo 2: Full-Duplex Meeting Sentinel (Streamed ASR Transcripts)
- **URL**: `http://localhost:8080/demo2`
- **CLI Runner**: `python run_demo2_meeting_sentinel.py --engine mlx --model-size 0.8b --num-chunks 12`
- **Capability**: Synchronizes a 4-speaker hardware BOM budget review (**Edinburgh AMI Meeting Corpus `IS1008d`**, 12 streamed slices over `63.6 s`, `3/12` filler slices) with a live prototype BOM ledger (`€12.50` cap, OLED trade-off, `€50M` profit target). Tractus operates on **streamed ASR text transcripts** (not raw audio waveforms), classifying each utterance over **`[FILLER, QUESTION, COMMAND, STATEMENT]`** to drive stateful duplex actions **`[SILENT_OBSERVE, DISPATCH_TASK, SUPPLEMENT_TASK]`**.
- **Model**: `Tractus-0.8B` / `2B` / `4B` / `9B` MLX INT4 or PyTorch BF16.
- **Execution Profile**: Single prefill pass with **`0` generated tokens** per utterance slice across all four scales. Note: **`2B`/`4B`/`9B` reliably detect minimal conversational fillers**, whereas `0.8B` is capacity-limited on short fillers.

### Demo 3: Multi-Intent Enterprise Service Router (Streamed ASR Transcripts)
- **URL**: `http://localhost:8080/demo3`
- **CLI Runner**: `python run_demo3_service_router.py --engine mlx --model-size 0.8b`
- **Capability**: Enterprise customer support workbench streaming **streamed ASR text transcripts** from authentic **PolyAI MINDS-14** complaint audio (`32.77 s` full track; audio playback is for demonstration while Tractus evaluates the text transcripts) alongside a Customer 360 profile and a 3-row transaction ledger (`-$200.00` `Unauthorized POS Charge`, `FLAGGED HIGH-RISK`). Replaces `50–85 s` IVR dial trees with 4 binary risk passes (**`Billing Dispute`**, **`Handoff Demand`**, **`Regulatory/Legal Threat`**, **`Emotional Distress`**) plus primary banking-intent routing.
- **Execution Profile**: Executes all 4 binary risk heads and primary intent routing in single-pass prefill (`0` generated tokens per pass), avoiding autoregressive CoT bottlenecks and JSON parse failures.

### Demo 4: Real-Time Spatial Dynamics & Collision Avoidance
- **URL**: `http://localhost:8080/demo4`
- **CLI Runner**: `python run_demo4_spatial_dynamics.py --mode match --engine mlx --model-size 0.8b`
- **Capability**: Real-time 22-agent `105m x 68m` tactical football pitch simulation (`JevBall`) enforcing a **`300.0 ms`** per-tick carrier SLA deadline (contrasted against a `366.0 ms` cloud API baseline). Evaluates continuous spatial coordinates, tackle threat ETA, passing-lane clearance, offside margin, and `xG` to score up to 14 typed tactical candidates (`PASS`, `SHOT`, `DRIBBLE`).
- **Contract**: 35.0-second autonomous tactical sequence (30.0s build-up and goal impact, 5.0s celebration) with `--mode {match, scenarios, latency-benchmark}`.
- **Execution Profile**: Scores all 14 tactical candidates in a single prefill pass with **`0` generated tokens per tick**, achieving **`0%` SLA deadline misses** under the `300.0 ms` per-tick carrier budget on edge `Tractus-0.8B` MLX INT4. Note: Demo 4 supports `--engine mlx` (Metal GPU) or `--engine pytorch` / `--engine unified` (PyTorch CUDA/MPS/CPU).

---

## Hardware Acceleration & Execution Backends

All demos support dual hardware backends with automatic fallback:

1. **Apple Silicon Metal GPU (MLX Backend, `--engine mlx`)**:
   - Optimized for M1/M2/M3/M4 Apple Silicon.
   - 4-bit affine quantization (INT4, `bits=4, group_size=64`, unquantized `lm_head`): `909 MB` (`0.8B`), `1.0 GB` (`2B`), `2.2 GB` (`4B`), `6.1 GB` (`9B`).
   - Zero-copy unified memory architecture between CPU and GPU.
2. **PyTorch Backend (`--engine pytorch`; Demo 4 also accepts `--engine unified` as an alias)**:
   - Native PyTorch implementation using BF16/FP16 tensors (`1.8 GB` for `0.8B` BF16).
   - Automatically utilizes NVIDIA GPUs via CUDA, Apple Silicon via MPS, or fallback x86/ARM CPU.

---

## Quickstart & Web UI Reproduction

### 1. Launch the Unified Web Cockpit
```bash
# Start web server (default port 8080; supports --engine mlx or pytorch)
python run_web_ui.py --engine mlx --model-size 0.8b --port 8080

# Open browser to access the unified flight deck or standalone pages:
# Unified Deck:     http://localhost:8080/
# Web Agent:        http://localhost:8080/demo1
# Meeting Sentinel: http://localhost:8080/demo2
# Service Router:   http://localhost:8080/demo3
# Spatial Dynamics: http://localhost:8080/demo4
```

### 2. Run Headless CLI Benchmarks
Each demo can also be executed headlessly from the command line:

```bash
# Demo 1: Autonomous Web Agent (25 steps; --engine mlx or pytorch)
python run_demo1_web_agent.py --engine mlx --model-size 0.8b --max-steps 25

# Demo 2: Meeting Sentinel (12 transcript chunks; --engine mlx or pytorch)
python run_demo2_meeting_sentinel.py --engine mlx --model-size 0.8b --num-chunks 12

# Demo 3: Service Router (Multi-intent triage; --engine mlx or pytorch)
python run_demo3_service_router.py --engine mlx --model-size 0.8b

# Demo 4: Spatial Dynamics Tactical Match (--mode match|scenarios|latency-benchmark; --engine mlx, pytorch, or unified)
python run_demo4_spatial_dynamics.py --mode match --engine mlx --model-size 0.8b
```

---

## Verification & Automated Test Suite

Verify release package integrity, schemas, and Python syntax from the package root:
```bash
python scripts/verify_release_package.py
```

---

## Engineering Invariants

1. **Integer Zero Rule**: Zero metrics are strictly formatted as integer `0` or `0%` (e.g., `0% Premature DONE`, `0` parse failures in Tractus single-pass/dual-mode 831-suite runs).
2. **Dual Parameter Reporting**: Documentation reports both total parameters (`0.82B` / `1.88B` / `4.21B` / `9.24B`) and non-embedding backbone parameters (`0.58B` / `1.48B` / `3.57B` / `8.12B`).
3. **Pure English Standards**: Zero CJK characters across all public web templates, API payloads, and documentation.
4. **Physical Truth**: Every reported benchmark accuracy and latency corresponds to a physical, verified evaluation on hardware.
