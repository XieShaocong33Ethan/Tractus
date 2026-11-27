---
license: other
license_name: modified-mit
license_link: LICENSE
language:
  - en
pipeline_tag: decision-model
library_name: transformers
tags:
  - decision-model
  - multimodal
  - classification
  - calibration
  - routing
  - system-one
  - dual-mode
  - qwen3.5
  - tractus
---

# Tractus: Open-Weight Unified Dual-Mode Decision Foundation Models (0.8B, 2B, 4B, 9B)

**Tractus** is an open-weight family of unified dual-mode decision foundation models (`0.8B`, `2B`, `4B`, `9B`) post-trained from [`Qwen/Qwen3.5`](https://huggingface.co/Qwen) base checkpoints with a `262,144`-token (`256K`) context window. Each scale ships in two paired configurations sharing the same hybrid topology (`3:1` Gated DeltaNet + Gated Attention):

- **Zero-Decode-Latency Single-Pass (`Tractus-decision-{0.8B,2B,4B,9B}`)**: Eliminates autoregressive decoding latency by scoring `2–255` runtime option keys in a **single prefill pass** with `0` generated tokens and a **99.9% smaller readout projection** (skipping the `248,320`-way vocabulary head) via **Thermostasis** (optional Binary Prior Shift $\rightarrow$ $\tanh$ logit softcapping `C = 30.0` $\rightarrow$ zero-sum centering $\rightarrow$ parameter-free 2D analytic $e$-thermostat with zero validation-fitted temperature scalars; $\tanh$ softcapping drives $T=1$ outlier-logit ECE reduction while zero-sum centering removes common-mode offsets, and the two optional static binary priors `+3.2` hallucination on RAGTruth and `-0.75` phishing on PhishNChips trigger only in Decision Index 0.3, with `z_0 = 0` across all 9 primary benchmarks and OOD `19,887`).
- **Zero-Re-Prefill Dual-Mode (`Tractus-{0.8B,2B,4B,9B}`)**: Resolves the majority of queries in `0` generated tokens (`71.0%` on 9B JevBench-231 and `479/831` on the Matched Commercial Suite — cutting matched single-GPU BF16 `batch=1` sequential median latency to `0.69 s` for `Tractus-9B` and `47.4 ms` single-template for `Tractus-decision-9B` vs. `26.00 s` at `2,925` mean generated tokens and `44` parse failures for `Qwen3.5-9B Thinking`), and when escalating via the Joint 2D Confidence-Entropy Phase Gate (`phase_zone == SYSTEM_2 and conf < 0.80`), reuses **100% of the live prefill KV cache** with **zero re-prefill** on a `3:1` hybrid backbone that cuts 256K BF16 KV memory **4×** (`34.36 GB -> 8.59 GB`), guarded by a Defensive Safety Ratchet (accepting $y^{\mathrm{S2}}$ iff $p_{y^{\mathrm{S2}}}^{\mathrm{S2}} \ge \max(0.4, 1/K)$).

---

## 1. Architecture & Operating Regimes

| Scale Tier | Single-Pass (`Tractus-decision`) | Dual-Mode (`Tractus`) | Total / Non-Emb Params | Layers (`DeltaNet + Attn`) | Hidden / FFN Dim | Recommended Default |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **9B** | [`Ethan33X/Tractus-decision-9B`](https://huggingface.co/Ethan33X/Tractus-decision-9B) | [`Ethan33X/Tractus-9B`](https://huggingface.co/Ethan33X/Tractus-9B) | 9.24B / 8.12B | 32 (`24 + 8`) | 4,096 / 12,288 | **Dual-Mode (`Tractus`, 8K)** |
| **4B** | [`Ethan33X/Tractus-decision-4B`](https://huggingface.co/Ethan33X/Tractus-decision-4B) | [`Ethan33X/Tractus-4B`](https://huggingface.co/Ethan33X/Tractus-4B) | 4.21B / 3.57B | 32 (`24 + 8`) | 2,560 / 9,216 | **Dual-Mode (`Tractus`, 8K)** |
| **2B** | [`Ethan33X/Tractus-decision-2B`](https://huggingface.co/Ethan33X/Tractus-decision-2B) | [`Ethan33X/Tractus-2B`](https://huggingface.co/Ethan33X/Tractus-2B) | 1.88B / 1.48B | 24 (`18 + 6`) | 2,048 / 6,144 | **Single-Pass (`Tractus-decision`)** |
| **0.8B** | [`Ethan33X/Tractus-decision-0.8B`](https://huggingface.co/Ethan33X/Tractus-decision-0.8B) | [`Ethan33X/Tractus-0.8B`](https://huggingface.co/Ethan33X/Tractus-0.8B) | 0.82B / 0.58B | 24 (`18 + 6`) | 1,024 / 3,584 | **Single-Pass (`Tractus-decision`)** |

---

## 2. Public Benchmark Results

### 2.1 Decision Index 0.3 & JevBench 231 (Unified `chat_template`, Zero OOF Fitting)

| Scale Tier | Decision Index 0.3 Skill (Raw / Breadth) | `Tractus-decision` JevBench (`231`) | `Tractus` Dual-Mode JevBench (`231`, E/S/H) | Zero-Token Exits (`Rec.`) |
| :--- | :---: | :---: | :---: | :---: |
| **9B** | **46.03** (59.32% / 44.71) | 79.22% (183/231) | **90.91% (210/231)** *(48/72/90)* | **71.0%** (`Dual`) |
| **4B** | **41.31** (55.56% / 40.00) | 76.19% (176/231) | **88.31% (204/231)** *(48/69/87)* | **43.7%** (`Dual`) |
| **2B** | **29.27** (46.83% / 26.73) | **69.26% (160/231)** *(48/66/46)* | 65.37% (151/231) *(48/58/45)* | **100%** (`Single-Pass`) |
| **0.8B** | **18.07** (37.66% / 17.10) | **72.73% (168/231)** *(48/66/54)* | 70.13% (162/231) *(HF 4K)* | **100%** (`Single-Pass`) |

*Notes*: Unified BF16 `256K` `chat_template` evaluation (`max_new_tokens=8192` on `2B/4B/9B` JevBench, HF `4096` at `0.8B`; the 1-item difference on 2B single-pass `160/231` vs `161/231` across separate runs reflects a BF16 left-padding/batching boundary shift rather than generation cap). Below `4B`, greedy System 2 rollouts suffer repetition collapse, so single-pass `Tractus-decision` is recommended by default. Full Decision Index 0.3 deliverables (`results.jsonl.gz`) live in [`Ethan33X/tractus-decision-index-0.3`](https://huggingface.co/datasets/Ethan33X/tractus-decision-index-0.3).

### 2.2 Unified 9-Benchmark `chat_template` Suite (`N=9,835` Items per Scale)

| Benchmark / Summary (`n`) | `0.8B decision` | `0.8B Tractus` | `2B decision` | `2B Tractus` | `4B decision` | `4B Tractus` | `9B decision` | `9B Tractus` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JevBench-231\*** (`231`, Top-1) | **72.73%** | 70.13% | **69.26%** | 65.37% | 76.19% | **88.31%** | 79.22% | **90.91%** |
| Global-MMLU-EN (`400`) | **49.00%** | 47.50% | **60.00%** | 59.00% | 79.25% | **82.75%** | 82.25% | **85.75%** |
| Global-MMLU-Lite (`1,200`) | **39.92%** | 38.92% | 48.42% | **49.00%** | 67.08% | **77.33%** | 73.50% | **81.67%** |
| Global-MMLU-NonEN (`2,200`) | 38.23% | **39.14%** | **43.77%** | 43.59% | 60.14% | **72.55%** | 66.86% | **79.00%** |
| When2Call (`2,000`) | **39.85%** | 38.85% | **58.75%** | 57.05% | 65.20% | **80.30%** | 76.05% | **82.20%** |
| Amazon ESCI (`2,000`, Macro-F1) | 26.66% | **27.19%** | **29.54%** | 28.02% | 41.07% | **46.39%** | 37.68% | **46.26%** |
| Amazon ESCI† (`2,000`, Top-1) | **52.00%** | 48.25% | **55.75%** | 52.05% | **61.60%** | 61.10% | 60.50% | **62.45%** |
| Mind2Web (`276`) | **63.77%** | **63.77%** | 74.28% | **75.36%** | **86.96%** | 84.78% | 86.23% | **87.32%** |
| PubMedQA (`1,000`) | 67.50% | **67.80%** | **72.20%** | 67.60% | **69.20%** | 69.00% | **75.60%** | 73.30% |
| New Yorker Caption (`528`) | **32.95%** | 31.82% | **45.27%** | 43.75% | 49.43% | **52.65%** | 65.72% | **67.05%** |
| **9-Task Macro Mean** | **47.84%** | 47.23% | **55.72%** | 54.30% | 66.06% | **72.67%** | 71.46% | **77.05%** |
| **Pooled 9-Task Acc.** (`N=9,835` Top-1) | **46.22%** | 45.19% | **54.91%** | 53.19% | 64.57% | **72.08%** | 70.53% | **76.18%** |

*\*JevBench-231 uses `max_new_tokens=8192` for `2B/4B/9B` and Hugging Face `4096` for `0.8B`; the remaining 8 tasks use `4096`. †9-Task Macro Mean uses ESCI Macro-F1; Pooled 9-Task Acc. uses ESCI Top-1.*

### 2.3 Frontier Commercial API Comparison (7 Shared Core Benchmarks + ECE)

Commercial API figures follow the public [DataCamp / Google Gemini Decision 1 evaluation](https://www.datacamp.com/blog/gemini-decision-1) (where refusals, safety blocks, and unsupported requests count as incorrect over the full denominator). Open-weight **Tractus** results use unified `chat_template` BF16 256K evaluation with **zero out-of-fold calibration fitting**. Both `Tractus` (Dual-Mode) and `Tractus-decision` (Single-Pass) are placed **first** in dedicated columns across the **7 shared core benchmarks** where all four commercial APIs publish scores (audio/video supplements omitted from the 7 shared evals; see the technical report appendix for full multimodal results):

#### Flagship Tiers (`9B` & `4B`) vs. Commercial APIs

| Benchmark (`Split / Metric`) | `Tractus-9B` (`Dual`) | `Tractus-decision-9B` | `Tractus-4B` (`Dual`) | `Tractus-decision-4B` | Gemini Decision 1 | TypeSafe Jev 1.14 | GPT-6 Luna (Decisions API) | Gemini 3.5 Flash-Lite (minimal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JevBench** (`Public / 231`) | 90.91% | 79.22% | 88.31% | 76.19% | **92.2%** | 90.0% | 90.9% | 86.6% |
| **Amazon ESCI** (`Macro-F1, n=2,000`) | 46.26% | 37.68% | 46.39% | 41.07% | **58.7%** | 56.7% | 54.7% | 47.2% |
| **New Yorker Caption** (`Text-only, n=528`) | 67.05% | 65.72% | 52.65% | 49.43% | **79.7%** | 72.0% | 68.4% | 73.7% |
| **Mind2Web Multimodal** (`n=276`) | **87.32%** | 86.23% | 84.78% | 86.96% | 86.2% | 86.8% | 81.8% | 78.0% |
| **PubMedQA** (`n=1,000`) | 73.30% | 75.60% | 69.00% | 69.20% | **79.8%** | 78.2% | 74.4% | 77.6% |
| **Global-MMLU-Lite** (`12 Non-EN, n=1,200`\*) | 81.67% | 73.50% | 77.33% | 67.08% | **86.0%** | 85.0% | 79.6% | 78.8% |
| **When2Call** (`n=2,000`) | **82.20%** | 76.05% | 80.30% | 65.20% | 79.9% | 82.0% | 70.4% | 60.9% |
| **Top-1 ECE** (`7 shared evals`, ↓) | 0.2228 | 0.0889 | 0.2609 | 0.1068 | **0.0250** | 0.0494 | 0.0721 | 0.2104 |

#### Edge Tiers (`2B` & `0.8B`) vs. Commercial APIs

| Benchmark (`Split / Metric`) | `Tractus-2B` (`Dual`) | `Tractus-decision-2B` | `Tractus-0.8B` (`Dual`) | `Tractus-decision-0.8B` | Gemini Decision 1 | TypeSafe Jev 1.14 | GPT-6 Luna (Decisions API) | Gemini 3.5 Flash-Lite (minimal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JevBench** (`Public / 231`) | 65.37% | 69.26% | 70.13% | 72.73% | **92.2%** | 90.0% | 90.9% | 86.6% |
| **Amazon ESCI** (`Macro-F1, n=2,000`) | 28.02% | 29.54% | 27.19% | 26.66% | **58.7%** | 56.7% | 54.7% | 47.2% |
| **New Yorker Caption** (`Text-only, n=528`) | 43.75% | 45.27% | 31.82% | 32.95% | **79.7%** | 72.0% | 68.4% | 73.7% |
| **Mind2Web Multimodal** (`n=276`) | 75.36% | 74.28% | 63.77% | 63.77% | 86.2% | **86.8%** | 81.8% | 78.0% |
| **PubMedQA** (`n=1,000`) | 67.60% | 72.20% | 67.80% | 67.50% | **79.8%** | 78.2% | 74.4% | 77.6% |
| **Global-MMLU-Lite** (`12 Non-EN, n=1,200`\*) | 49.00% | 48.42% | 38.92% | 39.92% | **86.0%** | 85.0% | 79.6% | 78.8% |
| **When2Call** (`n=2,000`) | 57.05% | 58.75% | 38.85% | 39.85% | 79.9% | **82.0%** | 70.4% | 60.9% |
| **Top-1 ECE** (`7 shared evals`, ↓) | 0.0747 | 0.0674 | 0.3080 | 0.2839 | **0.0250** | 0.0494 | 0.0721 | 0.2104 |

*\*On the broader 22 Non-English split (`n=2,200`), `Tractus` / `Tractus-decision` score `79.00%` / `66.86%` (`9B`), `72.55%` / `60.14%` (`4B`), `43.59%` / `43.77%` (`2B`), and `39.14%` / `38.23%` (`0.8B`). Tractus ECE values above reflect 15-bin sample-weighted raw calibration across the 7 shared unified-suite splits (`N=7,235`: `231 + 2,000 + 528 + 276 + 1,000 + 1,200 + 2,000`) with zero OOF fitting (`JevBench-231` dual-mode ECE is `0.0678` at `9B` and `0.0767` at `4B`); by contrast, the 831-item Matched Commercial Suite below reports 10-bin raw $e$-thermostat ECE over `N=831` items with 3-template selection (`0.0245` for `Tractus-decision-9B`, `0.0528` for `Tractus-9B` Dual).*

#### Global-MMLU-Lite Multilingual & Almanac Reference (Split-Separated)

Whereas DataCamp evaluates only the **12 Non-English** slice (`Gemini Decision 1`: `86.0%`, `TypeSafe Jev 1.14`: `85.0%`), public leaderboards such as **Benchmark Almanac** report **Global-MMLU-Lite Full (all 15 languages incl. English)**. Both Tractus's evaluated multilingual slices and the Almanac external models are reported cleanly in separate tables so splits are never conflated:

| Scale | `Tractus` (`Dual`, EN-400) | `Tractus-decision` (`EN-400`) | `Tractus` (`Dual`, 12 Non-EN) | `Tractus-decision` (`12 Non-EN`) | `Tractus` (`Dual`, 22 Non-EN) | `Tractus-decision` (`22 Non-EN`) | `Tractus-decision` (`23-lang Full, n=9,200`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **9B** | **85.75%** | 82.25% | **81.67%** | 73.50% | **79.00%** | 66.86% | 67.77% |
| **4B** | **82.75%** | 79.25% | **77.33%** | 67.08% | **72.55%** | 60.14% | 61.40% |
| **2B** | 59.00% | **60.00%** | **49.00%** | 48.42% | 43.59% | **43.77%** | 44.48% |
| **0.8B** | 47.50% | **49.00%** | 38.92% | **39.92%** | **39.14%** | 38.23% | 38.15% |

| External Reference (`Benchmark Almanac Full`, 15 langs incl. EN) | GPT-6 Sol | GPT-6 Astra | GPT-6 Luna | Gemma 4 31B IT | Gemini 3.5 Flash-Lite | Gemma 4 26B A4B | DeepSeek-V3.1 | Gemma 3 (`27B / 4B / 1B`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Score (15 langs incl. EN)** | **94.1%** | 92.6% | 92.0% | 90.7% | 89.5% | 85.7% | 80.4% | 76.3% / 65.1% / 34.2% |

*(Commercial decision specialists `Gemini Decision 1` and `TypeSafe Jev 1.14` publish only Non-English Global-MMLU-Lite scores: `—` on Full; dual-mode `Tractus` was evaluated on `EN-400`, `12 Non-EN`, and `22 Non-EN` above rather than the 9,200-item 23-language slice.)*

#### Matched Commercial & Qwen3.5 Suite (831 Shared; Single-Order Prefill, Raw Thermostasis ECE, Zero Fitted Temperature Scalars)

Evaluated on the **831-item Matched Commercial Suite** (`JevBench-231` + `6 x 100` shared tasks) at full `262,144`-token context (`BF16`, single-order prefill with zero option-permutation averaging and zero external cache injection; System 2 `max_new_tokens=8192`). **`Tractus-9B (GO)`** denotes **Gap-Only gating**, escalating whenever $\Delta z \le e$ without the `conf < 0.80` confidence guard. Official **Qwen3.5** baselines (`Qwen3.5-9B` Thinking, `Qwen3.5-4B` Thinking) are evaluated on the exact same 831 items using a single official prompt in BF16, whereas Tractus sweeps 3 prompt templates (`chat_template`, `route_b`, `decider`) and selects the per-task best template. Commercial API columns cite DataCamp's published full-split reference scores rather than re-evaluations on the `n=100` subsets:

##### Flagship (`9B`) vs. Official Qwen3.5 Baselines & Commercial APIs (831 Shared)

| Benchmark (`Split / Metric`) | `Tractus-decision-9B` | `Tractus-9B` (`Dual`) | `Tractus-9B` (`GO`) | Qwen3.5-9B (`Think`) | Qwen3.5-4B (`Think`) | Gemini Decision 1 | TypeSafe Jev 1.14 | GPT-6 Luna (Decisions API) | Gemini 3.5 Flash-Lite (minimal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JevBench** (`Public / 231`) | 80.52% | 88.74% | 92.21% | **93.07%** | 89.18% | 92.2% | 90.0% | 90.9% | 86.6% |
| **Amazon ESCI** (`Macro-F1, n=100`) | 63.12% | 59.23% | 55.75% | **65.23%** | 62.69% | 58.7% | 56.7% | 54.7% | 47.2% |
| **New Yorker Caption** (`n=100`) | 78.00% | 65.00% | 64.00% | 68.00% | 66.00% | **79.7%** | 72.0% | 68.4% | 73.7% |
| **Mind2Web Multimodal** (`n=100`) | **88.00%** | 85.00% | 85.00% | 79.00% | 78.00% | 86.2% | 86.8% | 81.8% | 78.0% |
| **PubMedQA** (`n=100`) | 76.00% | 70.00% | 65.00% | 74.00% | 70.00% | **79.8%** | 78.2% | 74.4% | 77.6% |
| **Global-MMLU-Lite** (`12 Non-EN, n=100`) | 67.00% | 77.00% | 79.00% | 72.00% | 71.00% | **86.0%** | 85.0% | 79.6% | 78.8% |
| **When2Call** (`n=100`) | 75.00% | **84.00%** | **84.00%** | **84.00%** | 81.00% | 79.9% | 82.0% | 70.4% | 60.9% |
| **Pooled Top-1 ECE** (`7 shared evals`, ↓) | **0.0245** | 0.0528 | 0.0280 | — | — | 0.0250 | 0.0494 | 0.0721 | 0.2104 |
| **Parse Fail / S2 Esc. or Mean Tok** | 0 / 0 | 0 / 352\* | 0 / 480\* | 44 / 2,925 | 65 / 2,925 | — | — | — | — |
| **Seq. Med. Lat (`batch=1`, Matched GPU, ↓)** | **47.4 ms** *(0.14 s 3-tpl)* | 0.69 s | 7.22 s | 26.00 s | 14.36 s | — | — | — | — |

##### Multi-Scale Tractus Scaling (`0.8B`–`9B`, Single-Order, 831 Shared)

| Benchmark (`Split / Metric`) | `0.8B decision` | `0.8B Dual` | `2B decision` | `2B Dual` | `4B decision` | `4B Dual` | `9B decision` | `9B Dual` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JevBench** (`Public / 231`) | **73.59%** | 72.29% | **71.43%** | 68.83% | 77.49% | **89.61%** | 80.52% | **88.74%** |
| **Amazon ESCI** (`Macro-F1, n=100`) | **32.96%** | 28.96% | **51.57%** | 48.06% | **67.10%** | 51.95% | **63.12%** | 59.23% |
| **New Yorker Caption** (`n=100`) | 36.00% | **39.00%** | **60.00%** | 57.00% | **66.00%** | 62.00% | **78.00%** | 65.00% |
| **Mind2Web Multimodal** (`n=100`) | **65.00%** | 64.00% | **74.00%** | **74.00%** | **84.00%** | 83.00% | **88.00%** | 85.00% |
| **PubMedQA** (`n=100`) | 65.00% | **67.00%** | **68.00%** | 66.00% | **70.00%** | 62.00% | **76.00%** | 70.00% |
| **Global-MMLU-Lite** (`12 Non-EN, n=100`) | **50.00%** | **50.00%** | **49.00%** | **49.00%** | 60.00% | **71.00%** | 67.00% | **77.00%** |
| **When2Call** (`n=100`) | **49.00%** | **49.00%** | **61.00%** | **61.00%** | 70.00% | **80.00%** | 75.00% | **84.00%** |
| **Pooled Top-1 ECE** (`7 shared evals`, ↓) | **0.0239** | 0.0385 | **0.0274** | 0.0389 | **0.0246** | 0.0499 | **0.0245** | 0.0528 |
| **S2 Escalated** (`Ratchet Rescued`) | 0 (`0`) | 327 (`131`) | 0 (`0`) | 559 (`224`) | 0 (`0`) | 614 (`7`) | 0 (`0`) | 352 (`4`) |

*\*In the 9B comparison, Tractus reports `Parse Fail / S2 Escalated*` (`*` out of 831 items, `0` parse failures; `4` Ratchet rescues in both `Tractus-9B` and `Tractus-9B (GO)`), while official Qwen3.5 baselines report `Parse Fail / Mean Generated Tokens`. Across all 831 items, `Tractus-decision-9B` completes in `0` generated tokens (`47.4 ms` single-template / `0.14 s` 3-template sequential `batch=1` median on matched single-GPU BF16 hardware) and `Tractus-9B` resolves `479/831` items in `0` generated tokens (`0.69 s` sequential median, `352` escalated, `0` parse failures), whereas `Qwen3.5-9B (Think)` averages `2,925` generated tokens (`26.00 s` sequential median, `44` parse failures) and `Qwen3.5-4B (Think)` takes `14.36 s` (`65` parse failures); `Tractus-9B (GO)` runs in `7.22 s` sequential median (`0` parse failures). Main table Pooled Top-1 ECE reports 10-bin raw Thermostasis $e$-thermostat calibration without validation-fitted temperature scalars; for reference, optional 5-fold out-of-fold temperature scaling yields (`raw / OOF`): `9B Dual` `0.0528 / 0.0836`; `9B GO` `0.0280 / 0.0269`; `9B dec.` `0.0245 / 0.0395`; `4B Dual` `0.0499 / 0.0107`; `4B dec.` `0.0246 / 0.0145`; `2B Dual` `0.0389 / 0.0347`; `2B dec.` `0.0274 / 0.0197`; `0.8B Dual` `0.0385 / 0.0313`; `0.8B dec.` `0.0239 / 0.0313`. JevBench-231 scores differ slightly between the unified `chat_template` HF 8K evaluation (`90.91%` 9B Dual, `88.31%` 4B Dual, `79.22%` 9B dec.) and the 831-item vLLM 8K suite (`88.74%` 9B Dual, `89.61%` 4B Dual, `80.52%` 9B dec.) on the exact same checkpoints due to prompt-template and serving-backend differences. Commercial API columns cite DataCamp's published full-split reference scores.*

---

## 3. Interactive Production Demos (4 Real-World Applications)

This repository includes **4 standalone interactive demos** and a **Unified Web Cockpit** (`demos/run_web_ui.py`). All demos execute authentic neural forward passes on Apple Silicon Metal GPU (MLX INT4) or PyTorch (CUDA/MPS) with zero mock responses.

### 3.1 Demo 1: Autonomous Web Navigation Agent (`demos/run_demo1_web_agent.py` & `/demo1`)

<p align="center">
  <img src="assets/demo1_web_agent.gif" alt="Demo 1: Autonomous Web Navigation Agent" width="100%" />
</p>

- **Scenarios**: The Web UI & Playwright suite (`/demo1`) navigates 4 interactive local web portals (**Google Flights** 10-flight matrix, **Casa Flora** property booking, **MegaStore** electronics checkout, **Global Service** 25+ country GDPR onboarding) plus **1 live public airline website (`https://hk.trip.com/flights/`)**, alongside causal refusal probes (impossible route `Atlantis` -> `BLOCKED` at `4B`/`9B` with `93.4%`/`77.7%` confidence; conservative `WAIT` at `0.8B`/`2B`). Standalone CLI (`demos/run_demo1_web_agent.py`) runs the closed-loop multi-step enterprise travel form (`MockWebApp`).
- **Measured Edge Performance (MLX INT4, `bits=4, group_size=64`, unquantized `lm_head`, Zero Autoregressive Decoding)**:

| Model (MLX INT4) | Weight Disk Size | Scenarios Passed | Step Accuracy | Premature DONE Rate | Gen. Tokens / Step | Inference Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Tractus-0.8B-INT4`** | **909 MB** | **100% (5 / 5)** | **100% (20 / 20)** | **0%** | **0** | 1 prefill pass |
| **`Tractus-2B-INT4`** | **1.0 GB** | **100% (5 / 5)** | **100% (20 / 20)** | **0%** | **0** | 1 prefill pass |
| **`Tractus-4B-INT4`** | **2.2 GB** | **100% (5 / 5)** | **100% (20 / 20)** | **0%** | **0** | 1 prefill pass |
| **`Tractus-9B-INT4`** | **6.1 GB** | **100% (5 / 5)** | **100% (20 / 20)** | **0%** | **0** | 1 prefill pass |

### 3.2 Demo 2: Full-Duplex Meeting Sentinel (`demos/run_demo2_meeting_sentinel.py` & `/demo2`)

<p align="center">
  <img src="assets/demo2_meeting_sentinel.gif" alt="Demo 2: Full-Duplex Meeting Sentinel" width="100%" />
</p>

- **Environment**: 4-speaker enterprise design & BOM budget review (**Edinburgh AMI Meeting Corpus `IS1008d`**, 12 streamed ASR text transcript slices over `63.6 s`, `3/12` filler slices; Tractus operates on streamed ASR text transcripts rather than raw audio waveforms) synchronized with a live prototype BOM cost ledger (`€12.50` cap, OLED display trade-off, `€50M` profit target).
- **What It Does**: Classifies each utterance slice in a single pass over **`[FILLER, QUESTION, COMMAND, STATEMENT]`**, driving stateful duplex actions `[SILENT_OBSERVE, DISPATCH_TASK, SUPPLEMENT_TASK]`. Eliminates autoregressive decoding latency by resolving each utterance slice in a single prefill pass with `0` generated tokens across all four scales (`2B`/`4B`/`9B` reliably detect minimal filler probes, whereas `0.8B` is capacity-limited on short fillers).

### 3.3 Demo 3: Multi-Intent Banking & Service Router (`demos/run_demo3_service_router.py` & `/demo3`)

<p align="center">
  <img src="assets/demo3_service_router.gif" alt="Demo 3: Multi-Intent Banking & Service Router" width="100%" />
</p>

- **Environment**: Enterprise customer support workbench streaming ASR text transcripts from authentic **PolyAI MINDS-14** complaint audio (`32.77 s` full track; audio playback is for demonstration while Tractus evaluates the streamed text transcripts) alongside a Customer 360 profile and a 3-row transaction ledger highlighting `-$200.00` (`Unauthorized POS Charge`, `FLAGGED HIGH-RISK`).
- **What It Does**: Replaces `50–85 s` multi-layer IVR dial trees with 4 binary risk/escalation passes (**`Billing Dispute`**, **`Handoff Demand`**, **`Regulatory/Legal Threat`**, **`Emotional Distress`**) plus primary banking-intent routing (`Emergency Card Freeze`, `Direct Debit`, `Billing Inquiry`, `Account Issue`). Executes all 4 binary risk heads and primary intent routing via single-pass prefill with `0` generated tokens per pass, avoiding autoregressive CoT bottlenecks and JSON parse failures.

### 3.4 Demo 4: Real-Time 22-Agent Spatial Dynamics — JevBall (`demos/run_demo4_spatial_dynamics.py` & `/demo4`)

<p align="center">
  <img src="assets/demo4_spatial_dynamics.gif" alt="Demo 4: Real-Time 22-Agent Spatial Dynamics" width="100%" />
</p>

- **Environment**: Real-time 22-agent `105m x 68m` tactical football pitch simulation (`JevBall`) enforcing a **`300.0 ms`** per-tick carrier SLA deadline (contrasted against a `366.0 ms` cloud API baseline).
- **What It Does**: Evaluates 22-agent coordinates, tackle threat ETA, passing-lane clearance, offside margin, and `xG` at each tick, scoring **up to 14 typed tactical candidates** (`PASS`, `SHOT`, `DRIBBLE` maneuvers such as through balls, passes to feet, lateral safe backpasses, clinical/curled shots, and solo dribbles). Scores all 14 tactical candidates in a single prefill pass with `0` generated tokens per simulation tick, achieving **`0%` SLA deadline misses** under the `300.0 ms` per-tick carrier budget on edge `Tractus-0.8B` MLX INT4 without autoregressive decoding delay.

### 3.5 Running the Demos Locally

```bash
# Launch the unified web cockpit (Web UI & Demos 1–3 support --engine mlx or pytorch; --model-size 0.8b/2b/4b/9b)
python demos/run_web_ui.py --engine mlx --model-size 0.8b --port 8080

# Or run individual standalone demos:
python demos/run_demo1_web_agent.py --engine mlx --model-size 0.8b --max-steps 25
python demos/run_demo2_meeting_sentinel.py --engine mlx --model-size 0.8b
python demos/run_demo3_service_router.py --engine mlx --model-size 0.8b
# Demo 4 accepts --engine mlx (Metal GPU) or --engine unified (PyTorch CUDA/MPS/CPU)
python demos/run_demo4_spatial_dynamics.py --mode match --engine mlx --model-size 0.8b
```

---

## 4. Quickstart

Run the packaged quickstart scripts:
```bash
python quickstart.py                               # Tractus-decision single-pass (Thermostasis: Optional Binary Prior Shift -> Softcap C=30 -> Zero-Sum Centering -> 2D e-Thermostat)
python quickstart_dual_mode.py --model Ethan33X/Tractus-4B  # Tractus dual-mode (System 1 + 2D phase-gated System 2)
```

Or invoke `transformers` directly with **Thermostasis** (Optional Binary Prior Shift $\rightarrow$ Softcap `C = 30.0` $\rightarrow$ Zero-Sum Centering $\rightarrow$ 2D Analytic $e$-Thermostat; requires `accelerate` for `device_map="auto"`):

```python
# Requires: pip install torch transformers accelerate
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Ethan33X/Tractus-decision-0.8B"  # Use Ethan33X/Tractus-4B or Tractus-9B for dual-mode System 1+2

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    device_map="auto",  # requires `accelerate`
).eval()

state = "My integration keeps failing with a 401 error since we rotated our API key this morning."
question = "Which support queue should handle this request?"
options = {
    "A": "Billing and subscription invoices",
    "B": "Technical API authentication support",
    "C": "New enterprise sales inquiries",
    "D": "Account closure and retention",
}

messages = [
    {"role": "system", "content": "You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain."},
    {"role": "user", "content": f"State:\n{state}\n\nQuestion:\n{question}\n\nOptions:\n" + "\n".join(f"{k}: {v}" for k, v in options.items()) + "\n\nReply with exactly one option key."},
]

try:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

keys = list(options)
key_ids = []
for k in keys:
    ids = tokenizer.encode(k, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Option key {k!r} must map to a single token, got {ids}.")
    key_ids.append(ids[0])
if len(set(key_ids)) != len(key_ids):
    raise ValueError("Option keys must map to distinct token IDs.")

with torch.inference_mode():
    terminal_logits = model(input_ids=inputs["input_ids"], logits_to_keep=1).logits[0, -1]

# Thermostasis: Optional Binary Prior Shift (asymmetric K=2 in Decision Index 0.3; z_0=0 here) -> Softcap (C=30.0) -> Zero-Sum Centering -> 2D Analytic e-Thermostat
capped = [30.0 * math.tanh(terminal_logits[tid].item() / 30.0) for tid in key_ids]
mean_z = sum(capped) / len(capped)
centered = [v - mean_z for v in capped]
s = sorted(centered, reverse=True)
dz = max(0.0, s[0] - s[1])
s_rival = math.log1p(sum(math.exp(z - s[1]) for z in s[2:])) if len(s) > 2 else 0.0
eff_gap = dz + math.exp(-1.0) * s_rival / (1.0 + math.exp(-(dz - s_rival)))
temp = math.exp(-1.0) + 1.0 / (1.0 + math.exp(-(math.e - eff_gap)))
exp_w = [math.exp((v - max(centered)) / temp) for v in centered]
raw_probs = {k: w / sum(exp_w) for k, w in zip(keys, exp_w)}
chosen = max(raw_probs, key=raw_probs.get)
probs = {k: round(p, 4) for k, p in raw_probs.items()}
print("Selected Option:", chosen, "| Probabilities:", probs)
```

---

## 5. License & Citation

### License
Copyright (c) 2026 Ethan Xie (Shaocong Xie). All rights reserved.
Licensed under the **Modified MIT License** (`license: other`, `license_name: modified-mit`, see [`LICENSE`](./LICENSE)):

- **SME & Developer Commercial Grant**: Free under standard MIT terms for commercial products and internal enterprise workflows when consolidated group trailing 12-month revenue across You and all direct/indirect Affiliates is **<= $10M USD** and consolidated MAU is **<= 10M** (must prominently display `"Powered by Tractus"` on UI/docs once product revenue exceeds **$1M USD** or **1M MAU**).
- **Commercial Scale & MaaS Gates**: A separate commercial license (`xieshaocongethan@gmail.com`) is required prior to deployment if (a) consolidated group trailing 12-month revenue exceeds **$10M USD** or **10M MAU**, or (b) You/Your Affiliates provide commercial **MaaS / hosted inference or fine-tuning API** endpoints whose trailing 12-month MaaS revenue exceeds **$100K USD**.
- **Beneficiary Attribution & Anti-Circumvention**: Custom client deployments are evaluated against the ultimate production beneficiary's consolidated group; entity splitting, SPV shells, or proxy wrappers immediately terminate license rights across all Affiliates. Preserves upstream `Qwen/Qwen3.5-Base` **Apache License 2.0** notice in [`LICENSE`](./LICENSE).

### Citation
```bibtex
@misc{xie2026tractus,
  title        = {Tractus: A Unified Dual-Mode Decision Foundation Model Family with Parameter-Free Thermostasis},
  author       = {Xie, Shaocong},
  year         = {2026},
  howpublished = {\url{https://github.com/XieShaocong33Ethan/Tractus}},
  note         = {Open Weights (0.8B/2B/4B/9B): \url{https://huggingface.co/Ethan33X/Tractus-9B}; Decision Index 0.3 Deliverables: \url{https://huggingface.co/datasets/Ethan33X/tractus-decision-index-0.3}}
}
```
