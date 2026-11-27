# Jev Decision Index 0.3 Benchmark & Official Submission Suite

The **Jev Decision Index 0.3** is the standard comprehensive evaluation suite for Decision Foundation Models, evaluating 37 indexed benchmarks (out of 43 active catalogs) spanning 5 core capability domains:

1. **Tools & Automation** (18.3% weight)
2. **Language Understanding** (25.8% weight)
3. **Knowledge & Reasoning** (25.8% weight)
4. **Arts & Human Taste** (10.0% weight)
5. **Retrieval & Classification** (20.0% weight)

---

## 1. Mathematical Formulation & Chance Correction

Decision Index employs strict chance-correction to eliminate random guessing artifacts across multi-choice and classification tasks:

$$\text{Skill} = \frac{\text{Raw Accuracy} - \text{Chance Baseline}}{1 - \text{Chance Baseline}} \times 100$$

When a model's accuracy equals the chance baseline (e.g., 50% on a binary task), its chance-corrected skill is 0. Negative scores are clamped to 0.

---

## 2. Official Results Summary Across the Tractus Family

All models in the **Tractus-decision** family are evaluated in a single causal forward pass (0 generated tokens, 0 Chain-of-Thought, 0 external sandboxes) via **Thermostasis** (optional Binary Prior Shift $\rightarrow$ $\tanh$ logit softcapping `C = 30.0` $\rightarrow$ zero-sum centering $\rightarrow$ parameter-free 2D analytic $e$-thermostat calibration; two optional static binary priors `+3.2` hallucination on RAGTruth and `-0.75` phishing on PhishNChips trigger only on those two asymmetric `K = 2` Decision Index 0.3 tasks):

| Model | Total Parameters | Non-Embedding Backbone | Decision Index 0.3 (Skill) | Raw Accuracy | Breadth Skill | Gen. Tokens | Official Deliverables |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Tractus-decision-9B` / `Tractus-9B`** | 9.24B | 8.12B | **46.03** | **59.32%** | 44.71 | **0** | [`Tractus-decision-9B/`](./Tractus-decision-9B) |
| **`Tractus-decision-4B` / `Tractus-4B`** | 4.21B | 3.57B | **41.31** | **55.56%** | 40.00 | **0** | [`Tractus-decision-4B/`](./Tractus-decision-4B) |
| **`Tractus-decision-2B` / `Tractus-2B`** | 1.88B | 1.48B | **29.27** | **46.83%** | 26.73 | **0** | [`Tractus-decision-2B/`](./Tractus-decision-2B) |
| **`Tractus-decision-0.8B` / `Tractus-0.8B`** | 0.82B | 0.58B | **18.07** | **37.66%** | 17.10 | **0** | [`Tractus-decision-0.8B/`](./Tractus-decision-0.8B) |

*Note: Single-pass `Tractus-decision` eliminates autoregressive decoding latency by scoring all candidate keys in a single prefill pass with `0` generated tokens and a `99.9%` smaller readout projection.*

### Capability Area Breakdown (Chance-Corrected Skill / Raw Accuracy)

| Capability Area | Suite Weight | `Tractus-decision-0.8B` | `Tractus-decision-2B` | `Tractus-decision-4B` | `Tractus-decision-9B` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tools & Automation** | 18.3% | **23.84** *(32.63% raw)* | **43.24** *(49.01% raw)* | **53.63** *(58.81% raw)* | **58.15** *(62.03% raw)* |
| **Language Understanding** | 25.8% | **27.65** *(48.99% raw)* | **41.16** *(59.31% raw)* | **53.19** *(67.67% raw)* | **60.40** *(73.21% raw)* |
| **Retrieval & Classification** | 20.0% | **15.26** *(33.30% raw)* | **31.79** *(49.65% raw)* | **37.97** *(53.24% raw)* | **37.44** *(54.77% raw)* |
| **Knowledge & Reasoning** | 25.8% | **10.45** *(33.72% raw)* | **11.52** *(33.10% raw)* | **26.06** *(44.01% raw)* | **31.71** *(47.33% raw)* |
| **Arts & Human Taste** | 10.0% | **8.10** *(36.48% raw)* | **13.79** *(40.48% raw)* | **34.19** *(52.87% raw)* | **40.95** *(58.53% raw)* |
| **Balanced Overall Index** | **100%** | **18.07** *(37.66% raw)* | **29.27** *(46.83% raw)* | **41.31** *(55.56% raw)* | **46.03** *(59.32% raw)* |

---

## 3. Submission Artifact Inventory

This directory contains the complete set of required artifacts for official submission to the Hugging Face Decision Index Leaderboard (both at root for `Tractus-decision-0.8B` and in scale-specific folders `Tractus-decision-{0.8B,2B,4B,9B}/`):

1. **`scores.json`**: Top-level aggregate scores, area breakdowns, request counts, and latency statistics.
2. **`benchmark-summary.json`**: Granular itemized performance metrics across all 37 indexed benchmarks (out of 43 active catalogs).
3. **`index.json`**: Official mathematical index formulation and benchmark weights.
4. **`environment.json`**: Hardware, CUDA version, driver, and engine fingerprint.
5. **`SUBMISSION_DRAFT.md`**: Ready-to-copy PR description and Hugging Face Space discussion post.
6. **`results_dataset_spec.md`**: Specifications for accessing and verifying the full `results.jsonl.gz` deliverables in [`Ethan33X/tractus-decision-index-0.3`](https://huggingface.co/datasets/Ethan33X/tractus-decision-index-0.3).
