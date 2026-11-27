# Submission Draft: Tractus Model Series (0.8B / 2B / 4B / 9B) on Decision Index 0.3

This document contains the submission text for opening a GitHub Pull Request or Hugging Face Space Discussion on the official Decision Index repository.

---

### PR Title
`[Submission] Tractus-decision Series (0.8B, 2B, 4B, 9B): Open-Weight Single-Pass Causal Decision Models`

### PR Body / Discussion Post

```markdown
### Model Information
- **Model Series**: Tractus-decision (`0.8B`, `2B`, `4B`, `9B`) & Dual-Mode Tractus (`0.8B`, `2B`, `4B`, `9B`)
- **Organization / Author**: Tractus Research (Shaocong Xie)
- **Model Repositories**:
  - `0.8B`: [Ethan33X/Tractus-decision-0.8B](https://huggingface.co/Ethan33X/Tractus-decision-0.8B) / [Ethan33X/Tractus-0.8B](https://huggingface.co/Ethan33X/Tractus-0.8B)
  - `2B`: [Ethan33X/Tractus-decision-2B](https://huggingface.co/Ethan33X/Tractus-decision-2B) / [Ethan33X/Tractus-2B](https://huggingface.co/Ethan33X/Tractus-2B)
  - `4B`: [Ethan33X/Tractus-decision-4B](https://huggingface.co/Ethan33X/Tractus-decision-4B) / [Ethan33X/Tractus-4B](https://huggingface.co/Ethan33X/Tractus-4B)
  - `9B`: [Ethan33X/Tractus-decision-9B](https://huggingface.co/Ethan33X/Tractus-decision-9B) / [Ethan33X/Tractus-9B](https://huggingface.co/Ethan33X/Tractus-9B)
- **Evaluation Deliverables Dataset**: [Ethan33X/tractus-decision-index-0.3](https://huggingface.co/datasets/Ethan33X/tractus-decision-index-0.3)
- **Pipeline Tag**: `decision-model`
- **Context Length**: 262,144 tokens (Native 256K, Zero Truncation)
- **License**: Modified MIT License (`<= $10M` consolidated revenue / `<= 10M` MAU; `<= $100K` commercial MaaS API; see `LICENSE`)

---

### Executive Summary

We submit official evaluation results for the **Tractus-decision** open-weight model family (`0.8B`, `2B`, `4B`, `9B`) on the **Decision Index 0.3** suite. Every model operates strictly as a pure System 1 single-pass causal decision engine:
- Evaluates normalized probability distributions directly at the terminal position in a single forward pass.
- Generates **0 free-form text tokens, uses 0 Chain-of-Thought (CoT), and invokes 0 external sandboxes**.
- Calibrated via **Thermostasis** on terminal option-key logits: optional Binary Prior Shift (two static priors `+3.2` RAGTruth hallucination and `-0.75` PhishNChips phishing on asymmetric `K = 2`) $\rightarrow$ $\tanh$ logit softcapping (`C = 30.0`) $\rightarrow$ zero-sum centering $\rightarrow$ parameter-free 2D analytic $e$-thermostat calibration.

---

### Benchmark Results Summary (Decision Index 0.3)

| Model | Total Params | Non-Embedding Backbone | Decision Index 0.3 (Skill) | Raw Accuracy | Breadth Skill | Gen. Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tractus-decision-9B** | 9.24B | 8.12B | **46.03** | **59.32%** | 44.71 | **0** |
| **Tractus-decision-4B** | 4.21B | 3.57B | **41.31** | **55.56%** | 40.00 | **0** |
| **Tractus-decision-2B** | 1.88B | 1.48B | **29.27** | **46.83%** | 26.73 | **0** |
| **Tractus-decision-0.8B** | 0.82B | 0.58B | **18.07** | **37.66%** | 17.10 | **0** |

*(Note: Single-pass `Tractus-decision` scores all candidate slots in a single forward pass via **Thermostasis**: optional Binary Prior Shift $\rightarrow$ $\tanh$ logit softcapping `C = 30.0` $\rightarrow$ zero-sum centering $\rightarrow$ parameter-free 2D analytic $e$-thermostat calibration).*

#### Domain Area Breakdown (Chance-Corrected Skill)

| Capability Area | Weight | Tractus-decision-0.8B | Tractus-decision-2B | Tractus-decision-4B | Tractus-decision-9B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tools & Automation** | 18.3% | **23.84** *(32.63% raw)* | **43.24** *(49.01% raw)* | **53.63** *(58.81% raw)* | **58.15** *(62.03% raw)* |
| **Language Understanding** | 25.8% | **27.65** *(48.99% raw)* | **41.16** *(59.31% raw)* | **53.19** *(67.67% raw)* | **60.40** *(73.21% raw)* |
| **Retrieval & Classification** | 20.0% | **15.26** *(33.30% raw)* | **31.79** *(49.65% raw)* | **37.97** *(53.24% raw)* | **37.44** *(54.77% raw)* |
| **Knowledge & Reasoning** | 25.8% | **10.45** *(33.72% raw)* | **11.52** *(33.10% raw)* | **26.06** *(44.01% raw)* | **31.71** *(47.33% raw)* |
| **Arts & Human Taste** | 10.0% | **8.10** *(36.48% raw)* | **13.79** *(40.48% raw)* | **34.19** *(52.87% raw)* | **40.95** *(58.53% raw)* |
| **Balanced Overall Index** | **100%** | **18.07** *(37.66% raw)* | **29.27** *(46.83% raw)* | **41.31** *(55.56% raw)* | **46.03** *(59.32% raw)* |

---

### Verification & Reproduction Artifacts

All evaluation deliverables across all four model scales (`index.json`, `scores.json`, `benchmark-summary.json`, `environment.json`, and `results.jsonl.gz`) are verified and available in [`Ethan33X/tractus-decision-index-0.3`](https://huggingface.co/datasets/Ethan33X/tractus-decision-index-0.3).

Thank you for reviewing our submission!
```
