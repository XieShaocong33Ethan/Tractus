# Official JevBench 231 Benchmark Submission & Audit Log

This document provides structured verification metadata for submitting the **Tractus** model family (`0.8B`, `2B`, `4B`, `9B`) to public JevBench leaderboards.

---

## 1. Model Identity & Endpoints

| Variant | Hugging Face Model ID | Base Model | Total Params | Non-Embedding |
| :--- | :--- | :--- | :---: | :---: |
| **`Tractus-9B` / `Tractus-decision-9B`** | [`Ethan33X/Tractus-9B`](https://huggingface.co/Ethan33X/Tractus-9B) / [`Tractus-decision-9B`](https://huggingface.co/Ethan33X/Tractus-decision-9B) | `Qwen/Qwen3.5-9B-Base` | 9.24B | 8.12B |
| **`Tractus-4B` / `Tractus-decision-4B`** | [`Ethan33X/Tractus-4B`](https://huggingface.co/Ethan33X/Tractus-4B) / [`Tractus-decision-4B`](https://huggingface.co/Ethan33X/Tractus-decision-4B) | `Qwen/Qwen3.5-4B-Base` | 4.21B | 3.57B |
| **`Tractus-2B` / `Tractus-decision-2B`** | [`Ethan33X/Tractus-2B`](https://huggingface.co/Ethan33X/Tractus-2B) / [`Tractus-decision-2B`](https://huggingface.co/Ethan33X/Tractus-decision-2B) | `Qwen/Qwen3.5-2B-Base` | 1.88B | 1.48B |
| **`Tractus-0.8B` / `Tractus-decision-0.8B`** | [`Ethan33X/Tractus-0.8B`](https://huggingface.co/Ethan33X/Tractus-0.8B) / [`Tractus-decision-0.8B`](https://huggingface.co/Ethan33X/Tractus-decision-0.8B) | `Qwen/Qwen3.5-0.8B-Base` | 0.82B | 0.58B |

---

## 2. Evaluation Protocol & Verification Outputs

- **Unified Evaluation Contract**: Canonical `chat_template` BF16 256K (`max_new_tokens=8192` for `2B/4B/9B`, HF `4096` for `0.8B`), zero out-of-fold fitting.
- **Single-Pass System 1 (`Tractus-decision`)**: Terminal option-key probability projection via **Thermostasis** (optional Binary Prior Shift $\rightarrow$ $\tanh$ logit softcapping `C = 30.0` $\rightarrow$ zero-sum centering $\rightarrow$ parameter-free 2D Analytic $e$-Thermostat calibration; `0` generated reasoning tokens, `z_0 = 0` on JevBench).
- **Dual-Mode System 1 + System 2 (`Tractus`)**: Same calibrated System 1 pass followed by in-place System 2 escalation when `phase_zone == SYSTEM_2 and conf < 0.80`, protected by the Defensive Safety Ratchet (`<=2B` recommended in `Tractus-decision` single-pass mode).

| Model & Mode | Overall (231) | Easy (48) | Standard (72) | Hard (111) | Raw Single-Pass PFR | Zero-Token Exits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Tractus-9B` (`Tractus`, 8K)** | **210 / 231 (90.91%)** | **48 / 48 (100%)** | **72 / 72 (100%)** | **90 / 111 (81.08%)** | 10.82% | **71.0%** |
| **`Tractus-decision-9B` (`Tractus-decision`)** | **183 / 231 (79.22%)** | **48 / 48 (100%)** | 70 / 72 (97.22%) | 65 / 111 (58.56%) | 10.82% | **100%** |
| **`Tractus-4B` (`Tractus`, 8K)** | **204 / 231 (88.31%)** | **48 / 48 (100%)** | 69 / 72 (95.83%) | **87 / 111 (78.38%)** | 14.29% | **43.7%** |
| **`Tractus-decision-4B` (`Tractus-decision`)** | **176 / 231 (76.19%)** | **48 / 48 (100%)** | 67 / 72 (93.06%) | 61 / 111 (54.95%) | 14.29% | **100%** |
| **`Tractus-decision-0.8B` (`Tractus-decision`, rec. <=2B)** | **168 / 231 (72.73%)** | **48 / 48 (100%)** | 66 / 72 (91.67%) | 54 / 111 (48.65%) | 16.02% | **100%** |
| **`Tractus-0.8B` (`Tractus`, HF 4K)** | 162 / 231 (70.13%) | **48 / 48 (100%)** | 65 / 72 (90.28%) | 49 / 111 (44.14%) | 16.02% | — |
| **`Tractus-decision-2B` (`Tractus-decision`, rec. <=2B)** | **160 / 231 (69.26%)** | **48 / 48 (100%)** | 66 / 72 (91.67%) | 46 / 111 (41.44%) | 15.58% | **100%** |
| **`Tractus-2B` (`Tractus`, 8K)** | 151 / 231 (65.37%) | **48 / 48 (100%)** | 58 / 72 (80.56%) | 45 / 111 (40.54%) | 15.58% | — |

---

## 3. Automated Integrity Check

Verify all deliverables match schema and invariants prior to submission:
```bash
python benchmarks/jevbench_231/eval_jevbench.py --verify-only
```
