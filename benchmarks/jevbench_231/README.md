# JevBench 231 Reproduction & Submission Package

**JevBench 231** is the authoritative benchmark for discrete symbolic reasoning, causal instruction auditing, and structured workflow control.

## 1. Benchmark Results Across the Tractus Family (Unified `chat_template` BF16 256K)

All evaluations use the unified `chat_template` BF16 256K contract (`max_new_tokens=8192` for `2B/4B/9B` in dual-mode `Tractus`; Hugging Face `4096` for `0.8B`), zero out-of-fold calibration fitting, and zero multi-pass option permutations:

| Model Pair | Scale | Recommended Mode | `Tractus-decision` (Single-Pass, 231) | `Tractus` (Dual-Mode, 231) | Easy / Standard / Hard (`Rec. Mode`) | Zero-Token Exits (`Rec.`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Tractus-9B` / `Tractus-decision-9B`** | 9.24B | **Dual-Mode (`Tractus`, 8K)** | 183 / 231 (79.22%) | **210 / 231 (90.91%)** | **48/48 \| 72/72 \| 90/111** | **71.0%** |
| **`Tractus-4B` / `Tractus-decision-4B`** | 4.21B | **Dual-Mode (`Tractus`, 8K)** | 176 / 231 (76.19%) | **204 / 231 (88.31%)** | **48/48 \| 69/72 \| 87/111** | **43.7%** |
| **`Tractus-decision-0.8B` / `Tractus-0.8B`** | 0.82B | **Single-Pass (`Tractus-decision`)** | **168 / 231 (72.73%)** | 162 / 231 (70.13%) | **48/48 \| 66/72 \| 54/111** | **100%** |
| **`Tractus-decision-2B` / `Tractus-2B`** | 1.88B | **Single-Pass (`Tractus-decision`)** | **160 / 231 (69.26%)** *(161 in separate BF16 batching pass)* | 151 / 231 (65.37%) | **48/48 \| 66/72 \| 46/111** | **100%** |

*Note: Single-pass `Tractus-decision` applies **Thermostasis** (optional Binary Prior Shift $\rightarrow$ $\tanh$ logit softcapping `C = 30.0` $\rightarrow$ zero-sum centering $\rightarrow$ parameter-free 2D analytic $e$-thermostat calibration; `z_0 = 0` on JevBench) with `0` generated reasoning tokens; the 1-item difference at 2B (`160/231` vs `161/231`) reflects a BF16 left-padding/batching boundary shift across evaluation runs rather than generation budget. `Tractus` pairs the same calibrated System 1 pass with a Joint 2D Confidence-Entropy Phase Gate (`phase_zone == SYSTEM_2 and conf < 0.80`) and Defensive Safety Ratchet.*

## 2. Verification & Live Inference Usage

```bash
# Verify archived scores.json & structural invariants across all 4 scales
python eval_jevbench.py --verify-only

# Run live single-pass or dual-mode inference against a Tractus checkpoint
python ../../quickstart.py --model Ethan33X/Tractus-decision-0.8B
python ../../quickstart_dual_mode.py --model Ethan33X/Tractus-4B
```

## 3. Contents
- `scores.json`: Structured metrics covering all 8 Tractus models (`Tractus-*` & `Tractus-decision-*`) on JevBench 231.
- `eval_jevbench.py`: Verification driver (`--verify-only`) auditing archived `scores.json` invariants.
- `SUBMISSION.md`: Standardized leaderboard submission metadata template.
