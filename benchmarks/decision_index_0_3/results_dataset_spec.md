# Results Dataset Specification: `results.jsonl.gz`

This document specifies the official evaluation trajectory deliverables for the **Tractus-decision** model series (`0.8B`, `2B`, `4B`, `9B`) on the **Decision Index 0.3** benchmark.

---

## 1. Official Dataset Repository

To keep the source repository lightweight, the compressed row-by-row probability logs (`results.jsonl.gz`) for all four scales are archived in the official Hugging Face Dataset repository:

- **Primary Distribution**: [Hugging Face Dataset `Ethan33X/tractus-decision-index-0.3`](https://huggingface.co/datasets/Ethan33X/tractus-decision-index-0.3)
- **Scale Directories**:
  - `Tractus-decision-0.8B/results.jsonl.gz` (Skill: **18.07**, Raw: **37.66%**)
  - `Tractus-decision-2B/results.jsonl.gz` (Skill: **29.27**, Raw: **46.83%**)
  - `Tractus-decision-4B/results.jsonl.gz` (Skill: **41.31**, Raw: **55.56%**)
  - `Tractus-decision-9B/results.jsonl.gz` (Skill: **46.03**, Raw: **59.32%**)

---

## 2. Line Schema Specification

Each line in `results.jsonl.gz` is a valid JSON object following the official Decision Index 0.3 schema:

```json
{
  "run_id": "1:BFCL-tool-selection:BFCL-tool-selection:live_simple:live_simple_186-111-0",
  "catalog_id": 1,
  "dataset": "BFCL",
  "status": "ok",
  "latency_ms": 114.6,
  "prediction": "A",
  "probabilities": {
    "A": 0.8412,
    "B": 0.0821,
    "C": 0.0452,
    "D": 0.0315
  },
  "ground_truth": "A",
  "correct": true
}
```

---

## 3. Upload & Verification Command

To sync or verify the results dataset with Hugging Face:

```bash
python scripts/upload_to_hf.py \
  --repo-id Ethan33X/tractus-decision-index-0.3 \
  --repo-type dataset \
  --folder benchmarks/decision_index_0_3
```
