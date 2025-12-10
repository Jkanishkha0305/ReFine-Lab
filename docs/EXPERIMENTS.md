# ReFine-Lab Experiments

## Active Experiments

### EXP-001: GRPO Fine-tuning on LLaMA-3.2-3B
- **Objective**: Improve instruction following on reasoning tasks
- **Status**: 🟡 Running (NYU BigPurple GPU cluster)
- **Config**: `configs/grpo_llama3b.yaml`
- **Baseline**: LLaMA-3.2-3B-Instruct (HuggingFace)
- **Eval**: MMLU, GSM8K, HellaSwag

### EXP-002: Knowledge Distillation — Qwen2.5-72B → Qwen2.5-7B
- **Objective**: Transfer reasoning capability to smaller model
- **Status**: 🟡 Running
- **Config**: `configs/distill_qwen.yaml`
- **Teacher**: Qwen2.5-72B | **Student**: Qwen2.5-7B

### EXP-003: Multi-turn RLHF with Reward Model
- **Objective**: Train reward model for multi-turn dialogue quality
- **Status**: 🔵 Planned
- **Config**: TBD

## Results

| Model | Method | MMLU | GSM8K | HellaSwag |
|-------|--------|------|-------|-----------|
| LLaMA-3.2-3B-Instruct (baseline) | — | 63.4 | 77.7 | 79.2 |
| LLaMA-3.2-3B (GRPO) | EXP-001 | TBD | TBD | TBD |
| Qwen2.5-7B (KD from 72B) | EXP-002 | TBD | TBD | TBD |

## HPC Setup (NYU BigPurple)

Cluster config: 2× NVIDIA A100 80GB (parallel strategy: LLaMA + Qwen simultaneously, then Phi-4 alone).
See `scripts/slurm/` for batch submission scripts.
