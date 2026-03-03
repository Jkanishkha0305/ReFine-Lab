# SLURM Batch Submission Scripts (BigPurple HPC)

## Quick Start

```bash
# On BigPurple login node:
cd ReFine-Lab
mkdir -p slurm_logs

# Option A: Pilot first (recommended, ~2 hrs, 1 GPU)
bash slurm/submit_all.sh --pilot

# Option B: Full pipeline with job chaining (~12 hrs total)
bash slurm/submit_all.sh
```

## Individual Jobs

```bash
# Submit individually:
sbatch slurm/00_setup.sbatch       # first-time setup (30 min)
sbatch slurm/01_sft.sbatch         # SFT all 3 models (3-4 hrs, 2 GPUs)
sbatch slurm/02_alignment.sbatch   # RV-GRPO + DPO + Judge (5-6 hrs, 2 GPUs)
sbatch slurm/03_eval.sbatch        # evaluation (1-2 hrs, 1 GPU)
sbatch slurm/pilot.sbatch          # quick pilot on Qwen only (2 hrs, 1 GPU)
```

## GPU Strategy

| Job | GPUs | Strategy |
|-----|------|----------|
| SFT | 2 | LLaMA + Qwen parallel on GPU 0/1, then Phi alone |
| Alignment | 2 | Same parallel strategy for each method |
| Eval | 1 | Sequential evaluation |
| Pilot | 1 | Single model (Qwen 1.5B) |

## Monitoring

```bash
squeue -u $USER              # check job status
scancel <JOB_ID>             # cancel a job
tail -f slurm_logs/sft_*.out # watch training progress
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS  # job details
```

## Time Estimates (2x A100 80GB)

| Stage | Time | GPU Hours |
|-------|------|-----------|
| SFT (3 models) | ~3 hrs | 6 |
| RV-GRPO (3 models) | ~3 hrs | 6 |
| DPO (3 models) | ~1.5 hrs | 3 |
| Judge-GRPO (3 models) | ~3 hrs | 6 |
| Evaluation | ~1.5 hrs | 1.5 |
| **Total** | **~12 hrs** | **~22.5** |

With 2 GPUs running models in parallel, wall-clock time is ~12 hours for the full pipeline.
