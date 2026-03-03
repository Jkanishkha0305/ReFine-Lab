# ReFine-Lab: Rubric-Verifiable GRPO for Behavioral Alignment in Mental Health SLMs

Extending GRPO beyond math/code to therapeutic conversation using clinically-grounded rubric rewards from Motivational Interviewing (MI). Train, align, evaluate, and deploy small language models (1-3.8B) for on-device mental health support.

## Research Question

> Can clinically-grounded rubric-verifiable rewards enable GRPO to produce measurably better therapeutic behavioral alignment in SLMs compared to SFT, DPO, and LLM-judge baselines?

## Models

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| LLaMA 3.2 | 1B | `meta-llama/Llama-3.2-1B-Instruct` |
| Qwen 2.5 | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Phi-4 Mini | 3.8B | `microsoft/Phi-4-mini-instruct` |

## Rubric-Verifiable Rewards (R1-R5)

The core contribution: 5 binary rewards grounded in Motivational Interviewing.

| Reward | Signal | How It's Verified |
|--------|--------|-------------------|
| R1: Open Question | Response asks open-ended (not yes/no) question | Regex + sentence parsing |
| R2: Emotion Reflection | Model reflects user's emotional state | NLI with DeBERTa |
| R3: No Premature Advice | No directive advice in early exchanges | POS tagging + keywords |
| R4: Validation Before Redirect | Validates feelings before suggesting | Pattern matching |
| R5: Length Appropriateness | Response is 50-200 tokens | Token count |

Combined reward: `R = w1*R1 + w2*R2 + w3*R3 + w4*R4 + w5*R5`, normalized to [0, 1].

## Quick Start

### 1. Setup (Lambda Labs / Cloud GPU)

```bash
git clone https://github.com/jkanishkha0305/ReFine-Lab.git
cd ReFine-Lab
cp .env.example .env   # fill in HF_TOKEN, GROQ_API_KEY, WANDB_API_KEY

# Standard setup
bash setup.sh

# Or faster setup with UV
bash setup_uv.sh
```

### 2. Verify Installation

```bash
source .venv/bin/activate
make check-gpu        # verify GPU
make test-rewards     # test rubric reward functions
```

### 3. Pilot Run (recommended first)

Quick validation on Qwen 1.5B only (~2 hours):

```bash
make pilot            # SFT (1K samples) + RV-GRPO (200 prompts) + DPO (500 samples)
make pilot-eval       # evaluate pilot models
```

### 4. Full Training Pipeline

```bash
# Step 0: Generate preference pairs for DPO (one-time, uses Groq free tier)
python data/generate_preferences.py --api_provider groq --num_samples 5000

# Step 1: SFT all 3 models
make train-sft

# Step 2: Alignment (run independently, all start from SFT checkpoints)
make train-grpo          # RV-GRPO (ours)
make train-dpo           # DPO baseline
make train-judge-grpo    # LLM-judge GRPO baseline

# Step 3: Evaluation
make eval                # rubric metrics + external benchmarks

# Step 4: Ablation study
make ablation            # leave-one-out analysis of R1-R5

# Step 5: Quantize & benchmark
make quantize            # GGUF Q4 for on-device deployment
```

Or run everything at once:

```bash
make all
```

## Pipeline Overview

```
MentalChat16K + ESConv
        |
    [prepare_sft_data.py]
        |
   SFT Dataset (~15K)
        |
   [train_sft.py] ──────────────────────────────────────────┐
        |                                                    |
   SFT Checkpoint                                           |
    /       |        \                                       |
   /        |         \                                      |
[RV-GRPO]  [DPO]  [LLM-Judge GRPO]                   [SFT baseline]
   |        |         |                                      |
   └────────┴─────────┴──────────────────────────────────────┘
                          |
                   [Evaluation]
              behavioral_metrics.py
             external_benchmarks.py
                   visualize.py
                          |
                   [Deployment]
                   quantize.py
                   benchmark.py
```

## Project Structure

```
ReFine-Lab/
├── data/
│   ├── rubric_rewards.py          # R1-R5 reward implementations (core contribution)
│   ├── prepare_sft_data.py        # merge MentalChat16K + ESConv
│   └── generate_preferences.py    # generate DPO pairs via Groq/Cerebras
│
├── experiments/
│   ├── sft/
│   │   └── train_sft.py           # SFT training for all 3 models
│   ├── grpo/
│   │   ├── train_rv_grpo.py       # MAIN: GRPO with rubric rewards
│   │   ├── train_llm_judge_grpo.py # baseline: GRPO with LLM-judge
│   │   └── run_ablation.py        # leave-one-out & weight sweep
│   └── alignment/
│       ├── train_dpo.py           # DPO baseline
│       ├── train_orpo.py          # ORPO (optional)
│       └── train_kto.py           # KTO (optional)
│
├── evaluation/
│   ├── behavioral_metrics.py      # compute R1-R5 on test set
│   ├── external_benchmarks.py     # CounselBench, MindEval, Empathy
│   ├── visualize.py               # radar charts, tables, figures
│   └── eval_config.json           # batch evaluation config
│
├── deployment/
│   ├── quantize.py                # LoRA merge + GGUF conversion
│   └── benchmark.py               # inference speed benchmarks
│
├── Makefile                       # all commands
├── setup.sh                       # Lambda Labs setup (pip)
├── setup_uv.sh                    # fast setup (UV)
├── pyproject.toml                 # Python packaging
├── requirements.txt               # pip fallback
└── .env.example                   # API keys template
```

## Experimental Design

4 methods x 3 models = 12 training runs:

| Method | Description | Script |
|--------|-------------|--------|
| SFT only | Supervised fine-tuning baseline | `train_sft.py` |
| SFT + DPO | Preference optimization baseline | `train_dpo.py` |
| SFT + GRPO (LLM-judge) | GRPO with generic reward model | `train_llm_judge_grpo.py` |
| **SFT + RV-GRPO (ours)** | **GRPO with rubric-verifiable rewards** | **`train_rv_grpo.py`** |

## Evaluation

### Behavioral Metrics (R1-R5)

Measured on 500 held-out test prompts:

- Open Question Rate
- Emotion Reflection Rate
- Premature Advice Rate (lower is better)
- Validation Rate
- Response Length Distribution

### External Benchmarks

- **CounselBench**: empathy, clinical accuracy, active listening, safety (1-5)
- **MindEval**: clinical competence, ethical conduct, therapeutic alliance (1-6)
- **Empathy Classification**: go_emotions emotion recognition

### Ablation Study

- Leave-one-out: remove each R1-R5 individually
- Weight sweep: optimize reward weight combinations

## Hardware

| Setup | GPU | What You Can Run |
|-------|-----|-----------------|
| Pilot | A10G (24GB) | `make pilot` — Qwen 1.5B only |
| Full | A100 (80GB) | `make all` — all 3 models, all methods |

Single GPU is sufficient. No multi-GPU needed.

**Estimated costs (Lambda Labs A100 @ $1.10/hr):**
- Pilot: ~$2-3
- Full pipeline: ~$70-90

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
HF_TOKEN=hf_...              # required (for gated models like LLaMA)
WANDB_API_KEY=...             # optional (experiment tracking)
GROQ_API_KEY=gsk_...          # for preference generation (free tier)
```

## Key Dependencies

- PyTorch >= 2.1.0
- Transformers >= 4.38.0
- TRL >= 0.14.0 (for GRPOTrainer)
- PEFT >= 0.8.0
- Accelerate >= 0.27.0

## Related Work

- [RLVER](https://hf.co/papers/2507.03112) — RLVR with emotion rewards (general EQ, 7B)
- [Clinical-R1](https://hf.co/papers/2512.00601) — Multi-objective GRPO for clinical reasoning
- [Psyche-R1](https://hf.co/papers/2508.10848) — Chinese psych LLM with GRPO
- [TIDE](https://hf.co/papers/2505.15065) — SLM empathy for PTSD (SFT only)
- [PsyLite](https://hf.co/papers/2506.21536) — Lightweight psych counseling with ORPO

## Citation

```bibtex
@misc{kanishkha2026rvgrpo,
  title={RV-GRPO: Rubric-Verifiable Group Relative Policy Optimization for Behavioral Alignment in Mental Health SLMs},
  author={Kanishkha, J},
  year={2026},
  url={https://github.com/jkanishkha0305/ReFine-Lab}
}
```

## License

MIT License — See [LICENSE](LICENSE) for details.

## Acknowledgments

Built with [HuggingFace TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft), and [Transformers](https://github.com/huggingface/transformers).
