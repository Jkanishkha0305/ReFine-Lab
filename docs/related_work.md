# Related Work & Reference Papers

Papers discussed during development of RV-GRPO, organized by relevance.

## Directly Relevant (Cite in Paper)

### GRPO & RLVR Foundations

- **DeepSeek-R1** — Original GRPO algorithm, verifiable rewards for math/code reasoning
  - https://arxiv.org/abs/2501.12948

- **RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents** (Jul 2025)
  - Closest competitor. RLVR with emotion rewards from simulated users. Qwen 7B + PPO/GRPO.
  - Our differentiator: domain-specific MI rubrics vs generic emotion scores, SLMs vs 7B.
  - https://hf.co/papers/2507.03112

- **Clinical-R1 (CRPO)** (Nov 2025)
  - Multi-objective verifiable GRPO for clinical reasoning. 3B model.
  - Rule-based rewards for accuracy/faithfulness, not behavioral alignment.
  - https://hf.co/papers/2512.00601

- **Psyche-R1** (Aug 2025)
  - Chinese psychological LLM. GRPO + SFT hybrid. 7B model.
  - Different domain (Chinese), uses GRPO for reasoning not behavioral alignment.
  - https://hf.co/papers/2508.10848

- **Multi-Objective GRPO for Safe and Aligned LLM Generation** (Mar 2025)
  - Multi-label reward regression model with GRPO. 0.5B, 7B, 14B models.
  - https://hf.co/papers/2503.21819

### Mental Health & Therapy LLMs

- **ChatCounselor** (Sep 2023)
  - LLM for mental health, trained on Psych8k (260 real therapy interviews).
  - SFT only, no RL. CounselingBench evaluation.
  - https://hf.co/papers/2309.15461

- **PsychēChat** (Jan 2026)
  - Emotion shift tracking + safety risk analysis for counseling.
  - Agent-based multi-module design, no GRPO/RL.
  - https://hf.co/papers/2601.12392

- **PsyLLM: Beyond Empathy** (May 2025)
  - Integrates diagnostic (DSM/ICD) + therapeutic reasoning (CBT, ACT).
  - Data synthesis pipeline. No GRPO.
  - https://hf.co/papers/2505.15715

- **TIDE: The Pursuit of Empathy — SLMs for PTSD** (May 2025)
  - SLMs (0.5-5B) for PTSD dialogue. SFT only. Found "empathy ceiling" in SLMs.
  - Our RV-GRPO could potentially break that ceiling with RL.
  - https://hf.co/papers/2505.15065

- **PsyLite** (Jun 2025)
  - Lightweight psych counseling on InternLM 7B. ORPO + conditional RAG.
  - GGUF Q4 deployment (5GB). Different alignment method (ORPO vs GRPO).
  - https://hf.co/papers/2506.21536

- **MCTSr-Zero: Self-Reflective Psychological Counseling** (May 2025)
  - MCTS + LLM for open-ended counseling dialogues. PsyEval benchmark.
  - https://hf.co/papers/2505.23229

### SLM & On-Device

- **Menta** (Dec 2025)
  - On-device SLM for mental health prediction (classification, not generation).
  - Closest to our deployment story.
  - https://arxiv.org/abs/2412.18129

### Evaluation Benchmarks

- **MindEval** — Multi-turn therapeutic evaluation benchmark
  - https://github.com/SWORDHealth/mind-eval

- **CounselBench** — Expert-evaluated counseling quality
  - https://github.com/llm-eval-mental-health/CounselBench

- **H2HTalk: Evaluating LLMs as Emotional Companion** (Jul 2025)
  - 4,650 scenarios, Secure Attachment Persona module, 50 LLMs benchmarked.
  - https://hf.co/papers/2507.03543

## GRPO Variants & Extensions (Background)

- **iGRPO: Self-Feedback-Driven LLM Reasoning** (Feb 2026)
  - Two-stage GRPO with model-generated drafts. SOTA on AIME24/25.
  - https://hf.co/papers/2602.09000

- **AMIR-GRPO: Inducing Implicit Preference Signals into GRPO** (Jan 2026)
  - Augments GRPO with DPO-style contrastive regularizer.
  - https://hf.co/papers/2601.03661

- **It Takes Two: Your GRPO Is Secretly DPO** (Oct 2025)
  - Shows 2-rollout GRPO matches 16-GRPO. 70% training time reduction.
  - https://hf.co/papers/2510.00977

- **GRPO for Speech Recognition** (Sep 2025)
  - Rule-based rewards for ASR. Shows GRPO works beyond text.
  - https://hf.co/papers/2509.01939

- **GRPO-CARE: Consistency-Aware RL for Multimodal Reasoning** (Jun 2025)
  - Two-tiered reward for answer correctness + reasoning coherence.
  - https://hf.co/papers/2506.16141

- **DeepVideo-R1: Video GRPO** (Jun 2025)
  - Reg-GRPO for video LLMs. Difficulty-aware data augmentation.
  - https://hf.co/papers/2506.07464

- **CheXPO-v2: Preference Optimization for Chest X-ray VLMs** (Dec 2025)
  - Knowledge Graph Consistency Reward with GRPO. Medical imaging.
  - https://hf.co/papers/2512.17213

## Surveys

- **A Technical Survey of RL Techniques for LLMs** (Jul 2025)
  - Comprehensive overview: PPO, GRPO, DPO, RLHF, RLAIF, RLVR.
  - https://hf.co/papers/2507.04136

## Personalization & Future Work

- **Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward** (Apr 2025)
  - Intrinsic motivation for user modeling. Education & fitness domains.
  - Relevant for per-user LoRA adapters (paper #2 direction).
  - https://hf.co/papers/2504.03206

## Datasets Used

- **ShenLab/MentalChat16K** — 9.7K mental health conversations (SFT primary)
  - https://huggingface.co/datasets/ShenLab/MentalChat16K

- **Amod/mental_health_counseling_conversations** — Real counselor conversations
  - https://huggingface.co/datasets/Amod/mental_health_counseling_conversations

- **tcabanski/mental_health_counseling_responses** — LLM-rated counseling responses
  - https://huggingface.co/datasets/tcabanski/mental_health_counseling_responses
