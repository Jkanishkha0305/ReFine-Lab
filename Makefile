# ============================================================
# ReFine-Lab: Reproducible Research Commands
# ============================================================
# Usage:
#   make setup          — First-time setup on Lambda Labs / cloud GPU
#   make train-sft      — Run SFT on all 3 models
#   make train-grpo     — Run RV-GRPO on all 3 models
#   make eval           — Run full evaluation suite
# ============================================================

SHELL := /bin/bash
PYTHON := python3
MODELS := llama qwen phi

# ── Setup ────────────────────────────────────────────────────

.PHONY: setup setup-uv check-gpu

setup:
	@bash setup.sh

setup-uv:
	@bash setup_uv.sh

check-gpu:
	@$(PYTHON) -c "\
	import torch; \
	print(f'PyTorch: {torch.__version__}'); \
	print(f'CUDA: {torch.cuda.is_available()}'); \
	print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); \
	print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB' if torch.cuda.is_available() else ''); \
	"

# ── Data Preparation ────────────────────────────────────────

.PHONY: data-sft data-preferences

data-sft:
	$(PYTHON) data/prepare_sft_data.py --output_dir ./data/processed

data-preferences:
	$(PYTHON) data/generate_preferences.py \
		--api_provider openai \
		--num_samples 5000 \
		--output_dir ./data/processed/preferences

# ── SFT Training ────────────────────────────────────────────

.PHONY: train-sft train-sft-llama train-sft-qwen train-sft-phi

train-sft: train-sft-llama train-sft-qwen train-sft-phi

train-sft-llama:
	$(PYTHON) experiments/sft/train_sft.py --model llama --use_wandb

train-sft-qwen:
	$(PYTHON) experiments/sft/train_sft.py --model qwen --use_wandb

train-sft-phi:
	$(PYTHON) experiments/sft/train_sft.py --model phi --use_wandb

# ── RV-GRPO Training (main method) ──────────────────────────

.PHONY: train-grpo train-grpo-llama train-grpo-qwen train-grpo-phi

train-grpo: train-grpo-llama train-grpo-qwen train-grpo-phi

train-grpo-llama:
	$(PYTHON) experiments/grpo/train_rv_grpo.py \
		--model llama \
		--sft_checkpoint ./outputs/sft/llama/final_model \
		--use_wandb

train-grpo-qwen:
	$(PYTHON) experiments/grpo/train_rv_grpo.py \
		--model qwen \
		--sft_checkpoint ./outputs/sft/qwen/final_model \
		--use_wandb

train-grpo-phi:
	$(PYTHON) experiments/grpo/train_rv_grpo.py \
		--model phi \
		--sft_checkpoint ./outputs/sft/phi/final_model \
		--use_wandb

# ── DPO Baseline ────────────────────────────────────────────

.PHONY: train-dpo train-dpo-llama train-dpo-qwen train-dpo-phi

train-dpo: train-dpo-llama train-dpo-qwen train-dpo-phi

train-dpo-llama:
	$(PYTHON) experiments/alignment/train_dpo.py \
		--model llama --use_wandb

train-dpo-qwen:
	$(PYTHON) experiments/alignment/train_dpo.py \
		--model qwen --use_wandb

train-dpo-phi:
	$(PYTHON) experiments/alignment/train_dpo.py \
		--model phi --use_wandb

# ── LLM-Judge GRPO Baseline ─────────────────────────────────

.PHONY: train-judge-grpo

train-judge-grpo:
	@for model in $(MODELS); do \
		$(PYTHON) experiments/grpo/train_llm_judge_grpo.py \
			--model $$model \
			--sft_checkpoint ./outputs/sft/$$model/final_model \
			--use_wandb; \
	done

# ── Pilot Run (quick test before full training) ─────────────

.PHONY: pilot

pilot:
	@echo "=== PILOT: Quick validation run on Qwen 1.5B ==="
	@echo ""
	@echo "Step 1/3: SFT (1K samples, 1 epoch)"
	$(PYTHON) experiments/sft/train_sft.py \
		--model qwen --max_samples 1000 --num_epochs 1 \
		--output_dir ./outputs/pilot/sft
	@echo ""
	@echo "Step 2/3: RV-GRPO (200 prompts, 4 generations)"
	$(PYTHON) experiments/grpo/train_rv_grpo.py \
		--model qwen \
		--sft_checkpoint ./outputs/pilot/sft/qwen/final_model \
		--max_samples 200 --num_generations 4 \
		--output_dir ./outputs/pilot/rv_grpo
	@echo ""
	@echo "Step 3/3: DPO (500 samples)"
	$(PYTHON) experiments/alignment/train_dpo.py \
		--model qwen --max_samples 500 --num_epochs 1 \
		--output_dir ./outputs/pilot/dpo
	@echo ""
	@echo "=== PILOT COMPLETE ==="
	@echo "Now run: make pilot-eval"

.PHONY: pilot-eval

pilot-eval:
	@echo "Evaluating pilot models..."
	$(PYTHON) evaluation/behavioral_metrics.py \
		--model_path ./outputs/pilot/sft/qwen/final_model \
		--model_id Qwen/Qwen2.5-1.5B-Instruct \
		--method sft --model_name qwen \
		--output_dir ./results/pilot
	$(PYTHON) evaluation/behavioral_metrics.py \
		--model_path ./outputs/pilot/rv_grpo/qwen/final \
		--model_id Qwen/Qwen2.5-1.5B-Instruct \
		--method rv-grpo --model_name qwen \
		--output_dir ./results/pilot
	$(PYTHON) evaluation/behavioral_metrics.py \
		--model_path ./outputs/pilot/dpo/qwen/final \
		--model_id Qwen/Qwen2.5-1.5B-Instruct \
		--method dpo --model_name qwen \
		--output_dir ./results/pilot

# ── Evaluation ──────────────────────────────────────────────

.PHONY: eval eval-rubric eval-benchmarks eval-inference

eval: eval-rubric eval-benchmarks

eval-rubric:
	$(PYTHON) evaluation/behavioral_metrics.py \
		--batch_config ./evaluation/eval_config.json \
		--output_dir ./results

eval-benchmarks:
	$(PYTHON) evaluation/external_benchmarks.py \
		--benchmark all \
		--batch_config ./evaluation/eval_config.json \
		--output_dir ./results/benchmarks

eval-inference:
	$(PYTHON) deployment/benchmark.py \
		--batch_config ./evaluation/eval_config.json \
		--output_dir ./results/benchmarks

# ── Ablation Study ──────────────────────────────────────────

.PHONY: ablation ablation-dry

ablation-dry:
	$(PYTHON) experiments/grpo/run_ablation.py \
		--mode leave_one_out \
		--model qwen \
		--sft_checkpoint ./outputs/sft/qwen/final_model \
		--dry_run

ablation:
	$(PYTHON) experiments/grpo/run_ablation.py \
		--mode leave_one_out \
		--model qwen \
		--sft_checkpoint ./outputs/sft/qwen/final_model \
		--use_wandb

# ── Deployment ──────────────────────────────────────────────

.PHONY: quantize

quantize:
	@for model in $(MODELS); do \
		$(PYTHON) deployment/quantize.py \
			--model_path ./outputs/rv_grpo/$$model/final \
			--model_id $$($(PYTHON) -c "from experiments.grpo.train_rv_grpo import MODEL_REGISTRY; print(MODEL_REGISTRY['$$model']['model_id'])") \
			--output_dir ./deployment/gguf/$$model \
			--quant_type q4_k_m; \
	done

# ── Full Pipeline ───────────────────────────────────────────

.PHONY: all

all: data-sft train-sft train-grpo train-dpo train-judge-grpo eval
	@echo "Full pipeline complete!"

# ── Utilities ───────────────────────────────────────────────

.PHONY: clean test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

test:
	$(PYTHON) -m pytest tests/ -v

test-rewards:
	$(PYTHON) -c "\
	from data.rubric_rewards import compute_rubric_reward; \
	good = compute_rubric_reward( \
		'I feel so depressed lately.', \
		'It sounds like you have been carrying a heavy weight. Depression can feel overwhelming. What has been the hardest part for you?', \
		return_breakdown=True \
	); \
	bad = compute_rubric_reward( \
		'I feel so depressed lately.', \
		'You should try exercising more. Here are 5 tips: 1) Go for walks 2) Eat better 3) Sleep more 4) Talk to friends 5) See a doctor.', \
		return_breakdown=True \
	); \
	print(f'Good response: {good[\"combined\"]:.3f}'); \
	print(f'Bad response:  {bad[\"combined\"]:.3f}'); \
	assert good['combined'] > bad['combined'], 'Reward function sanity check FAILED'; \
	print('Sanity check PASSED'); \
	"
