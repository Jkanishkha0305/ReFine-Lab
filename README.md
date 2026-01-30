# 🔬 ReFine-Lab: Your Complete Post-Training Learning Hub

A practical, beginner-friendly repository for learning **ALL** post-training techniques through hands-on experiments with small language models. Perfect for researchers, students, and practitioners who want to master LLM fine-tuning.

## 🎯 What You'll Learn

This repository covers **every major post-training technique** for LLMs:

- **Supervised Fine-Tuning (SFT)**: LoRA, QLoRA, Adapters, Prompt Tuning, Full Fine-Tuning
- **Knowledge Distillation**: Teacher-student training, Speculative KD
- **Preference Optimization**: DPO, ORPO, IPO, SimPO, KTO
- **RL-Based Methods**: PPO, GRPO, RLOO, RRHF
- **Full RLHF Pipeline**: Reward models → Policy optimization
- **Advanced Alignment**: Constitutional AI, Rejection Sampling, SPIN
- **Model Compression**: Quantization, Pruning, Model Merging
- **Dynamic Architectures**: Early exit, Mixture of Depths
- **Specialized Techniques**: Chain-of-Thought, Tool Use, Continual Learning
- **Safety & Control**: Unlearning, Model Editing
- **Optimization**: Flash Attention, Distributed Training, Fast Inference

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ReFine-Lab.git
cd ReFine-Lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for faster training
pip install flash-attn --no-build-isolation

# Optional: Setup Weights & Biases for experiment tracking
wandb login
```

### Your First Experiment (5 minutes)

Start with the simplest fine-tuning example:

```bash
# Open the first notebook
jupyter notebook notebooks/01_sft_lora_basics.ipynb
```

This notebook will guide you through:
1. Loading a small model (Gemma 270M)
2. Preparing a tiny instruction dataset
3. Fine-tuning with LoRA
4. Testing your fine-tuned model

**Expected time**: ~5 minutes on a single GPU
**Memory required**: ~4GB VRAM

## 📚 Learning Path

### Week 1-2: Foundations
1. **`01_sft_lora_basics.ipynb`** - Master LoRA fine-tuning
2. **`04_instruction_tuning.ipynb`** - Work with instruction datasets
3. **`07_data_quality.ipynb`** - Learn data filtering and quality scoring

### Week 3-4: Alignment Basics
4. **`12_dpo.ipynb`** - Direct Preference Optimization (easiest alignment method)
5. **`11_reward_models.ipynb`** - Understanding reward models
6. **`16_kto.ipynb`** - KTO (works with simple thumbs-up/down data)

### Week 5-6: Advanced RL
7. **`21_rlhf_pipeline.ipynb`** - Complete RLHF workflow
8. **`17_ppo.ipynb`**, **`18_grpo.ipynb`** - Compare RL methods

### Week 7-8: Specialization
9. **`09_knowledge_distillation.ipynb`** - Compress models
10. **`25_quantization.ipynb`** - Deploy efficiently
11. **Your own experiments!**

## 📁 Repository Structure

```
ReFine-Lab/
├── notebooks/              # 40+ learning notebooks
│   ├── 01_sft_lora_basics.ipynb
│   ├── 12_dpo.ipynb
│   └── ...
├── experiments/            # Production training scripts
│   ├── sft/
│   ├── dpo/
│   ├── ppo/
│   └── ...
├── configs/               # Configuration templates
│   ├── model_configs/     # Per-model configs
│   ├── dataset_configs/   # Dataset preparation
│   └── training_configs/  # Training hyperparameters
├── utils/                 # Shared utilities
│   ├── model_loader.py
│   ├── data_quality.py
│   ├── tracking.py
│   └── ...
├── datasets/             # Dataset preparation scripts
├── docs/                 # Comprehensive guides
│   ├── GETTING_STARTED.md
│   ├── TECHNIQUE_GUIDE.md
│   ├── hardware_guide.md
│   └── troubleshooting.md
└── requirements.txt
```

## 🎓 What's Included

### 40+ Jupyter Notebooks

Each notebook includes:
- 📖 Clear explanations of concepts
- 💻 Working code examples
- ⚡ Training time estimates
- 💾 Memory requirements
- 🎯 When to use this technique

### Ready-to-Run Training Scripts

Production-grade scripts for every technique:
- Command-line interfaces
- Config file support
- Experiment tracking integration
- Checkpoint management
- Multi-GPU support

### Model Support

Pre-configured for popular small language models:
- **Llama 3.2 1B** - Meta's efficient model
- **Phi-3 Mini** - Microsoft's small model
- **Gemma 2 270M/2B** - Google's compact models
- **Qwen 2.5 0.5B-7B** - Alibaba's multilingual models
- **LFM-2 1B** - Liquid AI's models

## 💻 Hardware Requirements

### Minimum (Learning & Experimentation)
- **GPU**: Single RTX 3060/3070 (12GB VRAM)
- **RAM**: 16GB
- **Models**: Gemma 270M, Qwen 0.5B with QLoRA
- **What you can do**: All techniques, small scale

### Recommended (Serious Experiments)
- **GPU**: RTX 4090 or 2x RTX 3090 (24GB+ VRAM)
- **RAM**: 32GB+
- **Models**: All models up to 3B with QLoRA, 1B full fine-tuning
- **What you can do**: Everything in the repo

### Optimal (Research & Production)
- **GPU**: A100 (40GB/80GB) or H100
- **RAM**: 64GB+
- **Models**: All models, including full fine-tuning of 7B models
- **What you can do**: Large-scale experiments, multi-GPU training

### Cloud Options
- **Google Colab Pro**: Sufficient for most notebooks ($10/month)
- **Lambda Labs**: ~$0.50-$1.10/hour for various GPUs
- **RunPod**: Flexible GPU rental
- **AWS/GCP/Azure**: Professional deployment

## 🔥 Key Features

✅ **Beginner-Friendly**: Start with no prior knowledge
✅ **Comprehensive**: Every major post-training technique
✅ **Practical**: Real, working code you can run today
✅ **Hardware-Aware**: Configs for different GPU setups
✅ **Research-Ready**: Experiment tracking, benchmarking, visualization
✅ **Well-Documented**: Extensive guides and troubleshooting
✅ **Active**: Following latest research and best practices

## 📊 Experiment Tracking

All training scripts integrate with:
- **Weights & Biases** (recommended) - Beautiful dashboards
- **TensorBoard** - Local visualization
- **MLflow** - Experiment management

Track automatically:
- Training/validation loss
- Learning rate schedules
- GPU memory usage
- Training speed (tokens/sec)
- Custom metrics
- Model checkpoints

## 🎯 Technique Comparison

| Technique | Difficulty | Data Needed | When to Use |
|-----------|-----------|-------------|-------------|
| **LoRA** | Easy | Instruction data | First choice for fine-tuning |
| **DPO** | Easy | Preference pairs | Easiest alignment method |
| **KTO** | Easy | Thumbs up/down | Minimal preference data |
| **PPO** | Medium | Reward model | Complex alignment needs |
| **RLHF** | Hard | Preferences + RM | Full research pipeline |
| **Distillation** | Medium | Teacher model | Model compression |
| **Quantization** | Easy | None | Fast inference |

See `docs/TECHNIQUE_GUIDE.md` for detailed comparison.

## 📖 Documentation

- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)**: Step-by-step beginner's guide
- **[TECHNIQUE_GUIDE.md](docs/TECHNIQUE_GUIDE.md)**: When to use which method
- **[hardware_guide.md](docs/hardware_guide.md)**: GPU requirements & optimization
- **[dataset_guide.md](docs/dataset_guide.md)**: Best datasets for each technique
- **[troubleshooting.md](docs/troubleshooting.md)**: Common issues & solutions
- **[papers.md](docs/papers.md)**: Key research papers (optional reading)

## 🤝 Contributing

Contributions are welcome! Areas we'd love help with:
- Adding new techniques
- Improving documentation
- Creating more example notebooks
- Optimizing training scripts
- Sharing your experiment results

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with amazing open-source libraries:
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace TRL](https://github.com/huggingface/trl)
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ReFine-Lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ReFine-Lab/discussions)
- **Documentation**: Check `docs/` directory

## 🌟 Star History

If this repository helps you, please consider giving it a ⭐!

---

**Ready to start?** Jump to [GETTING_STARTED.md](docs/GETTING_STARTED.md) or open your first notebook!
