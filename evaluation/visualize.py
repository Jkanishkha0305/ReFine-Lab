"""
Visualization for RV-GRPO Results

Generates radar charts, bar plots, and comparison tables for the paper.

Usage:
    python evaluation/visualize.py --results_file ./results/all_results.json
    python evaluation/visualize.py --results_file ./results/all_results.json --format pdf

Author: Jkanishkha0305
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


METRIC_LABELS = {
    "open_question": "Open\nQuestion",
    "emotion_reflection": "Emotion\nReflection",
    "no_premature_advice": "No Premature\nAdvice",
    "validation_before_redirect": "Validation\nBefore Redirect",
    "length_appropriate": "Length\nAppropriate",
}

METHOD_COLORS = {
    "sft": "#7f8c8d",
    "dpo": "#3498db",
    "grpo-judge": "#e67e22",
    "rv-grpo": "#e74c3c",
}

METHOD_LABELS = {
    "sft": "SFT Only",
    "dpo": "SFT + DPO",
    "grpo-judge": "GRPO (LLM-Judge)",
    "rv-grpo": "RV-GRPO (Ours)",
}


def plot_radar_chart(results, model_name: str, save_path: str):
    """Create radar chart comparing methods for a single model."""
    metrics = list(METRIC_LABELS.keys())
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    model_results = [r for r in results if r["model"] == model_name]

    for r in model_results:
        method = r["method"]
        values = [r["metrics"][m]["mean"] for m in metrics]
        values += values[:1]

        color = METHOD_COLORS.get(method, "#95a5a6")
        label = METHOD_LABELS.get(method, method)

        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], size=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Behavioral Profile: {model_name}", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved radar chart: {save_path}")


def plot_combined_bar_chart(results, save_path: str):
    """Create grouped bar chart comparing combined scores across all models/methods."""
    models = sorted(set(r["model"] for r in results))
    methods = sorted(set(r["method"] for r in results))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        scores = []
        for model in models:
            match = [r for r in results if r["model"] == model and r["method"] == method]
            if match:
                scores.append(match[0]["metrics"]["combined"]["mean"])
            else:
                scores.append(0)

        color = METHOD_COLORS.get(method, "#95a5a6")
        label = METHOD_LABELS.get(method, method)
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=label, color=color, edgecolor="white")

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Combined Rubric Score", fontsize=12)
    ax.set_title("RV-GRPO vs. Baselines: Combined Behavioral Score", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart: {save_path}")


def generate_latex_table(results) -> str:
    """Generate LaTeX table for the paper."""
    models = sorted(set(r["model"] for r in results))
    methods = sorted(set(r["method"] for r in results))
    metrics = list(METRIC_LABELS.keys()) + ["combined"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Behavioral metrics across models and alignment methods. "
        r"Higher is better for all metrics. \textbf{Bold} indicates best per model.}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll" + "c" * len(metrics) + "}",
        r"\toprule",
        r"Model & Method & R1:OQ & R2:ER & R3:NPA & R4:VBR & R5:Len & Combined \\",
        r"\midrule",
    ]

    for model in models:
        model_results = [r for r in results if r["model"] == model]

        # Find best method per metric
        best = {}
        for m in metrics:
            key = "mean" if m != "combined" else "mean"
            vals = {r["method"]: r["metrics"][m][key] for r in model_results}
            best[m] = max(vals, key=vals.get) if vals else None

        for r in model_results:
            method_label = METHOD_LABELS.get(r["method"], r["method"])
            row = f"{model} & {method_label}"
            for m in metrics:
                key = "mean"
                val = r["metrics"][m][key]
                if r["method"] == best.get(m):
                    row += f" & \\textbf{{{val:.3f}}}"
                else:
                    row += f" & {val:.3f}"
            row += r" \\"
            lines.append(row)

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize RV-GRPO results")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results_file) as f:
        results = json.load(f)

    # Radar charts per model
    models = set(r["model"] for r in results)
    for model in models:
        save_path = os.path.join(args.output_dir, f"radar_{model}.{args.format}")
        plot_radar_chart(results, model, save_path)

    # Combined bar chart
    bar_path = os.path.join(args.output_dir, f"combined_scores.{args.format}")
    plot_combined_bar_chart(results, bar_path)

    # LaTeX table
    latex = generate_latex_table(results)
    latex_path = os.path.join(args.output_dir, "results_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table: {latex_path}")
    print("\n" + latex)


if __name__ == "__main__":
    main()
