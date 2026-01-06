from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

LLM_CSV = Path("./results/llm_results.csv")
FINETUNE_CSV = Path("./results/finetune_results.csv")


def load_results(csv_path: Path, expected_columns: set[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = expected_columns.difference(df.columns)
    if missing:
        raise ValueError(f"File {csv_path.name} missing columns: {sorted(missing)}")

    if not pd.api.types.is_numeric_dtype(df["is_correct"]):
        df["is_correct"] = df["is_correct"].astype(bool)

    return df


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Groups by length and calculates the accuracy percentage."""
    grouped = df.groupby("length")
    accuracy = grouped["is_correct"].mean().reset_index()
    accuracy.rename(columns={"is_correct": "accuracy"}, inplace=True)
    accuracy["accuracy_percentage"] = accuracy["accuracy"] * 100
    return accuracy.sort_values("length")


def plot_comparison(
    llm_acc: pd.DataFrame, 
    ft_acc: pd.DataFrame, 
    output_path: Path | None = None
) -> None:
    """Plots both LLM and Fine-tune accuracy on the same graph."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        llm_acc["length"],
        llm_acc["accuracy_percentage"],
        marker="o",
        label="LLM (Baseline)",
        color="tab:blue",
        alpha=0.8
    )

    ax.plot(
        ft_acc["length"],
        ft_acc["accuracy_percentage"],
        marker="s",
        label="Fine-tuned Model",
        color="tab:orange",
        alpha=0.8
    )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cipher Accuracy Comparison: LLM vs. Fine-tuned")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    all_lengths = sorted(list(set(llm_acc["length"]) | set(ft_acc["length"])))
    ax.set_xticks(all_lengths)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_ylim(0, 105)
    ax.legend()

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)

def main() -> None:

    cols = {"length", "original_text", "true_cipher", "model_cipher", "is_correct"}

    try:
        df_llm = load_results(LLM_CSV, cols)
        df_ft = load_results(FINETUNE_CSV, cols)
        
        acc_llm = compute_accuracy(df_llm)
        acc_ft = compute_accuracy(df_ft)
        
        plot_comparison(acc_llm, acc_ft, None)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()