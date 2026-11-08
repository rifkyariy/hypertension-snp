from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHODS = [
    ("p5000_budget_500_main", "GateFM + budget", "#2563eb"),
    ("p5000_no_budget", "GateFM no budget", "#f97316"),
    ("p5000_pure_fm_like", "Pure FM", "#ef4444"),
]


def load_data(summary: Path) -> pd.DataFrame:
    df = pd.read_csv(summary)
    rows = []
    for experiment, label, color in METHODS:
        hit = df.loc[df["experiment"] == experiment]
        if hit.empty:
            raise RuntimeError(f"Missing experiment in summary: {experiment}")
        row = hit.iloc[0].copy()
        row["method_label"] = label
        row["color"] = color
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the original manuscript Figure S1 from the five-seed ablation summary.")
    parser.add_argument("--input", type=Path, required=True, help="Original defense_full_suite_summary.csv or mechanism_ablation_budget_only_summary.csv")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    df = load_data(args.input)
    df.to_csv(args.output / "mechanism_ablation_budget_only_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), gridspec_kw={"wspace": 0.26})
    fig.suptitle("Mechanism Ablation: Gate and Budget", fontsize=19, fontweight="bold", y=1.03)

    labels = df["method_label"].tolist()
    colors = df["color"].tolist()

    ax = axes[0]
    mean_rank = df["mean_gt_rank_mean"].astype(float).to_numpy()
    rank_std = df["mean_gt_rank_std"].astype(float).to_numpy()
    bars = ax.bar(labels, mean_rank, yerr=rank_std, capsize=5, color=colors, alpha=0.9, edgecolor="none")
    ax.set_ylabel("Mean GT rank (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(mean_rank + rank_std) * 1.15)
    for bar, value in zip(bars, mean_rank):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mean_rank + rank_std) * 0.035,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax = axes[1]
    gate_sum = df["gate_sum_mean"].astype(float).to_numpy()
    bars = ax.bar(labels, gate_sum, color=colors, alpha=0.9, edgecolor="none")
    ax.set_ylabel("Mean gate mass")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(gate_sum) * 1.12)
    for bar, value in zip(bars, gate_sum):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(gate_sum) * 0.025,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    note = (
        "Dataset: GAMETES pure 3-way, p=5000, 5 runs. "
        "GateFM + budget uses the same ranking score as the main experiment: gate x tri-embedding norm."
    )
    fig.text(0.01, -0.02, note, fontsize=10.5, color="#4b5563", ha="left", va="top")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    out_png = args.output / "figure_s1_gate_budget_ablation.png"
    out_pdf = args.output / "figure_s1_gate_budget_ablation.pdf"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(args.output / "figure_s1_gate_budget_ablation.svg", bbox_inches="tight")
    plt.close(fig)

    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
