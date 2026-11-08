"""Automatically render paper-style image figures after an authorized cohort workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RED = "#E74C3C"
BLUE = "#4E79A7"
GREEN = "#59A14F"


def export(fig: plt.Figure, out: Path, stem: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_results(root: Path) -> Path:
    selected = pd.read_csv(root / "consensus" / "stage1_consensus_top30.csv")
    triples = pd.read_csv(root / "stage2" / "stage2_fixed_pattern_triplets.csv")
    screened = pd.read_csv(root / "lower_order" / "stage2_triplets_with_lower_order_screen.csv")
    needed = {"snp_id", "mean_q_rank", "consensus_rank"}
    if not needed.issubset(selected.columns):
        raise ValueError("Consensus ranking fields are incomplete")
    needed = {"stage2_rank", "snp_a", "snp_b", "snp_c",
              "raw_p", "bonferroni_p", "odds_ratio", "or_ci_low", "or_ci_high"}
    if not needed.issubset(triples.columns):
        raise ValueError("Stage 2 results are missing required statistics")
    if "lower_order_screen_pass" not in screened:
        raise ValueError("Lower-order screening results missing")
    if len(triples) != 4060 or len(selected) != 30 or len(screened) != 4060:
        raise ValueError("Expected a complete Top-30 analysis with 4,060 triplets")

    out = root / "figures"
    out.mkdir(parents=True, exist_ok=False)
    selected = selected.sort_values("consensus_rank")
    ranked = selected.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.barh(ranked["snp_id"].astype(str), ranked["mean_q_rank"], color=RED)
    ax.set(xlabel="Mean SNP rank (lower is better)", title="Stage 1 — prioritized SNPs")
    ax.xaxis.grid(alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    export(fig, out, "stage1_snp_rank")

    first = triples.sort_values("stage2_rank").head(15).iloc[::-1]
    labels = first[["snp_a", "snp_b", "snp_c"]].astype(str).agg(" + ".join, axis=1)
    p = pd.to_numeric(first["raw_p"], errors="raise").to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid Stage 2 p values")
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.barh(labels, -np.log10(np.maximum(p, np.finfo(float).tiny)), color=BLUE)
    ax.set(xlabel="-log10(raw p)", title="Stage 2 — top fixed-pattern triplets")
    ax.xaxis.grid(alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    export(fig, out, "stage2_top_triplets")

    odds = triples.sort_values("stage2_rank").head(10).iloc[::-1]
    center = pd.to_numeric(odds.odds_ratio, errors="raise").to_numpy(float)
    low = pd.to_numeric(odds.or_ci_low, errors="raise").to_numpy(float)
    high = pd.to_numeric(odds.or_ci_high, errors="raise").to_numpy(float)
    if (not np.isfinite(center).all() or not np.isfinite(low).all()
            or not np.isfinite(high).all() or (low <= 0).any()
            or (center < low).any() or (center > high).any()):
        raise ValueError("Invalid odds-ratio confidence intervals")
    label = odds[["snp_a", "snp_b", "snp_c"]].astype(str).agg(" + ".join, axis=1)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ypos = np.arange(len(odds))
    ax.errorbar(center, ypos, xerr=[center-low, high-center],
                fmt="o", color=RED, ecolor=GREEN, elinewidth=1.5, capsize=3)
    ax.axvline(1, color="#6f7f88", ls="--", lw=1)
    ax.set_yticks(ypos, label)
    ax.set_xscale("log")
    ax.set(xlabel="Odds ratio and 95% CI (log scale)",
           title="Stage 2 — fixed-pattern association estimates")
    ax.xaxis.grid(alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    export(fig, out, "stage2_odds_ratio")

    print(f"[PASS] Cohort figures (PNG, SVG, PDF): {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    plot_results(args.run.resolve())


if __name__ == "__main__":
    main()
