"""Generate the descriptive 14,313-SNP p-value supplement.

The participant-level input is intentionally not distributed. Pass the existing
GWAS result CSV produced by the cohort analysis, with at least ``SNP`` and
``P_value`` columns. The publication export contains identifiers and test
statistics. Genomic positions require assembly-matched annotation and are
not copied from upstream plotting-coordinate columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/supplementary_data"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/supplementary_figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    aliases = {"snp": ("SNP", "snp"), "p": ("P_value", "pval", "p_value")}
    resolved = {
        key: next((name for name in names if name in df.columns), None)
        for key, names in aliases.items()
    }
    missing = [key for key, name in resolved.items() if name is None]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    p = pd.to_numeric(df[resolved["p"]], errors="coerce")
    if p.isna().any() or ((p <= 0) | (p > 1)).any():
        raise ValueError("P_value must contain finite values in (0, 1].")

    n = len(df)
    threshold = 0.05 / n
    out = pd.DataFrame({"SNP": df[resolved["snp"]]})
    optional_aliases = {
        "Chi_square": ("Chi_square", "stat"),
        "df": ("df",),
    }
    for output_name, names in optional_aliases.items():
        source_name = next((name for name in names if name in df.columns), None)
        if source_name is not None:
            out[output_name] = df[source_name]
    out["P_value"] = p
    out["Bonferroni_P"] = np.minimum(p * n, 1.0)
    out["minus_log10_P"] = -np.log10(p)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "Supplemental_Data_S1_single_SNP_pvalues.csv", index=False)

    observed = np.sort(p.to_numpy())
    expected = (np.arange(1, n + 1) - 0.5) / n
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(-np.log10(observed), bins=60, color="#315f89", edgecolor="white")
    axes[0].axvline(-np.log10(threshold), color="#b03a2e", linestyle="--")
    axes[0].set(xlabel=r"$-\log_{10}(p)$", ylabel="Number of SNPs", title="A  Distribution")
    axes[1].scatter(-np.log10(expected), -np.log10(observed), s=9, alpha=0.55, color="#315f89")
    limit = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, limit], [0, limit], color="#777777", linewidth=1)
    axes[1].set(xlabel="Expected", ylabel="Observed", title="B  Quantile-quantile plot")
    for suffix in ("png", "pdf"):
        fig.savefig(args.figure_dir / f"figure_s2_single_snp_pvalues.{suffix}", dpi=300)
    plt.close(fig)

    print(f"SNPs: {n:,}")
    print(f"Range: {p.min():.6g} to {p.max():.6g}; median: {p.median():.6g}")
    print(f"Bonferroni threshold: {threshold:.6g}; below threshold: {(p < threshold).sum():,}")


if __name__ == "__main__":
    main()
