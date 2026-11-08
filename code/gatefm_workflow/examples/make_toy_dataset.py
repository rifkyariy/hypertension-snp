from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic non-human genotype data for a smoke test."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=1200, type=int)
    parser.add_argument("--snps", default=30, type=int)
    parser.add_argument("--seed", default=20260815, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 100:
        raise ValueError("--samples must be at least 100.")
    if args.snps < 3:
        raise ValueError("--snps must be at least 3.")

    rng = np.random.default_rng(args.seed)
    minor_allele_frequencies = rng.uniform(0.15, 0.45, size=args.snps)
    X = np.column_stack(
        [
            rng.binomial(2, frequency, size=args.samples)
            for frequency in minor_allele_frequencies
        ]
    ).astype(np.int16)

    third_order_pattern = (X[:, 0] == 1) & (X[:, 1] == 1) & (X[:, 2] == 1)
    case_probability = np.where(third_order_pattern, 0.78, 0.20)
    y = rng.binomial(1, case_probability).astype(np.int64)
    if np.unique(y).size != 2:
        raise RuntimeError("Toy outcome unexpectedly contains only one class.")

    frame = pd.DataFrame(X, columns=[f"SNP{index:03d}" for index in range(args.snps)])
    frame["class"] = y
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    print(
        f"Wrote {args.samples} samples and {args.snps} SNPs to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
