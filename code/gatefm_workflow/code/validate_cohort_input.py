"""Validate an authorized genotype matrix without training or exporting samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro_io import load_genotype_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight authorized CSV/TSV genotype input without fitting GateFM."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--label-column")
    parser.add_argument("--case-value")
    parser.add_argument("--control-value")
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--expected-snps", type=int)
    args = parser.parse_args()

    X, y, names, label = load_genotype_dataset(
        args.data,
        label_column=args.label_column,
        case_value=args.case_value,
        control_value=args.control_value,
    )
    if args.expected_samples is not None and len(y) != args.expected_samples:
        raise ValueError(
            f"Sample-count mismatch: expected {args.expected_samples}, observed {len(y)}."
        )
    if args.expected_snps is not None and X.shape[1] != args.expected_snps:
        raise ValueError(
            f"SNP-count mismatch: expected {args.expected_snps}, observed {X.shape[1]}."
        )
    if X.shape[1] < 30:
        raise ValueError("The 10-fit human workflow requires at least 30 distinct SNPs.")
    counts = np.bincount(y, minlength=2)
    if min(counts) < 2:
        raise ValueError("Both outcome classes need samples for a stratified split.")
    print(json.dumps({
        "status": "valid",
        "samples": int(X.shape[0]),
        "snps": int(X.shape[1]),
        "controls": int(counts[0]),
        "cases": int(counts[1]),
        "outcome_column": label,
        "genotype_encoding": "0/1/2",
        "outcome_encoding": "0=control, 1=case (after explicit mapping when provided)",
        "note": "Preflight only: no training, no data rows exported."
    }, indent=2))


if __name__ == "__main__":
    main()