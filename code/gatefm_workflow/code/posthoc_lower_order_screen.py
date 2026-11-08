from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from repro_io import load_genotype_dataset, sha256_file, write_json
from run_stage2 import fixed_pattern_2x2, normalize_stage1_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply reproducible single- and pair-joint association screening."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--label-column", help="Explicit phenotype column.")
    parser.add_argument("--case-value", help="Source value meaning case (mapped to 1).")
    parser.add_argument("--control-value", help="Source value meaning control (mapped to 0).")
    parser.add_argument("--consensus-topk", required=True, type=Path)
    parser.add_argument("--stage2-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--triplet-adjusted-threshold", default=0.0002, type=float)
    parser.add_argument(
        "--lower-order-adjusted-threshold",
        default=0.0002,
        type=float,
    )
    parser.add_argument("--yates-correction", action="store_true")
    parser.add_argument("--zero-cell-correction", default=0.5, type=float)
    return parser.parse_args()


def compute_lower_order_screen(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    candidates: pd.DataFrame,
    triplets: pd.DataFrame,
    *,
    triplet_adjusted_threshold: float = 0.0002,
    lower_order_adjusted_threshold: float = 0.0002,
    yates_correction: bool = False,
    zero_cell_correction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    candidates, candidate_rule = normalize_stage1_candidates(candidates)
    required = {
        "global_idx",
        "snp_id",
        "representative_genotype",
        "candidate_rank",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Consensus file is missing columns: {sorted(missing)}")

    candidates = candidates.sort_values("candidate_rank", kind="stable").reset_index(drop=True)
    if (
        candidates["global_idx"].duplicated().any()
        or candidates["snp_id"].duplicated().any()
        or candidates["candidate_rank"].duplicated().any()
    ):
        raise ValueError("Lower-order candidate indices, SNP IDs and ranks must be distinct.")
    candidate_indices = candidates["global_idx"].astype(int).tolist()
    candidate_names = candidates["snp_id"].astype(str).tolist()
    candidate_genotypes = candidates["representative_genotype"].astype(int).tolist()
    fixed_genotype = dict(zip(candidate_indices, candidate_genotypes))

    for index, name, genotype in zip(
        candidate_indices,
        candidate_names,
        candidate_genotypes,
    ):
        if feature_names[index] != name:
            raise ValueError(
                f"SNP mapping mismatch at {index}: {feature_names[index]} != {name}"
            )
        if genotype not in (0, 1, 2):
            raise ValueError("representative_genotype must be 0, 1, or 2.")

    single_rows = []
    single_lookup = {}
    n_single_tests = len(candidate_indices)
    for index, name, genotype in zip(
        candidate_indices,
        candidate_names,
        candidate_genotypes,
    ):
        stats = fixed_pattern_2x2(
            X,
            y,
            (index,),
            (genotype,),
            yates_correction=yates_correction,
            zero_cell_correction=zero_cell_correction,
        )
        adjusted = min(1.0, stats["raw_p"] * n_single_tests)
        row = {
            "global_idx": index,
            "snp_id": name,
            "representative_genotype": genotype,
            **stats,
            "bonferroni_p": adjusted,
        }
        single_rows.append(row)
        single_lookup[index] = row

    pair_rows = []
    pair_lookup = {}
    n_pair_tests = math.comb(len(candidate_indices), 2)
    candidate_records = list(
        zip(candidate_indices, candidate_names, candidate_genotypes)
    )
    for (index_a, name_a, genotype_a), (index_b, name_b, genotype_b) in itertools.combinations(
        candidate_records,
        2,
    ):
        stats = fixed_pattern_2x2(
            X,
            y,
            (index_a, index_b),
            (genotype_a, genotype_b),
            yates_correction=yates_correction,
            zero_cell_correction=zero_cell_correction,
        )
        adjusted = min(1.0, stats["raw_p"] * n_pair_tests)
        key = tuple(sorted((index_a, index_b)))
        row = {
            "idx_a": index_a,
            "idx_b": index_b,
            "snp_a": name_a,
            "snp_b": name_b,
            "genotype_a": genotype_a,
            "genotype_b": genotype_b,
            **stats,
            "bonferroni_p": adjusted,
        }
        pair_rows.append(row)
        pair_lookup[key] = row

    screened_rows = []
    for row in triplets.to_dict(orient="records"):
        indices = [int(row["idx_a"]), int(row["idx_b"]), int(row["idx_c"])]
        actual_genotypes = [
            int(row["genotype_a"]),
            int(row["genotype_b"]),
            int(row["genotype_c"]),
        ]
        expected_genotypes = [fixed_genotype[index] for index in indices]
        if actual_genotypes != expected_genotypes:
            raise ValueError(
                "Stage-2 result contains a genotype pattern that differs from "
                "the frozen Stage-1 consensus."
            )

        single_adjusted = [single_lookup[index]["bonferroni_p"] for index in indices]
        pairs = [
            tuple(sorted((indices[0], indices[1]))),
            tuple(sorted((indices[0], indices[2]))),
            tuple(sorted((indices[1], indices[2]))),
        ]
        pair_adjusted = [pair_lookup[pair]["bonferroni_p"] for pair in pairs]
        single_pass = all(
            p >= lower_order_adjusted_threshold for p in single_adjusted
        )
        pair_pass = all(
            p >= lower_order_adjusted_threshold for p in pair_adjusted
        )
        triplet_pass = float(row["bonferroni_p"]) < triplet_adjusted_threshold
        row.update(
            {
                "single_a_bonferroni_p": single_adjusted[0],
                "single_b_bonferroni_p": single_adjusted[1],
                "single_c_bonferroni_p": single_adjusted[2],
                "pair_ab_bonferroni_p": pair_adjusted[0],
                "pair_ac_bonferroni_p": pair_adjusted[1],
                "pair_bc_bonferroni_p": pair_adjusted[2],
                "triplet_threshold_pass": triplet_pass,
                "single_screen_pass": single_pass,
                "pair_screen_pass": pair_pass,
                "lower_order_screen_pass": bool(
                    triplet_pass and single_pass and pair_pass
                ),
            }
        )
        screened_rows.append(row)

    singles = pd.DataFrame(single_rows)
    pairs = pd.DataFrame(pair_rows)
    screened = pd.DataFrame(screened_rows)
    summary = {
        "pattern_rule": "Stage-1-fixed genotype pattern versus its complement",
        "candidate_rule": candidate_rule,
        "triplet_adjusted_threshold": float(triplet_adjusted_threshold),
        "lower_order_adjusted_threshold": float(lower_order_adjusted_threshold),
        "n_single_tests": int(n_single_tests),
        "n_pair_tests": int(n_pair_tests),
        "n_triplets_total": int(len(screened)),
        "n_triplet_threshold_pass": int(screened["triplet_threshold_pass"].sum()),
        "n_after_single_screen": int(
            (screened["triplet_threshold_pass"] & screened["single_screen_pass"]).sum()
        ),
        "n_after_pair_screen": int(screened["lower_order_screen_pass"].sum()),
        "interpretation": (
            "Fixed-pattern higher-order candidates under the stated screen; "
            "the screen is not a hierarchical incremental-interaction test."
        ),
    }
    return singles, pairs, screened, summary


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    X, y, feature_names, _ = load_genotype_dataset(
        args.data, label_column=args.label_column,
        case_value=args.case_value, control_value=args.control_value,
    )
    candidates = pd.read_csv(args.consensus_topk).sort_values("consensus_rank")
    triplets = pd.read_csv(args.stage2_results)
    singles, pairs, screened, summary = compute_lower_order_screen(
        X,
        y,
        feature_names,
        candidates,
        triplets,
        triplet_adjusted_threshold=args.triplet_adjusted_threshold,
        lower_order_adjusted_threshold=args.lower_order_adjusted_threshold,
        yates_correction=args.yates_correction,
        zero_cell_correction=args.zero_cell_correction,
    )
    summary.update(
        {
            "data_sha256": sha256_file(args.data.resolve()),
            "candidate_file_sha256": sha256_file(args.consensus_topk.resolve()),
            "stage2_file_sha256": sha256_file(args.stage2_results.resolve()),
            "runtime_sec": float(time.perf_counter() - started),
        }
    )
    args.output.mkdir(parents=True, exist_ok=False)
    singles.to_csv(
        args.output / "lower_order_single_tests.csv",
        index=False,
    )
    pairs.to_csv(
        args.output / "lower_order_pair_fixed_pattern_tests.csv",
        index=False,
    )
    screened.to_csv(
        args.output / "stage2_triplets_with_lower_order_screen.csv",
        index=False,
    )
    screened[screened["lower_order_screen_pass"]].to_csv(
        args.output / "stage2_lower_order_screened_candidates.csv",
        index=False,
    )
    write_json(args.output / "lower_order_screen_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
