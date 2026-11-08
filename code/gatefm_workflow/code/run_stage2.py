from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from repro_io import (
    feature_order_sha256,
    load_genotype_dataset,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate fixed-pattern triplets from Stage-1 consensus."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--label-column", help="Explicit phenotype column.")
    parser.add_argument("--case-value", help="Source value meaning case (mapped to 1).")
    parser.add_argument("--control-value", help="Source value meaning control (mapped to 0).")
    candidate_group = parser.add_mutually_exclusive_group(required=True)
    candidate_group.add_argument(
        "--stage1-ranking",
        type=Path,
        help="One run's stage1_snp_ranking.csv; used for synthetic per-run Stage 2.",
    )
    candidate_group.add_argument(
        "--consensus-topk",
        type=Path,
        help="Cross-run consensus top-K file; used for the human-cohort analysis.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", default=None, type=int)
    parser.add_argument("--yates-correction", action="store_true")
    parser.add_argument("--zero-cell-correction", default=0.5, type=float)
    return parser.parse_args()


def normalize_stage1_candidates(candidates: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Normalize a single-run q ranking or a cross-run consensus ranking."""
    if "consensus_rank" in candidates.columns:
        rank_column = "consensus_rank"
        source = "cross-run mean complete-list q rank and genotype majority vote"
    elif "q_rank" in candidates.columns:
        rank_column = "q_rank"
        source = "single-run GateFM q rank and rho argmax"
    else:
        raise ValueError(
            "Stage-1 candidates must contain consensus_rank or q_rank."
        )
    normalized = candidates.copy()
    normalized["candidate_rank"] = normalized[rank_column].astype(int)
    return normalized, source


def fixed_pattern_2x2(
    X: np.ndarray,
    y: np.ndarray,
    indices: tuple[int, ...],
    genotypes: tuple[int, ...],
    *,
    yates_correction: bool,
    zero_cell_correction: float,
) -> dict:
    if not indices or len(indices) != len(genotypes):
        raise ValueError("indices and genotypes must have the same nonzero length.")
    matches = np.ones(len(y), dtype=bool)
    for index, genotype in zip(indices, genotypes):
        matches &= X[:, int(index)] == int(genotype)

    a = int(np.sum(matches & (y == 1)))
    b = int(np.sum(matches & (y == 0)))
    c = int(np.sum((~matches) & (y == 1)))
    d = int(np.sum((~matches) & (y == 0)))
    table = np.asarray([[a, b], [c, d]], dtype=np.float64)

    if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        chi2 = 0.0
        raw_p = 1.0
        expected = np.zeros_like(table)
    else:
        chi2, raw_p, _, expected = chi2_contingency(
            table,
            correction=bool(yates_correction),
        )

    corrected = bool(np.any(table == 0))
    effect_table = table.copy()
    if corrected:
        effect_table = effect_table + float(zero_cell_correction)
    aa, bb = effect_table[0]
    cc, dd = effect_table[1]
    odds_ratio = float((aa * dd) / (bb * cc))
    standard_error = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    log_or = math.log(odds_ratio)
    ci_low = float(math.exp(log_or - 1.96 * standard_error))
    ci_high = float(math.exp(log_or + 1.96 * standard_error))

    return {
        "case_pattern": a,
        "control_pattern": b,
        "case_other": c,
        "control_other": d,
        "pattern_frequency": float((a + b) / len(y)),
        "chi2": float(chi2),
        "raw_p": float(raw_p),
        "odds_ratio": odds_ratio,
        "or_ci_low": ci_low,
        "or_ci_high": ci_high,
        "zero_cell_correction_applied": corrected,
        "min_expected_count": float(expected.min()) if expected.size else 0.0,
        "sparse_expected_count_warning": bool(expected.size and expected.min() < 5.0),
    }


def compute_stage2_results(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    consensus_topk: pd.DataFrame,
    *,
    top_k: int | None = None,
    yates_correction: bool = False,
    zero_cell_correction: float = 0.5,
) -> pd.DataFrame:
    consensus_topk, inferred_source = normalize_stage1_candidates(consensus_topk)
    required = {
        "global_idx",
        "snp_id",
        "representative_genotype",
        "candidate_rank",
    }
    missing = required.difference(consensus_topk.columns)
    if missing:
        raise ValueError(f"Consensus file is missing columns: {sorted(missing)}")

    candidates = consensus_topk.sort_values("candidate_rank", kind="stable")
    if top_k is not None:
        candidates = candidates.head(int(top_k))
    candidates = candidates.reset_index(drop=True)
    if len(candidates) < 3:
        raise ValueError("At least three consensus SNPs are required.")

    if candidates["global_idx"].isna().any() or candidates["candidate_rank"].isna().any():
        raise ValueError("Candidate indices and ranks must not be missing.")
    for column in ("global_idx", "candidate_rank"):
        numeric = pd.to_numeric(candidates[column], errors="raise")
        if not np.isfinite(numeric).all() or (numeric != numeric.astype(np.int64)).any():
            raise ValueError(f"{column} must contain finite integer values.")
    if candidates["global_idx"].duplicated().any():
        raise ValueError("Stage-2 candidates must have distinct SNP indices.")
    if candidates["snp_id"].astype(str).str.casefold().duplicated().any():
        raise ValueError("Stage-2 candidates must have distinct SNP names.")
    if candidates["candidate_rank"].duplicated().any():
        raise ValueError("Stage-2 candidate ranks must be unique.")
    for row in candidates.itertuples(index=False):
        index = int(row.global_idx)
        if index < 0 or index >= X.shape[1]:
            raise IndexError(f"global_idx {index} is outside the dataset.")
        if str(feature_names[index]) != str(row.snp_id):
            raise ValueError(
                f"SNP mapping mismatch at global_idx={index}: "
                f"dataset={feature_names[index]!r}, consensus={row.snp_id!r}."
            )
        if int(row.representative_genotype) not in (0, 1, 2):
            raise ValueError("representative_genotype must be 0, 1, or 2.")

    n_tests = math.comb(len(candidates), 3)
    records = []
    for row_a, row_b, row_c in itertools.combinations(
        candidates.itertuples(index=False),
        3,
    ):
        indices = (
            int(row_a.global_idx),
            int(row_b.global_idx),
            int(row_c.global_idx),
        )
        genotypes = (
            int(row_a.representative_genotype),
            int(row_b.representative_genotype),
            int(row_c.representative_genotype),
        )
        stats = fixed_pattern_2x2(
            X,
            y,
            indices,
            genotypes,
            yates_correction=yates_correction,
            zero_cell_correction=zero_cell_correction,
        )
        records.append(
            {
                "snp_a": str(row_a.snp_id),
                "snp_b": str(row_b.snp_id),
                "snp_c": str(row_c.snp_id),
                "idx_a": indices[0],
                "idx_b": indices[1],
                "idx_c": indices[2],
                "genotype_a": genotypes[0],
                "genotype_b": genotypes[1],
                "genotype_c": genotypes[2],
                "representative_pattern": f"{genotypes[0]},{genotypes[1]},{genotypes[2]}",
                "pattern_source": inferred_source,
                **stats,
            }
        )

    results = pd.DataFrame(records)
    results["bonferroni_p"] = np.minimum(
        1.0,
        results["raw_p"].to_numpy(dtype=np.float64) * n_tests,
    )
    results = results.sort_values(
        ["raw_p", "idx_a", "idx_b", "idx_c"],
        kind="stable",
    ).reset_index(drop=True)
    results.insert(0, "stage2_rank", np.arange(1, len(results) + 1))
    return results


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    X, y, feature_names, _ = load_genotype_dataset(
        args.data, label_column=args.label_column,
        case_value=args.case_value, control_value=args.control_value,
    )
    candidate_path = (
        args.stage1_ranking
        if args.stage1_ranking is not None
        else args.consensus_topk
    )
    candidates = pd.read_csv(candidate_path)
    results = compute_stage2_results(
        X,
        y,
        feature_names,
        candidates,
        top_k=args.top_k,
        yates_correction=args.yates_correction,
        zero_cell_correction=args.zero_cell_correction,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    csv_path = args.output / "stage2_fixed_pattern_triplets.csv"
    results.to_csv(csv_path, index=False)
    normalized_candidates, candidate_rule = normalize_stage1_candidates(candidates)
    actual_candidate_count = min(
        len(normalized_candidates),
        int(args.top_k) if args.top_k is not None else len(normalized_candidates),
    )
    runtime_sec = time.perf_counter() - started
    summary = {
        "n_candidate_snps": int(actual_candidate_count),
        "n_triplets": int(len(results)),
        "pattern_rule": "one pre-specified representative genotype per SNP from Stage-1 embedding rho; no OR maximization",
        "candidate_rule": candidate_rule,
        "primary_test": "2x2 chi-square for fixed triplet genotype pattern versus all other patterns",
        "bonferroni_tests": int(len(results)),
        "chi_square_yates_correction": bool(args.yates_correction),
        "zero_cell_or_correction": float(args.zero_cell_correction),
        "data_sha256": sha256_file(args.data.resolve()),
        "candidate_file_sha256": sha256_file(candidate_path.resolve()),
        "feature_order_sha256": feature_order_sha256(feature_names),
        "runtime_sec": float(runtime_sec),
        "output_csv": str(csv_path.resolve()),
    }
    write_json(args.output / "stage2_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
