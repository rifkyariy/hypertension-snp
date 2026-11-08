from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repro_io import write_json


REQUIRED_COLUMNS = {
    "global_idx",
    "snp_id",
    "gate_value",
    "rho_g0",
    "rho_g1",
    "rho_g2",
    "representative_genotype",
    "q_score",
    "q_rank",
    "run_seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate complete Stage-1 rankings across independent runs."
    )
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", default=30, type=int)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument(
        "--require-run-manifests",
        action="store_true",
        help="Require and cross-check data_manifest.json beside every ranking file.",
    )
    parser.add_argument(
        "--pattern",
        default="run_*/stage1_snp_ranking.csv",
        help="Glob relative to --runs-root.",
    )
    return parser.parse_args()


def consensus_genotype(group: pd.DataFrame) -> tuple[int, bool, str]:
    votes = np.bincount(
        group["representative_genotype"].to_numpy(dtype=np.int64),
        minlength=3,
    )
    max_vote = int(votes.max())
    tied = np.flatnonzero(votes == max_vote)
    tie_occurred = len(tied) > 1
    if len(tied) == 1:
        selected = int(tied[0])
        reason = "majority_vote"
    else:
        rho_means = np.asarray(
            [
                float(group[f"rho_g{genotype}"].mean())
                for genotype in range(3)
            ],
            dtype=np.float64,
        )
        tied_rho = rho_means[tied]
        selected = int(tied[np.argmax(tied_rho)])
        reason = "vote_tie_mean_rho"
        if np.sum(np.isclose(tied_rho, tied_rho.max(), rtol=0.0, atol=1e-12)) > 1:
            selected = int(np.min(tied[np.isclose(tied_rho, tied_rho.max(), atol=1e-12)]))
            reason = "vote_and_rho_tie_smallest_genotype"
    return selected, tie_occurred, reason


def aggregate_rankings(
    ranking_paths: list[Path],
    *,
    top_k: int,
    expected_runs: int | None = None,
    require_run_manifests: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not ranking_paths:
        raise ValueError("No Stage-1 ranking files were supplied.")
    if expected_runs is not None and len(ranking_paths) != int(expected_runs):
        raise ValueError(
            f"Expected {int(expected_runs)} Stage-1 runs, found {len(ranking_paths)}."
        )

    frames = []
    reference = None
    data_hashes = set()
    feature_hashes = set()
    for run_number, path in enumerate(sorted(ranking_paths), start=1):
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["global_idx"].duplicated().any() or frame["snp_id"].duplicated().any():
            raise ValueError(f"Duplicate SNP indices or SNP IDs in {path}.")
        if frame["q_rank"].duplicated().any():
            raise ValueError(f"Duplicate q ranks in {path}.")
        if not np.array_equal(
            np.sort(frame["q_rank"].to_numpy(dtype=np.int64)),
            np.arange(1, len(frame) + 1, dtype=np.int64)
        ):
            raise ValueError(f"Stage-1 ranking must be a complete 1..P permutation: {path}.")
        if frame["run_seed"].nunique() != 1:
            raise ValueError(f"Ranking contains inconsistent run seeds: {path}.")
        frame = frame.sort_values("global_idx").reset_index(drop=True)
        current_reference = frame[["global_idx", "snp_id"]].copy()
        if reference is None:
            reference = current_reference
        elif not reference.equals(current_reference):
            raise ValueError(
                f"SNP index/name order differs between runs; first mismatch is in {path}."
            )
        frame["run_number"] = int(run_number)
        frame["ranking_path"] = str(path.resolve())
        frames.append(frame)
        manifest_path = path.with_name("data_manifest.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data_hashes.add(str(manifest["data_sha256"]))
            feature_hashes.add(str(manifest["feature_order_sha256"]))
        elif require_run_manifests:
            raise FileNotFoundError(
                f"Missing required Stage-1 data manifest: {manifest_path}"
            )

    unique_run_seeds = {
        int(frame["run_seed"].iloc[0])
        for frame in frames
    }
    if len(unique_run_seeds) != len(frames):
        raise ValueError("Repeated Stage-1 run seeds are not independent fits.")
    if len(data_hashes) > 1:
        raise ValueError("Stage-1 runs were generated from different data files.")
    if len(feature_hashes) > 1:
        raise ValueError("Stage-1 run manifests report different SNP orders.")

    long = pd.concat(frames, ignore_index=True)
    rows = []
    for (global_idx, snp_id), group in long.groupby(
        ["global_idx", "snp_id"],
        sort=True,
    ):
        representative, had_tie, tie_reason = consensus_genotype(group)
        votes = np.bincount(
            group["representative_genotype"].to_numpy(dtype=np.int64),
            minlength=3,
        )
        rows.append(
            {
                "global_idx": int(global_idx),
                "snp_id": str(snp_id),
                "n_runs": int(len(group)),
                "mean_q_rank": float(group["q_rank"].mean()),
                "std_q_rank": float(group["q_rank"].std(ddof=0)),
                "best_q_rank": int(group["q_rank"].min()),
                "worst_q_rank": int(group["q_rank"].max()),
                "mean_q_score": float(group["q_score"].mean()),
                "mean_gate_value": float(group["gate_value"].mean()),
                "vote_g0": int(votes[0]),
                "vote_g1": int(votes[1]),
                "vote_g2": int(votes[2]),
                "mean_rho_g0": float(group["rho_g0"].mean()),
                "mean_rho_g1": float(group["rho_g1"].mean()),
                "mean_rho_g2": float(group["rho_g2"].mean()),
                "representative_genotype": int(representative),
                "genotype_vote_tie": bool(had_tie),
                "genotype_decision_rule": tie_reason,
                "run_seeds": ",".join(
                    str(int(x)) for x in sorted(group["run_seed"].unique())
                ),
            }
        )

    consensus = pd.DataFrame(rows).sort_values(
        ["mean_q_rank", "mean_q_score", "global_idx"],
        ascending=[True, False, True],
        kind="stable",
    )
    consensus["consensus_rank"] = np.arange(1, len(consensus) + 1)
    consensus = consensus[
        ["consensus_rank"]
        + [column for column in consensus.columns if column != "consensus_rank"]
    ]
    selected = consensus.head(min(int(top_k), len(consensus))).copy()
    summary = {
        "n_runs": int(len(frames)),
        "n_snps": int(len(consensus)),
        "top_k": int(len(selected)),
        "expected_runs": None if expected_runs is None else int(expected_runs),
        "run_seeds": sorted(unique_run_seeds),
        "data_sha256": next(iter(data_hashes)) if data_hashes else None,
        "feature_order_sha256": next(iter(feature_hashes)) if feature_hashes else None,
        "ranking_paths": [str(path.resolve()) for path in sorted(ranking_paths)],
        "aggregation_rule": "ascending mean complete q_rank; tie by descending mean q_score; then global_idx",
        "genotype_rule": "majority vote across runs; tie by mean rho; final tie by smaller genotype code",
        "top_snp_ids": selected["snp_id"].tolist(),
    }
    return consensus, selected, summary


def main() -> None:
    args = parse_args()
    ranking_paths = sorted(args.runs_root.glob(args.pattern))
    consensus, selected, summary = aggregate_rankings(
        ranking_paths,
        top_k=args.top_k,
        expected_runs=args.expected_runs,
        require_run_manifests=args.require_run_manifests,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    consensus.to_csv(args.output / "stage1_consensus_all_snps.csv", index=False)
    selected.to_csv(args.output / f"stage1_consensus_top{len(selected)}.csv", index=False)
    write_json(args.output / "aggregation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
