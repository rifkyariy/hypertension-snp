from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_run_provenance import (
    manifest_dataset_sha256, resolved_run_config, verify_complete_run,
)


REQUIRED_MANIFEST_COLUMNS = {
    "dimension",
    "data_seed",
    "gt_snp_a",
    "gt_snp_b",
    "gt_snp_c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild GateFM rank, coverage, Hit@1, and runtime summaries."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-plan", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "config_thesis.json",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--latent-rank", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--non-reference", action="store_true",
                        help="Label custom-config complete runs as non-reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
    manifest["dimension"] = manifest["dimension"].astype(int)
    manifest["data_seed"] = manifest["data_seed"].astype(int)
    if manifest.duplicated(["dimension", "data_seed"]).any():
        raise ValueError("Duplicate dimension/data_seed entries in synthetic manifest.")
    lookup = {        (int(row.dimension), int(row.data_seed)): row
        for row in manifest.itertuples(index=False)
    }
    plan = json.loads(args.experiment_plan.read_text(encoding="utf-8"))
    dimensions = [int(x) for x in plan["synthetic"]["dimensions"]]
    data_seeds = [int(x) for x in plan["synthetic"]["data_seeds"]]
    model_seeds = [int(x) for x in plan["synthetic"]["model_initialization_seeds"]]

    rows = []
    checked_datasets: dict[tuple[int, int], str] = {}
    for stage1_path in sorted(args.runs_root.glob("p*/data_seed*/model_seed*/stage1/stage1_snp_ranking.csv")):
        model_root = stage1_path.parent.parent
        dimension = int(model_root.parent.parent.name.removeprefix("p"))
        data_seed = int(model_root.parent.name.removeprefix("data_seed"))
        model_seed = int(model_root.name.removeprefix("model_seed"))
        manifest_row = lookup.get((dimension, data_seed))
        if manifest_row is None:
            raise ValueError(f"No dataset manifest row for P={dimension}, data seed={data_seed}.")

        pair = (dimension, data_seed)
        if pair not in checked_datasets:
            _, checked_datasets[pair] = manifest_dataset_sha256(
                manifest_row, args.manifest
            )
        expected_config = resolved_run_config(
            args.config, dimension, epochs=args.epochs,
            batch_size=args.batch_size, latent_rank=args.latent_rank,
        )
        verify_complete_run(
            model_root, dimension, model_seed,
            checked_datasets[pair], expected_config,
        )

        ranking = pd.read_csv(stage1_path)
        gt_snps = [
            str(manifest_row.gt_snp_a),
            str(manifest_row.gt_snp_b),
            str(manifest_row.gt_snp_c),
        ]
        rank_map = dict(zip(ranking["snp_id"].astype(str), ranking["q_rank"].astype(int)))
        missing_gt = [snp for snp in gt_snps if snp not in rank_map]
        if missing_gt:
            raise ValueError(f"Ground-truth SNPs missing from ranking: {missing_gt}")
        gt_ranks = [rank_map[snp] for snp in gt_snps]

        stage2_path = model_root / "stage2" / "stage2_fixed_pattern_triplets.csv"
        stage2_summary_path = model_root / "stage2" / "stage2_summary.json"
        stage1_summary_path = stage1_path.with_name("run_summary.json")
        if not stage2_path.is_file() or not stage2_summary_path.is_file():
            raise FileNotFoundError(f"Missing Stage-2 output under {model_root}")
        stage2 = pd.read_csv(stage2_path)
        gt_set = frozenset(gt_snps)
        matching = stage2[
            stage2.apply(
                lambda row: frozenset((str(row.snp_a), str(row.snp_b), str(row.snp_c))) == gt_set,
                axis=1,
            )
        ]
        if len(matching) > 1:
            raise AssertionError("The generating triplet appeared more than once.")
        hit_at_1 = bool(len(matching) == 1 and int(matching.iloc[0]["stage2_rank"]) == 1)
        generating_triplet_rank = (
            int(matching.iloc[0]["stage2_rank"]) if len(matching) == 1 else None
        )
        stage1_summary = json.loads(stage1_summary_path.read_text(encoding="utf-8"))
        stage2_summary = json.loads(stage2_summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dimension": dimension,
                "data_seed": data_seed,
                "model_seed": model_seed,
                "gt_rank_a": gt_ranks[0],
                "gt_rank_b": gt_ranks[1],
                "gt_rank_c": gt_ranks[2],
                "mean_gt_rank": float(np.mean(gt_ranks)),
                "coverage_at_10": all(rank <= min(10, dimension) for rank in gt_ranks),
                "coverage_at_30": all(rank <= min(30, dimension) for rank in gt_ranks),
                "coverage_at_50": all(rank <= min(50, dimension) for rank in gt_ranks),
                "coverage_at_100": all(rank <= min(100, dimension) for rank in gt_ranks),
                "generating_triplet_stage2_rank": generating_triplet_rank,
                "hit_at_1": hit_at_1,
                "stage1_runtime_sec": float(stage1_summary["runtime_sec"]),
                "stage2_runtime_sec": float(stage2_summary["runtime_sec"]),
                "total_runtime_sec": float(stage1_summary["runtime_sec"])
                + float(stage2_summary["runtime_sec"]),
            }
        )

    per_run = pd.DataFrame(rows)
    if per_run.empty:
        raise ValueError("No completed synthetic runs were found.")
    reference_config = (
        Path(__file__).resolve().parent.parent / "configs" / "config_thesis.json"
    ).resolve()
    custom_configuration = (
        args.config.resolve() != reference_config
        or args.epochs is not None
        or args.batch_size is not None
        or args.latent_rank is not None
    )
    if custom_configuration and not args.non_reference:
        raise ValueError(
            "A custom benchmark configuration requires --non-reference "
            "for correctly scoped evaluation results."
        )
    expected = {        (dimension, data_seed, model_seed)
        for dimension in dimensions
        for data_seed in data_seeds
        for model_seed in model_seeds
    }
    observed = set(
        zip(
            per_run["dimension"].astype(int),
            per_run["data_seed"].astype(int),
            per_run["model_seed"].astype(int),
        )
    )
    missing_runs = sorted(expected.difference(observed))
    extra_runs = sorted(observed.difference(expected))
    if extra_runs:
        raise ValueError(f"Unexpected run identities outside the frozen plan: {extra_runs[:5]}")
    if missing_runs and not args.allow_incomplete:
        raise ValueError(
            f"The frozen plan is incomplete ({len(missing_runs)} missing runs). "
            "Use --allow-incomplete only for a labeled trend check."
        )

    summary = (
        per_run.groupby("dimension", as_index=False)
        .agg(
            n_runs=("mean_gt_rank", "size"),
            mean_gt_rank=("mean_gt_rank", "mean"),
            sd_gt_rank=("mean_gt_rank", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else np.nan),
            coverage_at_10=("coverage_at_10", "mean"),
            coverage_at_30=("coverage_at_30", "mean"),
            coverage_at_50=("coverage_at_50", "mean"),
            coverage_at_100=("coverage_at_100", "mean"),
            hit_at_1=("hit_at_1", "mean"),
            mean_runtime_sec=("total_runtime_sec", "mean"),
        )
        .sort_values("dimension")
    )
    scope = ("custom_configuration" if args.non_reference
             else "incomplete_trend_check" if missing_runs
             else "complete_frozen_plan")
    summary.insert(0, "result_scope", scope)
    args.output.mkdir(parents=True, exist_ok=False)
    per_run.sort_values(["dimension", "data_seed", "model_seed"]).to_csv(
        args.output / "synthetic_per_run_results.csv", index=False
    )
    summary.to_csv(args.output / "synthetic_dimension_summary.csv", index=False)
    (args.output / "evaluation_status.json").write_text(
        json.dumps(
            {
                "result_scope": scope,
                "observed_runs": len(observed),
                "expected_runs": len(expected),
                "missing_runs": [
                    [int(dimension), int(data_seed), int(model_seed)]
                    for dimension, data_seed, model_seed in missing_runs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
