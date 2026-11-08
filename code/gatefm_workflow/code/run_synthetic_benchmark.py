from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from verify_run_provenance import (
    manifest_dataset_sha256, resolved_run_config, verify_complete_run,
)


CODE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_ROOT.parent
REQUIRED_MANIFEST_COLUMNS = {
    "dimension",
    "data_seed",
    "data_path",
    "gt_snp_a",
    "gt_snp_b",
    "gt_snp_c",
}


def comma_ints(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen synthetic GateFM Stage-1 and per-run Stage-2 workflow."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "config_thesis.json",
    )
    parser.add_argument(
        "--experiment-plan",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "experiment_plan.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dimensions", default=None, help="Comma-separated subset.")
    parser.add_argument("--data-seeds", default=None, help="Comma-separated subset.")
    parser.add_argument("--model-seeds", default=None, help="Comma-separated subset.")
    parser.add_argument("--max-runs", default=None, type=int)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--latent-rank", type=int, default=None)
    parser.add_argument("--allow-config-override", action="store_true")
    parser.add_argument("--allow-incomplete-plan", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=CODE_ROOT)


def main() -> None:
    args = parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("--max-runs must be positive")
    override_values = (args.epochs, args.batch_size, args.latent_rank)
    if any(v is not None and v < 1 for v in override_values):
        raise ValueError("Training overrides must be positive")
    if any(v is not None for v in override_values) and not args.allow_config_override:
        raise ValueError("Overrides require --allow-config-override and are not formal paper runs")
    manifest_path = args.manifest.resolve()
    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
    manifest["dimension"] = manifest["dimension"].astype(int)
    manifest["data_seed"] = manifest["data_seed"].astype(int)
    if manifest.duplicated(["dimension", "data_seed"]).any():
        raise ValueError("Dataset manifest contains duplicate dimension/data_seed rows.")

    plan = json.loads(args.experiment_plan.read_text(encoding="utf-8"))
    formal_dimensions = set(int(x) for x in plan["synthetic"]["dimensions"])
    formal_data_seeds = set(int(x) for x in plan["synthetic"]["data_seeds"])
    formal_model_seeds = [
        int(x) for x in plan["synthetic"]["model_initialization_seeds"]
    ]
    dimensions_filter = comma_ints(args.dimensions)
    data_seed_filter = comma_ints(args.data_seeds)
    model_seed_filter = comma_ints(args.model_seeds)

    unknown_dimensions = set(manifest["dimension"]).difference(formal_dimensions)
    unknown_data_seeds = set(manifest["data_seed"]).difference(formal_data_seeds)
    if unknown_dimensions or unknown_data_seeds:
        raise ValueError(
            "Dataset manifest contains values outside the frozen plan: "
            f"dimensions={sorted(unknown_dimensions)}, "
            f"data_seeds={sorted(unknown_data_seeds)}"
        )

    selected = manifest.copy()
    if dimensions_filter is not None:
        selected = selected[selected["dimension"].isin(dimensions_filter)]
    if data_seed_filter is not None:
        selected = selected[selected["data_seed"].isin(data_seed_filter)]
    selected_model_seeds = [
        seed
        for seed in formal_model_seeds
        if model_seed_filter is None or seed in model_seed_filter
    ]
    if selected.empty or not selected_model_seeds:
        raise ValueError("The requested filters selected no synthetic runs.")

    observed_pairs = set(
        zip(selected["dimension"].astype(int), selected["data_seed"].astype(int))
    )
    expected_pairs = {
        (dimension, data_seed)
        for dimension in formal_dimensions
        for data_seed in formal_data_seeds
    }
    incomplete = (
        observed_pairs != expected_pairs
        or selected_model_seeds != formal_model_seeds
        or args.max_runs is not None
    )
    if incomplete and not args.allow_incomplete_plan:
        raise ValueError(
            "This is not the complete frozen plan. Add --allow-incomplete-plan "
            "for a labeled trend/smoke subset."
        )

    jobs = []
    for row in selected.sort_values(["dimension", "data_seed"]).itertuples(index=False):
        data_path, dataset_sha256 = manifest_dataset_sha256(row, manifest_path)
        for model_seed in selected_model_seeds:
            jobs.append(
                (int(row.dimension), int(row.data_seed), data_path, model_seed, dataset_sha256)
            )
    if args.max_runs is not None:
        jobs = jobs[: int(args.max_runs)]

    args.output_root.mkdir(parents=True, exist_ok=True)
    for index, (dimension, data_seed, data_path, model_seed, dataset_sha256) in enumerate(jobs, start=1):
        run_root = (
            args.output_root
            / f"p{dimension}"
            / f"data_seed{data_seed}"
            / f"model_seed{model_seed}"
        )
        stage1 = run_root / "stage1"
        stage2 = run_root / "stage2"
        expected_config = resolved_run_config(
            args.config, dimension, epochs=args.epochs,
            batch_size=args.batch_size, latent_rank=args.latent_rank,
        )
        if stage1.exists() or stage2.exists():
            if args.skip_existing:
                verify_complete_run(
                    run_root, dimension, model_seed, dataset_sha256, expected_config
                )
                print(f"[SKIP {index}/{len(jobs)}] Verified {run_root}", flush=True)
                continue
            raise FileExistsError(
                f"Partial or existing run directory requires manual review: {run_root}"
            )

        print(f"[JOB {index}/{len(jobs)}] P={dimension} data={data_seed} model={model_seed}")
        stage1_command = [
            sys.executable,
            str(CODE_ROOT / "run_stage1.py"),
            "--data", str(data_path), "--output", str(stage1),
            "--config", str(args.config.resolve()), "--seed", str(model_seed),
            "--device", args.device,
        ]
        for flag, value in (("epochs", args.epochs), ("batch-size", args.batch_size),
                            ("latent-rank", args.latent_rank)):
            if value is not None:
                stage1_command += ["--" + flag, str(value)]
        if any(v is not None for v in override_values):
            stage1_command.append("--allow-config-override")
        run_command(stage1_command)
        run_command(
            [
                sys.executable,
                str(CODE_ROOT / "run_stage2.py"),
                "--data",
                str(data_path),
                "--stage1-ranking",
                str(stage1 / "stage1_snp_ranking.csv"),
                "--output",
                str(stage2),
                "--top-k",
                "30",
            ]
        )
        verify_complete_run(run_root, dimension, model_seed, dataset_sha256, expected_config)

    print(f"[PASS] Completed {len(jobs)} requested synthetic run(s).")


if __name__ == "__main__":
    main()
