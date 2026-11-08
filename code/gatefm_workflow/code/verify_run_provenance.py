"""Verify complete synthetic-run provenance before reuse or benchmark aggregation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from repro_io import resolve_config, sha256_file


def resolved_run_config(
    config_path: Path,
    dimension: int,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    latent_rank: int | None = None,
) -> dict:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = resolve_config(raw, dimension)
    if epochs is not None:
        config["training"]["epochs"] = int(epochs)
    if batch_size is not None:
        config["training"]["batch_size"] = int(batch_size)
    if latent_rank is not None:
        config["model"]["latent_rank"] = int(latent_rank)
    return config


def manifest_dataset_sha256(row, manifest_path: Path) -> tuple[Path, str]:
    path = Path(str(row.data_path))
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    advertised = getattr(row, "dataset_sha256", None)
    if advertised is not None and pd.notna(advertised):
        if str(advertised).strip().lower() != actual_hash.lower():
            raise ValueError(
                f"Synthetic manifest checksum differs from dataset: "
                f"P={row.dimension}, seed={row.data_seed}."
            )
    return path, actual_hash


def verify_complete_run(
    run_root: Path,
    dimension: int,
    model_seed: int,
    data_sha256: str,
    config: dict,
) -> None:
    """Fail closed if complete Stage 1/2 outputs lack consistent source evidence."""
    root = Path(run_root)
    s1 = root / "stage1"
    s2 = root / "stage2"
    required = (
        s1 / "data_manifest.json",
        s1 / "config_resolved.json",
        s1 / "run_summary.json",
        s1 / "stage1_snp_ranking.csv",
        s1 / "model_checkpoint.pt",
        s2 / "stage2_summary.json",
        s2 / "stage2_fixed_pattern_triplets.csv",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty synthetic-run artifact: {path}")

    data_info = json.loads((s1 / "data_manifest.json").read_text(encoding="utf-8"))
    actual_config = json.loads((s1 / "config_resolved.json").read_text(encoding="utf-8"))
    stage1_summary = json.loads((s1 / "run_summary.json").read_text(encoding="utf-8"))
    stage2_summary = json.loads((s2 / "stage2_summary.json").read_text(encoding="utf-8"))

    if str(data_info.get("data_sha256", "")).lower() != data_sha256.lower():
        raise ValueError(f"Stage 1 source dataset checksum mismatch: {root}")
    if int(data_info.get("n_snps", -1)) != dimension:
        raise ValueError(f"Stage 1 source SNP dimension mismatch: {root}")
    if actual_config != config:
        raise ValueError(f"Stage 1 resolved training configuration mismatch: {root}")
    if int(stage1_summary.get("run_seed", -1)) != model_seed:
        raise ValueError(f"Stage 1 model seed mismatch: {root}")

    rank_path = s1 / "stage1_snp_ranking.csv"
    ranking = pd.read_csv(rank_path)
    for key in ("global_idx", "snp_id", "q_rank", "run_seed"):
        if key not in ranking.columns:
            raise ValueError(f"Stage 1 ranking is missing {key}: {root}")
    if len(ranking) != dimension or ranking["snp_id"].isna().any():
        raise ValueError(f"Stage 1 ranking is incomplete: {root}")
    if ranking["global_idx"].duplicated().any() or ranking["snp_id"].duplicated().any():
        raise ValueError(f"Stage 1 ranking contains duplicate SNPs: {root}")
    if not np.array_equal(
        np.sort(ranking["q_rank"].to_numpy(dtype=np.int64)),
        np.arange(1, dimension + 1, dtype=np.int64),
    ):
        raise ValueError(f"Stage 1 SNP ranks are not a complete permutation: {root}")
    if not (ranking["run_seed"].astype(np.int64) == model_seed).all():
        raise ValueError(f"Ranking seed mismatch: {root}")

    if str(stage2_summary.get("data_sha256", "")).lower() != data_sha256.lower():
        raise ValueError(f"Stage 2 source dataset checksum mismatch: {root}")
    if str(stage2_summary.get("candidate_file_sha256", "")).lower() != sha256_file(rank_path):
        raise ValueError(f"Stage 2 candidate ranking checksum mismatch: {root}")
    if stage2_summary.get("feature_order_sha256") != data_info.get("feature_order_sha256"):
        raise ValueError(f"Stage 2 feature-order checksum mismatch: {root}")
    candidate_count = min(30, dimension)
    triplet_count = math.comb(candidate_count, 3)
    if (
        int(stage2_summary.get("n_candidate_snps", -1)) != candidate_count
        or int(stage2_summary.get("n_triplets", -1)) != triplet_count
    ):
        raise ValueError(f"Stage 2 candidate or triplet count mismatch: {root}")
    if len(pd.read_csv(s2 / "stage2_fixed_pattern_triplets.csv")) != triplet_count:
        raise ValueError(f"Stage 2 results table is incomplete: {root}")