from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aggregate_stage1 import aggregate_rankings
from gatefm_model import GateFMModelConfig, ThirdOrderGateFM
from posthoc_lower_order_screen import compute_lower_order_screen
from run_stage2 import compute_stage2_results
from validate_contract import validate_config, validate_plan


def assert_close(name: str, actual, expected, atol: float = 1e-5) -> None:
    if not np.allclose(actual, expected, rtol=1e-5, atol=atol):
        maximum = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise AssertionError(f"{name} mismatch; max_abs_difference={maximum}")


def verify_fm_trick_and_scores() -> None:
    torch.manual_seed(7)
    model = ThirdOrderGateFM(
        GateFMModelConfig(
            n_snps=6,
            latent_rank=5,
            gate_temperature=0.9,
            gate_activation="hard_sigmoid",
            gamma_tri=0.25,
        )
    )
    X = torch.randint(0, 3, (11, 6), dtype=torch.long)
    fast = model.third_order_term(X).detach().numpy()
    direct = model.direct_third_order_term(X).detach().numpy()
    assert_close("third-order FM trick", fast, direct)
    expected_gate = torch.clamp(
        (model.gate_logit / model.gate_temperature + 1.0) / 2.0,
        min=0.0,
        max=1.0,
    )
    assert_close("hard-sigmoid gate", model.gates().detach().numpy(), expected_gate.detach().numpy())
    expected_logits = (
        model.bias.detach().cpu().numpy()[0] + model.gamma_tri * direct
    )
    assert_close("fixed third-order scale", model(X).detach().numpy(), expected_logits)

    artifacts = model.feature_artifacts()
    expected_rho = artifacts["gate"][:, None] * artifacts["embedding_norm"]
    expected_q = expected_rho.max(axis=1)
    expected_genotype = expected_rho.argmax(axis=1)
    assert_close("rho = gate * embedding norm", artifacts["rho"], expected_rho)
    assert_close("q = max rho", artifacts["q_score"], expected_q)
    if not np.array_equal(artifacts["representative_genotype"], expected_genotype):
        raise AssertionError("representative genotype is not argmax(rho).")


def verify_checkpoint_roundtrip() -> None:
    torch.manual_seed(11)
    config = GateFMModelConfig(n_snps=5, latent_rank=3, gate_temperature=1.2)
    model = ThirdOrderGateFM(config)
    X = torch.randint(0, 3, (9, 5), dtype=torch.long)
    before_logits = model(X).detach().numpy()
    before_artifacts = model.feature_artifacts()

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "checkpoint.pt"
        torch.save({"state": model.state_dict(), "config": config.__dict__}, path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        restored = ThirdOrderGateFM(GateFMModelConfig(**payload["config"]))
        restored.load_state_dict(payload["state"])
        after_logits = restored(X).detach().numpy()
        after_artifacts = restored.feature_artifacts()

    assert_close("checkpoint logits", after_logits, before_logits)
    for key in (
        "gate_logit",
        "gate",
        "embedding",
        "embedding_norm",
        "rho",
        "q_score",
        "representative_genotype",
        "q_rank",
    ):
        assert_close(f"checkpoint artifact {key}", after_artifacts[key], before_artifacts[key])


def make_ranking_frame(seed: int, rank_shift: int = 0) -> pd.DataFrame:
    n = 8
    rank = np.roll(np.arange(1, n + 1), rank_shift)
    q_score = 1.0 / rank
    representative = np.asarray([(index + seed) % 3 for index in range(n)])
    frame = pd.DataFrame(
        {
            "global_idx": np.arange(n),
            "snp_id": [f"SNP{index}" for index in range(n)],
            "gate_value": np.linspace(0.8, 0.2, n),
            "rho_g0": np.linspace(0.3, 0.1, n),
            "rho_g1": np.linspace(0.2, 0.4, n),
            "rho_g2": np.linspace(0.1, 0.2, n),
            "representative_genotype": representative,
            "q_score": q_score,
            "q_rank": rank,
            "run_seed": seed,
        }
    )
    return frame


def verify_aggregation() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        paths = []
        for run_number, seed in enumerate((101, 102, 103), start=1):
            path = root / f"run_{run_number:03d}.csv"
            make_ranking_frame(seed, rank_shift=run_number - 1).to_csv(path, index=False)
            paths.append(path)
        consensus, selected, summary = aggregate_rankings(paths, top_k=5)
    if len(consensus) != 8 or len(selected) != 5 or summary["n_runs"] != 3:
        raise AssertionError("Stage-1 aggregation dimensions are incorrect.")
    if not np.array_equal(
        consensus["consensus_rank"].to_numpy(),
        np.arange(1, 9),
    ):
        raise AssertionError("Consensus rank is not a complete 1..P ordering.")


def verify_stage2_fixed_patterns() -> None:
    rng = np.random.default_rng(23)
    n_samples = 180
    n_snps = 30
    X = rng.integers(0, 3, size=(n_samples, n_snps), dtype=np.int16)
    y = np.asarray([0, 1] * (n_samples // 2), dtype=np.int64)
    feature_names = [f"SNP{index}" for index in range(n_snps)]
    fixed_genotypes = np.asarray([index % 3 for index in range(n_snps)])
    consensus = pd.DataFrame(
        {
            "consensus_rank": np.arange(1, n_snps + 1),
            "global_idx": np.arange(n_snps),
            "snp_id": feature_names,
            "representative_genotype": fixed_genotypes,
        }
    )

    single_run = consensus.rename(columns={"consensus_rank": "q_rank"})
    single_run_results = compute_stage2_results(
        X,
        y,
        feature_names,
        single_run,
        top_k=30,
    )
    if len(single_run_results) != 4060:
        raise AssertionError("Single-run synthetic Stage 2 did not produce 4,060 triplets.")
    if set(single_run_results["pattern_source"]) != {
        "single-run GateFM q rank and rho argmax"
    }:
        raise AssertionError("Synthetic Stage 2 did not use the individual run's q ranking.")

    results = compute_stage2_results(
        X,
        y,
        feature_names,
        consensus,
        top_k=30,
    )
    if len(results) != 4060:
        raise AssertionError(f"Expected 4,060 triplets, found {len(results)}.")

    for row in results.itertuples(index=False):
        expected = (
            fixed_genotypes[int(row.idx_a)],
            fixed_genotypes[int(row.idx_b)],
            fixed_genotypes[int(row.idx_c)],
        )
        actual = (int(row.genotype_a), int(row.genotype_b), int(row.genotype_c))
        if actual != expected:
            raise AssertionError("Stage-2 changed an embedding-specified genotype pattern.")

    permuted_results = compute_stage2_results(
        X,
        y[::-1],
        feature_names,
        consensus,
        top_k=30,
    )
    key_columns = ["idx_a", "idx_b", "idx_c", "genotype_a", "genotype_b", "genotype_c"]
    left = results[key_columns].sort_values(["idx_a", "idx_b", "idx_c"]).reset_index(drop=True)
    right = (
        permuted_results[key_columns]
        .sort_values(["idx_a", "idx_b", "idx_c"])
        .reset_index(drop=True)
    )
    if not left.equals(right):
        raise AssertionError(
            "Stage-2 representative patterns changed after outcome labels were permuted."
        )


def verify_fixed_pattern_lower_order_screen() -> None:
    rng = np.random.default_rng(29)
    n_samples = 240
    n_snps = 30
    X = rng.integers(0, 3, size=(n_samples, n_snps), dtype=np.int16)
    y = np.asarray([0, 1] * (n_samples // 2), dtype=np.int64)
    feature_names = [f"SNP{index}" for index in range(n_snps)]
    fixed_genotypes = np.asarray([index % 3 for index in range(n_snps)])
    consensus = pd.DataFrame(
        {
            "consensus_rank": np.arange(1, n_snps + 1),
            "global_idx": np.arange(n_snps),
            "snp_id": feature_names,
            "representative_genotype": fixed_genotypes,
        }
    )
    triplets = compute_stage2_results(
        X,
        y,
        feature_names,
        consensus,
        top_k=30,
    )
    singles, pairs, screened, summary = compute_lower_order_screen(
        X,
        y,
        feature_names,
        consensus,
        triplets,
    )
    if len(singles) != 30 or len(pairs) != 435 or len(screened) != 4060:
        raise AssertionError("Lower-order screen dimensions are incorrect.")
    if summary["pattern_rule"] != "Stage-1-fixed genotype pattern versus its complement":
        raise AssertionError("Lower-order screen did not report the frozen-pattern rule.")
    first_pair = pairs.iloc[0]
    expected = (
        fixed_genotypes[int(first_pair.idx_a)],
        fixed_genotypes[int(first_pair.idx_b)],
    )
    actual = (int(first_pair.genotype_a), int(first_pair.genotype_b))
    if actual != expected:
        raise AssertionError("Pair screen changed a Stage-1-fixed genotype.")


def verify_frozen_contract() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (package_root / "configs" / "config_thesis.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (package_root / "configs" / "experiment_plan.json").read_text(encoding="utf-8")
    )
    validate_config(config)
    validate_plan(plan)


def main() -> None:
    checks = [
        ("frozen reference implementation configuration", verify_frozen_contract),
        ("FM trick and multiplicative score", verify_fm_trick_and_scores),
        ("checkpoint roundtrip", verify_checkpoint_roundtrip),
        ("10-run-style aggregation", verify_aggregation),
        ("fixed-pattern Stage-2 with 4,060 triplets", verify_stage2_fixed_patterns),
        ("fixed-pattern 30-single/435-pair screen", verify_fixed_pattern_lower_order_screen),
    ]
    for name, function in checks:
        function()
        print(f"[PASS] {name}")
    print("[PASS] All GateFM numerical regression checks completed.")


if __name__ == "__main__":
    main()
