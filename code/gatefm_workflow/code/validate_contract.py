from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro_io import resolve_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen code configuration and conceptual manuscript alignment."
    )
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
    parser.add_argument(
        "--manuscript-root",
        type=Path,
        default=None,
        help="Optional directory containing main.tex and supplement.tex.",
    )
    return parser.parse_args()


def require_equal(label: str, actual, expected) -> None:
    if isinstance(expected, float):
        if not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12):
            raise AssertionError(f"{label}: expected {expected}, found {actual}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")


def validate_config(config: dict) -> None:
    dimensions = {
        20: (16, 300, 64, 3.0, "n_snps", False),
        100: (32, 300, 64, 10.0, "n_snps", False),
        500: (256, 300, 64, 50.0, "half_n_snps", False),
        1000: (256, 300, 64, 100.0, "half_n_snps", True),
        5000: (512, 300, 64, 500.0, "half_n_snps", True),
        10000: (1024, 100, 16, 1000.0, "half_n_snps", True),
        14313: (20, 300, 128, 3.0, "half_n_snps", True),
    }
    for n_snps, expected in dimensions.items():
        latent_rank, epochs, batch_size, target_end, target_start, scheduled = expected
        resolved = resolve_config(config, n_snps)
        prefix = f"P={n_snps}"
        require_equal(prefix + " latent_rank", resolved["model"]["latent_rank"], latent_rank)
        require_equal(prefix + " epochs", resolved["training"]["epochs"], epochs)
        require_equal(prefix + " batch_size", resolved["training"]["batch_size"], batch_size)
        require_equal(prefix + " target_end", resolved["gate_constraint"]["target_end"], target_end)
        require_equal(prefix + " target_start", resolved["gate_constraint"]["target_start"], target_start)
        has_schedule = bool(resolved["training"].get("gate_learning_rate_schedule"))
        require_equal(prefix + " gate LR schedule", has_schedule, scheduled)

        require_equal(prefix + " gate_activation", resolved["model"]["gate_activation"], "hard_sigmoid")
        require_equal(prefix + " gate_temperature", resolved["model"]["gate_temperature"], 0.7)
        require_equal(prefix + " gamma_tri", resolved["model"]["gamma_tri"], 0.05)
        require_equal(prefix + " model LR", resolved["training"]["model_learning_rate"], 0.0001)
        require_equal(prefix + " weight decay", resolved["training"]["weight_decay"], 0.0001)
        require_equal(prefix + " validation fraction", resolved["training"]["validation_fraction"], 0.1)
        require_equal(prefix + " split seed", resolved["training"]["split_seed"], 42)
        require_equal(prefix + " final checkpoint", resolved["training"]["checkpoint_metric"], "final_epoch")
        require_equal(prefix + " constraint lambda", resolved["gate_constraint"]["lambda"], 0.001)
        require_equal(prefix + " anneal fraction", resolved["gate_constraint"]["anneal_fraction"], 0.9)
        require_equal(prefix + " no Yates", resolved["stage2"]["chi_square_yates_correction"], False)
        require_equal(prefix + " zero-cell correction", resolved["stage2"]["zero_cell_correction"], 0.5)


def validate_plan(plan: dict) -> None:
    require_equal(
        "synthetic dimensions",
        plan["synthetic"]["dimensions"],
        [20, 100, 500, 1000, 5000, 10000],
    )
    require_equal("synthetic data seeds", plan["synthetic"]["data_seeds"], [42, 100, 200, 300, 400])
    require_equal(
        "synthetic model seeds",
        plan["synthetic"]["model_initialization_seeds"],
        [3484015997, 3484015998, 3484015999],
    )
    require_equal("human run count", plan["human"]["consensus_runs"], 10)
    require_equal("human seed count", len(plan["human"]["model_initialization_seeds"]), 10)
    require_equal("human Stage-2 K", plan["human"]["stage2_top_k"], 30)


def validate_manuscript(root: Path) -> None:
    main_text = (root / "main.tex").read_text(encoding="utf-8")
    supplement_text = (root / "supplement.tex").read_text(encoding="utf-8")
    required_main = (
        "g_j=h(\\alpha_j/T)",
        "\\sum_{u<v<w}",
        "q_j=\\max",

    )
    required_supplement = (
        "\\overline{\\operatorname{rank}}(j)",
        "each selected SNP had one representative genotype state fixed",
        "\\binom{30}{2}=435",
    )
    for fragment in required_main:
        if fragment not in main_text:
            raise AssertionError(f"main.tex is missing method-contract fragment: {fragment}")
    # Both phrasings encode exactly one preselected pattern per tested triple.
    accepted_fixed_pattern_phrases = (
        "one Stage 1-fixed pattern per triple",
        "one Stage 1-fixed pattern each",
    )
    if not any(phrase in main_text for phrase in accepted_fixed_pattern_phrases):
        raise AssertionError("main.tex must state one Stage-1-fixed genotype pattern per triple")
    for fragment in required_supplement:
        if fragment not in supplement_text:
            raise AssertionError(
                f"supplement.tex is missing method-contract fragment: {fragment}"
            )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = json.loads(args.experiment_plan.read_text(encoding="utf-8"))
    validate_config(config)
    validate_plan(plan)
    if args.manuscript_root is not None:
        validate_manuscript(args.manuscript_root.resolve())
    if args.manuscript_root is not None:
        print("[PASS] Frozen configuration/plan and required manuscript method fragments checked.")
    else:
        print("[PASS] Frozen configuration/plan checked (no manuscript supplied).")


if __name__ == "__main__":
    main()
