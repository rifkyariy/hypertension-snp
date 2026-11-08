from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from gatefm_model import ThirdOrderGateFM, model_config_from_dict
from repro_io import (
    feature_order_sha256,
    load_genotype_dataset,
    resolve_config,
    set_reproducible_seed,
    sha256_file,
    write_json,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one configuration-controlled third-order GateFM run."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--label-column", help="Explicit phenotype column.")
    parser.add_argument("--case-value", help="Source value meaning case (mapped to 1).")
    parser.add_argument("--control-value", help="Source value meaning control (mapped to 0).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "config_thesis.json",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--latent-rank", type=int, default=None)
    parser.add_argument(
        "--allow-config-override",
        action="store_true",
        help="Required when overriding frozen reference epochs, batch size, or latent rank.",
    )
    return parser.parse_args()


def safe_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, float(np.mean(y_true))
    return (
        float(roc_auc_score(y_true, probabilities)),
        float(average_precision_score(y_true, probabilities)),
    )


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    dedicated_generator: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
    )
    generator = None
    if shuffle and dedicated_generator:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        persistent_workers=bool(num_workers > 0),
        generator=generator,
    )


@torch.no_grad()
def evaluate(
    model: ThirdOrderGateFM,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    labels = []
    probabilities = []
    loss_sum = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.long, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
        logits = model(xb).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="sum")
        prob = torch.sigmoid(logits)
        labels.append(yb.cpu())
        probabilities.append(prob.cpu())
        loss_sum += float(loss.item())
        n_samples += int(yb.numel())

    y_true = torch.cat(labels).numpy()
    y_prob = torch.cat(probabilities).numpy()
    auc, pr = safe_metrics(y_true, y_prob)
    return auc, pr, loss_sum / max(n_samples, 1)


def resolve_target_start(value: object, n_snps: int) -> float:
    if isinstance(value, str):
        if value == "n_snps":
            return float(n_snps)
        if value == "half_n_snps":
            return float(n_snps) / 2.0
        raise ValueError(
            "gate_constraint.target_start must be numeric, 'n_snps', or 'half_n_snps'."
        )
    return float(value)


def gate_target_at_epoch(
    epoch: int,
    epochs: int,
    *,
    target_start: float,
    target_end: float,
    anneal_fraction: float,
    anneal_power: float,
) -> float:
    anneal_epochs = max(float(epochs) * float(anneal_fraction), 1.0)
    progress = min(1.0, max(0.0, float(epoch) / anneal_epochs))
    progress = progress ** float(anneal_power)
    return float(target_start + (target_end - target_start) * progress)


def snapshot_state_cpu(model: ThirdOrderGateFM) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def gate_learning_rate_at_epoch(training_config: dict, epoch: int) -> float:
    schedule = training_config.get("gate_learning_rate_schedule")
    if not schedule:
        return float(training_config["gate_learning_rate"])

    selected = None
    for item in sorted(schedule, key=lambda row: int(row["start_epoch"])):
        if int(epoch) >= int(item["start_epoch"]):
            selected = float(item["learning_rate"])
        else:
            break
    if selected is None:
        raise ValueError(
            "gate_learning_rate_schedule must contain an entry starting at epoch 1."
        )
    return selected


def build_ranking_frame(
    artifacts: Dict[str, np.ndarray],
    feature_names: list[str],
    selected_epoch: int,
    seed: int,
) -> pd.DataFrame:
    n_snps = len(feature_names)
    rows = {
        "global_idx": np.arange(n_snps, dtype=np.int32),
        "snp_id": feature_names,
        "gate_logit": artifacts["gate_logit"],
        "gate_value": artifacts["gate"],
        "embedding_norm_g0": artifacts["embedding_norm"][:, 0],
        "embedding_norm_g1": artifacts["embedding_norm"][:, 1],
        "embedding_norm_g2": artifacts["embedding_norm"][:, 2],
        "rho_g0": artifacts["rho"][:, 0],
        "rho_g1": artifacts["rho"][:, 1],
        "rho_g2": artifacts["rho"][:, 2],
        "representative_genotype": artifacts["representative_genotype"],
        "q_score": artifacts["q_score"],
        "q_rank": artifacts["q_rank"],
        "selected_epoch": np.full(n_snps, int(selected_epoch), dtype=np.int32),
        "run_seed": np.full(n_snps, int(seed), dtype=np.int64),
    }
    return pd.DataFrame(rows).sort_values(
        ["q_rank", "global_idx"],
        kind="stable",
    )


def train_stage1(
    *,
    data_path: Path,
    output_dir: Path,
    config_path: Path,
    seed: int,
    device_arg: str = "auto",
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    latent_rank_override: int | None = None,
    allow_config_override: bool = False,
    label_column: str | None = None,
    case_value: str | None = None,
    control_value: str | None = None,
) -> dict:
    started = time.perf_counter()
    data_path = data_path.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    X, y, feature_names, label_column = load_genotype_dataset(
        data_path, label_column=label_column, case_value=case_value,
        control_value=control_value,
    )
    config = resolve_config(raw_config, X.shape[1])
    has_override = any(
        value is not None
        for value in (epochs_override, batch_size_override, latent_rank_override)
    )
    if has_override and not allow_config_override:
        raise ValueError(
            "Manuscript-locked settings cannot be overridden unless "
            "--allow-config-override is supplied. Use overrides only for tests."
        )
    if epochs_override is not None:
        config["training"]["epochs"] = int(epochs_override)
    if batch_size_override is not None:
        config["training"]["batch_size"] = int(batch_size_override)
    if latent_rank_override is not None:
        config["model"]["latent_rank"] = int(latent_rank_override)

    deterministic = bool(config["training"].get("deterministic", True))
    set_reproducible_seed(seed, deterministic=deterministic)
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    all_indices = np.arange(len(y), dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=float(config["training"]["validation_fraction"]),
        random_state=int(config["training"]["split_seed"]),
        stratify=y,
    )
    if bool(config["training"].get("sort_split_indices", True)):
        train_indices = np.sort(train_indices)
        validation_indices = np.sort(validation_indices)
    np.save(output_dir / "train_indices.npy", train_indices)
    np.save(output_dir / "validation_indices.npy", validation_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_validation = X[validation_indices]
    y_validation = y[validation_indices]

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = bool(device.type == "cuda")
    dedicated_loader_generator = bool(
        config["training"].get("dedicated_loader_generator", True)
    )
    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dedicated_generator=dedicated_loader_generator,
    )
    validation_loader = make_loader(
        X_validation,
        y_validation,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dedicated_generator=dedicated_loader_generator,
    )

    model_config = model_config_from_dict(config, X.shape[1])
    model = ThirdOrderGateFM(model_config).to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [model.E_tri, model.bias],
                "lr": float(config["training"]["model_learning_rate"]),
                "weight_decay": float(config["training"]["weight_decay"]),
            },
            {
                "params": [model.gate_logit],
                "lr": float(config["training"]["gate_learning_rate"]),
                "weight_decay": float(config["training"]["weight_decay"]),
            },
        ]
    )

    positive = float(np.sum(y_train == 1))
    negative = float(np.sum(y_train == 0))
    pos_weight = torch.tensor(
        negative / max(positive, 1.0),
        dtype=torch.float32,
        device=device,
    )

    epochs = int(config["training"]["epochs"])
    gate_cfg = config["gate_constraint"]
    target_start = resolve_target_start(gate_cfg["target_start"], X.shape[1])
    target_end = float(gate_cfg["target_end"])
    gate_lambda = float(gate_cfg["lambda"])
    anneal_fraction = float(gate_cfg.get("anneal_fraction", 1.0))
    anneal_power = float(gate_cfg.get("anneal_power", 1.0))

    history: list[dict] = []
    best_metric = -math.inf
    best_state = None
    best_epoch = None
    best_metrics = None
    evaluation_interval = int(config["training"].get("evaluation_interval", 1))
    if evaluation_interval <= 0:
        raise ValueError("training.evaluation_interval must be positive.")

    for epoch in range(1, epochs + 1):
        model.train()
        current_gate_lr = gate_learning_rate_at_epoch(
            config["training"],
            epoch,
        )
        optimizer.param_groups[1]["lr"] = current_gate_lr
        target = gate_target_at_epoch(
            epoch,
            epochs,
            target_start=target_start,
            target_end=target_end,
            anneal_fraction=anneal_fraction,
            anneal_power=anneal_power,
        )
        total_loss_sum = 0.0
        data_loss_sum = 0.0
        constraint_loss_sum = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.long, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(xb).view(-1)
            data_loss = F.binary_cross_entropy_with_logits(
                logits,
                yb,
                pos_weight=pos_weight,
            )
            soft_count = model.gates().sum()
            constraint_loss = gate_lambda * (soft_count - target).pow(2)
            loss = data_loss + constraint_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss_sum += float(loss.item())
            data_loss_sum += float(data_loss.item())
            constraint_loss_sum += float(constraint_loss.item())
            n_batches += 1

        should_evaluate = (
            epoch == 1
            or epoch == epochs
            or epoch % evaluation_interval == 0
        )
        if should_evaluate:
            train_auc, train_pr, train_eval_loss = evaluate(
                model,
                train_loader,
                device,
            )
            val_auc, val_pr, val_loss = evaluate(
                model,
                validation_loader,
                device,
            )
        else:
            train_auc = train_pr = train_eval_loss = math.nan
            val_auc = val_pr = val_loss = math.nan
        with torch.no_grad():
            soft_count_epoch = float(model.gates().sum().item())

        row = {
            "epoch": int(epoch),
            "train_objective": total_loss_sum / max(n_batches, 1),
            "train_data_loss": data_loss_sum / max(n_batches, 1),
            "gate_constraint_loss": constraint_loss_sum / max(n_batches, 1),
            "train_eval_bce": train_eval_loss,
            "val_bce": val_loss,
            "train_auc": train_auc,
            "train_pr": train_pr,
            "val_auc": val_auc,
            "val_pr": val_pr,
            "soft_count": soft_count_epoch,
            "target_soft_count": target,
            "gate_learning_rate": current_gate_lr,
            "metrics_evaluated": should_evaluate,
        }
        history.append(row)

        checkpoint_metric = str(config["training"]["checkpoint_metric"])
        if checkpoint_metric == "val_auc":
            if not should_evaluate:
                continue
            metric_value = val_auc
        elif checkpoint_metric == "val_pr":
            if not should_evaluate:
                continue
            metric_value = val_pr
        elif checkpoint_metric == "final_epoch":
            metric_value = float(epoch)
        else:
            raise ValueError(
                "checkpoint_metric must be 'val_auc', 'val_pr', or 'final_epoch'."
            )

        if metric_value > best_metric + 1e-12:
            best_metric = float(metric_value)
            best_epoch = int(epoch)
            best_state = snapshot_state_cpu(model)
            best_metrics = copy.deepcopy(row)

        if should_evaluate:
            print(
                f"epoch={epoch:04d} "
                f"loss={row['train_objective']:.6f} "
                f"soft={soft_count_epoch:.2f}/{target:.2f} "
                f"gate_lr={current_gate_lr:.6g} "
                f"train_auc={train_auc:.4f} val_auc={val_auc:.4f}",
                flush=True,
            )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError("No checkpoint was selected.")

    # All exported gates, embeddings, scores, and patterns come from this same state.
    model.load_state_dict(best_state)
    model.eval()
    artifacts = model.feature_artifacts()
    ranking = build_ranking_frame(artifacts, feature_names, best_epoch, seed)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    ranking.to_csv(output_dir / "stage1_snp_ranking.csv", index=False)
    np.save(output_dir / "embeddings_tri.npy", artifacts["embedding"])
    np.savez_compressed(
        output_dir / "stage1_snp_artifacts.npz",
        feature_names=np.asarray(feature_names, dtype=str),
        gate_logit=artifacts["gate_logit"],
        gate=artifacts["gate"],
        embedding_norm=artifacts["embedding_norm"],
        rho=artifacts["rho"],
        q_score=artifacts["q_score"],
        q_rank=artifacts["q_rank"],
        representative_genotype=artifacts["representative_genotype"],
    )

    checkpoint_payload = {
        "model_state_dict": best_state,
        "model_config": model_config.__dict__,
        "resolved_config": config,
        "feature_names": feature_names,
        "selected_epoch": int(best_epoch),
        "run_seed": int(seed),
        "best_metrics": best_metrics,
    }
    torch.save(checkpoint_payload, output_dir / "model_checkpoint.pt")

    manifest = {
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "feature_order_sha256": feature_order_sha256(feature_names),
        "label_column": label_column,
        "n_samples": int(X.shape[0]),
        "n_snps": int(X.shape[1]),
        "n_cases": int(np.sum(y == 1)),
        "n_controls": int(np.sum(y == 0)),
        "train_size": int(len(train_indices)),
        "validation_size": int(len(validation_indices)),
    }
    write_json(output_dir / "data_manifest.json", manifest)
    write_json(output_dir / "config_resolved.json", config)

    runtime_sec = time.perf_counter() - started
    summary = {
        "run_seed": int(seed),
        "device": str(device),
        "selected_epoch": int(best_epoch),
        "checkpoint_metric": config["training"]["checkpoint_metric"],
        "method_contract": {
            "gate_activation": model.gate_activation,
            "gate_temperature": model.gate_temperature,
            "gamma_tri": model.gamma_tri,
            "score": "q_j=max_k gate_j*L2(E_tri[j,k,:])",
            "checkpoint": config["training"]["checkpoint_metric"],
        },
        "best_metrics": best_metrics,
        "final_exported_soft_count": float(np.sum(artifacts["gate"])),
        "top30_snp_ids": ranking.head(30)["snp_id"].tolist(),
        "runtime_sec": runtime_sec,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda,
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = train_stage1(
        data_path=args.data,
        output_dir=args.output,
        config_path=args.config,
        seed=args.seed,
        device_arg=args.device,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        latent_rank_override=args.latent_rank,
        allow_config_override=args.allow_config_override,
        label_column=args.label_column,
        case_value=args.case_value,
        control_value=args.control_value,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
