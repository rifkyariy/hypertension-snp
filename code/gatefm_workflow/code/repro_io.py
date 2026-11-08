from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from decimal import Decimal, InvalidOperation
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch


LABEL_CANDIDATES = (
    "class",
    "label",
    "target",
    "y",
    "phenotype",
    "case_control",
)
IDENTIFIER_COLUMNS = {
    "id",
    "iid",
    "fid",
    "sample_id",
    "subject_id",
    "patient_id",
    "eid",
}


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".txt"}:
        raise ValueError(
            "Genotype input must be a pre-QC CSV or TSV table. "
            "Convert VCF/GEN/Parquet upstream in an authorized environment."
        )
    delimiter = "," if suffix == ".csv" else "\t"
    # pandas would silently rename duplicate columns (e.g. rsID to rsID.1).
    # Detect those before loading participant-level data.
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream, delimiter=delimiter), [])
    normalized = [str(name).strip().casefold() for name in header]
    if not header or not all(normalized):
        raise ValueError("Input table must have a nonempty header on every column.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate column names in genotype input are forbidden.")
    return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")


def load_genotype_dataset(
    path: Path,
    *,
    label_column: str | None = None,
    case_value: str | None = None,
    control_value: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, list[str], str]:
    """Load pre-QC genotype table without guessing phenotype direction.

    If values are not exactly 0=control and 1=case, callers MUST specify
    both --case-value and --control-value explicitly. Never infer polarity.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if (case_value is None) != (control_value is None):
        raise ValueError("case_value and control_value must be provided together.")

    df = _read_table(path)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if df.empty:
        raise ValueError("Input genotype table contains no samples.")
    matches = (
        [c for c in df.columns if c.casefold() == label_column.strip().casefold()]
        if label_column is not None
        else [c for c in df.columns if c.casefold() in LABEL_CANDIDATES]
    )
    if len(matches) != 1:
        raise ValueError(
            "Exactly one phenotype column is required; pass --label-column "
            "to select it explicitly. Never assume the last SNP is a label."
        )
    label_name = matches[0]
    labels = df[label_name]
    if labels.isna().any():
        raise ValueError("Outcome contains missing values.")
    if case_value is None:
        numeric = pd.to_numeric(labels, errors="raise")
        unique_y = set(np.unique(numeric).tolist())
        if unique_y != {0, 1}:
            raise ValueError(
                "Outcome must be 0=control, 1=case; for a different coding "
                "pass both --case-value and --control-value explicitly."
            )
        y = numeric.to_numpy(dtype=np.int64)
    else:
        def token(value: object) -> str:
            value_text = str(value).strip().casefold()
            try:
                numeric = Decimal(value_text)
            except InvalidOperation:
                return value_text
            return str(numeric.normalize()) if numeric.is_finite() else value_text

        case_token = token(case_value)
        control_token = token(control_value)
        if not case_token or not control_token or case_token == control_token:
            raise ValueError("Explicit case and control values must be different.")
        observed = labels.map(token)
        allowed = {case_token, control_token}
        if not set(observed.unique()).issubset(allowed):
            raise ValueError("Outcome contains values outside explicit case/control mapping.")
        if set(observed.unique()) != allowed:
            raise ValueError("Both explicit case and control classes must appear.")
        y = (observed == case_token).to_numpy(dtype=np.int64)

    features = df.drop(columns=[label_name])
    identifier_columns = [
        c for c in features.columns if c.strip().casefold() in IDENTIFIER_COLUMNS
    ]
    features = features.drop(columns=identifier_columns)
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            features[column] = pd.to_numeric(features[column], errors="raise")
    if features.isna().any().any():
        raise ValueError("Missing genotype values must be resolved upstream by QC.")
    genotype_values = np.unique(features.to_numpy())
    if not set(genotype_values.tolist()).issubset({0, 1, 2}):
        raise ValueError(
            "Genotype values must be complete integer 0/1/2 dosages; "
            "fractional, missing and imputation probabilities are unsupported."
        )
    X = np.ascontiguousarray(features.to_numpy(dtype=np.int16, copy=False))
    return X, y, [str(c) for c in features.columns], str(label_name)

def deep_update(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def resolve_config(config: dict, n_snps: int) -> dict:
    resolved = deepcopy(config)
    overrides = resolved.pop("dimension_overrides", {})
    override = overrides.get(str(int(n_snps)))
    if override is not None:
        resolved = deep_update(resolved, override)
    resolved["resolved_n_snps"] = int(n_snps)
    return resolved


def feature_order_sha256(feature_names: Iterable[str]) -> str:
    body = "\n".join(str(x) for x in feature_names).encode("utf-8")
    return hashlib.sha256(body).hexdigest()
