from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_ROOT.parent
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "data"
DEFAULT_DIMENSIONS = (20, 100, 500, 1000, 5000, 10000)
DEFAULT_DATA_SEEDS = (42, 100, 200, 300, 400)
EXPECTED_JAR_SHA256 = "5e38b0993f8915e3e3e4c225ce6b684455d02c6189877d6ae219d24ad69e27d7"


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the paper's pure-and-strict three-locus GAMETES datasets "
            "and a portable benchmark manifest."
        )
    )
    parser.add_argument("--gametes-jar", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--dimensions",
        type=comma_ints,
        default=list(DEFAULT_DIMENSIONS),
        help="Comma-separated dimensions; defaults to the six paper dimensions.",
    )
    parser.add_argument(
        "--data-seeds",
        type=comma_ints,
        default=list(DEFAULT_DATA_SEEDS),
        help="Comma-separated GAMETES seeds; defaults to the five paper seeds.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def paper_parameters(dimension: int, data_seed: int) -> dict[str, object]:
    if dimension not in DEFAULT_DIMENSIONS:
        raise ValueError(
            f"Dimension {dimension} is outside the frozen paper plan: {DEFAULT_DIMENSIONS}"
        )
    if data_seed not in DEFAULT_DATA_SEEDS:
        raise ValueError(
            f"Data seed {data_seed} is outside the frozen paper plan: {DEFAULT_DATA_SEEDS}"
        )
    return {
        "interaction_order": 3,
        "n_observations": 12000,
        "n_cases": 6000,
        "n_controls": 6000,
        "n_snps": dimension,
        "heritability": 0.10,
        "penetrance_model_prevalence": 0.50,
        "causal_locus_maf": 0.20,
        "background_maf_min": 0.01,
        "background_maf_max": 0.50 if dimension == 20 else 0.10,
        "edm_index": 1,
        "replicates": 1,
        "data_seed": data_seed,
    }


def gametes_command(java: str, jar: Path, params: dict[str, object]) -> list[str]:
    model_arg = (
        f"-h {float(params['heritability']):.6f} "
        f"-p {float(params['penetrance_model_prevalence']):.6f} "
        + " ".join(
            f"-a {float(params['causal_locus_maf']):.6f}"
            for _ in range(int(params["interaction_order"]))
        )
        + " -o model"
    )
    dataset_arg = (
        f"-n {float(params['background_maf_min']):.6f} "
        f"-x {float(params['background_maf_max']):.6f} "
        f"-a {int(params['n_snps'])} "
        f"-s {int(params['n_cases'])} "
        f"-w {int(params['n_controls'])} "
        f"-p {float(params['penetrance_model_prevalence']):.6f} "
        f"-r {int(params['replicates'])} "
        "-o dataset"
    )
    return [
        java,
        "-jar",
        str(jar),
        "-M",
        model_arg,
        "-D",
        dataset_arg,
        "-r",
        str(int(params["data_seed"])),
    ]


def convert_dataset(source: Path, destination: Path) -> tuple[list[str], int, dict[str, int]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    class_counts = {"0": 0, "1": 0}
    with source.open("r", newline="", encoding="utf-8-sig") as src, destination.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst, lineterminator="\n")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"GAMETES produced an empty file: {source}") from exc
        writer.writerow(header)
        expected_width = len(header)
        for row in reader:
            if len(row) != expected_width:
                raise RuntimeError(
                    f"Inconsistent GAMETES row width in {source}: "
                    f"expected {expected_width}, found {len(row)}"
                )
            writer.writerow(row)
            row_count += 1
            label = row[-1]
            class_counts[label] = class_counts.get(label, 0) + 1
    return header, row_count, class_counts


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copy2(source, destination)


def generate_one(
    java: str,
    jar: Path,
    jar_hash: str,
    data_root: Path,
    dimension: int,
    data_seed: int,
) -> Path:
    params = paper_parameters(dimension, data_seed)
    dataset_dir = data_root / "synthetic" / f"p{dimension}" / f"seed{data_seed}"
    dataset_path = dataset_dir / "dataset.csv"
    metadata_path = dataset_dir / "generation_metadata.json"
    if dataset_dir.exists():
        raise FileExistsError(
            f"Dataset directory already exists and was not skipped: {dataset_dir}"
        )

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"gametes_p{dimension}_seed{data_seed}_", dir=dataset_dir.parent
    ) as temp_name:
        temp_dir = Path(temp_name)
        command = gametes_command(java, jar, params)
        printable = command.copy()
        printable[0] = "java"
        printable[2] = "GAMETES_2.1.jar"
        print(f"[GENERATE] P={dimension} data_seed={data_seed}", flush=True)
        completed = subprocess.run(
            command,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = "\n".join(
                text.strip()
                for text in (completed.stdout or "", completed.stderr or "")
                if text.strip()
            )
            raise RuntimeError(
                f"GAMETES failed for P={dimension}, seed={data_seed}.\n{details}"
            )

        edm_index = int(params["edm_index"])
        source_dataset = (
            temp_dir
            / f"dataset_EDM-{edm_index}"
            / f"dataset_EDM-{edm_index}_1.txt"
        )
        if not source_dataset.is_file():
            raise FileNotFoundError(
                f"Expected GAMETES EDM-{edm_index} output was not created: {source_dataset}"
            )

        dataset_dir.mkdir(parents=False)
        header, row_count, class_counts = convert_dataset(source_dataset, dataset_path)
        feature_names = header[:-1]
        gt_names = [name for name in feature_names if name.startswith("M0P")]
        if len(feature_names) != dimension:
            raise RuntimeError(
                f"Expected {dimension} SNP columns, found {len(feature_names)}"
            )
        if len(gt_names) != 3:
            raise RuntimeError(f"Expected 3 generating SNPs, found {gt_names}")
        if row_count != int(params["n_observations"]):
            raise RuntimeError(f"Expected 12000 rows, found {row_count}")
        if class_counts.get("0", 0) != 6000 or class_counts.get("1", 0) != 6000:
            raise RuntimeError(f"Unexpected class counts: {class_counts}")

        copy_if_present(temp_dir / "model_Models.txt", dataset_dir / "model_Models.txt")
        copy_if_present(
            temp_dir / "model_EDM_Scores.txt", dataset_dir / "model_EDM_Scores.txt"
        )
        metadata = {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "GAMETES 2.1",
            "generator_source": "https://sourceforge.net/projects/gametes/files/GAMETES_2.1.jar/download",
            "generator_jar_sha256": jar_hash,
            "parameters": params,
            "selected_dataset": "EDM-1 replicate 1",
            "generating_snp_ids": gt_names,
            "outcome_column": header[-1],
            "observed_rows": row_count,
            "observed_class_counts": class_counts,
            "dataset_sha256": sha256_file(dataset_path),
            "gametes_command": printable,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return metadata_path


def existing_is_complete(data_root: Path, dimension: int, data_seed: int) -> bool:
    dataset_dir = data_root / "synthetic" / f"p{dimension}" / f"seed{data_seed}"
    dataset_path = dataset_dir / "dataset.csv"
    metadata_path = dataset_dir / "generation_metadata.json"
    if not dataset_dir.exists():
        return False
    if not dataset_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Incomplete existing dataset directory: {dataset_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != sha256_file(dataset_path):
        raise RuntimeError(f"Dataset checksum does not match metadata: {dataset_path}")
    return True


def rebuild_manifest(data_root: Path) -> Path:
    rows: list[dict[str, object]] = []
    for metadata_path in sorted((data_root / "synthetic").glob("p*/seed*/generation_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        params = metadata["parameters"]
        gt = metadata["generating_snp_ids"]
        dataset_path = metadata_path.parent / "dataset.csv"
        rows.append(
            {
                "dimension": int(params["n_snps"]),
                "data_seed": int(params["data_seed"]),
                "data_path": dataset_path.relative_to(data_root).as_posix(),
                "gt_snp_a": gt[0],
                "gt_snp_b": gt[1],
                "gt_snp_c": gt[2],
                "dataset_sha256": metadata["dataset_sha256"],
                "edm_index": int(params["edm_index"]),
            }
        )
    rows.sort(key=lambda row: (int(row["dimension"]), int(row["data_seed"])))
    manifest_path = data_root / "synthetic_dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dimension",
            "data_seed",
            "data_path",
            "gt_snp_a",
            "gt_snp_b",
            "gt_snp_c",
            "dataset_sha256",
            "edm_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    args = parse_args()
    dimensions = list(dict.fromkeys(args.dimensions))
    data_seeds = list(dict.fromkeys(args.data_seeds))
    for dimension in dimensions:
        for data_seed in data_seeds:
            paper_parameters(dimension, data_seed)

    jar = args.gametes_jar.resolve()
    if not jar.is_file():
        raise FileNotFoundError(jar)
    jar_hash = sha256_file(jar)
    if jar_hash.lower() != EXPECTED_JAR_SHA256:
        raise RuntimeError(
            "GAMETES jar checksum mismatch. "
            f"Expected {EXPECTED_JAR_SHA256}, found {jar_hash}."
        )
    java = shutil.which("java")
    if not java:
        raise RuntimeError("Java was not found on PATH. GAMETES requires a Java runtime.")

    data_root = args.output_root.resolve()
    if args.dry_run:
        for dimension in dimensions:
            for data_seed in data_seeds:
                params = paper_parameters(dimension, data_seed)
                command = gametes_command(java, jar, params)
                command[2] = "GAMETES_2.1.jar"
                print(
                    json.dumps(
                        {"parameters": params, "command": command}, ensure_ascii=False
                    )
                )
        print("[PASS] GAMETES paper-data configuration is valid; no data were written.")
        return

    data_root.mkdir(parents=True, exist_ok=True)
    for dimension in dimensions:
        for data_seed in data_seeds:
            if args.skip_existing and existing_is_complete(data_root, dimension, data_seed):
                print(f"[SKIP] P={dimension} data_seed={data_seed}", flush=True)
                continue
            generate_one(java, jar, jar_hash, data_root, dimension, data_seed)
    manifest_path = rebuild_manifest(data_root)
    print(f"[PASS] Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
