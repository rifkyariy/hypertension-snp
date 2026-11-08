"""Synthetic artifact provenance checks without research data or training."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from repro_io import sha256_file
from verify_run_provenance import verify_complete_run


class ProvenanceChecks(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.stage1 = self.root / "stage1"
        self.stage2 = self.root / "stage2"
        self.stage1.mkdir()
        self.stage2.mkdir()
        self.hash = "c" * 64
        self.config = {
            "model": {"latent_rank": 16},
            "training": {"epochs": 2},
            "resolved_n_snps": 3,
        }
        self.write_json(self.stage1 / "data_manifest.json", {
            "data_sha256": self.hash, "feature_order_sha256": "features-abc",
            "n_snps": 3,
        })
        self.write_json(self.stage1 / "config_resolved.json", self.config)
        self.write_json(self.stage1 / "run_summary.json", {"run_seed": 42})
        pd.DataFrame({
            "global_idx": [0, 1, 2],
            "snp_id": ["A", "B", "C"],
            "q_rank": [2, 1, 3],
            "run_seed": [42, 42, 42],
        }).to_csv(self.stage1 / "stage1_snp_ranking.csv", index=False)
        (self.stage1 / "model_checkpoint.pt").write_bytes(b"nonempty")
        pd.DataFrame({
            "snp_a": ["A"], "snp_b": ["B"], "snp_c": ["C"],
            "stage2_rank": [1],
        }).to_csv(self.stage2 / "stage2_fixed_pattern_triplets.csv", index=False)
        self.write_json(self.stage2 / "stage2_summary.json", {
            "data_sha256": self.hash,
            "candidate_file_sha256": sha256_file(
                self.stage1 / "stage1_snp_ranking.csv"
            ),
            "feature_order_sha256": "features-abc",
            "n_candidate_snps": 3, "n_triplets": 1,
        })

    @staticmethod
    def write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def check(self):
        verify_complete_run(self.root, 3, 42, self.hash, self.config)

    def test_consistent_run(self):
        self.check()

    def test_different_resolved_config_rejected(self):
        self.config["training"]["epochs"] = 300
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            self.check()

    def test_different_dataset_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset checksum"):
            verify_complete_run(self.root, 3, 42, "d" * 64, self.config)

    def test_candidate_file_changes_rejected(self):
        with (self.stage1 / "stage1_snp_ranking.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "candidate ranking checksum"):
            self.check()

    def test_incomplete_run_rejected(self):
        (self.stage2 / "stage2_summary.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            self.check()


if __name__ == "__main__":
    unittest.main()