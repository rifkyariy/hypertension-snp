"""Fail-closed cohort input and CLI regression tests. Synthetic data only."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))
from repro_io import load_genotype_dataset
from run_stage2 import compute_stage2_results
import run_workflow


class CohortInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_obj.cleanup)
        self.root = Path(self.tmp_obj.name)

    def write(self, frame: pd.DataFrame, name: str = "cohort.csv") -> Path:
        path = self.root / name
        frame.to_csv(path, index=False)
        return path

    def test_missing_label_is_not_last_column(self):
        path = self.write(pd.DataFrame({
            "A": [0, 1, 0, 1], "B": [1, 0, 1, 0],
            "C": [0, 0, 1, 1], "D": [1, 1, 0, 0],
        }))
        with self.assertRaisesRegex(ValueError, "phenotype column"):
            load_genotype_dataset(path)

    def test_nonstandard_outcomes_require_explicit_mapping(self):
        path = self.write(pd.DataFrame({
            "A": [0, 1, 2, 0], "B": [1, 1, 2, 0],
            "C": [2, 2, 1, 0], "case_control": [1, 2, 1, 2],
        }))
        with self.assertRaisesRegex(ValueError, "case-value"):
            load_genotype_dataset(path)
        X, y, names, label = load_genotype_dataset(
            path, label_column="case_control", case_value="1", control_value="2"
        )
        self.assertEqual(y.tolist(), [1, 0, 1, 0])
        self.assertEqual(X.shape, (4, 3))
        self.assertEqual(label, "case_control")

    def test_decimal_numeric_outcome_mapping(self):
        path = self.write(pd.DataFrame({
            "A": [0, 1, 2, 0], "B": [1, 1, 2, 0],
            "C": [2, 2, 1, 0],
            "phenotype": ["1.0", "2.0", "1.0", "2.0"],
        }))
        _, y, _, _ = load_genotype_dataset(
            path, case_value="1", control_value="2"
        )
        self.assertEqual(y.tolist(), [1, 0, 1, 0])
    def test_string_outcome_mapping(self):
        path = self.write(pd.DataFrame({
            "A": [0, 1, 2, 0], "B": [1, 1, 2, 0],
            "C": [2, 2, 1, 0], "phenotype": ["Case", "Control", "Case", "Control"],
        }))
        X, y, _, _ = load_genotype_dataset(
            path, case_value="case", control_value="control"
        )
        self.assertEqual(y.tolist(), [1, 0, 1, 0])

    def test_duplicate_column_names_are_rejected(self):
        path = self.root / "duplicate.csv"
        path.write_text("SNP,SNP,B,class\n0,1,2,0\n1,2,0,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate column"):
            load_genotype_dataset(path)

    def test_tab_separated_cohort_input(self):
        path = self.root / "cohort.tsv"
        pd.DataFrame({
            "A": [0, 1, 2, 0], "B": [1, 1, 2, 0],
            "C": [2, 2, 1, 0], "Class": [0, 1, 0, 1],
        }).to_csv(path, sep="\t", index=False)
        X, y, names, label = load_genotype_dataset(path)
        self.assertEqual(X.shape, (4, 3))
        self.assertEqual(y.tolist(), [0, 1, 0, 1])
        self.assertEqual(names, ["A", "B", "C"])
        self.assertEqual(label, "Class")
    def test_standard_case_control_and_id(self):
        path = self.write(pd.DataFrame({
            "sample_id": ["anon1", "anon2", "anon3", "anon4"],
            "A": [0, 1, 2, 0], "B": [1, 1, 2, 0],
            "C": [2, 2, 1, 0], "class": [1, 0, 1, 0],
        }))
        X, y, names, _ = load_genotype_dataset(path)
        self.assertEqual(names, ["A", "B", "C"])
        self.assertEqual(X.shape, (4, 3))
        self.assertEqual(y.tolist(), [1, 0, 1, 0])

    def test_stage2_rejects_duplicate_snp(self):
        X = np.asarray([
            [0, 1, 2], [1, 1, 1], [2, 2, 0], [0, 0, 1],
            [1, 0, 0], [2, 1, 2], [0, 0, 2], [1, 2, 1],
        ])
        y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        candidate = pd.DataFrame({
            "global_idx": [0, 0, 1], "snp_id": ["A", "A", "B"],
            "representative_genotype": [0, 1, 1],
            "consensus_rank": [1, 2, 3],
        })
        with self.assertRaisesRegex(ValueError, "distinct SNP"):
            compute_stage2_results(X, y, ["A", "B", "C"], candidate)

    def test_human_cli_passes_mapping_to_every_data_consumer(self):
        features = {f"rs{index}": [index % 3, (index + 1) % 3] * 2
                    for index in range(30)}
        features["case_control"] = [1, 2, 1, 2]
        path = self.write(pd.DataFrame(features))
        args = SimpleNamespace(
            mode="human", data=path, output=self.root / "out",
            config=ROOT / "configs/config_thesis.json", device="cpu",
            label_column="case_control", case_value="1", control_value="2",
            expected_samples=4, expected_snps=30,
            dimensions=None, data_seeds=None, model_seeds=None, max_runs=None,
            epochs=None, batch_size=None, latent_rank=None,
            manifest=None, comparators=None,
        )
        jobs = run_workflow.commands(args)
        self.assertEqual(len(jobs), 17)
        self.assertIn("validate_cohort_input.py", str(jobs[2][1]))
        for job in jobs:
            if Path(job[1]).name in {
                "run_stage1.py", "run_stage2.py", "posthoc_lower_order_screen.py",
                "validate_cohort_input.py",
            }:
                self.assertEqual(job[job.index("--case-value") + 1], "1")
                self.assertEqual(job[job.index("--control-value") + 1], "2")
                self.assertEqual(job[job.index("--label-column") + 1], "case_control")
        self.assertEqual(jobs[2][jobs[2].index("--expected-snps") + 1], "30")


if __name__ == "__main__":
    unittest.main()