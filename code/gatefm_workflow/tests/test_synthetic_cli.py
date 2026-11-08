"""Lightweight tests for the single-command GAMETES CLI; no datasets or torch needed."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_workflow


class GametesCommandTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.tmp = Path(self.workspace.name)
        self.manifest = self.tmp / "manifest.csv"
        self.manifest.write_text("dimension,data_seed,data_path,gt_snp_a,gt_snp_b,gt_snp_c\n",
                                 encoding="utf-8")
        self.args = SimpleNamespace(
            mode="synthetic", output=self.tmp/"output", config=ROOT/"configs/config_thesis.json",
            manifest=self.manifest, comparators=None, device="cpu",
            dimensions=None, data_seeds=None, model_seeds=None, max_runs=None,
            epochs=None, batch_size=None, latent_rank=None, data=None,
        )

    def test_workflow_uses_one_named_log(self):
        source = (ROOT / "run_workflow.py").read_text(encoding="utf-8")
        self.assertIn('run_log = args.output / "run.log"', source)
        self.assertNotIn('f"step_{', source)
        self.assertEqual(
            run_workflow.stage_label(
                [sys.executable, str(ROOT / "code" / "run_stage1.py"),
                 "--seed", "42"]),
            "Train GateFM Stage 1 (seed 42)")
        self.assertEqual(
            run_workflow.stage_label(
                [sys.executable, str(ROOT / "scripts" / "render_manuscript_figures.py")]),
            "Export benchmark figures")
    def test_default_keeps_formal_plan_and_figure_names(self):
        jobs = run_workflow.commands(self.args)
        self.assertEqual(len(jobs), 5)
        self.assertNotIn("--allow-incomplete-plan", jobs[2])
        self.assertNotIn("--allow-config-override", jobs[2])
        self.assertNotIn("--allow-incomplete", jobs[3])
        self.assertNotIn("--non-reference", jobs[3])
        self.assertNotIn("--allow-incomplete", jobs[4])
        self.assertEqual(jobs[4][-2:], ["--output", str(self.args.output.resolve()/"figures")])

    def test_subset_and_training_options_are_forwarded(self):
        self.args.dimensions = "20"
        self.args.data_seeds = "42"
        self.args.model_seeds = "3484015997"
        self.args.max_runs = 1
        self.args.epochs = 1
        self.args.batch_size = 64
        self.args.latent_rank = 16
        jobs = run_workflow.commands(self.args)
        for key, val in (("--dimensions", "20"), ("--data-seeds", "42"),
                         ("--model-seeds", "3484015997"), ("--max-runs", "1"),
                         ("--epochs", "1"), ("--batch-size", "64"),
                         ("--latent-rank", "16")):
            self.assertIn(key, jobs[2])
            self.assertEqual(jobs[2][jobs[2].index(key)+1], val)
        self.assertIn("--allow-incomplete-plan", jobs[2])
        self.assertIn("--allow-config-override", jobs[2])
        self.assertIn("--allow-incomplete", jobs[3])
        self.assertIn("--non-reference", jobs[3])
        self.assertIn("--allow-incomplete", jobs[4])

    def test_full_custom_training_is_still_nonreference(self):
        self.args.epochs = 10
        jobs = run_workflow.commands(self.args)
        self.assertNotIn("--allow-incomplete-plan", jobs[2])
        self.assertIn("--allow-config-override", jobs[2])
        self.assertIn("--non-reference", jobs[3])
        self.assertIn("--allow-incomplete", jobs[4])

    def test_invalid_and_misplaced_parameters_fail_before_creating_outputs(self):
        for arguments in (
            ["synthetic", "--max-runs", "0"],
            ["synthetic", "--epochs", "0"],
            ["synthetic", "--dimensions", "invalid"],
            ["human", "--epochs", "1"],
        ):
            with self.subTest(arguments=arguments):
                process = subprocess.run(
                    [sys.executable, str(ROOT/"run_workflow.py"), *arguments,
                     "--output", str(self.tmp/"invalid")],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse((self.tmp/"invalid").exists())

    def test_comparators_rejected_for_exploratory_runs(self):
        baseline = self.tmp/"comparators.csv"
        baseline.write_text("dimension,method,metric,value,is_estimated\n", encoding="utf-8")
        process = subprocess.run(
            [sys.executable, str(ROOT/"run_workflow.py"), "synthetic",
             "--manifest", str(self.manifest),
             "--output", str(self.tmp/"invalid"),
             "--dimensions", "20", "--comparators", str(baseline), "--execute"],
            capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("unmodified full formal experiment", process.stderr)
        self.assertFalse((self.tmp/"invalid").exists())


if __name__ == "__main__":
    unittest.main()