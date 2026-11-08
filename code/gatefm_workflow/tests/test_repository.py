"""Fast stdlib-only repository review gates; no scientific data or GPU needed."""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = (
    ".doc", ".docx", ".pdf", ".tex", ".vcf", ".gen", ".pt", ".pth",
    ".ckpt", ".npy", ".npz", ".h5", ".hdf5", ".pkl", ".zip", ".7z",
)


def tracked_paths() -> list[pathlib.Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return [
            pathlib.Path(item.decode("utf-8"))
            for item in result.stdout.split(b"\0") if item
        ]
    # A standalone source bundle inside a collaborator's code/ directory
    # is valid before the collaborator commits it to Git.
    excluded = {".git", ".venv", "__pycache__", "outputs", "data", "tools"}
    return sorted(
        path.relative_to(ROOT) for path in ROOT.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.relative_to(ROOT).parts)
    )


class RepositoryReview(unittest.TestCase):
    def test_required_files(self):
        required = (
            "README.md", "requirements.txt", ".gitignore",
            "run_workflow.py", "run_toy_demo.ps1", "generate_paper_data.ps1",
            "code/gatefm_model.py", "code/run_stage1.py",
            "code/aggregate_stage1.py", "code/run_stage2.py",
            "code/posthoc_lower_order_screen.py",
            "code/run_synthetic_benchmark.py",
            "code/evaluate_synthetic_benchmark.py",
            "code/verify_reproduction.py",
            "scripts/plot_toy_figures.py",
            "scripts/plot_cohort_figures.py",
            "scripts/render_manuscript_figures.py",
            "scripts/render_supplement_s1.py",
            "scripts/render_supplement_s2.py",
            "docs/METHOD.md", "docs/CODE_GUIDE.md",
            "docs/DATA_ACCESS.md",
        )
        for name in required:
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file(), name)

    def test_no_sensitive_or_generated_files_tracked(self):
        paths = tracked_paths()
        self.assertGreaterEqual(len(paths), 25)
        for path in paths:
            rel = path.as_posix()
            with self.subTest(file=rel):
                self.assertFalse(rel.startswith("manuscript/"), rel)
                self.assertFalse(rel.lower().endswith(BLOCKED_SUFFIXES), rel)
                self.assertFalse(
                    rel.startswith("data/"), rel
                )
                self.assertFalse(
                    rel.startswith("outputs/"), rel
                )

    def test_protocol_is_explicit(self):
        plan = json.loads((ROOT / "configs/experiment_plan.json").read_text())
        synthetic = plan["synthetic"]
        self.assertEqual(synthetic["dimensions"], [20, 100, 500, 1000, 5000, 10000])
        self.assertEqual(len(synthetic["data_seeds"]), 5)
        self.assertEqual(len(synthetic["model_initialization_seeds"]), 3)
        self.assertEqual(len(plan["human"]["model_initialization_seeds"]), 10)

    def test_docs_are_portable_and_links_exist(self):
        files = [ROOT / "README.md"] + list((ROOT / "docs").glob("*.md"))
        for page in files:
            content = page.read_text(encoding="utf-8-sig")
            self.assertNotRegex(content, r"[A-Z]:\\(?:Users|essay_hope_final|GateFM_)")
            self.assertNotIn("run_smoke_test.ps1", content)
            for href in re.findall(r"\]\(([^)]+)\)", content):
                if href.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                with self.subTest(page=page.name, link=href):
                    target = (page.parent / href.split("#", 1)[0]).resolve()
                    self.assertTrue(target.exists(), str(target))

    def test_public_documentation_is_english(self):
        for page in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
            content = page.read_text(encoding="utf-8-sig")
            self.assertNotRegex(content, r"[\u3400-\u9fff]", str(page))

    def test_figures_are_direct_image_outputs(self):
        figure_scripts = (
            "scripts/render_manuscript_figures.py",
            "scripts/plot_toy_figures.py",
            "scripts/plot_cohort_figures.py",
        )
        for script in figure_scripts:
            with self.subTest(script=script):
                content = (ROOT / script).read_text(encoding="utf-8")
                self.assertNotIn(".html", content)
                self.assertIn("figures", content)
        workflow = (ROOT / "run_workflow.py").read_text(encoding="utf-8")
        self.assertIn("FIGURES", workflow)
        self.assertNotIn("OPEN REPORT", workflow)
    def test_generated_data_and_tool_cache_are_ignored(self):
        for path in (
            "data/authorized_cohort.csv",
            "data/synthetic_dataset_manifest.csv",
            "outputs/test_run/workflow_status.json",
            "tools/GAMETES_2.1.jar",
        ):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "-C", str(ROOT), "check-ignore", "--no-index", "-q", path],
                    check=False,
                    capture_output=True,
                )
                if result.returncode == 128:
                    # GitHub ZIP downloads and pre-commit integration previews
                    # have no .git directory. Verify the bundled ignore rules.
                    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
                    matched = any(
                        path == rule or path.startswith(rule)
                        for rule in rules if rule and not rule.startswith("#")
                    )
                    self.assertTrue(matched, path)
                else:
                    self.assertEqual(result.returncode, 0, path)


if __name__ == "__main__":
    unittest.main()
