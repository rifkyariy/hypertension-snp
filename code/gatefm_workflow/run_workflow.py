"""GateFM workflow entry point. Plans by default; --execute starts work."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"


def commands(args):
    out = args.output.resolve()
    plan = ROOT / "configs" / "experiment_plan.json"
    cfg = args.config.resolve()
    def step(name, *items):
        return [sys.executable, str(CODE / name), *map(str, items)]
    def plot(name, *items):
        return [sys.executable, str(ROOT / "scripts" / name), *map(str, items)]
    jobs = [step("validate_contract.py"), step("verify_reproduction.py")]
    if args.mode == "synthetic":
        if args.manifest is None:
            raise ValueError("synthetic requires --manifest")
        manifest = args.manifest.resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        run_job = step("run_synthetic_benchmark.py", "--manifest", manifest,
                       "--output-root", out / "runs", "--config", cfg,
                       "--experiment-plan", plan, "--device", args.device)
        subset_requested = any(getattr(args, key) is not None
                               for key in ("dimensions", "data_seeds", "model_seeds", "max_runs"))
        overridden = any(getattr(args, key) is not None
                         for key in ("epochs", "batch_size", "latent_rank"))
        # Preserve the frozen 90-fit plan by default; allow safe, explicitly
        # labeled exploratory runs without changing config_thesis.json.
        for flag, value in (("dimensions", args.dimensions),
                            ("data-seeds", args.data_seeds),
                            ("model-seeds", args.model_seeds),
                            ("max-runs", args.max_runs),
                            ("epochs", args.epochs),
                            ("batch-size", args.batch_size),
                            ("latent-rank", args.latent_rank)):
            if value is not None:
                run_job.extend(("--" + flag, str(value)))
        if subset_requested:
            run_job.append("--allow-incomplete-plan")
        if overridden:
            run_job.append("--allow-config-override")
        eval_job = step("evaluate_synthetic_benchmark.py", "--manifest", manifest,
                        "--runs-root", out / "runs", "--output", out / "evaluation",
                        "--experiment-plan", plan, "--config", cfg)
        for flag, value in (
            ("epochs", args.epochs),
            ("batch-size", args.batch_size),
            ("latent-rank", args.latent_rank),
        ):
            if value is not None:
                eval_job.extend(("--" + flag, str(value)))
        if subset_requested:
            eval_job.append("--allow-incomplete")
        if overridden or cfg != (ROOT / "configs/config_thesis.json").resolve():
            eval_job.append("--non-reference")
        jobs.extend((run_job, eval_job))
        plot_command = plot("render_manuscript_figures.py", "benchmark",
                            "--evaluation", out / "evaluation", "--output", out / "figures")
        if subset_requested or overridden or cfg != (ROOT / "configs/config_thesis.json").resolve():
            plot_command.append("--allow-incomplete")
        if args.comparators is not None:
            if not args.comparators.is_file():
                raise FileNotFoundError(args.comparators)
            plot_command += ["--comparators", str(args.comparators.resolve())]
        jobs.append(plot_command)
        return jobs
    if args.mode == "smoke":
        data = out / "toy.csv"
        seeds = [42, 100, 200]
        jobs += [[sys.executable, str(ROOT / "examples" / "make_toy_dataset.py"),
                  "--output", str(data)]]
    else:
        if args.data is None or not args.data.is_file():
            raise ValueError("human requires --data pointing to an authorized genotype CSV")
        data = args.data.resolve()
        seeds = json.loads(plan.read_text(encoding="utf-8"))["human"]["model_initialization_seeds"]
        cohort_options = []
        for flag, value in (
            ("label-column", args.label_column),
            ("case-value", args.case_value),
            ("control-value", args.control_value),
        ):
            if value is not None:
                cohort_options.extend(("--" + flag, str(value)))
        preflight = step("validate_cohort_input.py", "--data", data, *cohort_options)
        if args.expected_samples is not None:
            preflight.extend(("--expected-samples", str(args.expected_samples)))
        if args.expected_snps is not None:
            preflight.extend(("--expected-snps", str(args.expected_snps)))
        jobs.append(preflight)
    for i, seed in enumerate(seeds, 1):
        job = step("run_stage1.py", "--data", data, "--output", out / "runs" / f"run_{i:03d}",
                   "--config", cfg, "--seed", seed, "--device", "cpu" if args.mode == "smoke" else args.device)
        if args.mode == "smoke":
            job += ["--epochs", "2", "--batch-size", "128", "--latent-rank", "8", "--allow-config-override"]
        if args.mode == "human":
            job.extend(cohort_options)
        jobs.append(job)
    consensus = out / "consensus" / "stage1_consensus_top30.csv"
    # Route the same approved phenotype encoding to every consumer.
    if args.mode == "human":
        stage2_flags = cohort_options
    else:
        stage2_flags = []
    jobs += [step("aggregate_stage1.py", "--runs-root", out / "runs", "--output", out / "consensus",
                  "--top-k", 30, "--expected-runs", len(seeds), "--require-run-manifests"),
             step("run_stage2.py", "--data", data, "--consensus-topk", consensus,
                  "--output", out / "stage2", "--top-k", 30, *stage2_flags),
             step("posthoc_lower_order_screen.py", "--data", data, "--consensus-topk", consensus,
                  "--stage2-results", out / "stage2" / "stage2_fixed_pattern_triplets.csv",
                  "--output", out / "lower_order", "--triplet-adjusted-threshold", "0.0002",
                  "--lower-order-adjusted-threshold", "0.0002", *stage2_flags)]
    if args.mode == "smoke":
        jobs.append(plot("plot_toy_figures.py", "--run", out))
    else:
        jobs.append(plot("plot_cohort_figures.py", "--run", out))
    return jobs


def stage_label(job):
    """Readable progress and log section names without altering command execution."""
    name = Path(job[1]).stem
    names = {
        "validate_contract": "Validate experiment configuration",
        "validate_cohort_input": "Preflight authorized cohort encoding and dimensions",
        "verify_reproduction": "Verify numerical implementation",
        "make_toy_dataset": "Generate smoke-test data",
        "run_synthetic_benchmark": "Train and analyze GAMETES datasets",
        "evaluate_synthetic_benchmark": "Summarize GAMETES metrics",
        "run_stage1": "Train GateFM Stage 1",
        "aggregate_stage1": "Aggregate SNP rankings",
        "run_stage2": "Evaluate SNP triplets (Stage 2)",
        "posthoc_lower_order_screen": "Screen single-SNP and pair effects",
        "plot_toy_figures": "Export smoke-test figures",
        "plot_cohort_figures": "Export cohort figures",
        "render_manuscript_figures": "Export benchmark figures",
    }
    title = names.get(name, name)
    if name == "run_stage1" and "--seed" in job:
        return f"{title} (seed {job[job.index('--seed') + 1]})"
    return title


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "synthetic", "human"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--label-column", help="Exact phenotype column; recommended for real cohorts.")
    parser.add_argument("--case-value", help="Original case value if not canonical 1.")
    parser.add_argument("--control-value", help="Original control value if not canonical 0.")
    parser.add_argument("--expected-samples", type=int, help="Optional preflight sample-count assertion.")
    parser.add_argument("--expected-snps", type=int, help="Optional preflight SNP-count assertion.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--comparators", type=Path,
                        help="Optional independent method metrics for complete benchmark figures")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config_thesis.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dimensions", help="GAMETES SNP dimensions, e.g. 20,100")
    parser.add_argument("--data-seeds", help="GAMETES dataset seeds, e.g. 42,100")
    parser.add_argument("--model-seeds", help="Model initialization seeds from experiment_plan.json")
    parser.add_argument("--max-runs", type=int, help="Maximum GAMETES model fits; partial results are marked preview")
    parser.add_argument("--epochs", type=int, help="Override training epochs (non-reference experiment)")
    parser.add_argument("--batch-size", type=int, help="Override training batch size (non-reference experiment)")
    parser.add_argument("--latent-rank", type=int, help="Override GateFM embedding dimension (non-reference experiment)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.mode != "human" and any(
        value is not None for value in (
            args.label_column, args.case_value, args.control_value,
            args.expected_samples, args.expected_snps,
        )
    ):
        parser.error("Cohort input mapping and expected counts require human mode.")
    if (args.case_value is None) != (args.control_value is None):
        parser.error("Both --case-value and --control-value are required together.")
    if any(value is not None and value < 1 for value in (
        args.expected_samples, args.expected_snps
    )):
        parser.error("Expected cohort dimensions must be positive.")
    if args.comparators is not None and args.mode != "synthetic":
        parser.error("--comparators is available only in synthetic mode")
    synthetic_options = ("dimensions", "data_seeds", "model_seeds", "max_runs",
                         "epochs", "batch_size", "latent_rank")
    if args.mode != "synthetic" and any(getattr(args, opt) is not None
                                       for opt in synthetic_options):
        parser.error("GAMETES subset and training override options require synthetic mode")
    for opt in ("max_runs", "epochs", "batch_size", "latent_rank"):
        value = getattr(args, opt)
        if value is not None and value < 1:
            parser.error(f"--{opt.replace('_', '-')} must be a positive integer")
    for opt in ("dimensions", "data_seeds", "model_seeds"):
        value = getattr(args, opt)
        if value is not None:
            try:
                ids = [int(v.strip()) for v in value.split(",")]
            except ValueError:
                parser.error(f"--{opt.replace('_', '-')} expects comma-separated integers")
            if not ids or any(v < 1 for v in ids):
                parser.error(f"--{opt.replace('_', '-')} expects positive integers")
    if args.mode == "synthetic" and args.comparators is not None:
        custom = any(getattr(args, opt) is not None for opt in synthetic_options)
        custom |= args.config.resolve() != (ROOT / "configs/config_thesis.json").resolve()
        if custom:
            parser.error("--comparators requires the unmodified full formal experiment")
    jobs = commands(args)
    if not args.execute:
        for i, job in enumerate(jobs, start=1):
            print(f"{i:02d}. {stage_label(job)}", flush=True)
            print("    " + subprocess.list2cmdline(job), flush=True)
    if not args.execute:
        print("PLAN ONLY. Add --execute to run. Default synthetic mode is the frozen 90-fit plan.")
        return
    if args.mode == "synthetic":
        custom = any(getattr(args, opt) is not None for opt in synthetic_options)
        custom |= args.config.resolve() != (ROOT / "configs/config_thesis.json").resolve()
        if custom:
            print("EXPLORATORY GAMETES RUN: selected subset or configured training options.",
                  flush=True)
    if args.output.exists():
        raise FileExistsError("Use a new output directory; existing results are never overwritten.")
    args.output.mkdir(parents=True)
    run_log = args.output / "run.log"
    record = {"mode": args.mode, "status": "running", "commands": jobs,
              "log": str(run_log.resolve()),
              "figures": str((args.output / "figures").resolve())}
    status_file = args.output / "workflow_status.json"
    def save():
        status_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    save()
    try:
        with run_log.open("w", encoding="utf-8") as stream:
            for i, job in enumerate(jobs, start=1):
                name = stage_label(job)
                record["current_step"] = i
                record["current_stage"] = name
                if i == len(jobs):
                    record["status"] = "analysis_complete"
                save()
                stream.write(f"=== {i}/{len(jobs)}: {name} ===\n")
                stream.write("Command: " + subprocess.list2cmdline(job) + "\n")
                stream.flush()
                print(f"[{i}/{len(jobs)}] {name}", flush=True)
                try:
                    subprocess.run(job, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                except BaseException as exc:
                    stream.write(f"FAILED: {name}: {exc}\n")
                    stream.flush()
                    raise
                stream.write(f"PASS: {name}\n\n")
                stream.flush()
        record["status"] = "completed"
        print(f"FIGURES {record['figures']}", flush=True)
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        print(f"FAILED: {record.get('current_stage', 'Workflow')} - see {run_log}",
              file=sys.stderr, flush=True)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
