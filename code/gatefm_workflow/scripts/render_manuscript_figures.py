"""Publication-style GateFM plots. Numerical values come exclusively from input CSVs.

Modes:
  reference: table1.csv, table2.csv and table3.csv from manuscript-compatible exports
  benchmark: completed 90-fit evaluation (or explicitly labeled exploratory subset)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMS = (20, 100, 500, 1000, 5000, 10000)
KS = (5, 10, 15, 20, 30, 50, 100)
STAGE1 = ("GateFM", "Distributed Transformer", "ReliefF", "MultiSURF")
STAGE2 = ("GateFM", "BitEpi", "MDR")
COLORS = {"GateFM": "#E74C3C", "Distributed Transformer": "#637b98",
          "ReliefF": "#d5a04c", "MultiSURF": "#8474a3",
          "BitEpi": "#4E79A7", "MDR": "#59A14F"}


def table(path, cols):
    data = pd.read_csv(path)
    missing = set(cols).difference(data)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if data.dimension.duplicated().any() or tuple(sorted(data.dimension.astype(int))) != DIMS:
        raise ValueError(f"{path.name}: need exactly six rows at P={DIMS}")
    return data.sort_values("dimension").reset_index(drop=True)


def numeric(seq, name, low=0, high=float("inf")):
    a = pd.to_numeric(seq, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(a).all() or np.any(a < low) or np.any(a > high):
        raise ValueError(f"{name}: invalid numeric range")
    return a


def boolean(value):
    text = str(value).strip().lower()
    if text in ("true", "1", "estimated"):
        return True
    if text in ("false", "0", "measured"):
        return False
    raise ValueError(f"Invalid measured/estimated marker: {value!r}")


def save(fig, folder, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_rank(t, folder):
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, name in enumerate(STAGE1):
        ax.bar(x + (i - 1.5) * 0.195, numeric(t[name], name), width=0.185,
               color=COLORS[name], label=name)
    ax.set(xticks=x, xticklabels=[f"{p:,}" for p in DIMS], xlabel="Number of SNPs (P)",
           ylabel="Mean rank of the three generating SNPs",
           title="Stage 1 mean generating-SNP rank")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    ax.yaxis.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, folder, "table1_mean_rank")


def plot_hit(t, folder):
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, name in enumerate(STAGE2):
        series = pd.to_numeric(t[f"{name}_hit_at_1"].replace("NE", np.nan), errors="coerce").to_numpy(dtype=float)
        if np.any((series[np.isfinite(series)] < 0) | (series[np.isfinite(series)] > 1)):
            raise ValueError(f"Invalid Hit@1 for {name}")
        ax.bar(x + (i-1)*0.24, series, 0.225, color=COLORS[name], label=name)
        for j, val in enumerate(series):
            if np.isnan(val):
                ax.text(x[j]+(i-1)*0.24, 0.04, "NE", fontsize=8, ha="center")
    ax.set(xticks=x, xticklabels=[f"{p:,}" for p in DIMS], xlabel="Number of SNPs (P)",
           ylabel="Hit@1", ylim=(0, 1.08), title="Stage 2 Hit@1 at Top-30")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.yaxis.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, folder, "table2_hit_at_1")


def plot_runtime(t, folder):
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    all_values = []
    for i, name in enumerate(STAGE2):
        vals = numeric(t[f"{name}_runtime_sec"], name, low=1e-12)
        all_values += vals.tolist()
        marks = ([False]*6 if name == "GateFM"
                 else [boolean(v) for v in t[f"{name}_runtime_is_estimated"]])
        for j, (value, estimated) in enumerate(zip(vals, marks)):
            ax.bar(x[j]+(i-1)*0.24, value, width=0.225,
                   color=COLORS[name], hatch="////" if estimated else "",
                   edgecolor="#313744" if estimated else "white", linewidth=0.7,
                   label=(name + (" (estimated)" if estimated else " (measured)"))
                         if j == 0 or (estimated and not any(marks[:j])) else None)
    ax.set_yscale("log")
    ax.set(xticks=x, xticklabels=[f"{p:,}" for p in DIMS], xlabel="Number of SNPs (P)",
           ylabel="Wall-clock runtime (seconds, log scale)", title="Runtime by SNP dimension")
    ax.set_ylim(10**(np.floor(np.log10(min(all_values)))-0.2),
                10**(np.ceil(np.log10(max(all_values)))+0.2))
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8.5)
    ax.yaxis.grid(alpha=0.28, which="major")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, folder, "figure2_runtime")


def plot_coverage(t, folder, *, name="table3_coverage_at_k", preview=False):
    dims = t["dimension"].astype(int).tolist()
    xx = np.arange(len(dims))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for k in KS:
        ax.plot(xx, numeric(t[f"coverage_at_{k}"], f"K={k}", high=1),
                marker="o", linewidth=1.5, label=f"K={k}")
    title = "Coverage by exhaustive-search width"
    if preview:
        title += " (exploratory; not manuscript)"
    ax.set(xticks=xx, xticklabels=[f"{p:,}" for p in dims],
           xlabel="Number of SNPs (P)", ylabel="Coverage of all three generating SNPs",
           ylim=(0, 1.05), title=title)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              frameon=False, fontsize=8.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, folder, name)


def render(t1, t2, t3, out, files=None):
    out.mkdir(parents=True, exist_ok=False)
    plot_rank(t1, out)
    plot_hit(t2, out)
    plot_runtime(t2, out)
    plot_coverage(t3, out)
    print(f"[PASS] Figure files (PNG, SVG, PDF): {out}")


def reference(args):
    paths = {x: args.tables.resolve() / f"{x}.csv" for x in ("table1", "table2", "table3")}
    t1 = table(paths["table1"], ("dimension", *STAGE1))
    t2 = table(paths["table2"], ("dimension", *(f"{m}_hit_at_1" for m in STAGE2),
                                 *(f"{m}_runtime_sec" for m in STAGE2),
                                 "BitEpi_runtime_is_estimated", "MDR_runtime_is_estimated"))
    t3 = table(paths["table3"], ("dimension", *(f"coverage_at_{k}" for k in KS)))
    render(t1, t2, t3, args.output.resolve(), paths)


def render_gatefm_only(summary, raw, source, out, *, preview=False):
    """Paper-layout charts driven only by the experiment's measured metrics.

    A subset/custom run is explicitly marked exploratory and uses preview filenames.
    """
    out.mkdir(parents=True, exist_ok=False)
    colors = COLORS["GateFM"]
    dimensions = summary.dimension.astype(int).tolist()
    x = np.arange(len(dimensions))
    labels = [f"{p:,}" for p in dimensions]
    prefix = "preview_" if preview else ""
    caption = " (exploratory; not manuscript)" if preview else ""
    charts = (
        ("table1_mean_rank", "Mean rank of generating SNPs",
         "Mean rank (lower is better)", numeric(summary["mean_gt_rank"], "mean rank"), False),
        ("table2_hit_at_1", "Stage 2 Hit@1 at Top-30",
         "Hit@1", numeric(summary["hit_at_1"], "Hit@1", high=1), False),
        ("figure2_runtime", "Runtime by SNP dimension",
         "Runtime (seconds, log scale)", numeric(summary["mean_runtime_sec"], "runtime", low=1e-12), True),
    )
    for stem, title, ylabel, values, logscale in charts:
        fig, ax = plt.subplots(figsize=(11, 6.2 if logscale else 4.8))
        ax.bar(x, values, width=0.27, color=colors, label="GateFM",
               edgecolor="white", linewidth=0.7)
        ax.set(xticks=x, xticklabels=labels, xlabel="Number of SNPs (P)",
               ylabel=ylabel, title=title + caption)
        if logscale:
            ax.set_yscale("log")
            ax.set_ylim(10**(np.floor(np.log10(min(values)))-0.2),
                        10**(np.ceil(np.log10(max(values)))+0.2))
        else:
            ax.set_ylim(0, 1.08 if stem == "table2_hit_at_1" else None)
        ax.yaxis.grid(alpha=0.27)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
        fig.tight_layout()
        save(fig, out, prefix + stem)
    coverage = []
    for rec in summary.itertuples(index=False):
        p = int(rec.dimension)
        matches = raw.loc[raw.dimension.astype(int) == p]
        if len(matches) != int(rec.n_runs) or len(matches) < 1:
            raise ValueError(f"P={p}: per-run rows must equal dimension summary n_runs")
        if not preview and len(matches) != 15:
            raise ValueError(f"P={p} expects 15 completed formal runs")
        row = {"dimension": p}
        for k in KS:
            count = sum(all(int(getattr(t, f"gt_rank_{g}")) <= min(k,p) for g in "abc")
                        for t in matches.itertuples(index=False))
            row[f"coverage_at_{k}"] = count / len(matches)
        if abs(row["coverage_at_30"] - float(rec.coverage_at_30)) > 1e-9:
            raise ValueError(f"Coverage@30 mismatch at P={p}")
        coverage.append(row)
    coverage = pd.DataFrame(coverage)
    plot_coverage(coverage, out, name=prefix + "table3_coverage_at_k",
                  preview=preview)
    if preview:
        print(f"[PASS] Exploratory (non-manuscript) GAMETES figures: {out}")
    else:
        print(f"[PASS] Formal GateFM figures (PNG, SVG, PDF): {out}")


def benchmark(args):
    source = args.evaluation.resolve()
    status = json.loads((source / "evaluation_status.json").read_text(encoding="utf-8"))
    is_formal = (status.get("result_scope") == "complete_frozen_plan"
                 and status.get("observed_runs") == 90
                 and status.get("expected_runs") == 90)
    if not is_formal and not args.allow_incomplete:
        raise ValueError("Formal paper figures require 90 frozen fits; "
                         "use --allow-incomplete for exploratory charts")
    required = {"dimension", "n_runs", "mean_gt_rank", "hit_at_1",
                "mean_runtime_sec", "coverage_at_30"}
    if is_formal:
        summary = table(source / "synthetic_dimension_summary.csv", required)
    else:
        summary = pd.read_csv(source / "synthetic_dimension_summary.csv")
        if not required.issubset(summary):
            raise ValueError(f"Missing summary columns: {sorted(required - set(summary))}")
        if (summary.dimension.duplicated().any()
                or not set(summary.dimension.astype(int)).issubset(DIMS)
                or summary.empty):
            raise ValueError("Exploratory dimensions must be unique, from the frozen dimensions")
        summary = summary.sort_values("dimension").reset_index(drop=True)
    if is_formal and not (summary.n_runs.astype(int) == 15).all():
        raise ValueError("Expected 15 model fits per dimension")
    raw = pd.read_csv(source / "synthetic_per_run_results.csv")
    if len(raw) != int(status.get("observed_runs", -1)):
        raise ValueError("Per-run rows do not match evaluation_status observed_runs")
    if (set(raw.dimension.astype(int)) != set(summary.dimension.astype(int))
            or sum(summary.n_runs.astype(int)) != len(raw)):
        raise ValueError("Per-run and summary dimensions/counts do not agree")
    if is_formal and (len(raw) != 90 or set(raw.dimension.astype(int)) != set(DIMS)):
        raise ValueError("Expected the 90 per-run benchmark rows")
    if not is_formal:
        if args.comparators is not None:
            raise ValueError("Comparator figure panels require a complete formal benchmark")
        render_gatefm_only(summary, raw, source, args.output.resolve(), preview=True)
        return
    if args.comparators is None:
        render_gatefm_only(summary, raw, source, args.output.resolve())
        return
    comp = pd.read_csv(args.comparators, keep_default_na=False)
    expected = {"dimension", "method", "metric", "value", "is_estimated"}
    if not expected.issubset(comp):
        raise ValueError(f"Comparator CSV requires {sorted(expected)}")
    if comp.duplicated(["dimension", "method", "metric"]).any():
        raise ValueError("Duplicate comparator record")
    lookup = {(int(r.dimension), r.method, r.metric): r for r in comp.itertuples(index=False)}
    ranks, results, coverage = [], [], []
    for rec in summary.itertuples(index=False):
        p = int(rec.dimension)
        rank = {"dimension": p, "GateFM": float(rec.mean_gt_rank)}
        for name in STAGE1[1:]:
            key = (p, name, "mean_rank")
            if key not in lookup: raise ValueError(f"Comparator absent: {key}")
            rank[name] = float(lookup[key].value)
        ranks.append(rank)
        row = {"dimension": p, "GateFM_hit_at_1": float(rec.hit_at_1),
               "GateFM_runtime_sec": float(rec.mean_runtime_sec)}
        for name in STAGE2[1:]:
            for metric in ("hit_at_1", "runtime_sec"):
                key = (p, name, metric)
                if key not in lookup: raise ValueError(f"Comparator absent: {key}")
                item = lookup[key]
                row[f"{name}_{metric}"] = ("NE" if str(item.value).strip().upper() == "NE"
                                          else float(item.value))
                if metric == "runtime_sec":
                    row[f"{name}_runtime_is_estimated"] = boolean(item.is_estimated)
        results.append(row)
        subset = raw[raw.dimension.astype(int) == p]
        cov = {"dimension": p}
        for k in KS:
            cov[f"coverage_at_{k}"] = sum(
                all(int(getattr(x, f"gt_rank_{z}")) <= min(p, k) for z in "abc")
                for x in subset.itertuples(index=False)) / 15
        if abs(cov["coverage_at_30"] - float(rec.coverage_at_30)) > 1e-9:
            raise ValueError(f"Coverage@30 mismatch at P={p}")
        coverage.append(cov)
    render(pd.DataFrame(ranks), pd.DataFrame(results), pd.DataFrame(coverage),
           args.output.resolve(), None)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    r = modes.add_parser("reference", help="Render a manuscript-compatible set of three input tables")
    r.add_argument("--tables", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    b = modes.add_parser("benchmark", help="Render completed 90-fit outputs with comparator evidence")
    b.add_argument("--evaluation", type=Path, required=True)
    b.add_argument("--comparators", type=Path, help="Independent comparator CSV, when available")
    b.add_argument("--allow-incomplete", action="store_true",
                   help="Render explicitly labeled exploratory images from subset/custom runs")
    b.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    {"reference": reference, "benchmark": benchmark}[args.mode](args)


if __name__ == "__main__":
    main()
