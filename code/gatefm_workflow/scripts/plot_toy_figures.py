"""Automatically render toy experiment figures in PNG, SVG and PDF."""
import argparse
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def csv_read(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def json_read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save_figure(fig, stem: Path):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=250)
    plt.close(fig)


def bars(out, heading, detail, labels, values, primary=()):
    values = [float(v) for v in values]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, max(4.3, 0.42 * len(labels) + 1.8)))
    colors = ["#E74C3C" if label in primary else "#4E79A7" for label in labels]
    ax.barh(x, values, color=colors)
    ax.set_yticks(x, labels)
    ax.invert_yaxis()
    ax.set_title(heading, loc="left", weight="bold")
    ax.set_xlabel(detail)
    ax.xaxis.grid(alpha=0.23)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, out.with_suffix(""))


def curve(out, cov, n):
    ks = (5, 10, 15, 20, 30)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(ks, [cov[k] for k in ks], color="#E74C3C", marker="o", linewidth=2)
    ax.set(xlabel="Top-K candidate SNPs", ylabel="Generating-triplet coverage",
           ylim=(0, 1.05), xticks=ks, title="Coverage by search width")
    ax.grid(alpha=0.23)
    fig.tight_layout()
    save_figure(fig, out.with_suffix(""))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True,type=Path)
    ap.add_argument("--true-snps",nargs=3,default=["SNP000","SNP001","SNP002"])
    args=ap.parse_args()
    root=args.run.resolve()
    if json_read(root/"workflow_status.json").get("status") not in ("completed", "analysis_complete"):
        raise ValueError("Analysis must finish before results can be plotted")
    ground=set(args.true_snps)
    runs=sorted(d for d in (root/"runs").iterdir()
                if d.is_dir() and (d/"stage1_snp_ranking.csv").is_file())
    if not runs:
        raise ValueError("No completed Stage 1 outputs")
    ranks=[]
    for d in runs:
        mapping={r["snp_id"]:int(float(r["q_rank"]))
                 for r in csv_read(d/"stage1_snp_ranking.csv")}
        if not ground.issubset(mapping):
            raise ValueError("Generating SNP missing from Stage 1 ranking")
        ranks.append(mapping)
    mean_ranks={s:sum(r[s] for r in ranks)/len(ranks) for s in ground}
    mean_gt_rank=sum(mean_ranks.values())/3
    coverage={k:sum(all(r[s]<=k for s in ground) for r in ranks)/len(ranks)
              for k in (5,10,15,20,30)}
    stage2=csv_read(root/"stage2"/"stage2_fixed_pattern_triplets.csv")
    st2=json_read(root/"stage2"/"stage2_summary.json")
    k=int(st2["n_candidate_snps"])
    assert len(stage2)==math.comb(k,3)==int(st2["n_triplets"])
    lower=json_read(root/"lower_order"/"lower_order_screen_summary.json")
    assert int(lower["n_single_tests"])==k
    assert int(lower["n_pair_tests"])==math.comb(k,2)
    true_row=next((r for r in stage2
                   if {r["snp_a"],r["snp_b"],r["snp_c"]}==ground),None)
    hit1=bool(true_row and int(true_row["stage2_rank"])==1)
    out=root/"figures"
    out.mkdir(exist_ok=True)
    bars(out/"toy_stage1_true_rank.svg",
         "Toy Stage 1 | Mean generating-SNP rank",
         "Mean q rank across three short model fits; smaller is better",
         args.true_snps,[mean_ranks[x] for x in args.true_snps],ground)
    curve(out/"toy_coverage_at_k.svg",coverage,len(runs))
    best=stage2[:15]
    labels=[f'#{r["stage2_rank"]} {r["snp_a"]}+{r["snp_b"]}+{r["snp_c"]}' for r in best]
    logp=[-math.log10(max(float(r["raw_p"]),1e-300)) for r in best]
    bars(out/"toy_stage2_top15.svg",
         "Toy Stage 2 | Top 15 fixed-pattern triplets",
         "-log10(raw p); top-ranked three-SNP hypotheses (not OR-max pattern mining)",
         labels,logp)
    runtimes=[float(json_read(d/"run_summary.json")["runtime_sec"]) for d in runs]
    bars(out/"toy_runtime_components.svg",
         "Toy | Runtime by pipeline component",
         "Single 30-SNP toy run, not Figure 2 runtime scaling across SNP dimensions",
         [d.name for d in runs]+["Stage 2","Lower order"],
         runtimes+[float(st2["runtime_sec"]),float(lower["runtime_sec"])])
    summary={
        "status":"completed", "scope":"TOY 30-SNP non-human functional test",
        "generating_snps":args.true_snps,"n_stage1_runs":len(runs),
        "stage1_true_snp_mean_rank":mean_gt_rank,
        "per_snp_mean_rank":mean_ranks,
        "coverage_at_k":{str(k):v for k,v in coverage.items()},
        "n_stage2_triplets":len(stage2),"n_single_tests":int(lower["n_single_tests"]),
        "n_pair_tests":int(lower["n_pair_tests"]),
        "stage2_true_triple_rank":int(true_row["stage2_rank"]) if true_row else None,
        "stage2_hit_at_1":hit1,
        "stage2_true_triple_raw_p":float(true_row["raw_p"]) if true_row else None,
        "stage1_runtime_sec":runtimes,
        "stage2_runtime_sec":float(st2["runtime_sec"]),
        "lower_order_runtime_sec":float(lower["runtime_sec"]),

    }
    (root/"toy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print("Toy figures (PNG, SVG, PDF):", out)


if __name__=="__main__":
    main()
