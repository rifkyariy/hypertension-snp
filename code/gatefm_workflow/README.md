# GateFM

**Supervised third-order factorization machine for high-dimensional three-SNP hypothesis prioritization**

GateFM learns genotype-specific embeddings and SNP-level gates to prioritize candidates (Stage 1), then evaluates one predefined genotype pattern for every selected three-SNP combination (Stage 2). This repository contains the model, experiment settings, synthetic-data generator, evaluation scripts, and reproducible examples.

**Manuscript:** *Supervised Third-Order Factorization for High-Dimensional Three-SNP Hypothesis Prioritization in Hypertension*


## Quick start: obtain your first result

Use **Windows PowerShell**, Git, and Python 3.11. The example runs on a CPU and generates its own data.

**Step 1 — Clone and install**

```powershell
git clone https://github.com/rifkyariy/hypertension-snp.git
cd hypertension-snp/code/gatefm_workflow
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Step 2 — Run the toy experiment**

```powershell
.\run_toy_demo.ps1 -Python .\.venv\Scripts\python.exe
```

**Step 3 — View the generated figures**

When the experiment completes, the figures are ready in the displayed `figures/` directory:

```text
Completed 10/10
COMPLETED ...\outputs\toy_<timestamp>
FIGURES ...\outputs\toy_<timestamp>\figures
```

The toy demonstration produces 1,200 samples with 30 SNPs, three short Stage 1 fits, **4,060** Stage 2 triplets, **30** single-SNP tests, **435** SNP-pair tests, and four charts exported as **PNG, SVG and PDF**.

**Cross-platform equivalent:**

```bash
python run_workflow.py smoke --output outputs/toy_001 --execute
```

To preview the plan, omit `--execute`. Use a new output directory for each run.

## The analysis pipeline

1. **Stage 1 — SNP ranking:** Train the third-order gated factorization machine. For SNP j, the ranking score is `q_j = max_k(g_j × ||E[j,k]||₂)`. Smaller ranking numbers indicate higher priority.
2. **Consensus:** Combine full-list rankings across independent model initializations and select a representative genotype state for each SNP.
3. **Stage 2 — triplet analysis:** Select the Top-K SNPs and enumerate `C(K,3)` combinations. Test one fixed genotype pattern per triplet; at K=30, the search covers **4,060** triplets.
4. **Lower-order analysis:** Summarize corresponding single- and two-SNP patterns.

See [Method](docs/METHOD.md) for the mathematical definitions and [Code guide](docs/CODE_GUIDE.md) for each Python file.

## Analyze your own SNP dataset

**Genotype table (CSV or TSV):** one row per sample, one column per SNP, and a binary phenotype column (for example, class). SNP genotypes use 0/1/2. For illustration:

```csv
SNP_A,SNP_B,SNP_C,class
0,1,2,0
1,0,1,1
2,2,0,0
0,1,1,1
```

Use your full dataset rather than these four illustrative rows. The reader accepts pre-QC CSV and tab-delimited TSV/TXT. It recognizes common outcome column names such as label, phenotype and case_control; an ambiguous/missing label must be resolved using --label-column. It never assumes that the final SNP column is the outcome. Standard sample ID columns are ignored; all remaining SNPs must be distinct, complete 0/1/2 integer genotypes. VCF, GEN and Parquet should be converted upstream by an authorized data custodian.

**Cohort input validation and preflight:** Default outcomes must be 0=control and 1=case. For any other encoding, explicitly supply BOTH --case-value and --control-value; the program never guesses disease direction. For example, only if the authorized source documentation confirms 1=case and 2=control, run the following. The expected dimensions are optional assertions for a 60,407-sample / 14,313-SNP cohort; replace them if the approved input has a different shape.

~~~powershell
.\.venv\Scripts\python.exe code\validate_cohort_input.py --data data\approved_cohort.csv --label-column HTN_STATUS --case-value 1 --control-value 2 --expected-samples 60407 --expected-snps 14313
~~~

This checks schema, SNP dosage, both label classes, sample/SNP counts and cohort polarity without training or exporting sample rows. Use the field name and class definitions that the data custodian actually confirms, not the placeholders above. For canonical class=0/1, omit --case-value and --control-value.

After a successful preflight, the complete 10-fit cohort workflow can use exactly the same mapping:

~~~powershell
.\.venv\Scripts\python.exe run_workflow.py human --data data\approved_cohort.csv --label-column HTN_STATUS --case-value 1 --control-value 2 --expected-samples 60407 --expected-snps 14313 --output outputs\authorized_cohort_001 --execute
~~~

The human workflow automatically validates the input before the ten independent training runs, and records progress and errors in run.log and workflow_status.json. Plan compute and storage capacity for high-dimensional datasets. Keep cohort inputs and derived results in approved local storage, following the associated research data-use and disclosure requirements.

### Additional execution options
**Option A — Single Stage 1 fit:** Store an authorized dataset as `data/my_genotypes.csv` and run:

```powershell
New-Item -ItemType Directory -Force data
# Copy your genotype CSV into data\my_genotypes.csv
.\.venv\Scripts\python.exe code\run_stage1.py --data data\my_genotypes.csv --output outputs\my_stage1_001 --config configs\config_thesis.json --seed 42 --device cpu
```

Inspect `outputs/my_stage1_001/stage1_snp_ranking.csv` for SNP scores and `q_rank`, alongside the checkpoint and training summary.

**Option B — Complete 10-fit workflow:** For a dataset with **at least 30 SNP columns** and sufficient samples in both classes for a stratified training/validation split:

```powershell
.\.venv\Scripts\python.exe run_workflow.py human --data data\my_genotypes.csv --output outputs\my_analysis_001
.\.venv\Scripts\python.exe run_workflow.py human --data data\my_genotypes.csv --output outputs\my_analysis_001 --execute
```

The first command previews the execution plan; the second performs 10 Stage 1 fits, consensus ranking, Stage 2, lower-order analysis, and automatic publication-style result plotting. Use a fresh output name for each subsequent experiment. Keep participant-level data in authorized local storage; `data/` and `outputs/` are excluded from Git.

## Find and interpret your results

The toy experiment creates a self-contained results folder:

```text
outputs/toy_<timestamp>/
├── workflow_status.json
├── toy_summary.json
├── run.log
├── toy.csv
├── runs/run_001/ ... run_003/
│   └── stage1_snp_ranking.csv
├── consensus/
│   ├── stage1_consensus_all_snps.csv
│   └── stage1_consensus_top30.csv
├── stage2/
│   └── stage2_fixed_pattern_triplets.csv
├── lower_order/
│   ├── lower_order_single_tests.csv
│   └── lower_order_pair_fixed_pattern_tests.csv
└── figures/
    ├── toy_stage1_true_rank.png / .svg / .pdf
    ├── toy_coverage_at_k.png / .svg / .pdf
    ├── toy_stage2_top15.png / .svg / .pdf
    └── toy_runtime_components.png / .svg / .pdf
```

| Output | Meaning |
|---|---|
| `workflow_status.json` | `"status": "completed"` confirms all planned steps finished. |
| `stage1_consensus_top30.csv` | `consensus_rank` orders the prioritized SNPs; lower numbers mean higher priority. |
| `stage2_fixed_pattern_triplets.csv` | `stage2_rank`, `raw_p`, `bonferroni_p`, `odds_ratio`, and confidence intervals for the tested triplets. |
| `figures/` | Publication-style images saved as PNG, SVG and PDF. |

If an experiment fails, check `workflow_status.json` for its status and `run.log` for the full, human-readable stage names and errors. All stage output is collected into this single log file; no `step_XX.log` files are generated for new runs. Existing older result folders are not modified. Each new run uses its own output directory, preserving previous results.

## Full GAMETES benchmark

The research configuration includes six SNP dimensions (**20, 100, 500, 1,000, 5,000, and 10,000**), five dataset seeds (**42, 100, 200, 300, 400**), and three model initialization seeds, totaling **30 synthetic datasets and 90 model fits**.

**Step 1 — Prepare GAMETES 2.1.** Install Java and check that it is on your PATH:

```powershell
java -version
.\generate_paper_data.ps1 -Python .\.venv\Scripts\python.exe
```

The generator obtains GAMETES, checks its SHA-256, creates 12,000-sample balanced case/control datasets, and writes `data/synthetic_dataset_manifest.csv`. Each manifest row records the SNP dimension, dataset seed, data path, three generating SNPs, and file checksum.

**Alternative JAR setup:** If SourceForge displays an intermediate download page in your terminal, [download GAMETES_2.1.jar from the official project](https://sourceforge.net/projects/gametes/files/GAMETES_2.1.jar/download) in a web browser and save the JAR in `tools/`. Check its SHA-256 with `Get-FileHash`. The expected value is:

```text
5E38B0993F8915E3E3E4C225CE6B684455D02C6189877D6AE219D24AD69E27D7
```

Then use the local file directly:

```powershell
.\generate_paper_data.ps1 -Python .\.venv\Scripts\python.exe -GametesJar .\tools\GAMETES_2.1.jar
```

You can first check a selected generator configuration with `-Dimensions 20 -DataSeeds 42 -DryRun` (append these options to the command above); this prints the configured GAMETES invocation and checks the JAR without generating datasets.

**Step 2 — Review and execute the benchmark.** The first command previews the run list; the second executes it.

```powershell
.\.venv\Scripts\python.exe run_workflow.py synthetic --manifest data\synthetic_dataset_manifest.csv --output outputs\benchmark_001
.\.venv\Scripts\python.exe run_workflow.py synthetic --manifest data\synthetic_dataset_manifest.csv --output outputs\benchmark_001 --execute
```

Use a new output directory for each experiment. The full six-dimension workload includes high-dimensional training and requires sufficient memory and compute resources.

**Quick GAMETES test — one model fit.** The same main command accepts a subset of the frozen plan and training overrides. This example requires only the P=20 / seed=42 dataset in the manifest:

```powershell
.\.venv\Scripts\python.exe run_workflow.py synthetic `
  --manifest data\synthetic_dataset_manifest.csv `
  --dimensions 20 --data-seeds 42 --model-seeds 3484015997 `
  --max-runs 1 --epochs 1 --batch-size 64 --latent-rank 16 `
  --device cpu --output outputs\gametes_quick_001 --execute
```

The command completes Stage 1, Stage 2, evaluation and image export. It creates PNG, SVG and PDF figures in outputs/gametes_quick_001/figures/ with preview_ filenames. The corresponding evaluation_status.json identifies the selected exploratory scope and the evaluation/ directory contains the metrics. For the full reference benchmark, use the documented 30-dataset / 90-fit command with the frozen settings.
**Supported synthetic command-line options:** `--dimensions`, `--data-seeds`, `--model-seeds`, `--max-runs`, `--epochs`, `--batch-size`, `--latent-rank`, `--device`, `--config`, and `--comparators` (comparators require the full unmodified study). Training overrides are forwarded to the Stage 1 CLI and never rewrite the frozen JSON configuration. Omit all subset/override flags for the formal 90-fit plan.

The GAMETES **data generator** is a separate prerequisite, with `-Dimensions` and `-DataSeeds` selectors. It intentionally fixes heritability, allele frequencies, and other model-generation parameters to the manuscript protocol; these are not exposed as free-form command-line parameters.

**Step 3 — View the figures.** The experiment automatically saves **PNG, SVG and PDF** files in `outputs/benchmark_001/figures/`:

```text
figure2_runtime.png
table1_mean_rank.png
table2_hit_at_1.png
table3_coverage_at_k.png
```

The corresponding vector versions (`.svg` and `.pdf`) are ready for manuscript preparation. The raw statistics remain in `outputs/benchmark_001/evaluation/`.

The figure layout follows the manuscript: Stage 1 generating-SNP mean rank, Stage 2 Hit@1, **Figure 2 runtime bars on a logarithmic axis**, and Coverage@K at seven widths. Figure values always come from the current GAMETES run. With GateFM results alone, each plot shows the GateFM series using the paper's presentation style.

To include the independent baseline methods **in the same automatically generated Figure 2**, supply the optional comparator CSV *as part of the experiment command*:

```powershell
.\.venv\Scripts\python.exe run_workflow.py synthetic --manifest data\synthetic_dataset_manifest.csv --comparators data\comparator_metrics.csv --output outputsenchmark_with_comparators_001 --execute
```

The comparator file contains `dimension,method,metric,value,is_estimated`. Supply `mean_rank` for Distributed Transformer, ReliefF and MultiSURF; `hit_at_1` and `runtime_sec` for BitEpi and MDR. Explicitly flag measured versus estimated runtime and use `NE` for unavailable baseline Hit@1. The same workflow command still runs the GateFM analysis; it then includes the supplied baseline values in the paper-style tables and charts.

**Your own SNP cohort:** `run_workflow.py human --data ... --output ... --execute` automatically creates `figures/stage1_snp_rank.*`, `figures/stage2_top_triplets.*` and `figures/stage2_odds_ratio.*`, with SNP rankings, triplet significance, odds ratios and 95% confidence intervals. This is a data-driven cohort report, using the paper's visual conventions.

Supplementary Figure S1 comes from the separate five-seed gate/budget ablation experiment; S2 comes from single-SNP statistics across the eligible cohort. Their original publication-style plotting scripts are also provided, with optional commands in [Code guide](docs/CODE_GUIDE.md). These specialist figures use their respective input datasets rather than the main GAMETES benchmark.

## Tests

```powershell
.\.venv\Scripts\python.exe code\validate_contract.py
.\.venv\Scripts\python.exe code\verify_reproduction.py
.\.venv\Scripts\python.exe -m unittest discover -s code -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Experiment configurations document the seeds, model parameters, and analysis settings. For direct numerical comparisons, align the input datasets, training conditions, ranking rule, and statistical procedure.

Stage 2 results are candidate genotype-pattern associations. Independent datasets provide additional support for biological interpretation.

## Citation and data access

When referencing this implementation, use the associated manuscript title above; journal citation information can be added following publication. Clinical data are handled separately under applicable access permissions. See [Data access](docs/DATA_ACCESS.md).

The repository is public, but no open-source license has been granted. See the [license statement](../../README.md#license).
