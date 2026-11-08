# Code guide

The following files implement the GateFM workflow. Start with `run_workflow.py` for end-to-end execution, or read `code/gatefm_model.py` for the model itself.

## Workflow

```text
Genotype CSV (samples x SNPs; genotypes 0/1/2) + binary outcome
              |
              v
       Stage 1: GateFM
              |
       q-score and SNP ranking
              |
     Per-run results / consensus
              |
       Top-K candidate SNPs
              |
    Stage 2: fixed-pattern triplets
              |
    Association and lower-order tests
              |
       Analysis CSV / JSON plus PNG / SVG / PDF figures
```

## Model, training and statistics

| File | Responsibility |
|---|---|
| `code/gatefm_model.py` | Third-order gated FM, embedding lookup, SNP gates, FM identity, score and genotype-state extraction. |
| `code/repro_io.py` | Genotype CSV parsing and validation, configuration resolution, deterministic seeding and file hashes. |
| `code/run_stage1.py` | One training run, final-epoch checkpoint and complete SNP ranking/score export. |
| `code/aggregate_stage1.py` | Multi-run consensus by mean full-list SNP rank and representative genotype vote. |
| `code/run_stage2.py` | Exhaustive selected triplets, fixed-pattern 2x2 analysis, p values, odds ratios and confidence intervals. |
| `code/posthoc_lower_order_screen.py` | Fixed-genotype single-SNP and pairwise association screens for candidate review. |

## Datasets, benchmark and evaluation

| File | Responsibility |
|---|---|
| `code/generate_gametes_data.py` | Generate simulation datasets through the verified GAMETES Java tool and automatically create a dataset manifest with paths, seeds and checksums. |
| `code/run_synthetic_benchmark.py` | Execute the formal or selected synthetic grid, forwarding optional epoch, batch-size and latent-rank overrides to Stage 1; subsets require a non-reference scope. |
| `code/evaluate_synthetic_benchmark.py` | Aggregate generating-SNP rank, Coverage@K, Stage 2 Hit@1 and run time, explicitly separating complete frozen plans from incomplete/custom runs. |
| `configs/config_thesis.json` | Frozen model, training, gate and Stage 2 settings with dimension-specific overrides. |
| `configs/experiment_plan.json` | Synthetic and cohort data/model seeds and evaluation scope. |
| `generate_paper_data.ps1` | Windows PowerShell helper to retrieve/verify GAMETES and generate simulation datasets. |


## Entry points and presentation

| File | Responsibility |
|---|---|
| `run_workflow.py` | Single entry point for smoke, 90-fit GAMETES and authorized cohort analyses; when executed, automatically generates matching result figures and publication-style image files. |
| `run_toy_demo.ps1` | Windows one-command toy data, training, analysis and automatically generated figures. |
| `examples/make_toy_dataset.py` | Generate reproducible non-human synthetic input for quick functional tests. |
| `scripts/plot_toy_figures.py` | Automatic last stage of the toy workflow: PNG/SVG/PDF diagnostic charts; summary data are saved with the run. |
| `scripts/render_manuscript_figures.py` | Final automatic GAMETES workflow stage: manuscript-style Stage 1, Stage 2, Figure 2 and coverage charts; adds comparator bars when independent data are supplied. |
| `scripts/plot_cohort_figures.py` | Automatic last stage of an authorized cohort run: SNP rank, Stage 2 significance, odds ratios and confidence intervals. |
| `scripts/render_supplement_s1.py` | Rebuild the original gate/budget ablation Figure S1 from its five-seed mechanism summary, preserving the publication plotting style. |
| `scripts/render_supplement_s2.py` | Rebuild the single-SNP p-value histogram, QQ plot and supplemental data from authorized SNP statistics. |
| `requirements.txt` | Pinned Python dependencies. |

## Validation

| File | Responsibility |
|---|---|
| `code/validate_contract.py` | Validate configuration and experiment seeds; optional check of manuscript method text supplied separately. |
| `code/verify_reproduction.py` | Numerical unit checks for FM identity, scoring, checkpoints, consensus and fixed-pattern statistics. |
| `code/test_archive_inputs.py` | Genotype input validation tests. |
| `tests/test_repository.py` | Source packaging, documentation, seed-plan and sensitive-data exclusion tests. |
| `code/validate_cohort_input.py` | Authorized-cohort table preflight, label mapping and expected dimension checks. |
| `code/verify_run_provenance.py` | Verify dataset checksums, resolved configurations, run identities and output consistency for benchmark aggregation. |
| `tests/test_human_input.py` | Regression checks for phenotype safety, duplicate SNPs and human workflow options. |

## Output layout

Each workflow execution creates a new `outputs/<run_name>/` directory. Its root has `workflow_status.json` and a single readable `run.log` (instead of scattered numbered step logs), while the actual results remain separated by `runs/`, `consensus/`, `stage2/`, `lower_order/`, `evaluation/` where applicable, and `figures/`. For GAMETES, all subset and training flags go through `run_workflow.py synthetic`. A selected or custom run automatically creates English `figures/preview_*.png` plus SVG/PDF and is explicitly marked exploratory. Omit subset/override flags to select the full frozen 90-fit evaluation and formal figure names.

For a self-contained introduction, run `run_toy_demo.ps1` and view the generated PNG, SVG or PDF images in `figures/`. The short demo verifies that the pipeline operates; full benchmark metrics depend on the configured datasets, model seeds and experiment protocol.
## Additional figure sources

For the separate ablation study, run the original manuscript S1 plotter after preparing the five-seed mechanism summary:

```powershell
.\.venv\Scripts\python.exe scripts\render_supplement_s1.py --input data\mechanism_ablation_budget_only_summary.csv --output outputs\figure_s1
```

For an authorized SNP-level p-value summary (`SNP` and `P_value`), run S2:

```powershell
.\.venv\Scripts\python.exe scripts\render_supplement_s2.py data\single_snp_pvalues.csv --output-dir outputs\figure_s2_data --figure-dir outputs\figure_s2
```

## Authorized cohort input validation

Before running the full human workflow, use code/validate_cohort_input.py on an authorized local pre-QC CSV/TSV genotype table. Specify the phenotype column if its name is non-standard, and always supply both --case-value and --control-value when the source uses anything other than 0=control, 1=case. The parser rejects missing labels instead of guessing the last column. Preflight may assert expected sample and SNP counts and runs automatically as the first data-dependent stage of run_workflow.py human. This does not expose or upload any participant records. Invalid data exits before model fitting and leaves run.log / workflow_status.json for diagnosis.

The data preparation contract begins with approved QC-completed 0/1/2 genotype tables. An authorized data custodian documents and performs upstream VCF/GEN conversion, imputation, SNP selection, QC and any clinical-covariate preparation.
