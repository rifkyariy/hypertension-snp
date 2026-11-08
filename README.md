# hypertension-snp — GateFM

Source code and method documentation for

> Huang S-H, Wang Y-C, Yunanto RA, Leu J-S, Tsai C-T.
> **Supervised Third-Order Factorization for High-Dimensional Three-SNP Hypothesis Prioritization in Hypertension.**

and the NTUST master's thesis *Epistasis Detection in High-Dimensional SNP Data* (Shih-Hung Huang, 2026).

> **Status:** the **runnable GateFM implementation** is in [code/gatefm_workflow/](code/gatefm_workflow/README.md).
> [docs/](docs/) holds the method documentation — model, statistics, datasets, experimental protocol and
> reference results.

## What GateFM does

GateFM is a **gated third-order factorization machine** that ranks SNPs by their participation in
third-order (three-locus) interactions without enumerating the `C(P,3)` triples. A second stage
then exhaustively tests every triple among the top-`K` SNPs with one model-specified genotype pattern
per triple.

```
(X, y) ──Stage 1: GateFM SNP ranking──▶ R ──Stage 2: top-K triple tests──▶ T
          O(P·r) per sample                     C(K,3) 2×2 tables
```

For the 14,313-SNP hypertension cohort, `C(14313,3) ≈ 4.89 × 10¹¹` triples are reduced to
`C(30,3) = 4,060` explicit records.

![workflow](docs/figures/fig1_workflow.png)

### Pipeline in 7 steps

1. **Encode** — genotype matrix `X ∈ {0,1,2}^{n×P}` treated as categorical states (one embedding per SNP per genotype).
2. **Train Stage 1** — GateFM: gated embeddings → factorized third-order term → logit; loss = class-weighted BCE + gate-count penalty.
3. **Rank SNPs** — score `q_j = max_k g_j‖v_{j,k}‖₂`; sort descending (cohort: mean rank over 10 fits).
4. **Select top-K** — `S_K` = first `K` SNPs (`K = 30` reporting width).
5. **Fix patterns** — representative genotype `k̂_j = argmax_k g_j‖v_{j,k}‖₂` per selected SNP (cohort: majority vote across fits; ties go to the larger mean `g_j‖v_{j,k}‖₂`, then the smaller genotype code).
6. **Test triples** — every `C(K,3)` triple → one 2×2 table → χ² p, Bonferroni p, OR (95% CI).
7. **Report** — triples sorted by raw p. The lower-order screen keeps triples with Bonferroni `p < 2 × 10⁻⁴` whose single-SNP and SNP-pair tests all have Bonferroni `p ≥ 2 × 10⁻⁴`. The synthetic benchmark skips this screen and runs Stage 2 separately for each fit.

## Quick start

Requires Python 3.11. The toy run works on a CPU and generates its own data. On Windows:

```powershell
cd code\gatefm_workflow
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_workflow.py smoke --output outputs\toy_001 --execute
```

On Linux/macOS:

```bash
cd code/gatefm_workflow
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_workflow.py smoke --output outputs/toy_001 --execute
```

Results and figures (PNG, SVG, PDF) go to the output directory. Use a new directory for each run; omit
`--execute` to preview the plan.

`run_workflow.py` has three modes:

| Mode | Runs |
|---|---|
| `smoke` | Toy check: 1,200 samples, 30 SNPs, three short Stage 1 fits, then Stage 2 and figures |
| `synthetic` | GAMETES benchmark: 30 datasets and 90 model fits (needs Java and GAMETES 2.1 to generate the data) |
| `human` | Authorized cohort: input preflight, 10 Stage 1 fits, consensus, Stage 2, lower-order tests and figures |

See the [workflow guide](code/gatefm_workflow/README.md) for the full commands.

> **Gate penalty:** the executable configuration pins `λ = 0.001`
> ([config_thesis.json](code/gatefm_workflow/configs/config_thesis.json)), while the thesis appendix and
> [docs/METHOD.md](docs/METHOD.md) document `λ = 0.01`. Cite the configuration used for each run.

## Documentation

| Doc | Contents |
|---|---|
| [code/gatefm_workflow/README.md](code/gatefm_workflow/README.md) | Usage: toy run, your own dataset, GAMETES benchmark, outputs, tests |
| [code/gatefm_workflow/docs/CODE_GUIDE.md](code/gatefm_workflow/docs/CODE_GUIDE.md) | What each source file, script and test does |
| [code/gatefm_workflow/docs/DATA_ACCESS.md](code/gatefm_workflow/docs/DATA_ACCESS.md) | Data access and distribution rules |
| [docs/METHOD.md](docs/METHOD.md) | Full method: equations, FM trick, gates, losses, scoring, Stage 2 statistics, metrics |
| [docs/DATASETS.md](docs/DATASETS.md) | GAMETES benchmark spec, Taiwan Biobank cohort, data formats, reference records |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Hyperparameters, experimental protocol, reference results (Tables 1–3, ablation) |
| [docs/API.md](docs/API.md) | Original pre-release implementation specification; the released entry points are in the code guide |

## Repository layout

```
hypertension-snp/
├── README.md
├── CITATION.cff
├── code/
│   ├── README.md
│   └── gatefm_workflow/
│       ├── run_workflow.py   # single entry point: smoke, synthetic, human
│       ├── code/             # model, Stage 1/2, consensus, GAMETES, evaluation, validation
│       ├── configs/          # frozen training and experiment settings
│       ├── scripts/          # result and manuscript figures
│       ├── examples/         # toy dataset generator
│       ├── tests/            # regression tests
│       └── docs/             # METHOD, CODE_GUIDE, DATA_ACCESS
└── docs/
    ├── METHOD.md
    ├── DATASETS.md
    ├── EXPERIMENTS.md
    ├── API.md
    └── figures/          # fig1_workflow, fig2_runtime, figS1_ablation, figS2_pvalues
```

## Reported environment

| Item | Version / spec |
|---|---|
| Python | 3.11.12 |
| PyTorch | 2.9.1+cu128 (CUDA 12.8) |
| NumPy / pandas / SciPy / scikit-learn | 2.4.3 / 3.0.1 / 1.17.1 / 1.8.0 |
| Matplotlib | 3.10.8 |
| Hardware | AMD Ryzen 7 9700X (8 cores), NVIDIA RTX 5070 (12 GB), 64 GB RAM |

## Data availability

- **Synthetic benchmark** — not distributed; regenerate with `code/gatefm_workflow/generate_paper_data.ps1`
  (see the [workflow guide](code/gatefm_workflow/README.md)). The specification is in [docs/DATASETS.md](docs/DATASETS.md).
- **Taiwan Biobank hypertension cohort** — controlled access. Individual-level data, derived outputs and
  model checkpoints **must not be committed** to this repository. Apply via <https://www.twbiobank.org.tw>
  (ethics: NTUH REC 201606128RINC). See [DATA_ACCESS.md](code/gatefm_workflow/docs/DATA_ACCESS.md).

## Citation

See [CITATION.cff](CITATION.cff). Archived release DOI: *[to confirm]*.

## License

*[to confirm]*
