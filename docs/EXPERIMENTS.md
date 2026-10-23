# Experiments and reference results

All numbers below are from the manuscript (v24) unless marked **Thesis** (`master.pdf`, Stage 2 at `K = 15`).
Use them to check a reproduction.

## Stage 1 hyperparameters (Thesis Table A-1)

Common: Adam, lr `1e-4`, `λ = 1e-2`, 90/10 train/val split, `γ₃ = 0.05`.

| `P` | rank `r` | epochs | batch | final `τ` |
|---|---|---|---|---|
| 20 | 16 | 300 | 64 | 3 |
| 100 | 32 | 300 | 64 | 10 |
| 500 | 256 | 300 | 64 | 50 |
| 1,000 | 256 | 300 | 64 | 100 |
| 5,000 | 512 | 300 | 64 | 500 |
| 10,000 | 1,024 | 100 | 16 | 1,000 |
| 14,313 (TWB) | *[to confirm]* | *[to confirm]* | *[to confirm]* | *[to confirm]* |

- `τ ≈ 0.1·P`. `P = 10,000` uses batch 16 / 100 epochs because of 12 GB VRAM, not tuning.
- Rank was chosen on an independent GAMETES seed (rank sweep 8 → 256 at `P = 1,000`), then frozen.
- *[to confirm]*: gate temperature `T`, `τ` schedule (start value and shape), TWB-cohort settings.

## Experimental protocol

1. **Simulation (Tables 1–3):** for each `P` × 5 seeds × 3 inits, train Stage 1, rank, compute mean GT rank and
   Coverage@K, run Stage 2 at `K = 30` (thesis: 15), record Hit@1 and wall-clock time.
2. **Baselines:** ReliefF, MultiSURF, Distributed Transformer as SNP rankers; BitEpi, MDR as exhaustive triple
   search; high-`P` runtimes extrapolated.
3. **Ablation:** `P = 5,000`, 1 run × 5 seeds, GateFM / no budget / Pure FM.
4. **Cohort:** 10 Stage 1 fits → consensus top 30 → 4,060 triples → lower-order screen.
5. **Descriptive scan:** 2×3 χ² for all 14,313 SNPs → Supplemental Data S1.

## Planned commands

Not runnable yet — these are the intended entry points from [API.md](API.md).

```bash
python scripts/run_simulation.py --config configs/gametes_p1000.yaml --seeds 42 100 200 300 400 --inits 3 --k 15 30
```

```bash
python scripts/run_ablation.py --config configs/gametes_p5000.yaml --seeds 42 100 200 300 400
```

```bash
python scripts/run_cohort.py --config configs/twb_cohort.yaml --runs 10 --k 30 --threshold 2e-4
```

```bash
python scripts/run_scan.py --data data/twb/genotypes.parquet --out results/supplemental_data_S1.csv
```

```bash
python scripts/make_figures.py --results results/
```

## Table 1 — Stage 1 mean GT rank (lower is better, ideal 2.00)

| `P` | GateFM | Distributed Transformer | ReliefF | MultiSURF |
|---|---|---|---|---|
| 20 | **2.98** | 13.33 | 4.60 | 5.73 |
| 100 | **6.51** | 53.07 | 54.60 | 68.60 |
| 500 | **4.33** | 294.93 | 265.27 | 280.87 |
| 1,000 | **7.00** | 514.13 | 547.07 | 570.13 |
| 5,000 | **13.20** | 2,314.47 | 3,065.20 | 3,041.00 |
| 10,000 | **247.20** | 3,640.07 | 4,216.07 | 6,050.60 |

### Per-seed GateFM mean GT rank (Thesis Tables A-2, A-3; mean of 3 inits)

| `P` | 42 | 100 | 200 | 300 | 400 | mean |
|---|---|---|---|---|---|---|
| 20 | 2.00 | 2.00 | 6.89 | 2.00 | 2.00 | 2.98 |
| 100 | 2.00 | 2.00 | 2.00 | 24.56 | 2.00 | 6.51 |
| 500 | 2.00 | 2.00 | 2.00 | 2.00 | 13.67 | 4.33 |
| 1,000 | 2.00 | 2.89 | 13.11 | 2.00 | 15.00 | 7.00 |
| 5,000 | 2.00 | 14.56 | 2.33 | 4.22 | 42.89 | 13.20 |
| 10,000 | 1227.44 | 2.22 | 2.00 | 2.33 | 2.00 | 247.20 |

`P = 10,000`, seed 42: one diverged Stage 1 run inflates the mean.

## Table 2 — Stage 2 Hit@1 and runtime at `K = 30`

| `P` | GateFM+top-30 Hit@1 | BitEpi | MDR | GateFM+top-30 s | BitEpi s | MDR s |
|---|---|---|---|---|---|---|
| 20 | 1.00 | 1.00 | 1.00 | 235.48 | 0.41 | 10.12 |
| 100 | 0.93 | 1.00 | 1.00 | 241.46 | 2.21 | 1,392.40 |
| 500 | 0.93 | 1.00 | NE | 355.45 | 231.80 | 178,320.86* |
| 1,000 | 0.87 | 1.00 | NE | 452.89 | 1,844.04 | 1,430,863.77* |
| 5,000 | 0.80 | NE | NE | 3,303.30 | 197,204.31* | 179,288,176.87* |
| 10,000 | 0.80 | NE | NE | 4,401.09 | 1,488,843.25* | 1,434,735,878.80* |

`*` extrapolated (see METHOD.md); NE = not estimable. `P = 20` uses `K' = 20` → 1,140 triples.

**Thesis (`K = 15`, 455 triples):** Hit@1 0.93 / 0.93 / 0.93 / 0.87 / 0.80 / 0.80; GateFM runtime
240.65 / 245.01 / 358.99 / 456.43 / 3,306.84 / 4,404.63 s.

## Table 3 — Coverage@K

| `P` | 5 | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| 20 | 0.93 | 0.93 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 |
| 100 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 1.00 |
| 500 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 1.00 | 1.00 |
| 1,000 | 0.80 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 | 1.00 |
| 5,000 | 0.67 | 0.73 | 0.80 | 0.80 | 0.80 | 0.80 | 0.93 |
| 10,000 | 0.73 | 0.80 | 0.80 | 0.80 | 0.80 | 0.87 | 0.87 |

## Ablation at `P = 5,000` (Figure S1; 1 run × 5 seeds)

| Variant | mean GT rank (SD) | mean gate mass |
|---|---|---|
| GateFM (`τ = 500`) | 84.33 (23.32) | 500.27 |
| GateFM no budget | 702.93 (280.17) | 2,223.74 |
| Pure FM | 1,522.80 (1,033.34) | 5,000.00 |

Different replication design from Table 1 — compare only within this table.

## Cohort application

- 14,313 SNPs → top 30 → 4,060 triple records; see [DATASETS.md](DATASETS.md) for Table 4 and Data S1 reference values.

## Figures

| File | Content | Planned script |
|---|---|---|
| `docs/figures/fig1_workflow.png` | two-stage workflow | static |
| `docs/figures/fig2_runtime.png` | runtime vs `P`, log scale | `make_figures.py` |
| `docs/figures/figS1_ablation.png` | gate / budget ablation | `make_figures.py` |
| `docs/figures/figS2_pvalues.png` | single-SNP p histogram + QQ | `make_figures.py` |
