# Implementation specification (planned API)

> **The code is not in this repository yet.** This document specifies the interface the released code is
> expected to follow, derived directly from [METHOD.md](METHOD.md). Names may change to match the original
> implementation when it is added.

Planned code layout:

```
gatefm/
├── data.py        # loading, encoding, splitting
├── model.py       # GateFM nn.Module
├── losses.py      # weighted BCE + gate-count loss
├── train.py       # Stage 1 training loop, τ schedule
├── ranking.py     # SNP scores, ranking, consensus across runs
├── stage2.py      # triple enumeration, 2×2 tests, lower-order screen
├── scan.py        # single-SNP 2×3 χ² scan (Supplemental Data S1)
├── metrics.py     # mean GT rank, Coverage@K, Hit@1
└── baselines/     # ReliefF, MultiSURF, BitEpi, MDR wrappers + runtime extrapolation
configs/           # one YAML per SNP dimension + cohort config
scripts/           # run_simulation.py, run_cohort.py, run_ablation.py, run_scan.py, make_figures.py
```

Package: `gatefm`. Array conventions: `X` is `np.ndarray[int8]` of shape `(n, P)` with values in `{0,1,2}`;
`y` is `np.ndarray[int8]` of shape `(n,)` with values in `{0,1}`. SNP indices are 0-based column indices of `X`.
Equation numbers refer to [METHOD.md](METHOD.md).

- [gatefm.data](#gatefmdata)
- [gatefm.model](#gatefmmodel)
- [gatefm.losses](#gatefmlosses)
- [gatefm.train](#gatefmtrain)
- [gatefm.ranking](#gatefmranking)
- [gatefm.stage2](#gatefmstage2)
- [gatefm.scan](#gatefmscan)
- [gatefm.metrics](#gatefmmetrics)
- [gatefm.baselines](#gatefmbaselines)
- [scripts/](#scripts)
- [Config schema](#config-schema)

---

## gatefm.data

### `load_gametes(path) -> (X, y, snp_ids, gt_idx)`
Read one GAMETES tab-separated dataset.

| Arg | Type | Description |
|---|---|---|
| `path` | `str \| Path` | GAMETES `.txt` file (header row of SNP names + `Class`) |

Returns `X (n,P) int8`, `y (n,) int8`, `snp_ids: list[str]`, `gt_idx: np.ndarray[int]` — column indices of the
3 generating SNPs (names starting with `M0P`). Raises `ValueError` if values outside `{0,1,2}` or no `Class` column.

### `load_genotype_table(path, label_col="label", id_col=None) -> (X, y, snp_ids)`
Read a cohort genotype table (CSV / Parquet) where each SNP is a column of dosages `0/1/2`.
Missing genotypes raise `ValueError` (the analysis assumes QC'd, complete data).

### `train_val_split(X, y, val_frac=0.1, seed=0, stratify=True) -> (X_tr, y_tr, X_va, y_va)`
Stratified 90/10 split used for Stage 1 training.

### `class GenotypeDataset(torch.utils.data.Dataset)`
`GenotypeDataset(X, y)` → items `(x: LongTensor[P], y: FloatTensor[])`. Stores `X` as `int8`; casts per batch.

### `pos_weight(y) -> float`
`N_control / N_case`. Used for the class-weighted BCE.

---

## gatefm.model

### `class GateFM(torch.nn.Module)`

```python
GateFM(n_snps: int, rank: int, temperature: float = 1.0, gamma3: float = 0.05,
       gate_init: float = 0.0, emb_init_std: float = 0.01, use_gates: bool = True,
       gate_fn: str = "hard_sigmoid")
```

| Arg | Description |
|---|---|
| `n_snps` | `P` |
| `rank` | latent rank `r` |
| `temperature` | `T` in `g_j = h(α_j / T)` |
| `gamma3` | fixed third-order scale `γ₃` |
| `gate_init` | initial `α_j` (0 → `g_j = 0.5` for hard-sigmoid) |
| `emb_init_std` | std of normal init of embeddings |
| `use_gates` | `False` → Pure FM ablation (`g_j ≡ 1`) |
| `gate_fn` | `"hard_sigmoid"` (manuscript) or `"sigmoid"` (thesis) |

Parameters

| Name | Shape | Meaning |
|---|---|---|
| `embedding.weight` | `(3P, r)` | `v_{j,k}` at row `3j + k` |
| `alpha` | `(P,)` | raw gate parameters `α_j` |
| `bias` | `()` | intercept `b` |

Methods

| Method | Returns | Description |
|---|---|---|
| `forward(x: LongTensor[B,P])` | `FloatTensor[B]` | logits `z_i` (Eq. FM trick + logit) |
| `gates()` | `FloatTensor[P]` | `g_j` |
| `gated_embeddings(x)` | `FloatTensor[B,P,r]` | `a_{ijf}` |
| `genotype_embeddings()` | `FloatTensor[P,3,r]` | `V_j` for all SNPs |
| `state_scores()` | `FloatTensor[P,3]` | `ρ_{j,k} = g_j‖v_{j,k}‖₂` |
| `snp_scores()` | `FloatTensor[P]` | `q_j = max_k ρ_{j,k}` |
| `representative_states()` | `LongTensor[P]` | `k̂_j = argmax_k ρ_{j,k}` |
| `gate_mass()` | `FloatTensor[]` | `Σ_j g_j` |

### `third_order_fm(a: Tensor[B,P,r]) -> Tensor[B,r]`
Pure function: `(p₁³ − 3p₁p₂ + 2p₃)/6` per latent dimension. Self-check against brute force for small `P`
lives in `tests/test_fm_trick.py`.

### `hard_sigmoid(t) -> Tensor`
`clamp((t + 1) / 2, 0, 1)`.

---

## gatefm.losses

### `bce_loss(logits, y, pos_weight=None) -> Tensor`
`torch.nn.functional.binary_cross_entropy_with_logits` with optional positive-class weight.

### `gate_count_loss(gates, tau, lam=1e-2) -> Tensor`
`lam * (gates.sum() - tau) ** 2`.

### `total_loss(model, logits, y, tau, lam, pos_weight=None, use_budget=True) -> (loss, parts: dict)`
`L = L_BCE + L_gate`. `use_budget=False` → "GateFM no budget" ablation. `parts` = `{"bce", "gate", "gate_mass"}`.

---

## gatefm.train

### `@dataclass TrainConfig`

| Field | Default | Description |
|---|---|---|
| `rank` | — | `r` |
| `epochs` | 300 | |
| `batch_size` | 64 | |
| `lr` | `1e-4` | Adam |
| `lam` | `1e-2` | `λ` |
| `tau_final` | — | final target gate mass (≈ `0.1·P`) |
| `tau_start` | `None` | initial target; `None` → `P` |
| `tau_schedule` | `"linear"` | how `τ_t` decreases from `tau_start` to `tau_final` |
| `temperature` | `1.0` | gate `T` |
| `gamma3` | `0.05` | |
| `val_frac` | `0.1` | |
| `class_weighted` | `False` | `True` for the cohort |
| `use_gates` / `use_budget` | `True` | ablation switches |
| `seed` | `0` | model init seed |
| `device` | `"cuda"` | |

`TrainConfig.from_yaml(path)` / `TrainConfig.to_yaml(path)`.

### `tau_at(epoch, cfg) -> float`
Target soft gate mass `τ_t` for an epoch.

### `train_gatefm(X, y, cfg, P_names=None, log_every=10) -> (model, history)`
Trains Stage 1. `history` is a DataFrame with columns
`epoch, tau, train_loss, train_bce, gate_loss, gate_mass, val_loss, val_auc, seconds`.
The returned model is in `eval()` mode on CPU.

### `save_run(model, history, cfg, out_dir)` / `load_run(out_dir) -> (model, history, cfg)`
Persist `model.pt`, `history.csv`, `config.yaml`.

---

## gatefm.ranking

### `rank_snps(model, snp_ids=None) -> pd.DataFrame`
Columns: `rank` (1 = best), `snp_idx`, `snp_id`, `q`, `k_hat`, `rho_0`, `rho_1`, `rho_2`, `gate`.
Sorted by decreasing `q`; ties → SNP input order.

### `consensus_ranking(rankings: list[pd.DataFrame]) -> pd.DataFrame`
Combines `R` runs. Columns: `consensus_rank`, `snp_idx`, `snp_id`, `mean_rank`, `mean_q`, `k_hat`
(majority vote), `k_hat_votes` (e.g. `"7/10"`), `mean_rho_0..2`.
Rank ties → larger `mean_q`, then input order. Vote ties → larger mean `ρ`, then smaller genotype code.

### `top_k(ranking, k) -> pd.DataFrame`
First `min(k, P)` rows.

---

## gatefm.stage2

### `enumerate_triples(k) -> np.ndarray[int, (C(k,3), 3)]`
`itertools.combinations(range(k), 3)` as an array (positions in the top-`K` list).

### `pattern_table(X, y, snps, pattern) -> (a, b, c, d)`
2×2 counts for one fixed pattern over any number of SNPs (1, 2 or 3).

### `chi2_2x2(a, b, c, d) -> (chi2, p)`
Pearson χ², 1 df, no continuity correction (`scipy.stats.chi2_contingency(..., correction=False)`).
Returns `p = 1.0` when a margin is zero.

### `odds_ratio_wald(a, b, c, d, alpha=0.05) -> (or_, lo, hi)`
`ad/bc` with Wald CI; adds 0.5 to all cells only if any cell is zero.

### `stage2_triples(X, y, ranking, k=30) -> pd.DataFrame`
Tests every triple in the top-`k`. One row per triple, sorted by `p_raw` then SNP indices:

| Column | Description |
|---|---|
| `rank` | 1 = smallest p |
| `snp_idx_1..3`, `snp_id_1..3` | SNPs in Stage 1 rank order |
| `pattern` | e.g. `"0,2,1"` |
| `a, b, c, d` | 2×2 counts (case-match, control-match, case-other, control-other) |
| `chi2`, `p_raw`, `p_bonf` | `m = C(k,3)` |
| `or`, `or_lo`, `or_hi` | Wald 95% CI |

### `lower_order_screen(X, y, ranking, triples, k=30, threshold=2e-4) -> pd.DataFrame`
Adds to the triple report: `p_bonf_single_1..3` (`m = k`), `p_bonf_pair_12/13/23` (`m = C(k,2)`) and
`retained: bool` (`p_bonf < threshold` and every component `≥ threshold`).

---

## gatefm.scan

### `single_snp_scan(X, y, snp_ids, positions=None) -> pd.DataFrame`
2×3 Pearson χ² per SNP, dropping empty genotype columns. Columns:
`snp_id`, `chrom`, `pos` (if `positions` given), `chi2`, `df`, `p`, `p_bonf`, `neg_log10_p`.
Written by `scripts/run_scan.py` as `supplemental_data_S1.csv`.

---

## gatefm.metrics

| Function | Returns |
|---|---|
| `gt_ranks(ranking, gt_idx) -> np.ndarray[3]` | 1-based ranks of the 3 generating SNPs |
| `mean_gt_rank(ranking, gt_idx) -> float` | mean of `gt_ranks` (ideal 2.0) |
| `coverage_at_k(ranking, gt_idx, k) -> bool` | all 3 generating SNPs in top `k` |
| `hit_at_1(report, gt_idx) -> bool` | generating triple is row 1 of the Stage 2 report |
| `summarize_runs(rows: list[dict]) -> pd.DataFrame` | per-dimension means over runs (15 per `P`) |

---

## gatefm.baselines

Each baseline returns either a ranking DataFrame (same schema as `rank_snps`, `q` = baseline score) or a
triple report (same schema as `stage2_triples`), plus wall-clock seconds.

| Function | Type | Notes |
|---|---|---|
| `relieff_rank(X, y, n_neighbors=10)` | SNP ranking | ReliefF (`skrebate`) |
| `multisurf_rank(X, y)` | SNP ranking | MultiSURF (`skrebate`) |
| `transformer_rank(...)` | SNP ranking | wrapper around the Distributed Transformer code (Graça et al. 2024) |
| `bitepi_search(X, y, order=3, bin="BitEpi")` | triple search | calls external BitEpi binary |
| `mdr_search(X, y, order=3)` | triple search | exhaustive MDR |
| `extrapolate_bitepi(P_obs, t_obs, P_new)` | seconds | log–log least squares on `P ∈ {100,500,1000}` |
| `extrapolate_mdr(t_100, P_new)` | seconds | `t_100 · C(P,3) / C(100,3)` |

---

## scripts/

| Script | Purpose | Key args |
|---|---|---|
| `run_simulation.py` | Stage 1 + Stage 2 + metrics on GAMETES | `--config`, `--data-dir`, `--seeds`, `--inits`, `--k 15 30 …` |
| `run_cohort.py` | 10 Stage 1 fits → consensus → Stage 2 + lower-order screen | `--config`, `--data`, `--runs 10`, `--k 30`, `--threshold 2e-4` |
| `run_ablation.py` | GateFM / no budget / Pure FM at `P = 5,000` | `--config`, `--seeds` |
| `run_baselines.py` | ReliefF, MultiSURF, Transformer, BitEpi, MDR | `--method`, `--P` |
| `run_scan.py` | Supplemental Data S1 | `--data`, `--out` |
| `make_figures.py` | Fig 1–2, S1–S2 | `--results` |

Outputs go to `results/<experiment>/<P>/<seed>/<init>/`: `model.pt`, `history.csv`, `config.yaml`,
`ranking.csv`, `triples_k{K}.csv`, `metrics.json`.

---

## Config schema

```yaml
# configs/gametes_p1000.yaml
data:
  kind: gametes            # gametes | table
  pattern: data/gametes/p1000/seed{seed}.txt
model:
  rank: 256
  temperature: 1.0         # [to confirm]
  gamma3: 0.05
  gate_fn: hard_sigmoid
train:
  epochs: 300
  batch_size: 64
  lr: 1.0e-4
  lam: 1.0e-2
  tau_final: 100
  tau_schedule: linear     # [to confirm]
  val_frac: 0.1
  class_weighted: false
stage2:
  k: [15, 30]
  coverage_k: [5, 10, 15, 20, 30, 50, 100]
seeds: [42, 100, 200, 300, 400]
inits: 3
```
