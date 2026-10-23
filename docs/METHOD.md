# Method

This document follows the manuscript (v24). Where the thesis (`master.pdf`) differs, it is noted as **Thesis:**.

## Notation

| Symbol | Meaning |
|---|---|
| `P` | number of SNPs |
| `n` | number of participants |
| `X ∈ {0,1,2}^{n×P}` | genotype matrix (counted-allele dosage) |
| `y ∈ {0,1}^n` | case (1) / control (0) label |
| `r` | third-order latent rank (embedding dimension) |
| `v_{j,k} ∈ ℝ^r` | embedding of SNP `j`, genotype state `k ∈ {0,1,2}` |
| `α_j` | raw gate parameter of SNP `j` |
| `g_j` | gate value of SNP `j` |
| `T` | gate temperature |
| `τ_t` | target soft gate mass (training stage `t`) |
| `λ` | gate-count loss weight |
| `γ₃` | fixed scaling of the third-order term (`0.05`) |
| `z_i`, `ŷ_i` | logit and probability for participant `i` |
| `ρ_{j,k}`, `q_j` | genotype-state score and SNP score |
| `R` | Stage 1 SNP ranking |
| `K`, `S_K` | Stage 2 search width and top-`K` SNP set |
| `T` (set) | ranked triple records |

## Stage 1 — GateFM

### Genotype representation

Genotypes are treated as **categorical**, not linear dosage (the 0/2 labels flip if the counted allele flips).
Each SNP has an embedding table `V_j = [v_{j,0}; v_{j,1}; v_{j,2}] ∈ ℝ^{3×r}`; participant `i` selects row `v_{j, x_ij}`.
Implementation: one `nn.Embedding(3·P, r)` indexed by `3·j + x_ij`.

### Gates

```
g_j = h(α_j / T),   h(t) = min{1, max{0, (t + 1)/2}}      (clipped linear hard-sigmoid)
```

One gate per SNP, shared by its three genotype embeddings.
**Thesis:** `g_j = σ(α_j)` (logistic sigmoid).

Gated embedding component: `a_{ijf} = g_j · v_{j, x_ij, f}`.

### Factorized third-order term (FM trick)

The sum over all distinct triples is the third elementary symmetric polynomial `e₃`, computed from
power sums `p_m = Σ_j a_{ijf}^m`:

```
e_{3,f} = Σ_{u<v<w} a_{iuf} a_{ivf} a_{iwf} = (p₁³ − 3 p₁ p₂ + 2 p₃) / 6
```

Cost per sample: `O(P·r)` instead of `O(P³·r)`.

### Logit

```
z_i = b + γ₃ · Σ_{f=1..r} e_{3,f},     ŷ_i = sigmoid(z_i),     γ₃ = 0.05
```

Stage 1 contains **only** the intercept and the third-order branch (no linear / pairwise terms);
lower-order associations are evaluated separately in Stage 2's lower-order screen.

### Loss

```
L_t = L_BCE + λ · (Σ_j g_j − τ_t)²
```

- `L_BCE`: binary cross-entropy. For the imbalanced cohort, class-weighted with
  `pos_weight = N_control / N_case` (TWB: 49,668 / 10,739 ≈ 4.625).
- `τ_t` is progressively reduced during training to make gates compete (final value ≈ 10% of `P`, see
  [EXPERIMENTS.md](EXPERIMENTS.md)).
- Optimizer: Adam, lr `1e-4`, `λ = 1e-2`, 90/10 train/validation split.

### SNP score and representative genotype

```
ρ_{j,k} = g_j · ‖v_{j,k}‖₂         q_j = max_k ρ_{j,k}         k̂_j = argmax_k ρ_{j,k}
```

SNPs are sorted by decreasing `q_j` → ranking `R = (s₁, …, s_P)`.

### Consensus across repeated fits (cohort)

With `R = 10` independent Stage 1 fits:

```
rank̄(j) = (1/R) Σ_r rank^{(r)}(j)        (lower is better)
k̂_j = mode_r argmax_k ρ^{(r)}_{j,k}
```

Tie-breaking: rank ties → larger cross-run mean `q_j`, then SNP input order.
Genotype-vote ties → larger cross-run mean `ρ` among tied states, then smaller genotype code.

## Stage 2 — top-K exhaustive fixed-pattern test

1. `K' = min(K, P)`; `S_K` = first `K'` SNPs of `R`.
2. Enumerate all `C(K',3)` triples `(u, v, w)`, `u < v < w` in rank order.
3. Pattern `c* = (k̂_u, k̂_v, k̂_w)` is **fixed before testing**. Indicator `R_i = 1[(x_iu, x_iv, x_iw) = c*]`.
4. 2×2 table:

   |  | case | control |
   |---|---|---|
   | matches `c*` | a | b |
   | does not match | c | d |

5. Statistics per triple:
   - Pearson χ² (1 df, **no** Yates correction) → raw `p`
   - `OR = ad / bc`; Wald 95% CI `exp(log OR ± 1.96·√(1/a + 1/b + 1/c + 1/d))`
   - 0.5 added to every cell for OR/CI **only** when any cell is zero
   - Bonferroni `p_Bonf = min(1, m·p)`, `m = C(K',3)` (one test per triple, not `27·C(K',3)`)
6. Sort by raw `p`; ties broken lexicographically by original SNP indices.

| K | triples | pairs |
|---|---|---|
| 15 | 455 | 105 |
| 20 (P=20, K=30) | 1,140 | 190 |
| 30 | 4,060 | 435 |
| 50 | 19,600 | 1,225 |
| 100 | 161,700 | 4,950 |

### Lower-order screen (candidate prioritization)

Threshold `θ = 2 × 10⁻⁴` (clinician-informed, exploratory).

- Single-SNP family: each of the `K` SNPs with its fixed state `k̂_j` → pattern vs complement 2×2 table, Bonferroni `m = K` (30).
- Pairwise family: each of the `C(K,2)` pairs with fixed states → 2×2 table, Bonferroni `m = C(K,2)` (435).
- A triple is **retained** iff `p_Bonf(triple) < θ` **and** all 3 component single-SNP and all 3 component
  pairwise adjusted p values are `≥ θ`.

Stage 1 selection and Stage 2 testing use the same participants, so p values describe selected cohort
patterns; inferential claims require independent validation / a hierarchical model with main, pairwise and
three-way terms.

## Single-SNP descriptive scan

For each SNP: 2×3 Pearson χ² (genotype × case/control), empty genotype columns dropped, df = (#non-empty columns − 1).
Bonferroni over all `P` SNPs (`0.05 / 14,313 = 3.493 × 10⁻⁶`). Output = Supplemental Data S1.

## Evaluation metrics (simulation)

With ground-truth set `S* = {j₁, j₂, j₃}` and `R = 15` runs (5 seeds × 3 inits):

```
mean GT rank = (1/R) Σ_r (1/3) Σ_{j∈S*} rank^{(r)}(j)          ideal = 2.00
Coverage@K   = (1/R) Σ_r 1[S* ⊆ S_K^{(r)}]
Hit@1        = (1/R) Σ_r 1[rank^{(r)}(t*) = 1]                  (completed runs only)
```

Runtime = wall-clock seconds (Stage 1 training + Stage 2 tests for GateFM).

### Baseline runtime extrapolation

- **BitEpi:** least-squares fit `log t = a + b·log P` on completed `P ∈ {100, 500, 1000}`; `t̂_P = exp(a + b log P)`.
- **MDR:** `t̂_P = t_100 · C(P,3) / C(100,3)`.

## Ablation variants (P = 5,000)

| Variant | Loss | Gates |
|---|---|---|
| GateFM | `L_BCE + L_gate` | learnable |
| GateFM no budget | `L_BCE` | learnable |
| Pure FM | `L_BCE` | fixed `g_j = 1` |
