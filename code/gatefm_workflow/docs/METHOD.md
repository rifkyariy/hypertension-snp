# Method specification

This document records the **implemented GateFM reference workflow**. Experiment settings and initialization seeds are stored in `configs/config_thesis.json` and `configs/experiment_plan.json`.

## Input and model

Each sample contains P SNP genotypes in {0, 1, 2} and a binary outcome. For SNP j and genotype k, the model learns a third-order embedding vector E[j,k] of rank r. Each SNP shares one gate across its three genotype states.

The configured hard-sigmoid gate is

```text
g[j] = clip((alpha[j] / T + 1) / 2, 0, 1),  T = 0.7
```

For each observation, the genotype-specific gated embedding for SNP j is u[j] = g[j] * E[j, genotype[j]]. The logit is an intercept plus a scaled third-order FM term with gamma3 = 0.05. For each latent dimension, the third-order sum is computed by the Newton-Girard identity:

```text
sum_{u<v<w} u[u] u[v] u[w]
  = ((sum u)^3 - 3 (sum u)(sum u^2) + 2 sum u^3) / 6
```

This is evaluated over latent dimensions, avoiding explicit enumeration of every SNP triple during model training.

## Training and Stage 1 ranking

Training uses weighted binary cross entropy with logits and a squared soft gate-count constraint (lambda = 0.001). The target gate mass is annealed over the first 90% of training epochs according to the resolved dimension-specific configuration.

The final-epoch checkpoint is used for SNP scoring. The score of SNP j is

```text
rho[j,k] = g[j] * ||E[j,k]||_2
q[j]     = max_{k in {0,1,2}} rho[j,k]
```

Sort by descending q to obtain the complete SNP ranking. The representative genotype is the k giving the maximum rho. Outputs include the resolved configuration, seeds, dataset hashes, model checkpoint, all SNP scores, rankings and representative genotype states.

For cohort consensus over multiple runs, compute the mean rank over each complete SNP list. Representative genotype states are selected by majority vote, breaking ties by mean rho and then smaller genotype code.

## Stage 2 association analysis

Let K = min(30, P). Stage 2 enumerates all C(K,3) unordered SNP triplets from the selected candidates.

For each triplet, its three representative genotype states are **fixed before association testing**. The method constructs a 2x2 table comparing the exact joint genotype pattern with all other genotype configurations and calculates a Pearson chi-square p value, an odds ratio and a 95% confidence interval.

The Pearson test uses no Yates correction. A 0.5 continuity adjustment applies to odds-ratio/interval calculations when a contingency-table cell is zero. Bonferroni adjustment uses the actual number of tested triplets. Ties in raw p value use the original SNP indices.

In the synthetic experiment, Stage 2 is evaluated separately for **each Stage 1 model fit**, using that fit's own SNP ranking and genotype assignments. For the cohort workflow, Stage 2 uses the consensus from 10 independently initialized Stage 1 fits.

## Lower-order screening

For the selected K SNPs, single-SNP and SNP-pair screens use the same preselected genotype states. At K=30 the screening set has 30 single-SNP and 435 pairwise fixed-pattern-versus-complement 2x2 tests. The respective Bonferroni families use the actual single/pair test counts. Screening thresholds are configurable and recorded.

These analyses support **hypothesis prioritization and candidate review**; a fixed-pattern association test alone is not a formal isolation of a three-way interaction term.

## Frozen experiment settings

- Synthetic SNP dimensions: 20, 100, 500, 1,000, 5,000, 10,000.
- Dataset seeds: 42, 100, 200, 300, 400.
- Model initialization seeds for each synthetic dataset: 3484015997, 3484015998, 3484015999.
- Cohort model initialization seeds: 3484015997 through 3484016006 (10 fits).
- Training/validation split: stratified 90%/10% with split seed 42.
- Checkpoint selection: final epoch.

Use the JSON configurations as the source of truth for dimension-specific rank, learning rate, epochs, batch size and target gate mass. Changes to model initialization or execution environment can affect numerical outcomes; comparisons also require consistent dataset generation, scoring, candidate selection and statistical procedures.