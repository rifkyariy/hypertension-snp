# Datasets

Two evaluation settings: a **GAMETES synthetic benchmark** with known generating loci, and the
**Taiwan Biobank (TWB) hypertension cohort** as an application without ground truth.

> **No individual-level data is stored in this repository.** `data/` and `results/` are git-ignored.

---

## 1. GAMETES synthetic benchmark

Generator: GAMETES (Urbanowicz et al., *BioData Min.* 2012;5:16, doi:10.1186/1756-0381-5-16).
Not distributed — regenerate with the settings below (Supplemental Table S1).

| Parameter | Value |
|---|---|
| Epistasis model | pure and strict, third order |
| Generating loci | 3 |
| Penetrance-table output | EDM-1 |
| Observations per dataset | 12,000 (6,000 cases / 6,000 controls) |
| Prevalence (generator setting) | 0.50 — not population hypertension prevalence |
| Heritability | 0.10 |
| Generating-locus MAF | 0.20 each |
| Background-locus MAF | 0.01–0.50 at `P = 20`; 0.01–0.10 at `P ≥ 100` |
| SNP dimensions `P` | 20, 100, 500, 1,000, 5,000, 10,000 |
| Data seeds | 42, 100, 200, 300, 400 |
| Model initializations | 3 per dataset → 15 runs per dimension |
| Genotype coding | counted-allele states 0, 1, 2 |

**Thesis:** background MAF reported as 0.01–0.10 for all dimensions.

A separate, independent GAMETES seed (not one of the five above) was used only for the rank sweep
(Figure A-1); keep it out of the main-result directory.

### File format

GAMETES writes one tab-separated file per dataset:

```
M0P0  M0P1  M0P2  N0  N1  ...  N{P-4}  Class
0     1     2     0   1   ...  0       1
```

- `M0P*` columns = the 3 generating SNPs (ground truth for evaluation).
- `N*` columns = background SNPs.
- `Class` = 1 case / 0 control.

Expected layout:

```
data/gametes/
├── p20/seed42.txt   seed100.txt  seed200.txt  seed300.txt  seed400.txt
├── p100/...
├── p500/...
├── p1000/...
├── p5000/...
├── p10000/...
└── rank_sweep/p1000_seed<independent>.txt
```

---

## 2. Taiwan Biobank hypertension cohort

| Item | Value |
|---|---|
| Source | Taiwan Biobank, population-based cohort (> 100,000 participants), predominantly Han Chinese |
| Access | controlled; apply at <https://www.twbiobank.org.tw> |
| Ethics | Research Ethics Committee, National Taiwan University Hospital, **201606128RINC** |
| Consent | written informed consent at enrollment; de-identified release |
| Phenotype | self-reported hypertension or antihypertensive medication use (self-report sensitivity > 80%, specificity > 90%) |
| Participants | **60,407** — 10,739 cases, 49,668 controls (≈ 1 : 4.6) |
| Same case-control set as | Chang SN et al., *J Hypertens* 2026;44:1812–1821, doi:10.1097/HJH.0000000000004395 |
| Population structure | prior PCA: no obvious stratification vs 1000 Genomes East Asian |

### SNP preprocessing (done upstream by the clinical collaborator)

1. Genotyping QC and whole-genome SNP set.
2. Keep SNPs with single-SNP association `p > 10⁻⁵` (remove major marginal effects).
3. LD pruning.
4. Result: **14,313 SNPs**, `C(14313, 3) ≈ 4.89 × 10¹¹` triples.

### Expected input format

`data/twb/genotypes.parquet` (or `.csv`):

| column | type | description |
|---|---|---|
| `label` | int {0,1} | hypertension case = 1 |
| `<snp_id>` × 14,313 | int8 {0,1,2} | counted-allele dosage; no missing values |

SNP identifiers are either rsIDs (`rs6428587`) or `chrom:pos:ref:alt` (`17:5913154:GT:G`).
Optional `data/twb/snp_positions.csv` with `snp_id, chrom, pos` for Supplemental Data S1.

### Analysis protocol

- 10 independent Stage 1 fits, class-weighted BCE (`pos_weight = 49,668 / 10,739`).
- Consensus = mean complete-list rank across runs; top 30 SNPs.
- Representative genotype = cross-run majority vote of `argmax_k ρ_{j,k}`.
- Stage 2: 4,060 triples, Bonferroni `m = 4,060`; lower-order screen with `m = 30` (single) and `m = 435` (pairs); threshold `2 × 10⁻⁴`.

---

## 3. Outputs that may be shared

| File | Contents | Shareable |
|---|---|---|
| `supplemental_data_S1.csv` | per-SNP `snp_id, chrom, pos, chi2, df, p, p_bonf, neg_log10_p` | yes (summary statistics) |
| `triples_k30.csv` (cohort) | triple records, counts, p, OR | summary only — check TWB release rules for small cell counts |
| `ranking.csv` (cohort) | SNP ranking and scores | yes |
| anything with participant rows | — | **no** |

### Reference values (Supplemental Data S1)

- p range `2.135 × 10⁻²³` – `0.989`, median `0.0164`
- 11,234 SNPs with `p < 0.05`
- 155 SNPs below Bonferroni `3.493 × 10⁻⁶`

### Reference triple records (Table 4, preliminary exploratory analysis)

| Rank | SNPs | Pattern | Raw p | Bonferroni p | OR (95% CI) |
|---|---|---|---|---|---|
| 1 | 17:5913154:GT:G; rs6428587; rs150290208 | 0, 2, 1 | 2.961 × 10⁻⁹ | 1.202 × 10⁻⁵ | 1.79 (1.47–2.18) |
| 2 | rs78137556; rs1436138; rs1410098 | 0, 1, 0 | 3.533 × 10⁻⁹ | 1.434 × 10⁻⁵ | 1.22 (1.14–1.30) |
| 3 | rs78137556; rs1436138; rs6843673 | 0, 1, 0 | 8.346 × 10⁻⁹ | 3.388 × 10⁻⁵ | 1.20 (1.13–1.27) |
| 4 | rs4714575; 17:67395857:A:G; rs1410098 | 0, 0, 0 | 2.235 × 10⁻⁸ | 9.074 × 10⁻⁵ | 1.23 (1.14–1.32) |
| 5 | 17:5913154:GT:G; rs1410098; rs150290208 | 2, 0, 1 | 2.390 × 10⁻⁸ | 9.703 × 10⁻⁵ | 1.42 (1.26–1.61) |

Candidate genes for record 1 (assembly to confirm): WSCD1 (chr17), ZNF326 (chr1), STAB2 (chr12).
These are exploratory; they describe selected cohort patterns and require independent validation.
