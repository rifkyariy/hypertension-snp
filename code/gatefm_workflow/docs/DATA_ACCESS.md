# Data access and responsible distribution

This repository is intended to hold **source code and public, non-human examples only**. It does not distribute manuscript drafts, clinical participant-level genotypes/phenotypes, original cohort VCF/GEN files, clinical scan tables, model checkpoints trained on participants or historical research-run outputs.

- Only authorized data custodians may run the human-cohort workflow on approved inputs. Repository access does **not** confer research-data permission.
- Keep cohort inputs and generated outputs out of Git. The `data/` and `outputs/` directories are ignored by Git and created locally by the workflow as needed. Never bypass the exclusions with `git add -f`.
- Outputs derived from sensitive data (including saved models, rankings, candidate tables and metadata) require a separate disclosure review before sharing.
- This repository is publicly viewable, but no open-source license has been granted. The right to redistribute code, figures or data has not yet been established. An open-source license should not be added without the authors' authorization.
- Researchers must comply with applicable institutional protocols, consent and data-use terms before additional cohort analyses or public releases.

See [Method specification](METHOD.md) for the model and statistical analysis.
