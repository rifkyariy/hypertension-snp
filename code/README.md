# GateFM source code

This directory contains the executable GateFM research implementation. The full runnable project is located in [gatefm_workflow/](gatefm_workflow/README.md). All source files and their required local configurations, examples, plotting tools and checks are kept inside this code/ directory so the existing manuscript repository layout can remain unchanged.

## Quick start

From the root of rifkyariy/hypertension-snp on Windows, with Python 3.11:

~~~powershell
cd code\gatefm_workflow
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_workflow.py smoke --output outputs\toy_001 --execute
~~~

On Linux/macOS, use Python 3.11, install requirements.txt, and run the following from code/gatefm_workflow/:

~~~bash
python run_workflow.py smoke --output outputs/toy_001 --execute
~~~

Outputs and plots are written to the chosen output directory. For new executions, choose a fresh output directory; existing results are not overwritten. See the [full usage and benchmark guide](gatefm_workflow/README.md).

## Scope and reproducibility

- gatefm_workflow/code/: Stage 1 model/training, independent-fit consensus, fixed-pattern Stage 2, single/pair screening, GAMETES generation/evaluation and numerical checks.
- gatefm_workflow/configs/: frozen training and experiment settings.
- gatefm_workflow/scripts/, examples/, tests/: figures, examples and regression tests.
- The ten-stage toy workflow provides a rapid functional check. The default GAMETES plan comprises 30 generated datasets and 90 independent model fits; optional subset runs are labeled exploratory.
- Participant-level datasets, generated model checkpoints, derived cohort outputs, and historical runs are managed separately within authorized research environments. Independent comparator summaries may be supplied to the figure-export workflow.
- The frozen executable configuration pins the squared gate-count penalty at lambda = 0.001. The thesis appendix and existing collaborator method notes also document a lambda = 0.01 setting; cite and retain the configuration used for each run.
The root-level manuscript documents and figures are maintained separately by the manuscript collaborators. The source implementation entry points and options are defined by the files in this directory.
## Authorized clinical cohort

For an approved genotype CSV or TSV table, use gatefm_workflow/code/validate_cohort_input.py first. The human workflow supports explicit --label-column, --case-value and --control-value arguments so the input schema and case/control direction are not inferred. Optional --expected-samples and --expected-snps checks can assert the expected cohort size. The human command runs this preflight before starting the ten independent fits, and writes run.log / workflow_status.json on execution failure. See gatefm_workflow/README.md for the full commands.

Human inputs, derived candidate tables, genotype rankings, model checkpoints, and analysis output stay out of Git and require authorized local access. Raw VCF/GEN processing and participant-level QC are upstream responsibilities of the data custodian, not embedded in this research handoff.
