# ProteinTalks

ProteinTalks is a neural ordinary differential equation model for perturbation-induced proteomic dynamics and drug efficacy or combination synergy. The Python package is `src/proteintalks`. The R analysis scripts are in `analysis/`.

## Setup

Python 3.10.6. From the repository root:

```bash
uv python install 3.10.6
uv sync
uv run ProteinTalks --help
```

Dependency versions match the former `ProteinTalks/requirements.txt`. The PyPI archive for torchdyn 1.0.4 omits the package source, so the lockfile installs that version from its Git tag. `setuptools>=69,<81` is included because the locked `lightning_fabric` still imports `pkg_resources`.

## Data and checkpoint

Download the public dataset into `data/raw`:

```bash
uv run python scripts/download_data.py
```

The trained weights shipped with this repository are `ckpts/best_checkpoint.pth`. Training and prediction arguments for the full dataset are in `docs/proteintalks/README.md`.

## Tests

```bash
uv run pytest
```

The tests check `--help`, then run prediction and one epoch of training on a few experiments from the downloaded dataset. The subset written for those runs is under `tests/input`. Logs and results go to `tests/output`.

## License

This repository does not have one license for every file.

The former ProteinTalks materials keep their MIT License, Copyright (c) 2024 yonggeli66. The license text is `src/proteintalks/LICENCE.md`. It covers only:

- `src/proteintalks/`
- `docs/proteintalks/`

The R scripts in `analysis/`, and the method notes below, were at the repository root before this reorganization. They are not covered by that MIT License, and they have no license yet.

`pyproject.toml` and `uv.lock` install the Python dependencies. They do not place the unlicensed files under the MIT License.

## Analysis scripts

These notes are the text of the original root README. Current filenames are in parentheses. See `analysis/README.md` for the old-to-new name map.

`analysis/chemmine.R` (formerly `ChemmineR.R`) generates 881-dimensional drug molecular fingerprints (DMFs) and 54-dimensional drug physicochemical properties (DPPs) for use as input features in machine-learning models.

`analysis/differential_expression.R` (formerly `Differentially expressed analysis.R`) defines a reusable function for calculating p values and fold changes (FCs). When a t-test cannot be performed, both the p value and FC are set to NA.

`analysis/pertscore.R` (formerly `PertScore calculation.R`) is run after `analysis/differential_expression.R`. It classifies proteins into three categories based on their p values and FCs: upregulated (1), downregulated (−1), and unchanged (0). This classification highlights the most recurrent changes in protein expression.

`analysis/mfuzz.R` (formerly `mFuzz analysis.R`): one-way analysis of variance (ANOVA) was used to identify differences among samples across different time points (p < 0.05). Within each GS grade, mean protein abundances were z-score normalized and used for fuzzy c-means clustering with the R package Mfuzz.
