# Analysis scripts

These files were at the repository root before this reorganization. They are not covered by the MIT License, and they have no license yet.

The MIT License covers only `src/proteintalks/` and `docs/proteintalks/` (the former ProteinTalks materials). The license text is `src/proteintalks/LICENCE.md`.

Only the filenames changed, to remove spaces. The script contents are unchanged.

| Current name | Previous name |
| --- | --- |
| `chemmine.R` | `ChemmineR.R` |
| `differential_expression.R` | `Differentially expressed analysis.R` |
| `pertscore.R` | `PertScore calculation.R` |
| `mfuzz.R` | `mFuzz analysis.R` |

## Method notes

The notes below are the text of the former root `README.md`. They are not covered by the MIT License either.

`chemmine.R` generates 881-dimensional drug molecular fingerprints (DMFs) and 54-dimensional drug physicochemical properties (DPPs) for use as input features in machine-learning models.

`differential_expression.R` defines a reusable function for calculating p values and fold changes (FCs). When a t-test cannot be performed, both the p value and FC are set to NA.

`pertscore.R` is run after `differential_expression.R`. It classifies proteins into three categories based on their p values and FCs: upregulated (1), downregulated (−1), and unchanged (0). This classification highlights the most recurrent changes in protein expression.

`mfuzz.R`: one-way analysis of variance (ANOVA) was used to identify differences among samples across different time points (p < 0.05). Within each GS grade, mean protein abundances were z-score normalized and used for fuzzy c-means clustering with the R package Mfuzz.
