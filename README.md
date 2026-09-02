The main analysis scripts described in the Methods section are provided below.

ChemmineR.R
This script generates 881-dimensional drug molecular fingerprints (DMFs) and 54-dimensional drug physicochemical properties (DPPs) for use as input features in machine-learning models.

Differentially expressed analysis.R
This script defines a reusable function for calculating p values and fold changes (FCs). When a t-test cannot be performed, both the p value and FC are set to NA.

PertScore calculation.R
This script is run after Differentially expressed analysis.R. It classifies proteins into three categories based on their p values and FCs: upregulated (1), downregulated (−1), and unchanged (0). This classification highlights the most recurrent changes in protein expression.

mFuzz analysis.R
One-way analysis of variance (ANOVA) was used to identify differences among samples across different time points (p < 0.05). Within each GS grade, mean protein abundances were z-score normalized and used for fuzzy c-means clustering with the R package Mfuzz.
