# PTV1
## Paper：
Here provide some main codes in analysis which mentioned in 'Method' section.

#### ChemmineR.R
For generating 881-dimensional drug molecular fingerprints (DMFs) and 55-dimensional drug physicochemical properties (DPPs) as the input features of machine learning models.
#### Differentially expressed analysis.R
Encapsulate the calculation of p-value and fold change(FC) in a function for code reuse; when T-test cannot be performed, both P-value and FC value are equal to NA.
#### PertScore calculation.R
It will be transfer after 'Differentially expressed analysis'. Divide proteins into three categories according to p-value and FC, upregulation: 1、 downregulation:-1 and None:0. Aiming to highlight the most recurrent protein expression changes.
#### mFuzz analysis.R
One-way analysis of variance (ANOVA) was used to determine differences between samples treated with different time points (p-value < 0.05). The average normalized protein quantities by z-score in each GS grade were used for fuzzy c-means clustering with the R package Mfuzz.

#### ppODE
ppODE:https://github.com/guomics-lab/PTV-1/tree/main/ProteinTalks
