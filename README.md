# PTV1
## Paper：《AI empowers perturbation proteomics for drug response and synergy prediction in breast cancer cells》
Here provide some analysis core code.

#### ChemmineR.R
For generating 881-dimensional drug molecular fingerprints (DMFs) and 55-dimensional drug physicochemical properties (DPPs) as the input features of machine learning models.
#### Differentially expressed analysis.R
Encapsulate the calculation of p-value and fold change in a function for code reuse; when T-test cannot be performed, both P-value and FC value are equal to NA.
#### PertScore calculation.R
It will be transfer after 'Differentially expressed analysis', 
#### mFuzz analysis.R
