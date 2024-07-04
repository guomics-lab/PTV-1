# PTV1
## Paper：
Here provide some analysis core code.

#### ChemmineR.R
For generating 881-dimensional drug molecular fingerprints (DMFs) and 55-dimensional drug physicochemical properties (DPPs) as the input features of machine learning models.
#### Differentially expressed analysis.R
Encapsulate the calculation of p-value and fold change(FC) in a function for code reuse; when T-test cannot be performed, both P-value and FC value are equal to NA.
#### PertScore calculation.R
It will be transfer after 'Differentially expressed analysis'. Divide proteins into three categories according to p-value and FC, upregulation: 1、 downregulation:-1 and None:0.
#### mFuzz analysis.R
