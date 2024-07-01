# describe the physicochemical properties of small molecule drugs
# https://www.bioconductor.org/packages/devel/bioc/vignettes/ChemmineR/inst/doc/ChemmineR.html
BiocManager::install("ChemmineR")
BiocManager::install("ChemmineOB")
library("ChemmineR")
lines <- readLines("~/Drugs_SMILE20231004.txt")
smiles = c()
i =1
for(line in lines) {
  print(i)
  sdfset = smiles2sdf(line)
  #print(header(sdfset[[1]]))
  smiles = c(smiles, line)
  i = i+1
}
smiles
sdfset = smiles2sdf(smiles)
result1 = sdfset# [[1]]
listCMTools()
#JoeLib Descriptors
job2 <- launchCMTool('PubChem Fingerprint Search', result1, 'Similarity Cutoff'= 0.95, 'Max Compounds Returned' = 10 )
result2 <- result(job2)
length(result2)
cid = c()
for (i in 1:length(result2)){
  if (i%%10 == 1){
    cid = c(cid, result2[i])
  }
}

job3 <- launchCMTool("pubchemID2SDF", cid)
result3 <- result(job3)
#header( result3[[1]] )

test = fp2bit(result3 )
fpmatrix = test@fpma
write.csv(fpmatrix, "~/fingerPrint.csv") # 881-dimensional drug molecular fingerprints (DMFs)

job4 <- launchCMTool("OpenBabel Descriptors", result3)
result4 <- result(job4)
#result4[1,]

jobTest = launchCMTool("JoeLib Descriptors", result3)
result5 = result(jobTest)
#result5[1,]
propma <- data.frame(MF=MF(result3, addH=FALSE), MW=MW(result3, addH=FALSE),
                     Ncharges=sapply(bonds(result3, type="charge"), length),
                     atomcountMA(result3, addH=FALSE),
                     groups(result3, type="countMA"),
                     rings(result3, upper=6, type="count", arom=TRUE))

dim(propma)
rownames(propma) = rownames(result4)
df = cbind(result4 , result5 )
df = cbind(df, propma)
dim(df)
df1 = df[intersect(colnames(df), colnames(example))]
write.csv(df1, "~/phychem.csv") # 55-dimensional drug physicochemical properties (DPPs)
