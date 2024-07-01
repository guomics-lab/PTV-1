# PertScore calculate after Differentially expressed analysis
ttest_out = ttest(dat, a, b,c,d)
sig = c()
for (i in 1:nrow(ttest_out)){
  if (is.na(ttest_out[i, "Pvalue"]) ){
    sig = c(sig, 0)
  }else if (ttest_out[i, "Pvalue"]<0.05 & ttest_out[i, "logFC"]>log2(1.2)){
    sig = c(sig, 1)
  }else if (ttest_out[i, "Pvalue"]<0.05 & ttest_out[i, "logFC"]< (-log2(1.2))){
    sig = c(sig, -1)
  }else{
    sig = c(sig, 0)
  }
}
ttest_out$UpDown = sig
