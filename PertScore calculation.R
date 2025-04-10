# PertScore calculate after Differentially expressed analysis
ttest = function(exp, a, b, c, d){
  Pvalue = c()
  FC = c()
  logFC = c()
  for(i in 1:nrow(exp)){
    # un-impute
    y = try(t.test(as.numeric(exp[i, a:b]), as.numeric(exp[i, c:d ])),silent=FALSE)
    if('try-error' %in% class(y))
    {
      Pvalue = c(Pvalue, NA)
    }else{
      y = t.test(as.numeric(exp[i, a:b]), as.numeric(exp[i, c:d ]))
      Pvalue = c(Pvalue, y$p.value)
    }
    pre = rowMeans( exp[i, a:b] , na.rm = TRUE)
    post = rowMeans( exp[i, c:d ] , na.rm = TRUE)
    FC =c(FC, pre/post) 
    logFC = c(logFC, log2(pre/post))
  }
  FDR=p.adjust(Pvalue, "BH")
  out<-cbind(exp, Pvalue, FDR, FC, logFC)
  out
}

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
