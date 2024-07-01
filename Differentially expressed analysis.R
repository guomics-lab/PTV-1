# this function is single drug datasets, a is the starting index of 1st group, b is the ending index of 1st group, c is the starting index of 2nd group, d is the ending index of 2nd group
ttest <- function(exp, a, b, c, d){
  Pvalue <- c()
  FC = c()
  logFC = c()
  for(i in 1:nrow(exp)){
    y = try(t.test(as.numeric(exp[i, a:b]), as.numeric(exp[i, c:d ])), silent=FALSE)
    if('try-error' %in% class(y))
    {
      Pvalue = c(Pvalue, NA)
      FC = c(FC, NA)
      logFC = c(logFC, NA)
    }else{
      y = t.test(as.numeric(exp[i, a:b]), as.numeric(exp[i, c:d ]))
      Pvalue = c(Pvalue, y$p.value)
      pre = rowMeans(exp[i, a:b] , na.rm = TRUE)
      post = rowMeans(exp[i, c:d ] , na.rm = TRUE)
      FC =c(FC, pre-post) 
      logFC = c(logFC, pre-post)
    }
  }
  FDR=p.adjust(Pvalue, "BH")
  out<-cbind(exp, Pvalue, FDR, FC, logFC)
  out
}
# this function is combine drug datasets, a is the starting index of 1st group, b is the ending index of 1st group, c is the starting index of 2nd group, d is the ending index of 2nd group
ttest.D <- function(exp, a, b, c, d){
  Pvalue <- c()
  FC = c()
  logFC = c()
  for(i in 1:nrow(exp)){
    y = try(t.test(as.numeric(exp[i, a:b]), as.numeric(exp[i, c:d ])), silent=FALSE)
    if('try-error' %in% class(y))
    {
      Pvalue = c(Pvalue, NA)
      FC = c(FC, NA)
      logFC = c(logFC, NA)
    }else{
      y = t.test(as.numeric(exp[i, a:b]), as.numeric(exp[i, c:d ]))
      Pvalue = c(Pvalue, y$p.value)
      pre = rowMeans(exp[i, a:b] , na.rm = TRUE)
      post = rowMeans(exp[i, c:d ] , na.rm = TRUE)
      FC =c(FC, pre/post) 
      logFC = c(logFC, log2(pre/post))
    }
  }
  FDR=p.adjust(Pvalue, "BH")
  out<-cbind(exp, Pvalue, FDR, FC, logFC)
  out
}
