## One-way analysis of variance (ANOVA) 
anova_func <- function(exp, a, b, c, d){
  type<- c(rep("0hrs", a),rep("6hrs", b),rep("24hrs", c),rep("48hrs", d) ) 
  Pvalue = c()
  for(i in 1:nrow(exp)){
    if (  sum(!(is.na(exp[i,  1:a ]))) <2 | sum(!(is.na(exp[i,  (a+1):(a+b)]))) <2 | sum(!(is.na(exp[i,  (a+b+1):(a+b+c)]))) <2 | sum(!(is.na(exp[i,  (a+b+c+1):(a+b+c+d)]))) <2){
      Pvalue = c(Pvalue, NA)
      next
    }
    y = try(aov(as.numeric(exp[i,  ]) ~ type), silent=FALSE)
    if('try-error' %in% class(y))
    {
      Pvalue = c(Pvalue, NA)
    }else{
      y = aov(as.numeric(exp[i,  ]) ~ type)
      if (dim(summary(y)[[1]])[2] != 5){
        Pvalue[i]<- NA
        next
      }
      Pvalue[i]<- summary(y)[[1]][,5][1]
    }
  }
  FDR=p.adjust(Pvalue, "BH")
  out<-cbind(exp, Pvalue, FDR )
  out
}

## Mfuzz
library(Mfuzz)
library(RColorBrewer)
mycol <- c("cyan","yellow","orangered")
mycolor <- colorRampPalette(mycol)(100)
out = data.frame(anova_func(dat, length(sample0), length(sample6), length(sample24), length(sample48)))
genesymbol = c()
for (gene in rownames(out)){
  genesymbol = c(genesymbol, df_pre[gene, "geneSymbol"])
}
out$geneSymbol = genesymbol
write.csv(out[c("geneSymbol", "Pvalue", "FDR")], paste0("~/prottalk/code/vulnerableGeneComparison/ANOVA_Mfuzz/Basal", basal, "_", sb,  "_", cell, "_anova.csv"))
out = out[!is.na(out$Pvalue), ]
out = subset(out, out$Pvalue<0.05)
if (nrow(out) < 3){next}
mfuzzInput = data.frame(hrs0 = rowMeans(out[1:length(sample0)], na.rm = T), 
                        hrs6 = rowMeans(out[ (1+length(sample0)): (length(sample6)+length(sample0))], na.rm = T),
                        hrs24 = rowMeans(out[ (1+length(sample6)+length(sample0)):(length(sample6)+length(sample0) + length(sample24))], na.rm = T),
                        hrs48 = rowMeans(out[ (1+length(sample6)+length(sample0) + length(sample24)):(length(sample0) +length(sample6)+ length(sample24)+length(sample48))], na.rm = T))

set.seed(2023)
mat <- as.matrix(mfuzzInput)
dt <- new("ExpressionSet", exprs = mat)
# dt <- filter.NA(dt, thres=0.25)
dt.f <- fill.NA(dt , mode="mean")
tmp <- filter.std(dt.f, min.std=0)
dt.s <- standardise(tmp)
m1 <- mestimate(dt.s)
cl <- mfuzz(dt.s, c = 3, m = m1 )
pdf(paste0("~/prottalk/code/vulnerableGeneComparison/ANOVA_Mfuzz/Basal", basal, "_", sb, "_", cell, "_Mfuzz.pdf"), width = 10)
mfuzz.plot(dt.s,cl, mfrow=c(2,4),
           new.window= FALSE,
           time.labels = colnames(dt.s),
           colo = mycolor)
dev.off()
protein_cluster <- cl$cluster
protein_cluster <- cbind(mat[names(protein_cluster), ], protein_cluster)
write.csv(protein_cluster, paste0("~/prottalk/code/vulnerableGeneComparison/ANOVA_Mfuzz/Basal", basal, "_", sb, "_", cell, "_MfuzzCluster.csv"))
member = cl$membership
write.csv(member, paste0("~/prottalk/code/vulnerableGeneComparison/ANOVA_Mfuzz/Basal", basal, "_", sb, "_", cell, "_MfuzzClusterMembership.csv"))
