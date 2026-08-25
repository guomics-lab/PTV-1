rm(list = ls());gc()
library(openxlsx)
library(reshape2)
library(pheatmap)

#### B ####
df1 = read.xlsx("D:/chh/2023workProject/prottalk/code/kinase_target/target_anova/up.xlsx", sheet = "heatmap")
colnames(df1)
df1 = df1[1:3]
colnames(df1)  = c("Proteins","Description","LogP")
colnames(df1)
df2 = read.xlsx("D:/chh/2023workProject/prottalk/code/differ analysis20230822/subtype/ANOVA/Subtype.xlsx", sheet = "Sheet2")
colnames(df2)
df2 = df2[1:3]
df = rbind(df1, df2)
temp = dcast(df,Proteins~Description, value.var = "LogP")
rownames(temp ) = temp$Proteins
temp = temp[2:ncol(temp)]

rownames(temp)
temp = temp[c("top_C1", "top_C2", "top_C3", "HDAC_C1", "HDAC_C2/3", "CDK2_C1", "CDK2_C2", "CDK2_C3" , "egfr", "mek", "pi3k","src"), ]
rownames(temp) = c("TOP_C1", "TOP_C2", "TOP_C3", "HDAC_C1", "HDAC_C2/3", "CDK2_C1", "CDK2_C2", "CDK2_C3" , "EGFR", "MEK", "PI3K", "SRC")
dim(temp)
#temp[is.na(temp)] = 0
# temp[is.na(temp)] = min(df$LogP)
pheatmap(t(temp), scale = "none",
         # color = colorRampPalette(c("blue", "white","red" ))(100),
         # color = colorRampPalette(c( "#689A62", "#D6E4D5" , "#30472D"))(100),
         # color = colorRampPalette(c("#4A9C4A" , "#72A26E", "#BEE0BE" ))(100),
         # color = colorRampPalette(c("#367236" , "#78BF78", "#BEE0BE" ))(100),
         # color = colorRampPalette(c("#285528" , "#78BF78", "#BEE0BE" ))(100),
         color = colorRampPalette(c("#426E38", "#4A9C4A", "#A7D5A7"  ))(100),
         cluster_rows = F,
         cluster_cols = F,
         cellheight = 10,
         cellwidth = 10,
         fontsize_row = 9,
         angle_col =  90 ,#c("270", "0", "45", "90", "315")
         na_col = "white" )

pheatmap(t(temp), scale = "none",
         color = colorRampPalette(c("#D17101", "#FEA134", "#FFC684"  ))(100),
         cluster_rows = F,
         cluster_cols = F,
         cellheight = 10, 
         cellwidth = 10,
         fontsize_row = 9,
         angle_col =  90 ,#c("270", "0", "45", "90", "315")
         na_col = "white" )
#### C ####
matrixB = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/matrix.csv", row.names = 1)
matrixB[1:3, 1:3]
drug_cell_drugs = read.xlsx("D:/chh/2023workProject/prottalk/database/drug_cell_inhi24_IC50_MOA_combo_20231113-1549.xlsx" )
sampleInfoB = read.csv("D:/chh/2023workProject/prottalk/database/iDrug_sample_final_info20231207_NY.csv")
sampleInfoB = subset(sampleInfoB, Sample_ID %in% rownames(matrixB))
sampleInfoB = sampleInfoB[order(sampleInfoB$pert_id, sampleInfoB$protein_plate, sampleInfoB$pert_time), ]
unique(sampleInfoB$protein_plate)
sampleInfoB$pct = paste0(sampleInfoB$pert_id, '_', sampleInfoB$protein_plate, '_', sampleInfoB$pert_time)

df_pre = read.csv("//172.16.13.136/share/members/chenghonghan/PTV1/proof_check20260608/fig2/protein_gene.csv", row.names = 1)
head(df_pre )


sb = "cdk inhibitor"
cluster1 = c("MCAT", "ASPH", "LRRC59", "GNB4", "NECTIN2", "PACSIN2", "DHRS4", "NOMO2", "AIDA", "UQCRB", "NQO1", "SAP30BP", "SPTBN2", "ZNF326", "ARHGAP29", "TMX1", 
             "GAR1", "APOC3", "UNC119B", "SURF2", "KRT1", "RHEB", "SNRPD1", "PRKAR2A", "SDHAF2", "IGFBP2", "RBM4B", "EIF2AK4", "MFGE8", "ZMPSTE24", "GRB2", "PIGU", 
             "MCU", "STAT6", "FTH1", "GOLGB1", "WRNIP1", "PSMD13", "BCAM", "CLCN7", "SDF2")
cluster2 = c("RHOG", "DDX6", "FDXR", "UBAP2L", "YAP1", "PUS1", "CAST", "POLR2F", "CBR1", "ATG5", "UTP4", "S100A10", "SH3PXD2A", "TUBB4B", "QKI", "CAD",
             "PPP2R5D", "OGA", "MRPL3", "NRDC", "GTF3C4", "RFC3")
print(length(cluster1), length(cluster2))
dd = drug_cell_drugs[drug_cell_drugs$Subtype2 %in% sb , 'Pert_ID']
dd

sample0 = sampleInfoB[sampleInfoB$pert_time %in% 0, 'Sample_ID']

temp = sampleInfoB[sampleInfoB$pert_id %in% dd, ]

sample62448_Y = sampleInfoB[sampleInfoB$pert_id %in% dd & sampleInfoB$pert_time>0 & sampleInfoB$NY %in% 'Y', 'Sample_ID']
sample62448_N = sampleInfoB[sampleInfoB$pert_id %in% dd & sampleInfoB$pert_time>0 & sampleInfoB$NY %in% 'N', 'Sample_ID']
print(len(sample6_24_48_Y), len(sample6_24_48_N))

cl = 1
pros = cluster1
uni = uniprotID_geneSymbol[uniprotID_geneSymbol$GeneSymbol %in% pros, 1]
uni

## Y
temp_mat = matrixB[c(sample62448_Y, sample0), uni]
narow = rowSums(!is.na(temp_mat))
nacol = colSums(!is.na(temp_mat))
temp_mat = temp_mat[narow>0, nacol>0]
temp_info = sampleInfoB[sampleInfoB$Sample_ID %in% rownames(temp_mat), ]
nrow(temp_info)

techrep = sum(grepl('rep', temp_info$Sample_ID))
temp_info = subset(temp_info, !grepl('rep', temp_info$Sample_ID))
temp1 = temp_info %>% distinct(pct, .keep_all = T)
biorep = nrow(temp_info)-nrow(temp1)
print(paste0(c('cluster1 Y techrep:', techrep, ' biorep:', biorep), collapse = ''))

## N
temp_mat = matrixB[c(sample62448_N, sample0), uni]
narow = rowSums(!is.na(temp_mat))
nacol = colSums(!is.na(temp_mat))
temp_mat = temp_mat[narow>0, nacol>0]
temp_info = sampleInfoB[sampleInfoB$Sample_ID %in% rownames(temp_mat), ]
nrow(temp_info)

techrep = sum(grepl('rep', temp_info$Sample_ID))
temp_info = subset(temp_info, !grepl('rep', temp_info$Sample_ID))
temp1 = temp_info %>% distinct(pct, .keep_all = T)
biorep = nrow(temp_info)-nrow(temp1)
print(paste0(c('cluster1 N techrep:', techrep, ' biorep:', biorep), collapse = ''))


cl = 2
pros = cluster2
uni = uniprotID_geneSymbol[uniprotID_geneSymbol$GeneSymbol %in% pros, 1]
uni

## Y
temp_mat = matrixB[c(sample62448_Y, sample0), uni]
narow = rowSums(!is.na(temp_mat))
nacol = colSums(!is.na(temp_mat))
temp_mat = temp_mat[narow>0, nacol>0]
temp_info = sampleInfoB[sampleInfoB$Sample_ID %in% rownames(temp_mat), ]
nrow(temp_info)

techrep = sum(grepl('rep', temp_info$Sample_ID))
temp_info = subset(temp_info, !grepl('rep', temp_info$Sample_ID))
temp1 = temp_info %>% distinct(pct, .keep_all = T)
biorep = nrow(temp_info)-nrow(temp1)
print(paste0(c('cluster2 Y techrep:', techrep, ' biorep:', biorep), collapse = ''))

## N
temp_mat = matrixB[c(sample62448_N, sample0), uni]
narow = rowSums(!is.na(temp_mat))
nacol = colSums(!is.na(temp_mat))
temp_mat = temp_mat[narow>0, nacol>0]
temp_info = sampleInfoB[sampleInfoB$Sample_ID %in% rownames(temp_mat), ]
nrow(temp_info)

techrep = sum(grepl('rep', temp_info$Sample_ID))
temp_info = subset(temp_info, !grepl('rep', temp_info$Sample_ID))
temp1 = temp_info %>% distinct(pct, .keep_all = T)
biorep = nrow(temp_info)-nrow(temp1)
print(paste0(c('cluster2 N techrep:', techrep, ' biorep:', biorep), collapse = ''))



#### D ####
matrixB = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/matrix.csv", row.names = 1)
matrixB[1:3, 1:3]
drug_cell_drugs = read.xlsx("D:/chh/2023workProject/prottalk/database/drug_cell_inhi24_IC50_MOA_combo_20231113-1549.xlsx" )
sampleInfoB = read.csv("D:/chh/2023workProject/prottalk/database/iDrug_sample_final_info20231207_NY.csv")
sampleInfoB = subset(sampleInfoB, Sample_ID %in% rownames(matrixB))
sampleInfoB = sampleInfoB[order(sampleInfoB$pert_id, sampleInfoB$protein_plate, sampleInfoB$pert_time), ]
unique(sampleInfoB$protein_plate)
sampleInfoB$pct = paste0(sampleInfoB$pert_id, '_', sampleInfoB$protein_plate, '_', sampleInfoB$pert_time)

unique(sampleInfoB$NY)
uniprotID_geneSymbol = read.xlsx("D:/chh/2023workProject/prottalk/database/uniprotID_geneSymbol_geneID.xlsx")

pros = uniprotID_geneSymbol[uniprotID_geneSymbol$GeneSymbol %in% c('ATG3', 'BTF3', 'PAK1', 'NDE1', 'ELAVL2', 'CKS2'), 1]
pros
drugpros = list()
drugpros[['#38']] = c('Q9NXR1', 'Q13153')

drugpros[['#53']] = c('P20290')
drugpros[['#8']] = c('P33552')
drugpros[['#69']] = c('Q9NT62')
drugpros[['#9']] = c('Q12926')
drugpros



sample0 = sampleInfoB[sampleInfoB$pert_time %in% 0, 'Sample_ID']
for (d in names(drugpros)) {
  # d = '#38'
  sample62448_Y = sampleInfoB[sampleInfoB$pert_id %in% d & sampleInfoB$pert_time>0 & sampleInfoB$NY %in% 'Y', 'Sample_ID']
  sample62448_N = sampleInfoB[sampleInfoB$pert_id %in% d & sampleInfoB$pert_time>0 & sampleInfoB$NY %in% 'N', 'Sample_ID']
  
  for (pro in drugpros[[d]]) {
    # pro = 'Q13153'
    print(paste0(c(d, pro, uniprotID_geneSymbol[uniprotID_geneSymbol$UniprotID %in% pro, 2] ), collapse = '_'))
    ## Y
    temp = matrixB[c(sample62448_Y, sample0), c(pro, 'A0A024RBG1')]
    colnames(temp)[1] = 'target'
    temp = subset(temp, !is.na(target))
    temp = subset(sampleInfoB , Sample_ID %in% rownames(temp))
    
    techrep = sum(grepl('rep', temp$Sample_ID))
    temp = subset(temp, !grepl('rep', temp$Sample_ID))
    temp1 = temp%>%distinct(pct, .keep_all = T)
    biorep = nrow(temp)-nrow(temp1)
    print(paste0(c('Y techrep:', techrep, ' biorep:', biorep), collapse = ''))
    
    ## N
    temp = matrixB[c(sample62448_N, sample0), c(pro, 'A0A024RBG1')]
    colnames(temp)[1] = 'target'
    temp = subset(temp, !is.na(target))
    temp = subset(sampleInfoB , Sample_ID %in% rownames(temp))
    
    techrep = sum(grepl('rep', temp$Sample_ID))
    temp = subset(temp, !grepl('rep', temp$Sample_ID))
    temp1 = temp %>% distinct(pct, .keep_all = T)
    biorep = nrow(temp)-nrow(temp1)
    print(paste0(c('N techrep:', techrep, ' biorep:', biorep), collapse = ''))
  }
}

