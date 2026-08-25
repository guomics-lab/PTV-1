rm(list = ls());gc()
library(openxlsx)
library(ggplot2)
library(ggsci)
library(dplyr)
library(reshape2)
library(patchwork)
# mat
df1 = read.csv('D:/chh/2023workProject/prottalk/code/PTV1_BD_cv_cor/matrix_BD_combine20260616.csv')
dim(df1)
df1[1:4, 1:3]
rownames(df1) = df1$X
df1 = df1[-1]
df1 = log2(df1)
df1[1:4, 1:3]

# info
df1_info = read.csv("D:/chh/2023workProject/prottalk/code/PTV1_BD_cv_cor/matrix_BD_combine_info20260616.csv")

df1_info$s1 = gsub('_rep', 'rep', df1_info$Column.Names, ignore.case = T)
temp = gsub('.*_', '', df1_info$s1)
df1_info$s1 = ifelse(grepl('ool|HCC1937', df1_info$s1, ignore.case = T), df1_info$s1, temp)
df1_info[df1_info$s1 %in% 1 & !grepl('pool', df1_info$s1), ]
df1_info[df1_info$s1 %in% 2 & !grepl('pool', df1_info$s1), ]
df1_info[df1_info$Column.Names %in%  'WAF20221206sunr_idrug_phase1_15min_sw_5255_2', 2] = '5255rep'
df1_info[df1_info$Column.Names %in%  'WAF20221210sunr_idrug_phase1_15min_sw_5334_2', 2] = '5334rep'
df1_info[df1_info$Column.Names %in%  'WAF20230330sunr_idrug_phase2_15min_sw_B3230_2', 2] = 'B3230rep'
df1_info[df1_info$Column.Names %in%  'WAF20230330sunr_idrug_phase2_15min_sw_B3231_2', 2] = 'B3231rep'

df1_info$s2 = gsub('rep', '', df1_info$s1)

df1_info$pool = sapply(df1_info$s2, function(x){
  stringi::stri_length(x)
})
df1_info$pool = ifelse(df1_info$pool>20, 'pool', NA)

pool_info = subset(df1_info, !is.na(df1_info$pool))
pool_info$Column.Names = gsub('-', '.', pool_info$Column.Names)
# "D20230813sunr_idrug_phase3_20ms_15min_1ug_SWATH_M453meisen_6H_pool1"
D20230813sunr_idrug_phase3_20ms_15min_1ug_SWATH_M453meisen_6H_pool

sinfo1 = read.xlsx("D:/chh/2023workProject/prottalk/database/matrixB_sinfo_20250905.xlsx")
head(sinfo1)
sinfo1 = sinfo1 %>% distinct(Sample_ID, .keep_all = T)
colnames(sinfo1)
sinfo1 = sinfo1[c("Sample_ID", "protein_plate", "pert_id", "pert_time")]

df1_info = subset(df1_info, is.na(pool) & !grepl('-', df1_info$Column.Names))
df1_info = df1_info[1:3]
nrow(df1_info)

colnames(df1_info)[3]= 'Sample_ID'
df1_info = merge(df1_info, sinfo1, by = 'Sample_ID', all.x = T)
nrow(df1_info)# 16443

sinfo2 = read.xlsx("D:/chh/2023workProject/prottalk/database/iDrug_D_combo_group.info.xlsx")
rownames(sinfo2) = sinfo2$Sample_ID
colnames(sinfo2)
head(sinfo2)

for (i in 1:nrow(df1_info)) {
  # i=13126
  if(df1_info[i, "Sample_ID"] %in% sinfo2$Sample_ID){
    tmp = subset(sinfo2, Sample_ID %in% df1_info[i, "Sample_ID"])
    tmp$pert_id = ifelse(tmp$Anchor_id> tmp$Library_id, paste0(tmp$Anchor_id, '_', tmp$Library_id), paste0(tmp$Library_id, '_', tmp$Anchor_id))
    colnames(tmp)
    df1_info[i, c('pert_id', "protein_plate", "Anchor_id", "Library_id", "pert_time")] = tmp[1, c('pert_id', "protein_plate", "Anchor_id", "Library_id", "pert_time")]
  }
}

unique(df1_info$pert_id)
df1_info[is.na(df1_info$pert_id), ]

df1_info = subset(df1_info, !is.na(pert_id))
nrow(df1_info)# 15827

freq = data.frame(table(df1_info$s1))
freq = subset(freq, Freq>1)

freq = data.frame(table(df1_info$Sample_ID))
freq = subset(freq, Freq>1)
techrep = subset(df1_info, Sample_ID %in% freq$Var1)
for(v in freq$Var1){
  # v= 'D5521'
  tmp = subset(df1_info, Sample_ID %in% v)
  tmp = tmp[order(tmp$s1), ]
  # break
  df1_info[df1_info$Column.Names %in% tmp$Column.Names[-1], 'biorep_techrep'] ='techrep'
}
unique(df1_info$protein_plate)

df1_info[df1_info$protein_plate %in% c('M453(meisen）', "MDA-MB-453" ), 'protein_plate'] = "MDA-MB-453"
df1_info[df1_info$protein_plate %in% c("MDA-MB-453(ATCC)", "M453(ATCC)" ), 'protein_plate'] = "MDA-MB-453-ATCC"

df1_info$pct = paste0(df1_info$pert_id, '_', df1_info$protein_plate, '_', df1_info$pert_time)

biorep = subset(df1_info, is.na(biorep_techrep))
nrow(biorep)
unique_sample = biorep %>% distinct(pct, .keep_all = T)
nrow(unique_sample)
freq = data.frame(table(biorep$pct))
freq = subset(freq, Freq>1)# 4492 condition
biorep = subset(biorep, pct %in% freq$Var1)
biorep = biorep[order(biorep$pct),]
nrow(biorep)

for (v in unique(biorep$pct)) {
  tmp = subset(biorep, pct %in% v)
  df1_info[df1_info$Column.Names %in% tmp$Column.Names[2:nrow(tmp)], 'biorep_techrep'] = 'biorep'
  # break
}
table(df1_info$biorep_techrep)

nrow(df1_info) + nrow(pool_info)
nrow(biorep) - nrow(unique_sample)# 10029
table(df1_info$biorep_techrep)# 1166
nrow(pool_info)# 618

#### pool #####
pool = df1[pool_info$Column.Names]
pool[1:3, 1:2]

pool_naimpute = pool
narow = rowSums(!is.na(pool_naimpute))
pool_naimpute = pool_naimpute[narow>0, ]
pool_naimpute[is.na(pool_naimpute)] = 0.8 * min(pool_naimpute, na.rm = T) #0

set.seed(2026)
dfpca = prcomp(t(pool_naimpute))#, scale. = T
dfpca_pc = data.frame(dfpca$x)[1:3]#
head(dfpca_pc)

pca_p = ggplot(dfpca_pc, aes(PC1, PC2 ))+
  geom_point(size=0.5)+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_vline(xintercept = c(60, 153))+
  geom_hline(yintercept = c(20, 50))+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black'), 
        axis.title.x = element_blank() 
        )
pca_p

clu1 = dfpca_pc[dfpca_pc$PC1<60 & dfpca_pc$PC2<50,  ]
clu2 = rbind(dfpca_pc[dfpca_pc$PC1>60 & dfpca_pc$PC1<153 & dfpca_pc$PC2<50 & dfpca_pc$PC2>20,  ], 
             dfpca_pc[dfpca_pc$PC1>153 & dfpca_pc$PC2<20,  ]
             )
clu3 = unique(c(rownames(dfpca_pc[dfpca_pc$PC2>50,  ]), 
                rownames(dfpca_pc[dfpca_pc$PC1>153 & dfpca_pc$PC2>20,  ])
))

pca_p = ggplot(clu2, aes(PC1, PC2 ))+
  geom_point()+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_vline(xintercept = c(60, 153))+
  geom_hline(yintercept = c(20, 50))+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black'), 
        axis.title.x = element_blank() 
  )
pca_p

pool_3clusters = list()
pool_3clusters[[1]] = rownames(clu1)# 302
pool_3clusters[[2]] = rownames(clu2)
pool_3clusters[[3]] = clu3

all_cor_res = c()
for (i in 1:3) {
  # i=2
  poolcor = cor(pool[ pool_3clusters[[i]]  ], use ="pairwise.complete.obs" )
  diag(poolcor) = NA
  poolcor[upper.tri(poolcor)] = NA
  poolcor = melt(poolcor)
  poolcor = subset(poolcor, !is.na(value))
  
  all_cor_res = rbind(all_cor_res, poolcor )
  
  cvpool = apply(pool[ pool_3clusters[[i]]  ], 1, function(x){
    sd(x, na.rm = T)/mean(x, na.rm = T)
  })
  
  cvpool = data.frame(val = cvpool)
  # col=c("coral2","dodgerblue3","burlywood2"),names=c("pool","tench_rep","bio_rep")
  poolcv_p = ggplot(cvpool, aes('pool', val))+
    geom_violin(fill = "coral2")+
    geom_boxplot(width = 0.1, outlier.color = NA)+# , size=1
    ylim(0,1) +
    annotate('text', x= 'pool' , y  = 0.5, label = paste0('Median=',round(median(cvpool$val, na.rm = T), 4)))+
    labs(  y = 'Coefficient of variation')+
    theme_classic()+
    theme(text = element_text(size = 15),
          axis.text  = element_text(size = 15, color = 'black'), axis.title.x = element_blank() )
  
  poolcor_p = ggplot(poolcor, aes('pool', value))+
    geom_violin(fill = "coral2")+
    geom_boxplot(width = 0.1, outlier.color = NA)+# , size=1
    ylim(0,1) +
    annotate('text', x= 'pool' , y  = 0.5, label = paste0('Median=',round(median(poolcor$value, na.rm = T), 4)))+
    labs(  y = 'Pearson correlation')+
    theme_classic()+
    theme(text = element_text(size = 15),
          axis.text  = element_text(size = 15, color = 'black'), axis.title.x = element_blank() )
  # ggsave(paste0('//172.16.13.136/share/members/chenghonghan/PTV1/proof_check20260608/EDF1_BC_zhangzhi/EDF1_C_pool_clu', i, '.pdf'), poolcor_p+poolcv_p, width = 6.5, height = 3 )
}
head(all_cor_res)

poolcor_p = ggplot(all_cor_res, aes('pool', value))+
  geom_violin(fill = "coral2")+
  geom_boxplot(width = 0.1, outlier.color = NA)+# , size=1
  ylim(0,1) +
  annotate('text', x= 'pool' , y  = 0.5, label = paste0('Median=',round(median(all_cor_res$value, na.rm = T), 4)))+
  labs(  y = 'Pearson correlation')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black'), axis.title.x = element_blank() )
poolcor_p
ggsave(paste0('//172.16.13.136/share/members/chenghonghan/PTV1/proof_check20260608/EDF1_BC_zhangzhi/EDF1_C_pool.pdf'), poolcor_p, width = 3, height = 3 )
#### techrep #####
techcv = c()
techcor = c()
for (s in unique(techrep$Sample_ID)) {
  # s = "D35"
  ss = techrep[techrep$Sample_ID %in% s, 2]
  if(length(ss)<2){
    print(s);next
  }
  tmp = df1[ss]
  tmp = apply(tmp, 1, function(x){
    sd(x, na.rm = T)/mean(x, na.rm = T)
  })
  techcv = c(techcv, na.omit(tmp))
  
  tmp = df1[ss]
  tmp = cor(df1[ss], use ="pairwise.complete.obs" )
  diag(tmp) = NA
  tmp[upper.tri(tmp)] = NA
  tmp = melt(tmp)
  tmp = subset(tmp, !is.na(value))
  techcor = rbind(techcor, tmp)
}


techrep_cv_dat = data.frame(techcv)
colnames(techrep_cv_dat) = 'val'
head(techrep_cv_dat)

# col=c("coral2","dodgerblue3","burlywood2"),names=c("pool","tench_rep","bio_rep")
techcv_p = ggplot(biorep_cv, aes('tech', val))+
  geom_violin(fill = "dodgerblue3")+
  geom_boxplot(width = 0.1, outlier.color = NA)+# , size=1
  ylim(0,1) +
  annotate('text', x= 'tech' , y  = 0.5, label = paste0('Median=',round(median(techrep_cv_dat$val, na.rm = T), 4)))+
  labs(  y = 'Coefficient of variation')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black'), axis.title.x = element_blank() )
techcv_p


head(techcor)
techcor_p = ggplot(techcor, aes('tech', value))+
  geom_violin(fill = "dodgerblue3")+
  geom_boxplot(width = 0.1, outlier.color = NA)+# , size=1
  ylim(0,1) +
  annotate('text', x= 'tech' , y  = 0.5, label = paste0('Median=',round(median(techcor$value, na.rm = T), 4)))+
  labs(  y = 'Pearson correlation')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black'), axis.title.x = element_blank() )
techcor_p

#### biorep #####
biorep_cor = c()
biorep_cv = c()
for (i in unique(biorep$pct) ) {
  # i = '#56_BT20_48'
  tmp = subset(biorep, pct %in% i)
  # break
  if(nrow(tmp)<2){
    print(i)
    next
  }
  # print(i)
  tmp = df1[tmp$Column.Names]
  tmp_cor = cor(tmp, use ="pairwise.complete.obs" )
  diag(tmp_cor) = NA
  tmp_cor[upper.tri(tmp_cor)] = NA
  tmp_cor = data.frame(melt(tmp_cor))
  tmp_cor = subset(tmp_cor, !is.na(value))
  biorep_cor = rbind(biorep_cor, tmp_cor)
  
  tmpcv = apply(tmp, 1, function(x){
    sd(x, na.rm = T)/mean(x, na.rm = T)
  })
  biorep_cv = c(biorep_cv, tmpcv)
}
head(biorep_cor)
nrow(biorep_cor)

biorep_cv = data.frame(biorep_cv)
colnames(biorep_cv) = 'val'
head(biorep_cv)

# col=c("coral2","dodgerblue3","burlywood2"),names=c("pool","tench_rep","bio_rep")
biorepcv = ggplot(biorep_cv, aes('bio', val))+
  geom_violin(fill = "burlywood2")+
  geom_boxplot(width = 0.1, outlier.color = NA)+# , size=1
  ylim(0,1) +
  annotate('text', x= 'bio' , y  = 0.5, label = paste0('Median=',round(median(biorep_cv$val, na.rm = T), 4)))+
  labs(  y = 'Coefficient of variation')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black'), axis.title.x = element_blank() )
biorepcv



biorepcor = ggplot(biorep_cor, aes('bio', value))+
  geom_violin(fill = "burlywood2")+
  geom_boxplot(width = 0.1, outlier.color = NA)+# , size=1
  ylim(0,1) +
  annotate('text', x= 'bio' , y  = 0.5, label = paste0('Median=',round(median(biorep_cor$value, na.rm = T), 4)))+
  labs(  y = 'Pearson correlation')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black'), axis.title.x = element_blank() )
biorepcor


library(patchwork)
ggsave('D:/chh/2023workProject/prottalk/code/EDF1_zhangzhi/EDF1_C20260617.pdf', 
       poolcv_p + techcv_p + biorepcv +
         poolcor_p + techcor_p + biorepcor, width = 8, height = 5)
