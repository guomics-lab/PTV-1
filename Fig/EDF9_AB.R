library(openxlsx)
library(ggplot2)
library(ggsci)

ptv1_ffpe_human = read.csv('z:/members/sunr/others/PTV1/2025_EGraw/FFPE/PTV1_FFPE_mzML_human_report.pg_matrix.tsv', sep = '\t')
ptv1_ffpe_human[1:2, 1:6]
rownames(ptv1_ffpe_human) = ptv1_ffpe_human$Protein.Group
ptv1_ffpe_human = ptv1_ffpe_human[6:ncol(ptv1_ffpe_human)]

ptv1_ffpe_humaninfo = data.frame(filename = colnames(ptv1_ffpe_human), row.names = colnames(ptv1_ffpe_human))
ptv1_ffpe_humaninfo$identify_num = colSums(!is.na(ptv1_ffpe_human))

ptv1_ffpe_humaninfo$sid = as.character(
  lapply(ptv1_ffpe_humaninfo$filename, function(x){
    x = gsub('.mzML', '', x, fixed = T)
    x = gsub('_rep', 'rep', x, fixed = T)
    x = strsplit(x, '_')[[1]]
    n = length(x)
    paste0(x[(n-1):n], collapse = '_')
  })
)
ptv1_ffpe_humaninfo[ptv1_ffpe_humaninfo$filename == 'WAF20250717qianlj_PTV1_15min_SWATH_POOL.mzML', 'sid'] = 'b7_pool'

ptv1_ffpe_humaninfo$pool = ifelse(grepl('pool',ptv1_ffpe_humaninfo$filename ,ignore.case = T), 'pool', NA)
ptv1_ffpe_humaninfo$machine = ifelse(grepl('^D',ptv1_ffpe_humaninfo$filename ), 'D', 'WAF')

ptv1_ffpe_humaninfo$batch = as.character(lapply(ptv1_ffpe_humaninfo$sid, function(x){
  strsplit(x, '_')[[1]][1]
}))

table(ptv1_ffpe_humaninfo$batch)
ptv1_ffpe_humaninfo[ptv1_ffpe_humaninfo$batch == 'b39', ]

ptv1_ffpe_humaninfo = subset(ptv1_ffpe_humaninfo, identify_num>0)
ptv1_ffpe_humaninfo$batch.ID = gsub('rep', '', ptv1_ffpe_humaninfo$sid)

colnames(ffpe_info)

ptv1_ffpe_humaninfo = merge(ptv1_ffpe_humaninfo, ffpe_info[c("病理号", "batch.ID", "drug", "drugTimes", "年龄", "transfertime", "deadtime","是否转移", "是否死亡" )], by = c('batch.ID'), all.x = T)
rownames(ptv1_ffpe_humaninfo) = ptv1_ffpe_humaninfo$filename

ptv1_ffpe_humaninfo = ptv1_ffpe_humaninfo[order(ptv1_ffpe_humaninfo$batch.ID, ptv1_ffpe_humaninfo$identify_num, decreasing = T),]

runmulti = data.frame(table(ptv1_ffpe_humaninfo$sid))
runmulti = subset(runmulti, Freq>1)
runmulti = subset(ptv1_ffpe_humaninfo, sid %in% runmulti$Var1)
runmulti$sid

ptv1_ffpe_humaninfo = ptv1_ffpe_humaninfo %>% distinct(sid, .keep_all = T)
# write.xlsx(ptv1_ffpe_humaninfo, "D:/chh/2025workProject/20250506PTV1/FFPE/ptv1_ffpe_humaninfo.xlsx")
ptv1_ffpe_humaninfo = read.xlsx(  "D:/chh/2025workProject/20250506PTV1/FFPE/ptv1_ffpe_humaninfo.xlsx")

runmulti = data.frame(table(ptv1_ffpe_humaninfo$sid))
runmulti = subset(runmulti, Freq>1)
runmulti = subset(ptv1_ffpe_humaninfo, sid %in% runmulti$Var1)
runmulti$sid

ptv1_ffpe_human = ptv1_ffpe_human[ptv1_ffpe_humaninfo$filename]
narow = rowSums(!is.na(ptv1_ffpe_human))
ptv1_ffpe_human = ptv1_ffpe_human[narow!=0,]

dim(ptv1_ffpe_human)
temp = ptv1_ffpe_human
temp = subset(temp, !grepl(';', rownames(temp)) )
temp = subset(temp, !grepl('SWISS', rownames(temp)) )
temp = subset(temp, !grepl('TREMBL', rownames(temp)) )
temp = subset(temp, !grepl('ENSEMBL', rownames(temp)) )
narow = rowSums(!is.na(temp))
temp = temp[narow!=0, ]
dim(temp)
temp[1:10, 1:2]


# rep 只做correlation CV
# pool样本也是做一下
# PCA/TSNE/UMAP 标签是batch number, 质谱编号
######## human ########
######## pool ########
ptv1_ffpe_human = temp
poolinfo = subset(ptv1_ffpe_humaninfo, !is.na(pool))

temp = ptv1_ffpe_human[poolinfo$filename]
temp = cor(temp,  use = "pairwise.complete.obs")
diag(temp) = NA
temp[upper.tri(temp)] = NA
temp = melt(temp)
temp = subset(temp, !is.na(value))
poolcor = data.frame(temp)
head(poolcor)
# dir.create('D:/chh/2025workProject/20250506PTV1/FFPE/')
write.xlsx(poolcor, "D:/chh/2025workProject/20250506PTV1/FFPE/human_poolcor.xlsx", rowNames = T)


temp = ptv1_ffpe_human[poolinfo$filename]
temp = data.frame(apply(as.matrix(log2(temp)), 1, function(x){# log2(temp)
  sd(x, na.rm = T)/mean(x, na.rm = T)
}))
poolcv = temp;colnames(poolcv) = 'temp'
poolcv = subset(poolcv, !is.na(poolcv$temp))
head(poolcv)
median(poolcv$temp)
write.xlsx(poolcv, "D:/chh/2025workProject/20250506PTV1/FFPE/human_poolcv_log2.xlsx", rowNames = T)


p = ggplot(poolcv, aes(x = 'pool cv', y = temp))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.05 )+
  ylim(0,1)+
  annotate('text', x= 'pool cv' , y  = median(poolcv$temp, na.rm = T), label = paste0('Median=',round(median(poolcv$temp, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+ # poolms_log2_cv
  theme_classic()+
  theme(text = element_text(size = 15),
        # axis.text.x = element_text(size = 15),
        # axis.text.y = element_text(size = 15)
  )
ggsave('D:/chh/2025workProject/20250506PTV1/FFPE/human_poolcv_log2.pdf', p)

p = ggplot(poolcor[poolcor$value > 0.75, ], aes(x = 'pool cor', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.1)+
  ylim(0,1)+
  annotate('text', x= 'pool cor' , y  = median(poolcor$value, na.rm = T), label = paste0('Median=',round(median(poolcor$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15))
ggsave('D:/chh/2025workProject/20250506PTV1/FFPE/human_poolcor.pdf', p)


techrep = ptv1_ffpe_humaninfo[grepl('rep', ptv1_ffpe_humaninfo$sid), 'sid']
biorep = data.frame(table(ptv1_ffpe_humaninfo[!grepl('rep', ptv1_ffpe_humaninfo$sid), '病理号'] ))
biorep = subset(biorep, Freq>1)
biorep
biorep = subset(ptv1_ffpe_humaninfo, 病理号 %in% biorep$Var1 & !grepl('rep', sid))

######## biorep cv cor ########
ptv1_ffpe_human = temp
biorep_cor = c()
biorep_cv = c()
for (i in unique(biorep$病理号)) {
  ss = biorep[biorep$病理号 %in% i, 'filename']
  temp = ptv1_ffpe_human[ss]
  temp = cor(temp,  use = "pairwise.complete.obs")
  diag(temp) = NA
  temp[upper.tri(temp)] = NA
  temp = melt(temp)
  temp = subset(temp, !is.na(value))
  biorep_cor = rbind(biorep_cor, temp)
  
  temp = ptv1_ffpe_human[ss]
  temp = data.frame(apply(as.matrix(log2(temp)), 1, function(x){# log2(temp)
    sd(x, na.rm = T)/mean(x, na.rm = T)
  }))
  temp$bid= ss[1]
  colnames(temp)[1] = 'value'
  biorep_cv = rbind(biorep_cv, temp)
}
median(biorep_cv$value, na.rm = T)

write.xlsx(biorep_cv, "D:/chh/2025workProject/20250506PTV1/FFPE/human_biorep_cv_log2.xlsx", rowNames = T)
write.xlsx(biorep_cor, "D:/chh/2025workProject/20250506PTV1/FFPE/human_biorep_cor.xlsx", rowNames = T)


p = ggplot(biorep_cor[biorep_cor$value>= 0.9, ], aes(x = 'biorep cor', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.1)+
  ylim(0,1)+
  annotate('text', x= 'biorep cor' , y  = median(biorep_cor$value, na.rm = T), label = paste0('Median=',round(median(biorep_cor$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15))
ggsave("D:/chh/2025workProject/20250506PTV1/FFPE/human_biorep_cor.pdf", p)

ss = biorep_cor[biorep_cor$value< 0.9, 1:2]
biorep_cv = subset(biorep_cv, !(bid %in% ss))
p = ggplot(biorep_cv, aes(x = 'biorep cv', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.05 )+
  ylim(0,1)+
  annotate('text', x= 'biorep cv' , y  = median(biorep_cv$value, na.rm = T), label = paste0('Median=',round(median(biorep_cv$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+ # poolms_log2_cv
  theme_classic()+
  theme(text = element_text(size = 15),
        # axis.text.x = element_text(size = 15),
        # axis.text.y = element_text(size = 15)
  )
ggsave("D:/chh/2025workProject/20250506PTV1/FFPE/human_biorep_cv_log2.pdf", p)

######## techrep cv cor ########
techrep = gsub('rep', '', techrep)

techrep_cor = c()
techrep_cv = c()
for (i in unique(techrep)) {
  ss = ptv1_ffpe_humaninfo[ptv1_ffpe_humaninfo$batch.ID %in% i, 'filename'] #techrep[techrep$Cell_plate %in% i, 'filename']
  if(length(ss)<2){
    print(ss)
    next
  }
  temp = ptv1_ffpe_human[ss]
  temp = cor(temp,  use = "pairwise.complete.obs")
  diag(temp) = NA
  temp[upper.tri(temp)] = NA
  temp = melt(temp)
  temp = subset(temp, !is.na(value))
  techrep_cor = rbind(techrep_cor, temp)
  
  temp = ptv1_ffpe_human[ss]
  temp = data.frame(apply(as.matrix(log2(temp)), 1, function(x){# log2(temp)
    sd(x, na.rm = T)/mean(x, na.rm = T)
  }))
  temp$bid = ss[1]
  colnames(temp)[1] = 'value'
  techrep_cv = rbind(techrep_cv, temp)
}
write.xlsx(techrep_cv, "D:/chh/2025workProject/20250506PTV1/FFPE/human_techrep_cv_log2.xlsx", rowNames = T)
write.xlsx(techrep_cor, "D:/chh/2025workProject/20250506PTV1/FFPE/human_techrep_cor.xlsx", rowNames = T)


p = ggplot(techrep_cv, aes(x = 'techrep cv', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.05 )+
  ylim(0,1)+
  annotate('text', x= 'techrep cv' , y  = median(techrep_cv$value, na.rm = T), label = paste0('Median=',round(median(techrep_cv$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+ # poolms_log2_cv
  theme_classic()+
  theme(text = element_text(size = 15),
        # axis.text.x = element_text(size = 15),
        # axis.text.y = element_text(size = 15)
  )
ggsave("D:/chh/2025workProject/20250506PTV1/FFPE/human_techrep_cv_log2.pdf", p)
p = ggplot(techrep_cor, aes(x = 'techrep cor', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.1)+
  ylim(0,1)+
  annotate('text', x= 'techrep cor' , y  = median(techrep_cor$value, na.rm = T), label = paste0('Median=',round(median(techrep_cor$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15))
ggsave("D:/chh/2025workProject/20250506PTV1/FFPE/human_techrep_cor.pdf", p)
