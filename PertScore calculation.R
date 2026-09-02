rm(list = ls())
library(openxlsx)
library(ggplot2)
library(ggsci)
library(foreach)
library(doParallel)

#### EF ####
D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/ptv1_Ttest_drug_cell_time20260324.R

#### 20260320 subtype time ####
files = list.files('D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260302/drug_cell_time/', pattern = '*xlsx')
files_obs = files[grepl('hrs_ttest20260307', files)]
files_obs = sort(files_obs)
temp = read.xlsx(paste0('D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260302/drug_cell_time/', files_obs[1]), rowNames = T)
cl <- makeCluster(6)  # 创建一个4核心的集群
registerDoParallel(cl)
result <- foreach(f = files_obs ) %dopar% {
  library(openxlsx)
  temp = read.xlsx(paste0('D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260302/drug_cell_time/', f), rowNames = T)
  temp$sig = ifelse(is.na(temp$FDR), 0, ifelse(temp$logFC> log2(1.2) & temp$FDR<0.05, 1,
                                               ifelse( temp$logFC< (-log2(1.2)) & temp$FDR<0.05, -1, 0)))
  temp$sig
}
stopCluster(cl)
res= data.frame(do.call(cbind, lapply(result, function(x) c(x, rep(NA, max(lengths(result)) - length(x))))), row.names = rownames(temp) )
colnames(res) = files_obs
all_obs = res

###### empirical_FDR abs else #3 ####
S_obs = data.frame(row.names = rownames(all_obs))
for (sb2 in c( "ALK", "antimitotic", "hormonal agent", "Kinase", "CDK",  "Topoisomerase", 'PARP') ) {
  if(is.na(sb2)) next
  # sb2 = "ALK"
  d = unique(sampleInfoB[sampleInfoB$subtype2 %in% sb2, 'pert_id'])
  temp1 = all_obs[grepl(paste0(paste0(d,"_"), collapse = '|'), colnames(all_obs))]
  # print(colnames(temp1))
  
  for (i in c(6,24, 48)) {
    S_obs[ paste0(sb2, '_', i)] = rowSums(temp1[grepl(paste0('_', i, 'hrs'), colnames(temp1))], na.rm = T)
  }
  # break
}
head(S_obs)
sort(colnames(S_obs))
# write.xlsx(S_obs, 'D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260302_sb2time_pertscore20260320.xlsx', sheetName = 'obs', rowNames = T)

# _drugsum_20260309
files = list.files('D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/', pattern = '*xlsx')
files_null = files[!grepl('_drugsum_', files) & grepl('seeds', files) & grepl('20260324', files)]
files_null = sort(files_null)
files_null[1:10]

set.seed(2026)
n_seeds = sort(sample(1:20000, 500))
n_seeds[1:10]

S_null_matrix_sum = data.frame(matrix(0, nrow = nrow(all_obs), ncol = 10*length(colnames(S_obs) )), row.names = rownames(all_obs))
cols = unlist(lapply(colnames(S_obs), function(x){
  paste0(x, '_', n_seeds[1:10])
}))
colnames(S_null_matrix_sum) = cols

for (f in files_null) {
  # f = files_null[1]
  temp = read.xlsx(paste0('D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/', f), rowNames = T)
  colnames(temp) = gsub('X.', '#', colnames(temp), fixed = T)
  temp[is.na(temp)] = 0
  d = gsub('_.*', '', colnames(temp)[1])
  seed_i = strsplit(f, '_')[[1]][3]
  sb2 = drug_info[drug_info$Pert_ID %in% d, 'Subtype2']
  if(!(sb2 %in% c( "ALK", "antimitotic", "hormonal agent", "Kinase", "CDK",  "Topoisomerase", 'PARP') )) next
  
  for (tt in c(6,24, 48)) {
    # tt = 6
    temp1 = temp[grepl(paste0(tt, '$'), colnames(temp))]
    S_null_matrix_sum[rownames(temp1), paste0(sb2, '_', tt, '_', seed_i)] = S_null_matrix_sum[rownames(temp1),paste0(sb2, '_', tt, '_', seed_i)] + rowSums(temp1, na.rm = T)
  }
  
}
S_null_matrix_sum[1:3, 1:10]
dim(S_null_matrix_sum)

rownames(S_null_matrix_sum) = gsub('[^0-9A-Za-z]', '.', rownames(S_null_matrix_sum))
setdiff(rownames(PARP_null_matrix_sum), rownames(S_null_matrix_sum))

# c( "ALK", "antimitotic", "hormonal agent", "Kinase", "CDK",  "Topoisomerase", 'PARP')
Topoisomerase_null_matrix_sum = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/1000seeds_Topoisomerase.csv", row.names = 1)
Topoisomerase_null_matrix_sum = cbind(Topoisomerase_null_matrix_sum, S_null_matrix_sum[rownames(Topoisomerase_null_matrix_sum), grepl('Topoisomerase', colnames(S_null_matrix_sum))])
dim(Topoisomerase_null_matrix_sum)
Kinase_null_matrix_sum = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/1000seeds_Kinase.csv", row.names = 1)
Kinase_null_matrix_sum = cbind(Kinase_null_matrix_sum, S_null_matrix_sum[rownames(Kinase_null_matrix_sum), grepl('Kinase', colnames(S_null_matrix_sum), ignore.case = T)])
dim(Kinase_null_matrix_sum)
PARP_null_matrix_sum = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/1000seeds_PARP.csv", row.names = 1)
PARP_null_matrix_sum = cbind(PARP_null_matrix_sum, S_null_matrix_sum[rownames(PARP_null_matrix_sum), grepl('PARP', colnames(S_null_matrix_sum))])
dim(PARP_null_matrix_sum)
CDK_null_matrix_sum = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/1000seeds_CDK.csv", row.names = 1)
CDK_null_matrix_sum = cbind(CDK_null_matrix_sum, S_null_matrix_sum[rownames(CDK_null_matrix_sum), grepl('CDK', colnames(S_null_matrix_sum))])
dim(CDK_null_matrix_sum)
hormonal_null_matrix_sum = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/1000seeds_hormonal.csv", row.names = 1)
hormonal_null_matrix_sum = cbind(hormonal_null_matrix_sum, S_null_matrix_sum[rownames(hormonal_null_matrix_sum), grepl('hormonal', colnames(S_null_matrix_sum))])
dim(hormonal_null_matrix_sum)
ALK_null_matrix_sum = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/1000seeds_ALK.csv", row.names = 1)
ALK_null_matrix_sum = cbind(ALK_null_matrix_sum, S_null_matrix_sum[rownames(ALK_null_matrix_sum), grepl('ALK', colnames(S_null_matrix_sum))])
dim(ALK_null_matrix_sum)
antimitotic_null_matrix_sum = read.csv("D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324/1000seeds_antimitotic.csv", row.names = 1)
antimitotic_null_matrix_sum = cbind(antimitotic_null_matrix_sum, S_null_matrix_sum[rownames(antimitotic_null_matrix_sum), grepl('antimitotic', colnames(S_null_matrix_sum))])
dim(Topoisomerase_null_matrix_sum)

Topoisomerase_null_matrix_sum[1:3, 1:10]

c( "ALK", "antimitotic", "hormonal agent", "Kinase", "CDK",  "Topoisomerase", 'PARP')
empirical_FDR_res = c()
protein_FDR_sb2 = data.frame(row.names = rownames(S_obs))
for (sb2_tt in colnames(S_obs)) {
  # sb2_tt = "ALK_6"
  S_obs_sum = abs(S_obs[, sb2_tt])
  names(S_obs_sum) = rownames(S_obs)
  
  if(grepl('ALK', sb2_tt)){
    S_null_matrix_sb2 = ALK_null_matrix_sum[grepl(paste0(sb2_tt, '_'), colnames(ALK_null_matrix_sum))]
  }else if(grepl('antimitotic', sb2_tt)){
    S_null_matrix_sb2 = antimitotic_null_matrix_sum[grepl(paste0(sb2_tt, '_'), colnames(antimitotic_null_matrix_sum))]
  }else if(grepl('hormonal', sb2_tt)){
    S_null_matrix_sb2 = hormonal_null_matrix_sum[grepl(paste0(sb2_tt, '_'), colnames(hormonal_null_matrix_sum))]
  }else if(grepl('Kinase', sb2_tt)){
    S_null_matrix_sb2 = Kinase_null_matrix_sum[grepl(paste0(sb2_tt, '_'), colnames(Kinase_null_matrix_sum))]
  }else if(grepl('CDK', sb2_tt)){
    S_null_matrix_sb2 = CDK_null_matrix_sum[grepl(paste0(sb2_tt, '_'), colnames(CDK_null_matrix_sum))]
  }else if(grepl('Topoisomerase', sb2_tt)){
    S_null_matrix_sb2 = Topoisomerase_null_matrix_sum[grepl(paste0(sb2_tt, '_'), colnames(Topoisomerase_null_matrix_sum))]
  }else if(grepl('PARP', sb2_tt)){
    S_null_matrix_sb2 = PARP_null_matrix_sum[grepl(paste0(sb2_tt, '_'), colnames(PARP_null_matrix_sum))]
  }
  S_null_matrix_sb2 = abs(S_null_matrix_sb2)
  # S_null_matrix_sb2[1:3, 1:10]
  
  unique_scores <- sort(unique(S_obs_sum), decreasing = TRUE)
  FDR_table <- data.frame(score = unique_scores,
                          R_obs = NA,
                          V_null = NA,
                          FDR = NA)
  for (i in seq_along(unique_scores)) {
    # i=1
    s0 <- unique_scores[i]
    
    R_obs <- sum(S_obs_sum >= s0)# 真实数据
    Vb <- colSums(S_null_matrix_sb2 >= s0)# null 每次 permutation 的 exceedance 数量
    
    E_V <- mean(Vb)# 期望假阳性数量
    
    FDR_value <- E_V / R_obs
    
    FDR_table$R_obs[i] <- R_obs
    FDR_table$V_null[i] <- E_V
    FDR_table$FDR[i] <- min(FDR_value, 1)
  }
  protein_FDR <- rep(NA, length(S_obs_sum))
  names(protein_FDR) <- names(S_obs_sum)
  for (i in 1:nrow(FDR_table)) {
    s0 <- FDR_table$score[i]
    protein_FDR[S_obs_sum == s0] <- FDR_table$FDR[i]
  }
  protein_FDR_result <- data.frame(
    protein = names(S_obs_sum),
    pertscore = S_obs_sum,
    empirical_FDR = protein_FDR
  )
  protein_FDR_result$Gene = human_gene[rownames(protein_FDR_result), 2]
  protein_FDR_sb2[rownames(protein_FDR_result), sb2_tt] = protein_FDR_result$empirical_FDR
  
  empirical_FDR_res = rbind(empirical_FDR_res, c(sb2_tt, 
                                                 nrow(subset(protein_FDR_result, empirical_FDR<0.01)),
                                                 nrow(subset(protein_FDR_result, empirical_FDR<0.05)),
                                                 nrow(subset(protein_FDR_result, empirical_FDR<0.1))))
}
protein_FDR_sb2$Gene = human_gene[rownames(protein_FDR_sb2), 2]
gc()
xlsx::write.xlsx(protein_FDR_sb2, 'D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324_1000seeds_sb2time_pertscore.xlsx', sheetName = 'protein_FDR_sb2' )

empirical_FDR_res = data.frame(empirical_FDR_res)
colnames(empirical_FDR_res) = c('subtype', 'empirical_FDR0.01', 'empirical_FDR0.05', 'empirical_FDR0.1')
for (i in c('empirical_FDR0.01', 'empirical_FDR0.05', 'empirical_FDR0.1')) {
  empirical_FDR_res[i] = as.numeric(empirical_FDR_res[, i])
}
xlsx::write.xlsx(empirical_FDR_res, 'D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324_1000seeds_sb2time_pertscore.xlsx', sheetName = 'empirical_FDR_count', append = T)

dim(protein_FDR_sb2)


#### all in one ####
seeds1000 = unique(gsub('.*_', '_', colnames(Topoisomerase_null_matrix_sum)))
S_null_1000seed = data.frame(row.names = rownames(Topoisomerase_null_matrix_sum))
for (i in seeds1000) {
  # i = seeds1000[1]
  S_null_1000seed[,i] = ALK_null_matrix_sum[, grepl(paste0(i, '$'), colnames(ALK_null_matrix_sum))]+
    Kinase_null_matrix_sum[, grepl(paste0(i, '$'), colnames(Kinase_null_matrix_sum))]+
    CDK_null_matrix_sum[, grepl(paste0(i, '$'), colnames(CDK_null_matrix_sum))]+
    PARP_null_matrix_sum[, grepl(paste0(i, '$'), colnames(PARP_null_matrix_sum))]+
    Topoisomerase_null_matrix_sum[, grepl(paste0(i, '$'), colnames(Topoisomerase_null_matrix_sum))]+
    antimitotic_null_matrix_sum[, grepl(paste0(i, '$'), colnames(antimitotic_null_matrix_sum))]
  
}
S_null_1000seed[1:10, 1:10]
S_null_1000seed[1:10, 995:1000]


S_obs[1:10, 1:5]
colnames(S_obs)
S_obs_sum = abs(rowSums(S_obs[!grepl('hormonal', colnames(S_obs))]))
names(S_obs_sum) = rownames(S_obs)
S_null_matrix_sb2 = abs(S_null_1000seed)
dim(S_null_matrix_sb2)
# S_null_matrix_sb2[1:3, 1:10]

unique_scores <- sort(unique(S_obs_sum), decreasing = TRUE)
FDR_table <- data.frame(score = unique_scores,
                        R_obs = NA,
                        V_null = NA,
                        FDR = NA)
for (i in seq_along(unique_scores)) {
  # i=1
  s0 <- unique_scores[i]
  
  R_obs <- sum(S_obs_sum >= s0)# 真实数据
  Vb <- colSums(S_null_matrix_sb2 >= s0)# null 每次 permutation 的 exceedance 数量
  
  E_V <- mean(Vb)# 期望假阳性数量
  
  FDR_value <- E_V / R_obs
  
  FDR_table$R_obs[i] <- R_obs
  FDR_table$V_null[i] <- E_V
  FDR_table$FDR[i] <- min(FDR_value, 1)
}
protein_FDR <- rep(NA, length(S_obs_sum))
names(protein_FDR) <- names(S_obs_sum)
for (i in 1:nrow(FDR_table)) {
  s0 <- FDR_table$score[i]
  protein_FDR[S_obs_sum == s0] <- FDR_table$FDR[i]
}
protein_FDR_result <- data.frame(
  protein = names(S_obs_sum),
  pertscore = S_obs_sum,
  empirical_FDR = protein_FDR
)
# protein_FDR_result$Gene = human_gene[rownames(protein_FDR_result), 2]
nrow(subset(protein_FDR_result, empirical_FDR<0.05))

S_obs_6sb2 = data.frame( row.names = rownames(S_obs))
for (i in c( "ALK", "antimitotic", "Kinase", "CDK",  "Topoisomerase", 'PARP')) {
  S_obs_6sb2[,i] = rowSums(S_obs[grepl(i, colnames(S_obs))])
}
write.xlsx(S_obs_6sb2, 'D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324_1000seeds_pertscore.xlsx', rowNames = T)
xlsx::write.xlsx(protein_FDR_result, 'D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324_1000seeds_pertscore.xlsx',append = T, sheetName = 'empirical_FDR')

S_obs_6sb2 = read.xlsx('D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324_1000seeds_pertscore.xlsx', sheet = 1, rowNames = T)
protein_FDR_result = read.xlsx('D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324_1000seeds_pertscore.xlsx', sheet = 'empirical_FDR', rowNames = T)

S_obs_6sb2 = subset(S_obs_6sb2, !grepl(':', rownames(S_obs_6sb2)) & !grepl(';', rownames(S_obs_6sb2)))
protein_FDR_result = subset(protein_FDR_result, !grepl(':', rownames(protein_FDR_result)) & !grepl(';', rownames(protein_FDR_result)))

library(UpSetR)
times = list()
for (cutoff in c(7,8,9,10,11,12 )) {
  df1 = S_obs_6sb2
  df1[abs(df1)<=cutoff] = NA
  narow = rowSums(!is.na(df1) )
  df1 = df1[narow!=0, ]
  times[[paste0('cutoff', cutoff)]] = rownames(df1)
}
names(times) #<- c("6hrs", "24hrs", "48hrs")

upset(fromList(times), 
      order.by = "freq",  # 主坐标系排序
      number.angles = 0,  # 柱标倾角
      nsets = 10,
      point.size = 3,  # 点大小
      line.size = 1,  # 线粗细
      #sets.x.label = "Datasets Size",  # x 标题
      set_size.show = T,
      #main.bar.color = "gray",
      #sets.bar.color = "gray",
      mainbar.y.label = "Count of Intersection",  # y 标题
      sets.x.label = "Datasets Size",  # x 标题
      text.scale = c(1.5, 1.5, 1.5, 1.5, 1.5, 1.5), # y 标题大小，y 刻度标签大小，datasetSize标题大小，datasetSize刻度标签大小，datasetSize分类标签大小，柱数字 大小
)

# empirical_FDR<0.05
upset_mat_FDR0.05 = protein_FDR_result[protein_FDR_result$empirical_FDR<0.05, ]
df1 = S_obs_6sb2[rownames(upset_mat_FDR0.05), ]
df1[abs(df1)<=10] = NA
narow = rowSums(!is.na(df1) )
df1 = df1[narow!=0, ]
df1$geneSymbol = human_gene[rownames(df1), 2]
colnames(df1)
df1 = df1[c("geneSymbol", "ALK", "antimitotic", "CDK", "Kinase", "PARP",  "Topoisomerase")]
write.xlsx(df1, 'D:/chh/2023workProject/prottalk/code/differ analysis20230822/Ttest20260324_1000seeds_pertscore_empiricalFDR0.05_absthan10.xlsx', rowNames = T)


times = list()
for (cutoff in c(7,8,9,10,11,12 )) {
  df1 = S_obs_6sb2[rownames(upset_mat_FDR0.05), ]
  df1[abs(df1)<=cutoff] = NA
  narow = rowSums(!is.na(df1) )
  df1 = df1[narow!=0, ]
  times[[paste0('cutoff', cutoff)]] = rownames(df1)
}
names(times) #<- c("6hrs", "24hrs", "48hrs")

upset(fromList(times), 
      order.by = "freq",  # 主坐标系排序
      number.angles = 0,  # 柱标倾角
      nsets = 10,
      point.size = 3,  # 点大小
      line.size = 1,  # 线粗细
      #sets.x.label = "Datasets Size",  # x 标题
      set_size.show = T,
      #main.bar.color = "gray",
      #sets.bar.color = "gray",
      mainbar.y.label = "Count of Intersection",  # y 标题
      sets.x.label = "Datasets Size",  # x 标题
      text.scale = c(1.5, 1.5, 1.5, 1.5, 1.5, 1.5), # y 标题大小，y 刻度标签大小，datasetSize标题大小，datasetSize刻度标签大小，datasetSize分类标签大小，柱数字 大小
)

library(ggsci)
line_dat =c()
for (i in names(times) ) {
  # i = "cutoff7"
  temp = upset_mat_FDR0.05[times[[i]], ]
  temp = temp[order(temp$empirical_FDR, decreasing = T), ]
  line_dat = rbind(line_dat, data.frame(rank_n = 1:nrow(temp), empirical_FDR = temp$empirical_FDR, cutoff = i))
}
line_dat = data.frame(line_dat)
head(line_dat)
line_dat$cutoff = factor(line_dat$cutoff, levels = unique(line_dat$cutoff))

ggplot(line_dat, aes(rank_n, empirical_FDR, color = cutoff))+
  geom_line()+
  scale_color_d3()+
  # geom_hline(yintercept = 0.05, linetype = "dashed")+
  theme_classic()+
  theme(text = element_text(size = 15, color = 'black'),
        axis.text = element_text(size = 15, color = 'black') 
  )

library(ggrepel)
df1 = S_obs_6sb2[rownames(upset_mat_FDR0.05), ]
df1$Gene = human_gene[rownames(df1), 2]

for (i in colnames(df1)[1:6]) {
  # i = 'ALK'
  df1.1 = df1[order(df1[i]), ]
  df1.1$rank = 1:nrow(df1.1)
  df1.1 = df1.1[c(i, 'rank', 'Gene')]
  df1.1 = subset(df1.1, !is.na(df1.1$Gene))
  colnames(df1.1)[1] = 'Pertscore'
  p1 = ggplot(df1.1, aes(rank, Pertscore))+
    geom_point(size=1)+
    geom_text_repel(data =df1.1[c(1:10, nrow(df1.1):(nrow(df1.1)-9)), ] , aes(rank, Pertscore, label =  Gene), color = 'red', max.overlaps = Inf)+
    labs(title = i)+
    theme_classic()+
    theme(text = element_text(size = 15, color = 'black'),
          axis.text = element_text(size = 15, color = 'black'),
          axis.title = element_blank()
    )
  ggsave(paste0('//172.16.13.136/share/members/sunr/PTV1/PTV1_honghan/empirical_FDR/subtype_pertscore_plot_FDR0.05_', i, '.pdf'), p1)
}
