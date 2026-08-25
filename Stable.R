library(openxlsx)
library(reshape2)
library(dplyr)
library(ggplot2)
library(ggsci)
library(ggpubr)
#### TableS5 ####
df1 = read.xlsx("D:/chh/2025workProject/20250506PTV1/PTV1_check/04_sTable/TableS5_cross_cell_20251030.xlsx", sheet = 4)
head(df1)
for (i in 3:ncol(df1)) {
  df1[,i] = as.numeric(df1[,i])
}

# df1_sum = df1 %>% 
#   group_by(Model) %>%
#   summarise(
#     mean_prc = mean(AUPRC, na.rm = T),
#     sd_prc = sd(AUPRC, na.rm = T),
#     se_prc = sd(AUPRC, na.rm = T)/sqrt(length(na.omit(AUPRC))),
#     mean_roc = mean(AUROC, na.rm = T),
#     sd_roc = sd(AUROC, na.rm = T),
#     mean_acc = mean(Accuracy, na.rm = T),
#     sd_acc = sd(Accuracy, na.rm = T),
#     val_num = n()
#   )
# df1_sum$se_prc = df1_sum$sd_prc/sqrt(df1_sum$val_num)
# df1_sum$se_roc = df1_sum$sd_roc/sqrt(df1_sum$val_num)
# df1_sum$se_acc = df1_sum$sd_acc/sqrt(df1_sum$val_num)
# df1_sum


temp = subset(df1, !is.na(df1$AUPRC))
colnames(temp) = gsub('AUPRC', 'val', colnames(temp))
PRC_sum = temp %>% 
  group_by(Model) %>%
  summarise(
    mean_val = mean(val, na.rm = T),
    sd_val = sd(val, na.rm = T),
    val_num = n()
  )
PRC_sum$se_val = PRC_sum$sd_val/sqrt(PRC_sum$val_num)
PRC_sum

colnames(df1)
temp = subset(df1, !is.na(df1$AUROC))# "Accuracy"  "AUROC"     "AUPRC"
colnames(temp) = gsub('AUROC', 'val', colnames(temp))
PRC_sum = temp %>% 
  group_by(Model) %>%
  summarise(
    mean_val = mean(val, na.rm = T),
    sd_val = sd(val, na.rm = T),
    val_num = n()
  )
PRC_sum$se_val = PRC_sum$sd_val/sqrt(PRC_sum$val_num)
PRC_sum

temp = subset(df1, !is.na(df1$Accuracy))# "Accuracy"  "AUROC"     "AUPRC"
colnames(temp) = gsub('Accuracy', 'val', colnames(temp))
PRC_sum = temp %>% 
  group_by(Model) %>%
  summarise(
    mean_val = mean(val, na.rm = T),
    sd_val = sd(val, na.rm = T),
    val_num = n()
  )
PRC_sum$se_val = PRC_sum$sd_val/sqrt(PRC_sum$val_num)
PRC_sum



unique(df1$Model)
res = c()
for (mm in c("genecompass", "geneformer", "uce", "KNN", "Random Forest", "Bootstrap", "Deepsynergy")) {
  for (item in c("Accuracy", "AUPRC", "AUROC" )) {
    pre = df1[df1$Model %in% "ppODE (SWA)", item]
    post = df1[df1$Model %in% mm, item]
    p1 = t.test(pre, post)
    # print(c(mm, item, p1$p.value))
    res = rbind(res, c(mm, item, p1$p.value))
  }
}
View(res)


dfc = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/00_check/PTV1_check/04_sTable/TableS5_cross_cell_20251030.xlsx", sheet = 4)
head(dfc)
dfc[3:ncol(dfc)] = apply(dfc[3:ncol(dfc)], 2, function(x) as.numeric(x))
colnames(dfc)
dfc[500:505, ]
for (i in c("Accuracy",  "AUROC",  "AUPRC", "R2.score",  "Pearson")) {
  dfc[, i] = round(dfc[, i], 3)
}

write.xlsx(dfc, "//172.16.13.136/share/members/chenghonghan/PTV1/proof_check20260608/stable5_C_20260615-1035.xlsx")

#### TableS6 ####
# df_ts6 = read.csv("D:/chh/2025workProject/20250506PTV1/PTV1_check/04_sTable/TableS6_cross_drug20251030.xlsx")
df_ts6 = read.csv("D:/chh/2025workProject/20250506PTV1/ptds123_cross_drug_0922/crossdrug_combined_metrics.csv")
head(df_ts6)
for (i in c('AUPRC', 'AUROC', 'Accuracy' )) {
  df_ts6[i] = as.numeric(df_ts6[,i])
}
df_ts6$method = gsub('uce', 'UCE', df_ts6$method)
df_ts6$method = gsub('gf', 'Geneformer', df_ts6$method)
df_ts6$method = gsub('gc', 'GeneCompass', df_ts6$method)
df_ts6$method = gsub('ppODE (SWA)', 'ProteinTalks', df_ts6$method, fixed = T)
unique(df_ts6$method)

stat_AUPRC <- df_ts6[!is.na(df_ts6$AUPRC), ] %>%
  group_by( method) %>%
  summarise(
    n= n(), 
    mean=mean(AUPRC, na.rm = T), sd=sd(AUPRC, na.rm = T), se=sd(AUPRC, na.rm = T)/sqrt(n)
  )
stat_AUPRC = data.frame(stat_AUPRC)

stat_AUROC <- df_ts6[!is.na(df_ts6$AUROC), ] %>%
  group_by( method) %>%
  summarise(
    n=n( ), 
    mean=mean(AUROC, na.rm = T), sd=sd(AUROC, na.rm = T), se=sd(AUROC, na.rm = T)/sqrt(n)
  )
stat_AUROC = data.frame(stat_AUROC)

stat_Accuracy <- df_ts6[!is.na(df_ts6$Accuracy), ] %>%
  group_by( method ) %>%
  summarise(
    n=n( ), 
    mean=mean(Accuracy, na.rm = T), sd=sd(Accuracy, na.rm = T), se=sd(Accuracy, na.rm = T)/sqrt(n)
  )
stat_Accuracy = data.frame(stat_Accuracy)

temp = stat_AUPRC
rownames(temp) = temp$method
temp$AUPRC = paste0(signif(temp$mean, 3), '±', signif(temp$se, 3))
stat_res = temp

temp = stat_AUROC
rownames(temp) = temp$method
stat_res[rownames(temp), 'AUROC'] = paste0(signif(temp$mean, 3), '±', round(temp$se, 3))

temp = stat_Accuracy
rownames(temp) = temp$method
stat_res[rownames(temp), 'Accuracy'] = paste0(signif(temp$mean, 3), '±', signif(temp$se, 3))
stat_res

hists = combn(c('ProteinTalks', "GeneCompass", "Geneformer" , "UCE", 'Deepsynergy', 'Bootstrap', 'Random Forest', 'KNN'), 2)
hist_pair = list()
for (i in seq(1, length(hists), 2)) {
  if(grepl('ProteinTalks', hists[i], ignore.case = T) | grepl('ProteinTalks', hists[i+1], ignore.case = T)){
    
    hist_pair[[length(hist_pair)+1]] = hists[i:(i+1)]
  }
  # if(length(hist_pair)==7)break
}
hist_pair

temp = df_ts6#[c('drug', 'method',  'Accuracy')]
temp = subset(temp, method %in% c('ProteinTalks', "GeneCompass", "Geneformer" , "UCE", 'Deepsynergy', 'Bootstrap', 'Random Forest', 'KNN' ))
temp$method = factor(temp$method, levels = c('ProteinTalks', "GeneCompass", "Geneformer" , "UCE", 'Deepsynergy', 'Bootstrap', 'Random Forest', 'KNN' ))
ggplot(temp, aes(method, AUPRC, fill = method))+
  geom_bar(stat = "summary", fun ="mean", position = position_dodge(), width = 0.5) +#, alpha=0.7
  stat_summary(fun.data = 'mean_se', geom = "errorbar", colour = "black",
               width = 0.15,position = position_dodge( .9))+
  geom_signif( # 添加显著性标签
    comparisons = hist_pair, # 可以试试用combn(c("A", "B", "C" ), 2)生成
    step_increase = 0.1, na.rm = T, tip_length = 0,
    test="t.test",  # "t 检验，比较两组（参数）" = "t.test","Wilcoxon 符号秩检验，比较两组（非参数）" = "wilcox.test"
    map_signif_level = T   # 标签样式F为数字，T为*号
  )+ 
  geom_text(aes(method, mean, label = paste0(round(mean, 3), '±', round(se, 3))), 
            data = subset(stat_AUPRC, method %in% c('ProteinTalks', "GeneCompass", "Geneformer" , "UCE", 'Deepsynergy', 'Bootstrap', 'Random Forest', 'KNN' ) ),
            position = position_dodge(0.9), size = 3 , vjust = -1)+
  scale_y_continuous(n.breaks = 10)+
  theme_bw()+
  theme(text = element_text(size = 15 ),
        axis.text.x = element_text(size = 13, angle = 90, hjust = 1, color = 'black'),#
        axis.text.y = element_text(size = 13, color = 'black'),
        panel.grid = element_blank())+
  scale_fill_manual(values = c('#E7211A', "#67AD91", '#4E4EBE' , '#F7E673', '#8B64A8' , '#89CCC0', '#ADD26A', '#7FABCB','#B4C7EC' ))+
  labs(x = 'ppODE vs other methods')

ts6_B=c()
for (i in c("GeneCompass", "Geneformer", "UCE", "KNN", "Bootstrap", "Random Forest", "Deepsynergy")) {
  pre = subset(df_ts6, method %in% 'ProteinTalks')
  post = subset(df_ts6, method %in% i)
  for (j  in c('AUPRC', 'AUROC','Accuracy')) {
    pval = t.test(pre[, j], post[, j])
    ts6_B = rbind(ts6_B, c(i, j, signif(pval$p.value, 3)))
  }
}
View(ts6_B)

stat_AUPRC <- df_ts6[!is.na(df_ts6$AUPRC), ] %>%
  group_by( method, drug) %>%
  summarise(
    n= n(), 
    mean=mean(AUPRC, na.rm = T), sd=sd(AUPRC, na.rm = T), se=sd(AUPRC, na.rm = T)/sqrt(n)
  )
stat_AUPRC = data.frame(stat_AUPRC)

stat_AUROC <- df_ts6[!is.na(df_ts6$AUROC), ] %>%
  group_by( method, drug) %>%
  summarise(
    n=n( ), 
    mean=mean(AUROC, na.rm = T), sd=sd(AUROC, na.rm = T), se=sd(AUROC, na.rm = T)/sqrt(n)
  )
stat_AUROC = data.frame(stat_AUROC)

stat_Accuracy <- df_ts6[!is.na(df_ts6$Accuracy), ] %>%
  group_by( method, drug ) %>%
  summarise(
    n=n( ), 
    mean=mean(Accuracy, na.rm = T), sd=sd(Accuracy, na.rm = T), se=sd(Accuracy, na.rm = T)/sqrt(n)
  )
stat_Accuracy = data.frame(stat_Accuracy)

temp = stat_AUPRC
rownames(temp) = paste0(temp$method, temp$drug)
temp$AUPRC = paste0(signif(temp$mean, 3), '±', signif(temp$se, 3))
stat_res = temp

temp = stat_AUROC
rownames(temp) = paste0(temp$method, temp$drug)
stat_res[rownames(temp), 'AUROC'] = paste0(signif(temp$mean, 3), '±', round(temp$se, 3))

temp = stat_Accuracy
rownames(temp) = paste0(temp$method, temp$drug)
stat_res[rownames(temp), 'Accuracy'] = paste0(signif(temp$mean, 3), '±', signif(temp$se, 3))
stat_res[rownames(temp), c('method', 'drug')] = temp[c('method', 'drug')]
View(stat_res) # tableS6 C
# write.xlsx(stat_res, 'D:/chh/2025workProject/20250506PTV1/ptds123_cross_drug_0922/method_drug_stat.xlsx')


# for (i in c("GeneCompass", "Geneformer", "UCE", "KNN", "Bootstrap", "Random Forest", "Deepsynergy")) {
#   for(d in unique(df_ts6$drug)){
#     pre = subset(df_ts6, method %in% 'ProteinTalks' & drug == d)
#     post = subset(df_ts6, method %in% i & drug == d)
#     if(nrow(pre)<2 | nrow(post) < 2) next
#     for (j  in c('AUPRC', 'AUROC','Accuracy')) {
#       y = try(t.test(pre[, j], post[, j]), silent = T)
#       if('try-error' %in% class(y)) next
#       pval = t.test(pre[, j], post[, j])
#       print(c(i, d, j, signif(pval$p.value, 3)))
#     }
#   }
# }


#### TableS7 ####
df1 = read.csv("//172.16.13.136/share/members/sunr/PTV1/00_check/PTV1_check_20260603/Figures/Fig_PTDS4_2026/Fig_PTDS4_2026/Fig_PTDS4_2026_metricsPERRUN.csv")
head(df1)
df1_summ = df1 %>% 
  group_by(method) %>%
  summarise(
    mean_AUPRC = mean(AUPRC),
    se_AUPRC = sd(AUPRC)/sqrt(5),
    mean_AUROC = mean(AUROC),
    se_AUROC = sd(AUROC)/sqrt(5)
  )
df1_summ = df1_summ[df1_summ$method %in% c( "GeneCompass", "Geneformer", "ProteinTalks", "UCE"), ]
# df1_summ$se_AUPRC = df1_summ$sd_AUPRC/sqrt(5)
# df1_summ$se_AUROC = df1_summ$sd_AUROC/sqrt(5)
df1_summ$meanseAUPRC = paste0(signif(df1_summ$mean_AUPRC, 3), ' ± ', signif(df1_summ$se_AUPRC, 3))
df1_summ$meanseAUROC = paste0(signif(df1_summ$mean_AUROC, 3), ' ± ', signif(df1_summ$se_AUROC, 3))
df1_summ

#### TableS8 ####
"D:\chh\2025workProject\20250506PTV1\PTV1_check\04_sTable\TableS8_PTDS-5_info_20251030.xlsx"
#### TableS9 2025 ####

#### TableS9 2026 ####
"D:/chh/2025workProject/20250506PTV1/PTV1_check/04_sTable/TableS9_PTDS-5_20250917.xlsx"
ptds5_val = read.csv("D:/chh/2025workProject/20250506PTV1/ptds5_significance_excel/ptds5_ppODE_ML_benchmark_res.csv")
ptds5_val = subset(ptds5_val, Models %in% 'ppODE_swa2')
colnames(ptds5_val)
ptds5_val = ptds5_val[c("Models", "Accuracy", "AUROC", "AUPRC" )]
temp = read.xlsx("D:/chh/2025workProject/20250506PTV1/ptds5_significance_excel/ptds5_scfm_as_input_runs_detail.xlsx")
colnames(temp)
temp = temp[c("foundation_model" , "accuracy" ,"auroc" , "auprc")]
colnames(temp) = c("Models", "Accuracy", "AUROC", "AUPRC" )
ptds5_val = rbind(ptds5_val, temp)
head(ptds5_val)
for (i in c('Accuracy',  'AUROC',  'AUPRC' )) {
  # i='uce'
  p1 = t.test(ptds5_val[ptds5_val$Models %in% 'ppODE_swa2',  i], ptds5_val[ptds5_val$Models %in% "genecompass",  i], )
  p2 = t.test(ptds5_val[ptds5_val$Models %in% 'ppODE_swa2',  i], ptds5_val[ptds5_val$Models %in% "geneformer",  i])
  p3 = t.test(ptds5_val[ptds5_val$Models %in% 'ppODE_swa2',  i], ptds5_val[ptds5_val$Models %in%  "uce" ,  i])
  print(c(i, p1$p.value, p2$p.value, p3$p.value))
}

temp = melt(ptds5_val, id.vars = 'Models')


res = ptds5_val
head(res)
stat_AUPRC <- res[!is.na(res$AUPRC), ] %>%
  group_by( Models) %>%#, drug
  summarise(
    n=n( ), 
    mean=mean(AUPRC, na.rm = T), sd=sd(AUPRC, na.rm = T), se=sd(AUPRC, na.rm = T)/sqrt(n)
  )
stat_AUPRC = data.frame(stat_AUPRC)
stat_AUPRC
unique(stat_AUPRC$Models)

stat_AUROC <- res[!is.na(res$AUROC), ] %>%
  group_by( Models) %>%
  summarise(
    n=n( ), 
    mean=mean(AUROC, na.rm = T), sd=sd(AUROC, na.rm = T), se=sd(AUROC, na.rm = T)/sqrt(n)
  )
stat_AUROC = data.frame(stat_AUROC)

stat_Accuracy <- res[!is.na(res$Accuracy), ] %>%
  group_by( Models ) %>%
  summarise(
    n=n( ), 
    mean=mean(Accuracy, na.rm = T), sd=sd(Accuracy, na.rm = T), se=sd(Accuracy, na.rm = T)/sqrt(n)
  )
stat_Accuracy = data.frame(stat_Accuracy)

head(stat_AUPRC)

temp = stat_AUPRC
rownames(temp) = paste0(temp$Models)#, '_', temp$drug
temp$AUPRC = paste0(round(temp$mean, 4), '±', round(temp$sd, 4))
ptds_count = temp

temp = stat_Accuracy
rownames(temp) = paste0(temp$Models)#, '_', temp$drug
ptds_count[rownames(temp), 'Accuracy'] = paste0(round(temp$mean, 4), '±', round(temp$sd, 4))

temp = stat_AUROC
rownames(temp) = paste0(temp$Models)#, '_', temp$drug
ptds_count[rownames(temp), 'AUROC'] = paste0(round(temp$mean, 4), '±', round(temp$sd, 4))
colnames(ptds_count)
head(ptds_count)

temp = stat_AUPRC
temp$Models = factor(temp$Models, levels = c("ppODE_swa2", "genecompass", "geneformer", 'uce' ))
p = ggplot(temp, aes(Models, mean))+
  geom_bar(stat = 'identity', position = "dodge", width = 0.5)+
  geom_errorbar(aes(ymax = mean+se, ymin = mean-se), position = position_dodge(0.9), width = 0.2)+
  geom_text(aes(label = paste0(round(mean, 3), ' ± ', round(se, 3))),
            position = position_dodge(0.9), size = 3 , vjust = -1)+
  theme_bw()+ #scale_y_continuous(n.breaks = 8)+
  ylim(0,1)+
  theme(text = element_text(size = 15 ),
        axis.text.x = element_text(size = 13, angle = 90, hjust = 1, color = 'black'),#
        axis.text.y = element_text(size = 13, color = 'black'),
        panel.grid = element_blank())
p


temp = stat_AUROC
temp$Models = factor(temp$Models, levels = c("ppODE_swa2", "genecompass", "geneformer", 'uce' ))
p = ggplot(temp, aes(Models, mean))+
  geom_bar(stat = 'identity', position = "dodge", width = 0.5)+
  geom_errorbar(aes(ymax = mean+se, ymin = mean-se), position = position_dodge(0.9), width = 0.2)+
  geom_text(aes(label = paste0(round(mean, 3), ' ± ', round(se, 3))),
            position = position_dodge(0.9), size = 3 , vjust = -1)+
  theme_bw()+ #scale_y_continuous(n.breaks = 8)+
  ylim(0,1)+
  theme(text = element_text(size = 15 ),
        axis.text.x = element_text(size = 13, angle = 90, hjust = 1, color = 'black'),#
        axis.text.y = element_text(size = 13, color = 'black'),
        panel.grid = element_blank())
p

temp = stat_Accuracy
temp$Models = factor(temp$Models, levels = c("ppODE_swa2", "genecompass", "geneformer", 'uce' ))
p = ggplot(temp, aes(Models, mean))+
  geom_bar(stat = 'identity', position = "dodge", width = 0.5)+
  geom_errorbar(aes(ymax = mean+se, ymin = mean-se), position = position_dodge(0.9), width = 0.2)+
  geom_text(aes(label = paste0(round(mean, 3), ' ± ', round(se, 3))),
            position = position_dodge(0.9), size = 3 , vjust = -1)+
  theme_bw()+ #scale_y_continuous(n.breaks = 8)+
  ylim(0,1)+
  theme(text = element_text(size = 15 ),
        axis.text.x = element_text(size = 13, angle = 90, hjust = 1, color = 'black'),#
        axis.text.y = element_text(size = 13, color = 'black'),
        panel.grid = element_blank())
p

#### TableS10 ####

#### TableS11 ####
#### TableS12 ####
res = read.xlsx('D:/chh/2025workProject/20250506PTV1/L/pdx2015nm_ML/pdx2015nm_combined_metrics.xlsx')
for (i in c('Accuracy', 'AUROC', 'AUPRC')) {
  res[i] = as.numeric(res[,i])
}
unique(res$Models)
res = subset(res,  !(Models %in% c( 'ppode_nonswa', 'ppode_swa',  'Deepsynergy','Logistic Regression','SGD' )))
res[res$Models %in% 'Bootstrap (Bagging with Decision Tree)', 'Models'] = 'Bootstrap'
res = res[res$Models != 'Logistic Regression', ]
#head(res)
unique(res$Models)
## method 
stat_AUPRC <- res[!is.na(res$AUPRC), ] %>%
  group_by( Models) %>%#, drug
  summarise(
    n=n( ), 
    mean=mean(AUPRC, na.rm = T), sd=sd(AUPRC, na.rm = T), se=sd(AUPRC, na.rm = T)/sqrt(n)
  )
stat_AUPRC = data.frame(stat_AUPRC)

stat_AUROC <- res[!is.na(res$AUROC), ] %>%
  group_by( Models) %>%
  summarise(
    n=n( ), 
    mean=mean(AUROC, na.rm = T), sd=sd(AUROC, na.rm = T), se=sd(AUROC, na.rm = T)/sqrt(n)
  )
stat_AUROC = data.frame(stat_AUROC)

stat_Accuracy <- res[!is.na(res$Accuracy), ] %>%
  group_by( Models ) %>%
  summarise(
    n=n( ), 
    mean=mean(Accuracy, na.rm = T), sd=sd(Accuracy, na.rm = T), se=sd(Accuracy, na.rm = T)/sqrt(n)
  )
stat_Accuracy = data.frame(stat_Accuracy)

head(stat_AUPRC)

temp = stat_AUPRC
#temp = subset(temp, drug %in% drugs &  Models %in% c('ppODE (SWA)', 'gc', 'gf', 'uce') )
rownames(temp) = paste0(temp$Models)#, '_', temp$drug
temp$AUPRC = paste0(signif(temp$mean, 3), '±', signif(temp$se, 3))
pdxnm_count = temp

temp = stat_Accuracy
# temp = subset(temp, drug %in% drugs &  Models %in% c('ppODE (SWA)', 'gc', 'gf', 'uce') )
rownames(temp) = paste0(temp$Models)#, '_', temp$drug
pdxnm_count[rownames(temp), 'Accuracy'] = paste0(signif(temp$mean, 3), '±', signif(temp$se, 3))

temp = stat_AUROC
# temp = subset(temp, drug %in% drugs &  Models %in% c('ppODE (SWA)', 'gc', 'gf', 'uce') )
rownames(temp) = paste0(temp$Models)#, '_', temp$drug
pdxnm_count[rownames(temp), 'AUROC'] = paste0(signif(temp$mean, 3), '±', signif(temp$se, 3))
colnames(pdxnm_count)
head(pdxnm_count)

# write.xlsx(pdxnm_count[c("Models","Accuracy","AUPRC", "AUROC")], 'D:/chh/2025workProject/20250506PTV1/L/pdx2015nm_ML/pdx2015nm_combined_metrics_meansd.xlsx')


hists = combn(unique(res$Models), 2)
hist_pair = list()
for (i in seq(1, length(hists), 2)) {
  if(grepl('ppODE', hists[i], ignore.case = T) | grepl('ppODE', hists[i+1], ignore.case = T)){
    
    hist_pair[[length(hist_pair)+1]] = hists[i:(i+1)]
  }
  # if(length(hist_pair)==7)break
}
hist_pair


temp = c()
for (pair in 1:length(hist_pair)) {
  pair = hist_pair[[pair]]
  i = pair[1]
  j = pair[2]
  
  pvalacc = t.test(res[res$Models == i, 'Accuracy'], res[res$Models == j, 'Accuracy'])
  pvalprc = t.test(res[res$Models == i, 'AUPRC'], res[res$Models == j, 'AUPRC'])
  pvalroc = t.test(res[res$Models == i, 'AUROC'], res[res$Models == j, 'AUROC'])
  temp = rbind(temp, c(i, j, pvalacc$p.value, pvalprc$p.value, pvalroc$p.value))
  #print(c(i, j, pvalacc$p.value, pvalprc$p.value, pvalroc$p.value))
}
temp = data.frame(temp)
temp = temp[order(temp$X1, temp$X2), ]
colnames(temp)[3:5] = c("Accuracy", "AUPRC", "AUROC")
temp$mm = paste0(temp$X1, ' vs ', temp$X2)
temp1 = melt(temp[3:6], id.vars = 'mm')
head(temp1)
temp1$value = as.numeric(temp1$value)
temp1$value = signif(temp1$value, 3)
for (i in 1:nrow(temp1)) {
  # if(temp1[i, 'value']<0.00001) temp1[i, 'Significant'] = '***** (p < 0.00001)'
  # else if(temp1[i, 'value']<0.0001) temp1[i, 'Significant'] = '**** (p < 0.0001)'
  # else if(temp1[i, 'value']<0.001) temp1[i, 'Significant'] = '*** (p < 0.001)'
  if(temp1[i, 'value']<0.001) temp1[i, 'Significant'] = '*** (p < 0.001)'
  else if(temp1[i, 'value']<0.01) temp1[i, 'Significant'] = '** (p < 0.01)'
  else if(temp1[i, 'value']<0.05) temp1[i, 'Significant'] = '* (p < 0.05)'
  else temp1[i, 'Significant'] = 'ns (p >= 0.05)'
}
View(temp1)
# write.xlsx(temp1, 'D:/chh/2025workProject/20250506PTV1/L/pdx2015nm_ML/pdx2015nm_combined_metrics_ttest.xlsx')


temp = res
unique(temp$Models)
temp$Models = factor(temp$Models, levels = c('ppODE (SwaSwa)', 'genecompass', 'geneformer', 'uce', 'Deepsynergy', 'Bootstrap', 'KNN', 'Random Forest' ))
ggplot(temp, aes(Models, AUPRC, fill = Models))+
  geom_bar(stat = "summary", fun ="mean", position = position_dodge(), alpha=1, width = 0.5) +
  #ylim(0, 1.5)+
  scale_fill_manual(values = c('#E7211A', "#67AD91", '#4E4EBE' , '#F7E673', '#8B64A8', '#ADD26A' , '#89CCC0', '#7FABCB' ))+
  stat_summary(fun.data = 'mean_se', geom = "errorbar", colour = "black",
               width = 0.15,position = position_dodge( .9)) +
  geom_signif( # 添加显著性标签
    comparisons = hist_pair, # 可以试试用combn(c("A", "B", "C" ), 2)生成
    step_increase = 0.1, na.rm = T, tip_length = 0,
    test="t.test",  # "t 检验，比较两组（参数）" = "t.test","Wilcoxon 符号秩检验，比较两组（非参数）" = "wilcox.test"
    map_signif_level = T   # 标签样式F为数字，T为*号
  )+  geom_text(aes(Models, mean, label = paste0(signif(mean, 3), '±', signif(se, 3))), 
                data = subset(stat_AUPRC, Models %in% c('ppODE (SwaSwa)', 'genecompass', 'geneformer', 'uce', 'Bootstrap', 'Random Forest', 'KNN' ) ),
                position = position_dodge(0.9), size = 3 , vjust = -1)+
  scale_y_continuous(n.breaks = 10)+
  theme_bw()+
  theme(text = element_text(size = 15 ),
        axis.text.x = element_text(size = 13, angle = 90, hjust = 1, color = 'black'),#
        axis.text.y = element_text(size = 13, color = 'black'),
        panel.grid = element_blank())+
  labs(x = '')
#### TableS13 ####
# D:/chh/2025workProject/20250506PTV1/Prognosis/PTV1_Prognosis_unicox_liutongFFPE.R
df_ffpe = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/PTV1_rebuttal/00_PTV1_revised_202602/504FFPE_model/ptv1_ffpe_501patient_matinfo_20260410-1102.xlsx", sheet = 1)
colnames(df_ffpe)[1:10]
df_info = df_ffpe[1:9]

#### TableS14 ####