rm(list = ls());gc()
library(openxlsx)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(reshape2)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#### Supplementary information Figure 1 ####
path_cur = getwd()
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
ttest_res = c()
for (i in c('Accuracy',  'AUROC',  'AUPRC' )) {
  # i='uce'
  p1 = t.test(ptds5_val[ptds5_val$Models %in% 'ppODE_swa2',  i], ptds5_val[ptds5_val$Models %in% "genecompass",  i], )
  p2 = t.test(ptds5_val[ptds5_val$Models %in% 'ppODE_swa2',  i], ptds5_val[ptds5_val$Models %in% "geneformer",  i])
  p3 = t.test(ptds5_val[ptds5_val$Models %in% 'ppODE_swa2',  i], ptds5_val[ptds5_val$Models %in%  "uce" ,  i])
  print(c(i, p1$p.value, p2$p.value, p3$p.value))
  ttest_res = rbind(ttest_res, c(i, p1$p.value, p2$p.value, p3$p.value))
}

ttest_res = data.frame(ttest_res)
ttest_res[2:ncol(ttest_res)] = apply(ttest_res[2:ncol(ttest_res)], 2, as.numeric)
colnames(ttest_res)[2:4] = c("Genecompass", "Geneformer", "UCE" )
write.xlsx(ttest_res, 'D:/chh/2025workProject/20250506PTV1/ptds5_significance_excel/ptds5_AUPRC_AUROC_ACC_ttest20260611.xlsx')
temp = melt(ptds5_val, id.vars = 'Models')


res = ptds5_val
res$Models[res$Models %in% "ppODE_swa2"] = 'ProteinTalks'
res$Models[res$Models %in% "genecompass"] = 'Genecompass'
res$Models[res$Models %in% "geneformer"] = 'Geneformer'
res$Models[res$Models %in% "uce"] = 'UCE'
res$Models
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
temp$Models = factor(temp$Models, levels = c("ProteinTalks", "Genecompass", "Geneformer", 'UCE' ))
p1 = ggplot(temp, aes(Models, mean ))+
  geom_bar(aes(color = Models), stat = 'identity', position = "dodge", width = 0.5, fill = 'white')+
  geom_errorbar(aes(ymax = mean+se, ymin = mean-se), position = position_dodge(0.9), width = 0.2)+
  geom_text(aes(label = paste0(round(mean, 3), ' ± ', round(se, 3))),
            position = position_dodge(0.9), size = 3 , vjust = -1)+
  theme_bw()+ #scale_y_continuous(n.breaks = 8)+
  ylim(0,1)+
  geom_point(data = res, aes(Models, AUPRC, color = Models))+
  scale_color_manual(values = c('#E7211A', '#4E4EBE' ,"#67AD91",  '#F7E673' ))+
  theme(text = element_text(size = 15 ),
        axis.text.x = element_text(size = 13, angle = 90, hjust = 1, color = 'black'),#
        axis.text.y = element_text(size = 13, color = 'black'),
        panel.grid = element_blank(),
        legend.position = 'none' , axis.title = element_blank()
        )
p1
# ggsave(paste0('D:/chh/2025workProject/20250506PTV1/ptds5_significance_excel/ptds5_AUPRC_20260331.pdf'), p, width = 4, height = 4)


temp = stat_Accuracy
temp$Models = factor(temp$Models, levels =  c("ProteinTalks", "Genecompass", "Geneformer", 'UCE' ))
p2 = ggplot(temp, aes(Models, mean ))+
  geom_bar(aes(color = Models), stat = 'identity', position = "dodge", width = 0.5, fill = 'white')+
  geom_errorbar(aes(ymax = mean+se, ymin = mean-se), position = position_dodge(0.9), width = 0.2)+
  geom_text(aes(label = paste0(round(mean, 3), ' ± ', round(se, 3))),
            position = position_dodge(0.9), size = 3 , vjust = -1)+
  theme_bw()+ #scale_y_continuous(n.breaks = 8)+
  ylim(0,1)+
  geom_point(data = res, aes(Models, Accuracy, color = Models))+
  scale_color_manual(values = c('#E7211A', '#4E4EBE' ,"#67AD91",  '#F7E673' ))+
  theme(text = element_text(size = 15 ),
        axis.text.x = element_text(size = 13, angle = 90, hjust = 1, color = 'black'),#
        axis.text.y = element_text(size = 13, color = 'black'),
        panel.grid = element_blank(),
        legend.position = 'none' , axis.title = element_blank()
  )
# ggsave(paste0('D:/chh/2025workProject/20250506PTV1/ptds5_significance_excel/ptds5_Accuracy_20260331.pdf'), p, width = 4, height = 4)


temp = stat_AUROC
temp$Models = factor(temp$Models, levels =  c("ProteinTalks", "Genecompass", "Geneformer", 'UCE' ))
p3 = ggplot(temp, aes(Models, mean ))+
  geom_bar(aes(color = Models), stat = 'identity', position = "dodge", width = 0.5, fill = 'white')+
  geom_errorbar(aes(ymax = mean+se, ymin = mean-se), position = position_dodge(0.9), width = 0.2)+
  geom_text(aes(label = paste0(round(mean, 3), ' ± ', round(se, 3))),
            position = position_dodge(0.9), size = 3 , vjust = -1)+
  theme_bw()+ #scale_y_continuous(n.breaks = 8)+
  ylim(0,1)+
  geom_point(data = res, aes(Models, AUROC, color = Models))+
  scale_color_manual(values = c('#E7211A', '#4E4EBE' ,"#67AD91",  '#F7E673' ))+
  theme(text = element_text(size = 15 ),
        axis.text.x = element_text(size = 13, angle = 90, hjust = 1, color = 'black'),#
        axis.text.y = element_text(size = 13, color = 'black'),
        panel.grid = element_blank(),
        legend.position = 'none' , axis.title = element_blank()
  )
# ggsave(paste0('D:/chh/2025workProject/20250506PTV1/ptds5_significance_excel/ptds5_AUROC_20260331.pdf'), p, width = 4, height = 4)

p1+p3+p2
ggsave(paste0('D:/chh/2025workProject/20250506PTV1/ptds5_significance_excel/ptds5_AUPRC_AUROC_ACC20260609-1905.pdf'), p1+p3+p2, width = 9.5)
#### Supplementary information Figure 2 ####
library(ggpubr)
rm(list = ls());gc()
# sheets = readxl::excel_sheets("D:/chh/2025workProject/20250506PTV1/PTV1_15prot/西湖IHC评分.xlsx")
# sheets
# IHC = c()
# for (i in 1:length(sheets)) {
#   # i=1
#   temp = read.xlsx("D:/chh/2025workProject/20250506PTV1/PTV1_15prot/西湖IHC评分.xlsx", sheet = i)
#   # print(colnames(temp))
#   
#   pre = temp[grepl('high', temp$患者, ignore.case = T) , 4]
#   post = temp[grepl('low', temp$患者, ignore.case = T) , 4]
#   p1 = t.test(pre, post)
#   
#   pre = temp[grepl('high', temp$患者, ignore.case = T) , 8]
#   post = temp[grepl('low', temp$患者, ignore.case = T) , 8]
#   p2 = t.test(pre, post)
#   
#   pre = temp[grepl('high', temp$患者, ignore.case = T) , 9]
#   post = temp[grepl('low', temp$患者, ignore.case = T) , 9]
#   p3 = t.test(pre, post)
#   print(sheets[i])#, '总分1'
#   print(c(p1$p.value, p2$p.value, p3$p.value))
#   IHC = rbind(IHC, cbind(protein = sheets[i],Group = temp$患者, value = temp$`总分1（=染色强度*阳性细胞比例）`, type = rep('sum1', nrow(temp))))
#   IHC = rbind(IHC, cbind(protein = sheets[i],Group = temp$患者, value = temp$`总分2（=染色强度*阳性细胞比例）`, type = rep('sum2', nrow(temp))))
#   IHC = rbind(IHC, cbind(protein = sheets[i], Group = temp$患者, value = temp$平均分, type = rep('mean_val', nrow(temp))))
# }
# IHC = data.frame(IHC)
# IHC$value = as.numeric(IHC$value)
# IHC$Group = gsub('.*-', '', IHC$Group)
# IHC$Group = toupper(gsub('[^a-zA-z]', '', IHC$Group))
# head(IHC)
# IHC = subset(IHC, protein %in% c("PALLD",   "RCOR3",   "ZFYVE19" ))
# 
# plot_list = list()
# for (i  in unique(IHC$protein)) {
#   # i = 'PALLD'
#   p = ggplot(IHC[IHC$protein %in% i, ], aes(type, value, color = Group))+
#     geom_boxplot( ) +
#     geom_point(position = position_jitterdodge(jitter.width = 0.2))+
#     stat_compare_means(method = "t.test", label = "p.signif")+
#     labs(title = i)+
#     theme_classic()+
#     theme(axis.text = element_text(size = 13, color = 'black'))
#   # p
#   plot_list[[length(plot_list)+1]] = p
#   # # ggsave(paste0("D:/chh/2025workProject/20250506PTV1/PTV1_15prot/IHC_", i, "boxplot.pdf"), p)
#   
# }
# ggarrange(plotlist = plot_list)

sheets = readxl::excel_sheets("D:/chh/2025workProject/20250506PTV1/PTV1_15prot/西湖IHC评分_20260407.xlsx")
sheets
IHC = c()
for (i in 1:length(sheets)) {
  # i=1
  print(i)
  temp = read.xlsx("D:/chh/2025workProject/20250506PTV1/PTV1_15prot/西湖IHC评分_20260407.xlsx", sheet = i)
  # print(colnames(temp))
  # temp$结果= gsub('高', 'high', temp$结果)
  # temp$结果= gsub('低', 'low', temp$结果)
  # temp$患者 = paste0(temp$患者, temp$结果)
  temp = subset(temp, !is.na(temp$`总分1（=染色强度*阳性细胞比例）`))
  temp$pid = gsub(' .*', '', temp$患者)
  temp$患者 = ifelse(!grepl('high|low', temp$患者, ignore.case = T), paste0(temp$患者, temp$结果), temp$患者 )
  temp$患者= ifelse( grepl('high|高', temp$患者, ignore.case = T), 'high', temp$患者 )
  temp$患者= ifelse( grepl('low|低', temp$患者, ignore.case = T), 'low', temp$患者 )
  
  pre = temp[grepl('high', temp$患者, ignore.case = T) , 4]
  post = temp[grepl('low', temp$患者, ignore.case = T) , 4]
  p1 = t.test(pre, post)
  
  pre = temp[grepl('high', temp$患者, ignore.case = T) , 8]
  post = temp[grepl('low', temp$患者, ignore.case = T) , 8]
  p2 = t.test(pre, post)
  
  pre = temp[grepl('high', temp$患者, ignore.case = T) , 9]
  post = temp[grepl('low', temp$患者, ignore.case = T) , 9]
  p3 = t.test(pre, post)
  print(sheets[i])#, '总分1'
  print(c(p1$p.value, p2$p.value, p3$p.value))
  IHC = rbind(IHC, cbind(protein = sheets[i], pid = temp$pid, Group = temp$患者, value = temp$`总分1（=染色强度*阳性细胞比例）`, type = rep('sum1', nrow(temp))))
  IHC = rbind(IHC, cbind(protein = sheets[i], pid = temp$pid,Group = temp$患者, value = temp$`总分2（=染色强度*阳性细胞比例）`, type = rep('sum2', nrow(temp))))
  IHC = rbind(IHC, cbind(protein = sheets[i], pid = temp$pid, Group = temp$患者, value = temp$平均分, type = rep('mean_val', nrow(temp))))
}
IHC = data.frame(IHC)
IHC$value = as.numeric(IHC$value)
IHC = subset(IHC, !is.na(value))
IHC$Group = gsub('.*-', '', IHC$Group)
IHC = subset(IHC, Group %in% c("low", "high"))
IHC$Group = toupper(gsub('[^a-zA-z]', '', IHC$Group))
head(IHC)
unique(IHC$protein)
IHC = subset(IHC, protein %in% c("PALLD",   "RCOR3",   "ZFYVE19" ))

IHC$type = factor(IHC$type, levels = c('sum1', 'sum2', 'mean_val'))

IHC$pgt = paste0(IHC$protein, '_', IHC$Group, '_', IHC$type)
table(IHC$pgt)
IHC = IHC[order(IHC$protein, IHC$type, IHC$Group, IHC$value),]
plot_list = list()
for (i in unique(IHC$protein)) {
  # i = "PALLD"
  temp = IHC[IHC$protein %in% i, ]
  p = ggplot(temp, aes(type, value, color = Group))+
    geom_boxplot( outlier.color = NA ) +
    geom_point(position = position_jitterdodge(), size=0.7 )+#jitter.width = 0.5
    scale_color_manual(values = c('#F69634', '#637AA6'))+
    stat_compare_means(method = "t.test", hide.ns = T)+#,label = "p.signif"
    labs(y = 'H-scores', title = paste0(i, ' N=', length(unique(temp$pid))))+
    theme_classic()+
    theme(text = element_text(size = 13, color = 'black'),
          axis.text = element_text(size = 13, color = 'black'),
          axis.title.x = element_blank()#, legend.position = 'none'
    )
  plot_list[[length(plot_list)+1]] = p
  # # ggsave(paste0("D:/chh/2025workProject/20250506PTV1/PTV1_15prot/IHC_", i, "_boxplot_20260407.pdf"), p)
}
ggarrange(plotlist = plot_list)

ggsave(paste0("D:/chh/2025workProject/20250506PTV1/PTV1_15prot/IHC_boxplot_20260611-1353.pdf"), ggarrange(plotlist = plot_list))

unique(IHC$pid)

#### SI Figure 3 In vitro validation of PALLD biological function in modulating drug synergy ####
library(DoseFinding)
library(ggpubr)
library(dplyr)
# C
ttest_try = function(exp, a, b, c, d){
  Pvalue <- c()
  logFC = c()
  for(i in 1:nrow(exp)){
    y = try(t.test(as.numeric(exp[i, a:b]), as.numeric(exp[i, c:d ])), silent=T)
    if('try-error' %in% class(y)) # 判断当前循环的try语句中的表达式是否运行正确
    {
      Pvalue = c(Pvalue, NA)
    }else{
      y = t.test(as.numeric(exp[i, a:b]), as.numeric(exp[i, c:d ]))# 默认var.equal = FALSE
      Pvalue[i]<- y$p.value
    }
  }
  pre = rowMeans(2**exp[, a:b], na.rm = TRUE)
  post = rowMeans(2**exp[, c:d ], na.rm = TRUE)
  FC = pre/post
  logFC = log2(FC)
  
  FDR = p.adjust(Pvalue, "BH")
  
  out<-cbind(exp, Pvalue, FDR, FC, logFC)
  return(data.frame(out)[c('Pvalue', 'FDR', 'FC', 'logFC')])
}
human_gene = read.xlsx("Z:/members/xuezhangzhi/chh/uniprotkb_Human_AND_reviewed_true_AND_m_2025_10_22.xlsx", rowNames = T)


prot15 = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/PTV1_rebuttal/00_PTV1_revised_202602/15prot/15prot.xlsx")
prot15$Gene = human_gene[prot15$Entry, 2]
prot15$siRNA = gsub('_.*', '', prot15$Entry.Name)
rownames(prot15) = prot15$siRNA


sinfo = read.xlsx('D:/chh/2025workProject/20250506PTV1/Celllinevalidation/PTV1补充样本样本信息表_20260309_zhanggm.xlsx')
sinfo1 = read.xlsx('D:/chh/2025workProject/20250506PTV1/Celllinevalidation/20260313-PTV1-P4-1937-1395-HCC70-M453.xlsx')
sinfo = rbind(sinfo[colnames(sinfo1)], sinfo1)
rownames(sinfo) = paste0('s', sinfo$样本id)
colnames(sinfo)
colnames(sinfo)[2] = 'cell'
sinfo$siRNA = gsub('SYNE-', 'SYNE2-', sinfo$siRNA)
sinfo$siRNA1 = gsub('-.*', '', sinfo$siRNA)
sinfo$gene = prot15[sinfo$siRNA1, 'Gene']
head(sinfo)


ptv1_15prot_CAB_O = read.csv("//172.16.13.136/share/members/sunr/PTV1_15prot_raw/PTV1_15prot_report20260618.pg_matrix.tsv", sep = '\t', row.names = 1)
ptv1_15prot_CAB_O$Genes = human_gene[rownames(ptv1_15prot_CAB_O), 2]
ptv1_15prot_CAB_O[1:3, 1:3]

sinfo1 = data.frame(filename = colnames(ptv1_15prot_CAB_O))
sinfo1$filename1 = as.character(lapply(sinfo1$filename, function(x){
  x = strsplit(x, '.', fixed = T)[[1]]
  x[length(x)-1]
}))
dim(sinfo1)
sinfo1 = sinfo1 %>% distinct(filename1, .keep_all = T)
dim(sinfo1)
sinfo1 = sinfo1[3:nrow(sinfo1), ]
sinfo1$filename1 = gsub('_rep', 'rep', sinfo1$filename1)

sinfo1$id = gsub('.*_', '', sinfo1$filename1)
sinfo1$id = gsub('\\..*', '', sinfo1$id )
sinfo1$id = paste0('s', sinfo1$id)
rownames(sinfo1) = sinfo1$id

sinfo1['s20260312205210', 'id'] = 's219rep'
rownames(sinfo1) = sinfo1$id
sinfo1$id = gsub('rep', '_', sinfo1$id)
sinfo1$id = gsub('_.*', '', sinfo1$id)

sinfo1 = cbind(sinfo1, sinfo[sinfo1$id, 2:ncol(sinfo)])

unique(sinfo1$gene)
intersect(ptv1_15prot_CAB_O$Genes, sinfo1$gene)


ttest_res = c()
for (pp in unique(sinfo1$cell)) {
  # 
  if(is.na(pp)) next
  for (g in unique(sinfo1$siRNA)) {
    if(is.na(g)) next
    if(g == 'NC')next
    # pp='HCC38';g = "PALLD-1"
    g1 = prot15[gsub('-.*', '', g), 'Gene']
    if(!(g1 %in% ptv1_15prot_CAB_O$Genes)) next
    
    temp = subset(sinfo1, cell %in% pp & siRNA %in% c(g, 'NC'))
    temp[temp$siRNA %in% g, 'gene'] = 'ko'
    temp = temp[order(temp$gene), ]
    m = nrow(temp[temp$gene %in% 'ko', ])
    print(c(pp, g, g1,m))
    if(m<2) next
    
    ttest_out = ttest_try(log2(ptv1_15prot_CAB_O[temp$filename]), 1, m, m+1, nrow(temp))
    # ttest_out = cbind(log2(ptv1_15prot_CAB_O[temp$filename]), ttest_out)
    
    ttest_out$KO_num = rowSums(!is.na(ptv1_15prot_CAB_O[temp$filename[1:2]]))
    ttest_out$NC_num = rowSums(!is.na(ptv1_15prot_CAB_O[temp$filename[3:4]]))
    
    ttest_out$sig_FDR = ifelse(is.na(ttest_out$FDR), 0, ifelse(ttest_out$logFC> log2(1.2) & ttest_out$FDR<0.05, 1,
                                                               ifelse( ttest_out$logFC< (-log2(1.2)) & ttest_out$FDR<0.05, -1, 0)))
    ttest_out$sig_Pvalue = ifelse(is.na(ttest_out$Pvalue), 0, ifelse(ttest_out$logFC> log2(1.2) & ttest_out$Pvalue<0.05, 1,
                                                                     ifelse( ttest_out$logFC< (-log2(1.2)) & ttest_out$Pvalue<0.05, -1, 0)))
    ttest_out$gene = human_gene[rownames(ttest_out), 2]
    ttest_out = subset(ttest_out, gene %in% g1, )
    ttest_out$uniprotId = rownames(ttest_out)
    ttest_out$cell = pp
    ttest_out$siRNA = g
    
    ttest_res = rbind(ttest_res, ttest_out)
    
  }
}
colnames(ttest_res)
ttest_res[ttest_res$cell %in% 'HCC38' & ttest_res$gene %in% 'PALLD', ]

temp = merge(sinfo1 , ttest_res[c("Pvalue","FDR", "FC","logFC","KO_num","NC_num", "uniprotId", "cell","siRNA")] , by = c("cell", "siRNA"), all.x = T)
temp = temp[order(temp$cell, temp$siRNA), ]
colnames(temp)
# temp[is.na(temp$filename), 14:19] = NA

write.xlsx(data.frame(temp), paste0('Z:/members/sunr/PTV1_15prot_raw/PTV1_Celllinevalidation_siRNA_PTV1_15prot20260618_ttest.xlsx'), rowNames = F)
write.xlsx(sinfo1, paste0('Z:/members/sunr/PTV1_15prot_raw/PTV1_Celllinevalidation_siRNA_PTV1_15prot20260618_info.xlsx'), rowNames = F)


sinfo1 = read.xlsx(paste0('Z:/members/sunr/PTV1_15prot_raw/PTV1_Celllinevalidation_siRNA_PTV1_15prot20260618_info.xlsx'), rowNames = F)
unique(sinfo1$cell)
HCC38 = subset(sinfo1, cell %in% 'HCC38' & sinfo1$siRNA1 %in% c('NC', 'PALLD'))
rownames(HCC38) = HCC38$filename

ptv1_15prot_CAB_O[1:3, 1:3]
HCC38_mat = ptv1_15prot_CAB_O[HCC38$filename]
HCC38_mat$pros = rownames(HCC38_mat)
HCC38_mat = data.frame(melt(HCC38_mat, id.vars = 'pros'))
head(HCC38_mat)
HCC38_mat = subset(HCC38_mat, pros %in% c('Q8WX93'))
HCC38_mat$value = log2(HCC38_mat$value)
HCC38_mat$Group = HCC38[HCC38_mat$variable,  'siRNA1']
HCC38_mat
HCC38_mat$Group = factor(HCC38_mat$Group, levels = c('NC', 'PALLD'))
p = ggplot(HCC38_mat, aes(Group, value, color = Group  ))+
  geom_boxplot(outlier.color = NA)+
  geom_point(position = position_jitterdodge(jitter.width = 0.5))+
  stat_compare_means( method = 't.test', hide.ns = T, na.rm = T )+#, comparisons = list(c("nc_com1",  "palld_com1"), c("nc_com2",  "palld_com2"))
  labs(y = 'log2(intensity)', title = 'HCC38 PALLD' )+
  theme_classic() +
  theme(text = element_text(size = 15, color = 'black'),
        axis.text = element_text(size = 13 , color = 'black'),
        axis.title.x = element_blank(),
        legend.position = 'none'
  )
p
ggsave(paste0('Z:/members/sunr/PTV1_15prot_raw/HCC38_PALLD_boxplot_20260618.pdf'), p, width = 4.5)


# E
path_cur = 'Z:/members/sunr/PTV1_15prot_raw/inhibition_ratio/20260411_0412/'
files = list.files('Z:/members/sunr/PTV1_15prot_raw/inhibition_ratio/20260411_0412/' )
files  = files[!grepl('~', files) & grepl('xlsx', files)  & !grepl('inhibition', files)]
files# = files[1:4]
condose = data.frame(com = c('com1-1', 'com1-2', 'com1-3', 'com1-4', 'com1-5', 'com1-6', 'com1-7',
                             'com2-1', 'com2-2', 'com2-3', 'com2-4', 'com2-5', 'com2-6', 'com2-7',
                             'Doc-1', 'Doc-2', 'Doc-3', 'Doc-4', 'Doc-5', 'Doc-6', 'Doc-7', 
                             'car 10uM', 'car 40uM', 'car 20uM', 'car 80uM' ), 
                     dose = c(paste0(c(0.3, 1.2, 4.8, 19.2, 76.8, 153.6, 307.2), ' ', 10),
                              paste0(c(0.3, 1.2, 4.8, 19.2, 76.8, 153.6, 307.2), ' ', 40),
                              # '0.3 10', '1.2 10', '4.8 10', '19.2 10', '76.8 10', '153.6 10', '307.2 10', 
                              #        '0.3 40', '1.2 40', '4.8 40', '19.2 40', '76.8 40', '153.6 40', '307.2 40', 
                              0.3, 1.2, 4.8, 19.2, 76.8, 153.6, 307.2,
                              10, 40, 20, 80))
rownames(condose) = condose$com


f = "1806-nc.xlsx"
dat1 = read.xlsx( paste0(path_cur, f ) )
dat1[dat1=='' | dat1==' '] = NA
dat1 = subset(dat1, dat1$X2 %in% c(LETTERS[1:8]))
dat1_group =  dat1[9:16, 3:14]
dat1_group$id = paste0('s', 1:nrow(dat1_group))
dat1_group = data.frame(melt(dat1_group, id.vars = 'id'))
colnames(dat1_group)[3] = 'group'
dat1_group$dose = condose[dat1_group$group, 2]
dat1_group_ref = dat1_group

inhibition_ratio = c()
for (f in files) {
  # f = files[1]
  # f = "1806-nc.xlsx"
  dat1 = read.xlsx( paste0(path_cur, f ) )
  dat1[dat1=='' | dat1==' '] = NA
  dat1 = subset(dat1, dat1$X2 %in% c(LETTERS[1:8]))
  
  dat1_val =  dat1[1:8, 3:14]
  dat1_val$id = paste0('s', 1:nrow(dat1_val))
  dat1_val = data.frame(melt(dat1_val, id.vars = 'id'))
  
  if(nrow(dat1)> 8) {
    dat1_group =  dat1[9:16, 3:14]
    dat1_group$id = paste0('s', 1:nrow(dat1_group))
    dat1_group = data.frame(melt(dat1_group, id.vars = 'id'))
    colnames(dat1_group)[3] = 'group'
    dat1_group$dose = condose[dat1_group$group, 2]
    
    dat1 = data.frame(merge(dat1_val, dat1_group , by = c('id', 'variable')))
  }else{
    dat1 = data.frame(merge(dat1_val, dat1_group_ref , by = c('id', 'variable')))
  }
  
  
  dat1$filename = f
  print(dim(dat1))
  # dat1 = subset(dat1, !is.na(group))
  # dat1 = dat1[order(dat1$filename, dat1$group), ]
  
  inhibition_ratio = rbind(inhibition_ratio, dat1) 
}
inhibition_ratio[inhibition_ratio=='' | inhibition_ratio==' '] = NA
inhibition_ratio = subset(inhibition_ratio, !is.na(group))
# inhibition_ratio$group = gsub('med', 'ctrl', inhibition_ratio$group)
unique(inhibition_ratio$group)
inhibition_ratio$variable = as.character(inhibition_ratio$variable)
inhibition_ratio = inhibition_ratio[order(inhibition_ratio$filename, inhibition_ratio$group), ]
inhibition_ratio$rep123 = rep(c('rep1', 'rep2', 'rep3'), nrow(inhibition_ratio)/3)

inhibition_ratio$lib_dose = as.numeric(gsub(' .*', '', inhibition_ratio$dose))
inhibition_ratio$Anchor_dose = as.numeric(gsub('.* ', '', inhibition_ratio$dose))
inhibition_ratio$value = as.numeric(inhibition_ratio$value)

inhibition_ratio[grepl('car', inhibition_ratio$group), 'lib_dose'] = 0
inhibition_ratio[grepl('Doc', inhibition_ratio$group), 'Anchor_dose'] = 0


inhibition_ratio$drug = gsub(' .*', '', inhibition_ratio$group)
inhibition_ratio$drug = gsub('-.*', '', inhibition_ratio$drug)
inhibition_ratio$drug = ifelse(grepl('com1', inhibition_ratio$drug), 'com1', inhibition_ratio$drug)
inhibition_ratio$drug = ifelse(grepl('com2', inhibition_ratio$drug), 'com2', inhibition_ratio$drug)
inhibition_ratio$drug = ifelse(grepl('DMSO', inhibition_ratio$group), 'DMSO', inhibition_ratio$drug)
inhibition_ratio$drug = ifelse(grepl('Media', inhibition_ratio$group), 'Media', inhibition_ratio$drug)

inhibition_ratio$libraryID = inhibition_ratio$drug
inhibition_ratio$AnchorID = inhibition_ratio$drug
inhibition_ratio[grepl('car', inhibition_ratio$group), 'libraryID'] = NA
inhibition_ratio[grepl('Doc', inhibition_ratio$group), 'AnchorID'] = NA
inhibition_ratio[grepl('com', inhibition_ratio$group), 'libraryID'] = 'Doc'
inhibition_ratio[grepl('com', inhibition_ratio$group), 'AnchorID'] = 'car'


Emax_res = c()
for (f in unique(inhibition_ratio$filename)) {
  
  # f = "38-palld.xlsx"
  dat1 = subset(inhibition_ratio, filename %in% f)
  dat1$Group  = paste0(dat1$group, '_', dat1$rep123)
  df1 = data.frame(row.names = c("0.3", "1.2", "4.8", "19.2", "76.8", "153.6", "307.2", "10", "40") )
  # df1['10', c("car 10_rep1", "car 10_rep2", "car 10_rep3") ] = dat1[dat1$Group %in% c("car 10uM_rep1", "car 10uM_rep2", "car 10uM_rep3"), 'value']
  # df1['40', c("car 40_rep1", "car 40_rep2", "car 40_rep3") ] = dat1[dat1$Group %in% c("car 40uM_rep1", "car 40uM_rep2", "car 40uM_rep3"), 'value']
  
  unique(dat1$rep123)
  for (rep_i in c("rep1", "rep2", "rep3")) {
    print(rep_i)
    dose = subset(dat1, Anchor_dose %in% 10 & dat1$drug %in% 'com1' &  rep123 %in% rep_i)
    df1[as.character(dose$lib_dose), paste0('com1_10_', rep_i)] = dose$value
  }
  for (rep_i in c("rep1", "rep2", "rep3")) {
    print(rep_i)
    dose = subset(dat1, Anchor_dose %in% 40 & dat1$drug %in% 'com2' &  rep123 %in% rep_i)
    df1[as.character(dose$lib_dose), paste0('com2_40_', rep_i)] = dose$value
  }
  
  
  for (rep_i in c("rep1", "rep2", "rep3")) {
    print(rep_i)
    dose = subset(dat1, dat1$drug %in% 'Doc' &  rep123 %in% rep_i)
    df1[as.character(dose$lib_dose), paste0('Doc_', rep_i)] = dose$value
  }

  
  for (rep_i in c("rep1", "rep2", "rep3")) {
    df1_rep_i = df1[grepl(rep_i, colnames(df1))]
    
    for(i in 1:ncol(df1_rep_i)){
      df1_rep_i[i ] = df1_rep_i[i]/ na.omit(df1_rep_i[,i])[1]
    }
    
    df1_rep_i$dose = as.numeric(rownames(df1_rep_i))
    df1_rep_i = df1_rep_i[order(df1_rep_i$dose, decreasing = F), ]
    
    df1.1 = melt( df1_rep_i, id.vars = 'dose' )
    colnames(df1.1) = c("dose","Drug",  "value")
    df1.1$value = as.numeric(df1.1$value)
    df1.1$dose = as.numeric(df1.1$dose)
    df1.1 = df1.1[!is.na(df1.1$value),]
    df1.1$Drug = as.character(df1.1$Drug)
    
    Emax = c()
    for ( j in unique(df1.1$Drug)){
      temp.1.1  = df1.1[df1.1$Drug == j, ]
      temp.1.1$value = 1 - temp.1.1$value
      emax0 <- fitMod(dose, value, data = temp.1.1,  model = "emax" )
      Emax = rbind(Emax, emax0$coefs)
    }
    Emax = data.frame(Emax, row.names =  unique(df1.1$Drug))
    Emax$rep123 = unique(df1.1$Drug)
    Emax$filename = f
    Emax[paste0('com1_10_', rep_i), 'delta_com1'] = Emax[paste0('com1_10_', rep_i), 'eMax'] - Emax[paste0('Doc_', rep_i), 'eMax']
    Emax[paste0('com2_40_', rep_i), 'delta_com2'] = Emax[paste0('com2_40_', rep_i), 'eMax'] - Emax[paste0('Doc_', rep_i), 'eMax']
    
    Emax[paste0('com1_10_', rep_i), 'div_com1'] = Emax[paste0('com1_10_', rep_i), 'eMax'] / Emax[paste0('Doc_', rep_i), 'eMax']
    Emax[paste0('com2_40_', rep_i), 'div_com2'] = Emax[paste0('com2_40_', rep_i), 'eMax'] / Emax[paste0('Doc_', rep_i), 'eMax']
    
    
    Emax_res = rbind(Emax_res, Emax[c('filename',  'rep123', 'eMax', 'delta_com1', 'delta_com2', 'div_com1', 'div_com2')])
  }
}

Emax_res = Emax_res[order(Emax_res$rep123), ]
Emax_res = Emax_res[order(Emax_res$filename), ]
unique(Emax_res$filename)

Emax_res$cell = gsub('-.*', '', Emax_res$filename)
unique(Emax_res$cell)
write.xlsx(Emax_res, 'Z:/members/sunr/PTV1_15prot_raw/inhibition_ratio/20260411_0412/inhibition_ratio_5cell_Emax_res20260414.xlsx')


# Emax_res = read.xlsx('Z:/members/sunr/PTV1_15prot_raw/inhibition_ratio/20260411_0412/inhibition_ratio_5cell_Emax_res.xlsx')
Emax_res = read.xlsx('Z:/members/sunr/PTV1_15prot_raw/inhibition_ratio/20260411_0412/inhibition_ratio_5cell_Emax_res20260414.xlsx')
head(Emax_res)
Emax_res$cell2 = gsub('.*-', '', Emax_res$filename)
Emax_res$cell2 = gsub('.xlsx', '', Emax_res$cell2)

cl = 38
temp = subset(Emax_res, cell %in% cl)
temp$delta_com = ifelse(is.na(temp$delta_com1), NA, 'com1')
temp$delta_com = ifelse(is.na(temp$delta_com2), temp$delta_com, 'com2')

temp$delta_emax = temp$delta_com1
temp$delta_emax = ifelse(is.na(temp$delta_com2), temp$delta_emax, temp$delta_com2)

temp = subset(temp, !is.na(delta_emax))
temp$cmp = paste0(temp$cell2, '_', temp$delta_com)
p = ggplot(temp, aes(cmp, delta_emax, color =  delta_com ))+
  geom_boxplot()+
  geom_point(position = position_jitterdodge(jitter.width = 0.2))+
  
  stat_compare_means(
    method = 't.test', comparisons = list(c("nc_com1",  "palld_com1"), c("nc_com2",  "palld_com2")) )+
  theme_classic() +
  labs( title = cl )+
  theme(text = element_text(size = 15, color = 'black'),
        axis.text = element_text(size = 13 , color = 'black'),
        axis.title.x = element_blank(), legend.position = 'none'
  )
p
# ggsave(paste0('Z:/members/sunr/PTV1_15prot_raw/inhibition_ratio/20260411_0412/inhibition_ratio_5cell_', cl,'_Emax_boxplot.pdf'), p)
ggsave(paste0('Z:/members/sunr/PTV1_15prot_raw/inhibition_ratio/20260411_0412/inhibition_ratio_5cell_', cl,'_Emax_boxplot20260609-2001.pdf'), p, width = 5)
