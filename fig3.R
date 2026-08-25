library(openxlsx)
library(reshape2)
library(dplyr)
library(ggplot2)
library(ggsci)
library(ggpubr)
#### C PTNC ####
df1 = read.csv("//172.16.13.136/share/members/sunr/PTV1/00_check/PTV1_check_20260603/Figures/Fig_PTDS4_2026/Fig_PTDS4_2026/Fig_PTDS4_2026_metricsPERRUN.csv" )
head(df1)
for (i in 4:ncol(df1)) {
  df1[,i] = as.numeric(df1[,i])
}

# AUPRC
temp = subset(df1, !is.na(df1$AUPRC))
colnames(temp) = gsub('AUPRC', 'val', colnames(temp))
PRC_sum = temp %>% 
  group_by(method ) %>%
  summarise(
    mean_val = mean(val, na.rm = T),
    sd_val = sd(val, na.rm = T),
    val_num = n()
  )
PRC_sum$se_val = PRC_sum$sd_val/sqrt(PRC_sum$val_num)
PRC_sum

# AUROC
temp = subset(df1, !is.na(df1$AUROC))# "Accuracy"  "AUROC"     "AUPRC"
colnames(temp) = gsub('AUROC', 'val', colnames(temp))
PRC_sum = temp %>% 
  group_by(method ) %>%
  summarise(
    mean_val = mean(val, na.rm = T),
    sd_val = sd(val, na.rm = T),
    val_num = n()
  )
PRC_sum$se_val = PRC_sum$sd_val/sqrt(PRC_sum$val_num)
PRC_sum
