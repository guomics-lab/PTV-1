rm(list = ls());gc()
library(openxlsx)
library(reshape2)
library(ggplot2)
library(ggsci)
library(pheatmap)

#### shap_dynamic_change shap_dynamic_change #####
files = list.files('Z:/members/sunr/PTV1/PTV1_rebuttal/analysis/shap_dynamic_change/shap_dynamic_change', pattern = '*csv', full.names = T)
f = files[1]# negative
temp = read.csv(f, row.names = 1)
temp = temp[order(temp$abs_sum_change, decreasing = T), ]
negative = temp[1:20,]
rownames(negative) = negative$protein

f = files[2]# positive
temp = read.csv(f, row.names = 1)
temp = temp[order(temp$abs_sum_change, decreasing = T), ]
rownames(temp) = temp$protein
positive = temp
rownames(positive) = positive$protein
rownames(positive) = gsub(':', '.', rownames(positive), fixed = T)
rownames(positive) = gsub('-', '.', rownames(positive), fixed = T)


uni_label = temp
uni_label = uni_label[2:1]
uni_label = rbind(uni_label, c("Q14956", 'GPNMB'))
uni_label = rbind(uni_label, c("Q12789", 'GTF3C1'))
uni_label = rbind(uni_label, c("P11166", 'SLC2A1'))
uni_label = rbind(uni_label, c("P21589", 'NT5E'))
rownames(uni_label) = uni_label$protein
rownames(uni_label) = gsub(':', '.', rownames(uni_label), fixed = T)
rownames(uni_label) = gsub('-', '.', rownames(uni_label), fixed = T)

files = list.files('Z:/members/sunr/PTV1/PTV1_rebuttal/analysis/shap_multitime_heatmap&barplot20250825/antimitotic', pattern = '*csv', full.names = T)
files
temp = read.csv(files[1], row.names = 1)
temp = data.frame(t(temp))
col_pros = colnames(temp)
rows_pros = rownames(temp)
temp[] = lapply(temp, as.numeric)
input = temp
colnames(input) = paste0('col', 1:ncol(input))
dim(temp)

for (f in files[2:6]) {
  temp = read.csv(f, row.names = 1)
  temp = data.frame(t(temp))
  temp[] = lapply(temp, as.numeric)
  print(c(f, dim(temp)))
  colnames(temp) = paste0('col', 1:ncol(temp) + ncol(input))
  input[rownames(temp), colnames(temp)] = temp
}
ncol(input)/6
quantile(unlist(input), probs = c(0.05, 0.95))
# pdf(gsub('csv', '.pathwaycluster.20250806-1605.pdf', f), width = 10, height = 8)

type1 = strsplit(files[5], '/', fixed = T)[[1]];type1 = type1[length(type1)]
temp = input[((95*4)+1):(95*5)]
colnames(temp) = col_pros
rownames(temp) = rows_pros
temp = t(temp[intersect(rownames(uni_label), rows_pros ), rownames(positive)])
temp = data.frame(temp[1:50, ])
differ = setdiff(colnames(temp), rownames(temp))
differ = setdiff(differ, c("ENSEMBL.ENSBTAP00000034412", "SWISS.PROT.P19001","TREMBL.Q1RMK2"))
temp_positive1  = temp#[differ]

type1 = strsplit(files[4], '/', fixed = T)[[1]];type1 = type1[length(type1)]
temp = input[((95*3)+1):(95*4)]
colnames(temp) = col_pros
rownames(temp) = rows_pros
temp = t(temp[intersect(rownames(uni_label), rows_pros ), rownames(positive)])
temp = data.frame(temp[1:50, ])
# temp_positive0 = temp[differ]
temp_positive0 = temp[positive[1:20, 2], positive[1:20, 2]]

p0 = pheatmap(temp_positive0 ,
              colorRampPalette(c( "#13A0AD", "white", "red"))(1000 ),
              # scale = 'none',
              cluster_rows = F, cluster_cols = F,
              breaks = c(seq(-8.732562e-06,  8.779657e-06 , length=1000)),
              fontsize = 10,
              border_color = NA,
              cellheight= 8,
              cellwidth = 8,
              fontsize_row = 4,
              fontsize_col= 4,
              show_rownames = T, show_colnames = T,
              labels_row = uni_label[rownames(temp_positive0), 'gene'],
              labels_col = uni_label[colnames(temp_positive0), 'gene'],
              # annotation_col = comboinfo[c(6)],
              # annotation_row = comboinfo[c(6)],
              # annotation_colors = combo_colors,
              main = 'point0')

roworder = rownames(temp_positive0[p0$tree_row$order,])
colorder = colnames(temp_positive0[p0$tree_col$order])

roworder = rownames(temp_positive0 )
colorder = colnames(temp_positive0 )

p1 = pheatmap(temp_positive1[roworder, colorder], #,
              colorRampPalette(c( "#13A0AD", "white", "red"))(1000 ),
              # scale = 'none',
              cluster_rows = F, cluster_cols = F,
              breaks = c(seq(-8.732562e-06,  8.779657e-06 , length=1000)),
              fontsize = 10,
              border_color = NA,
              cellheight= 12,
              cellwidth = 10,
              fontsize_row = 11,
              fontsize_col= 9,
              show_rownames = T, show_colnames = T,
              labels_row = uni_label[roworder, 'gene'],
              labels_col = uni_label[colorder, 'gene'],
              # annotation_col = comboinfo[c(6)],
              # annotation_row = comboinfo[c(6)],
              # annotation_colors = combo_colors,
              main = 'point1')

#### lolipop ####
library(ggplot2)
files = list.files('Z:/members/sunr/PTV1/PTV1_rebuttal/analysis/shap_dynamic_change/shap_dynamic_change', pattern = '*csv', full.names = T)
files
for (f in files[1:2]) {
  # f = files[1]
  temp = read.csv(f, row.names = 1)
  head(temp)
  temp = temp[order(temp$abs_sum_change, decreasing = T), ]
  temp$gene = factor(temp$gene, levels = temp$gene)
  ## fig 1 ：基础图形
  fig3 = ggplot(temp[1:50, ], aes(x = gene, y = abs_sum_change)) +
    geom_segment( aes(x = gene, xend = gene, y = 0, yend = abs_sum_change),color = "black")+ #控制线段的参数，见下
    geom_point(size = 2, pch = 21, bg = 5, col = 1) + #控制散点的参数
    theme_classic()+ labs(x = '')+
    theme(text = element_text(color = 'black', size = 8),
          axis.text = element_text(color = 'black', size = 8),
          axis.text.x = element_text(color = 'black', angle = 90, hjust = 1, size = 8, vjust = 0.5))
  ggsave(gsub('csv', '_top50_lolipop.pdf', f), fig3)
}

#geom_segment中的参数用于控制线条相关参数
#x=xv，xend=xv 表示x轴的线条起始位置x和终止位置xend都是xv（没有线条）
#类似的，y = 1, yend = yv表示y轴线段起始点为y=1，种植点为每个类别的值即yv


#### D ####
# source：D:/chh/2023workProject/prottalk/code/IC50/fig5/ic50_anctor_h1299A20_use_gdsc_20240604_20250908.R

#20240304 ic50 curve
rm(list = ls());gc()
library(dplyr)
library(ggplot2)
library(nlme)
library(gdscIC50)
# source("D:/chh/2023workProject/prottalk/code/IC50/fig5/ic_50_function_20240327.R")
# source("//172.16.13.136/share/members/chenghonghan/PTV1/proof_check20260608/fig2/fig7_ic50_function.R")
source("./fig4_ic50_function.R")
# library(tidyr)
# source("ic_50_function_20240327.R")
library(doMC)
registerDoMC()
library(magrittr)

dose = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/PTV1_honghan/validation/Docetaxel_bt20.xlsx", sheet = 1)
dose = dose[11:22, ]
dose = dose[seq(2, nrow(dose), 2), 4:13]
rownames(dose) = c(paste0('WT_rep', 1:3), paste0('AKR1C3_rep', 1:3))
dose = data.frame(t(dose))
dose$doseum = c(0.0000125,	0.00125,	0.0125,	0.125,	0.25,	1.25,	2.5,	5,	25,	50)
dose = data.frame(melt(dose, id.vars = 'doseum'))
dose$variable = as.character(dose$variable)
dose$drug1 = as.character(lapply(as.character(dose$variable), function(x){
  strsplit(x, '_')[[1]][1]
}))
dose$variable = as.character(lapply(dose$variable, function(x){
  strsplit(x, '_')[[1]][2]
}))

wt = 'WT';kd = 'AKR1C3'
##### pchisq #####
library(lme4)
# pid = unique(dose$patient)

pid_cur = 'BT20'
temp1 = dose
temp1$x = temp1$doseum

unique(temp1$drug1)
unique(temp1$x)

WT <- lmer(value ~  doseum + (1|variable), data = temp1[temp1$drug1 == 'WT', ])
AKR1C3 <- lmer(value ~ doseum + (1|variable), data = temp1[temp1$drug1 == 'AKR1C3', ])

pid_cur
1 - pchisq((-2 * (logLik(WT) - logLik(AKR1C3))), df = 1)


dfDat = temp1[c('drug1', 'value', 'x')]
colnames(dfDat)[1:2] = c('drug', 'y')
dfDat$CL = 'No'
dfDat$maxc = max(dose$doseum)
dfDat$ANCHOR_VIAB = 1
dfDat$x = (log(dfDat$x/dfDat$maxc)/log(2))+length(unique(dose$doseum))
dfDat$y = 1-dfDat$y # important
dfDat$y[dfDat$y<0]<-0
dfDat$y[dfDat$y>1]<-1

dfDat$drug_spec<-"DRUG_ID_lib+maxc"
dfDat$DRUG_ID_lib<-dfDat$drug
gDat <- nlme::groupedData(y ~ x | drug/CL, data = dfDat, FUN = mean,
                          labels = list(x = "Concentration", y = "Viability"),
                          units = list(x = "uM/l", y = "percentage killed"))
gDat$type<-paste(gDat$CL, gDat$DRUG_ID_lib, sep="_")

nlme_stats<-c()
for (i in 1:length(unique(gDat$type))) { #length(unique(gDat$type))
  indsel<-gDat[gDat$type==unique(gDat$type)[i],]
  # ANCHOR_VIAB<-mean(indsel$ANCHOR_VIAB,na.rm=T)
  fmMod1_lib <- try(  fitModel(indsel, vStart = c(8.886464, 1.495953), bLargeScale = FALSE, bSilent = TRUE), silent = F)
  if("try-error" %in% class(fmMod1_lib)) {
    next
  }
  nlme_stats_lib <- calcNlmeStats(fmMod1_lib, indsel)
  nlme_stats_lib$ic50_true<-exp(nlme_stats_lib$IC50)
  nlme_stats<-rbind(nlme_stats,nlme_stats_lib)
}
unique(nlme_stats$ic50_true)
View(nlme_stats)
#协同判定
nlme_stats1 <- nlme_stats# [nlme_stats$RMSE<0.2, ]
plot_data <- nlme_stats1 %>%
  mutate_(lx = ~log(getConcFromX(x, maxc)),
          lxmid = ~log(getConcFromX(xmid, maxc))
  )
unique(plot_data$drug)

plot_data$drug1<-plot_data$drug
# plot_data$drug1<-gsub("high_10_","",plot_data$drug1)
# plot_data$drug1<-gsub("low_2_","",plot_data$drug1)
plot_data$drug1<-sapply(strsplit(plot_data$drug1,"_"),function(e){e[length(e)]})

plot_data<-plot_data[plot_data$IC50!=Inf,]
plot_data<-plot_data[plot_data$IC50!=-Inf,]


drug_Cisplatin<-plot_data[plot_data$drug== "WT" , ] #"CAY10603"
drug_Cisplatin$ythat_1<-mean(drug_Cisplatin$ANCHOR_VIAB)-drug_Cisplatin$yhat
drug_Cisplatin$y_1<-mean(drug_Cisplatin$ANCHOR_VIAB)-drug_Cisplatin$y
plot_Cisplatin_xmid <- mean(drug_Cisplatin$xmid)
plot_Cisplatin_scal <- mean(drug_Cisplatin$scal)
plot_Cisplatin_maxc <- mean(drug_Cisplatin$maxc)
plot_Cisplatin_ANCHOR_VIAB<- mean(drug_Cisplatin$ANCHOR_VIAB)
# plot_Cisplatin_x <- 1 - plot_scal * log((1 - 1e-3) / 1e-3) + plot_xmid
# plot_Cisplatin_x <- log(getConcFromX(plot_Cisplatin_x, plot_maxc))
# plot_Cisplatin_x <- min(c(plotmat_bliss$lx, plot_Cisplatin_x))

drug_CAY10603<-plot_data[plot_data$drug== "AKR1C3", ] #"CAY10603"
drug_CAY10603$ythat_1<-mean(drug_CAY10603$ANCHOR_VIAB)-drug_CAY10603$yhat
drug_CAY10603$y_1<-mean(drug_CAY10603$ANCHOR_VIAB)-drug_CAY10603$y
plot_CAY10603_xmid <- mean(drug_CAY10603$xmid)
plot_CAY10603_scal <- mean(drug_CAY10603$scal)
plot_CAY10603_maxc <- mean(drug_CAY10603$maxc)
plot_CAY10603_ANCHOR_VIAB<- mean(drug_CAY10603$ANCHOR_VIAB)
# plot_CAY10603_x <- 1 - plot_scal * log((1 - 1e-3) / 1e-3) + plot_xmid
# plot_CAY10603_x <- log(getConcFromX(plot_CAY10603_x, plot_maxc))
# plot_CAY10603_x <- min(c(plotmat_bliss$lx, plot_CAY10603_x))


plotmat_bliss<-rbind(drug_Cisplatin[,-11], drug_CAY10603[,-11] )
delta_Emax=mean(plotmat_bliss$yhat[plotmat_bliss$x==max(plotmat_bliss$x,na.rm=T)&plotmat_bliss$drug=="WT"])-mean(plotmat_bliss$yhat[plotmat_bliss$x==max(plotmat_bliss$x,na.rm=T)&plotmat_bliss$drug=="AKR1C3"])

plot_xmid <- mean(plotmat_bliss$xmid)
plot_scal <- mean(plotmat_bliss$scal)
plot_maxc <- mean(plotmat_bliss$maxc)
plot_ANCHOR_VIAB<- mean(plotmat_bliss$ANCHOR_VIAB)


plot_low_x <- 1 - plot_scal * log((1 - 1e-3) / 1e-3) + plot_xmid
plot_low_x <- log(getConcFromX(plot_low_x, plot_maxc))
plot_low_x <- min(c(plotmat_bliss$lx, plot_low_x))

plot_high_x <- 1 - plot_scal * log(1e-3 / (1 - 1e-3)) + plot_xmid
plot_high_x <- log(getConcFromX(plot_high_x, plot_maxc))
plot_high_x <- max(c(plotmat_bliss$lx, plot_high_x))

p_high<-ggplot(plotmat_bliss,aes(x = lx, y = ythat_1, group=drug, col=drug))+
  geom_point()+
  scale_color_manual(values=c("purple","orange","green", 'lightblue',"red")) +
  scale_x_continuous(limits =c(-15, 15) , n.breaks = 10)+# c(plot_low_x, plot_high_x)
  scale_y_continuous(limits = c(0, 1)) +
  stat_function(aes_(x =~lx), fun = l3_model2, colour="purple" ,
                args = list(maxc = plot_CAY10603_maxc,
                            xmid = plot_CAY10603_xmid,
                            scal = plot_CAY10603_scal,
                            ANCHOR_VIAB=plot_CAY10603_ANCHOR_VIAB)) +
  stat_function(aes_(x =~lx), fun = l3_model2, colour="orange" ,
                args = list(maxc = plot_Cisplatin_maxc ,
                            xmid = plot_Cisplatin_xmid,
                            scal = plot_Cisplatin_scal,
                            ANCHOR_VIAB=plot_Cisplatin_ANCHOR_VIAB))+
  # geom_point(aes_(x = plotmat_bliss$lx, y = plotmat_bliss$y_1, group=plotmat_bliss$drug), shape = 1)+
  theme_classic()+
  labs(y = "Response: normalized intensity", x =  expression(Dose/log[e]~mu*M), title = paste0('Patient: ', pid_cur ))+
  theme(text = element_text(size = 15))+
  geom_point(aes_(x = plotmat_bliss$lxmid, y = plotmat_bliss$ANCHOR_VIAB/2), shape = 2)#, color = "black"
# p_high #+annotate("label", x = plotmat_bliss$IC50 + 1, y = 0.5, hjust = "left",
#label = sprintf("IC50==%.3f~log[e]~mu*M", plotmat_bliss$IC50), parse = T)
#label = sprintf("IC50==%.3f~mu*M", plotmat_bliss$ic50_true), parse = T)

temp = data.frame(nlme_stats[c("drug", "ic50_true", 'IC50')])
temp = distinct(temp)
temp$pid = pid_cur
temp
#     drug  ic50_true      IC50  pid
# 1     WT 5.79212956  1.756500 BT20
# 2 AKR1C3 0.08363914 -2.481244 BT20
delta_ic50<-temp$IC50[temp$drug=="WT"]-temp$IC50[temp$drug=="AKR1C3"]


p_high<-p_high+annotate('text', x = -7, y = 0.5, label = paste0('WT IC50=', signif(temp[temp$drug == 'WT', 'IC50'], 3))) +
  annotate('text', x = -7, y = 0.4, label = paste0('AKR1C3 IC50=', signif(temp[temp$drug == 'AKR1C3', 'IC50'], 3))) +
  annotate('text', x = -7, y = 0.3, label = paste0('delta IC50=', signif(delta_ic50, 3))) +
  annotate('text', x = -7, y = 0.2, label = paste0('delta Emax=', signif(delta_Emax, 3)))

p_high
ggsave("./260622ptv1_fig4D_one_point.pdf",p_high,width = 8,height = 6)

# ==================== 绘图 ====================
p_high <- ggplot(plotmat_bliss, aes(x = lx, y = ythat_1, group = drug, col = drug)) +
  # ---- 拟合值点（实心圆） ----
geom_point() +
  # ---- 生物重复原始点（空心圆，半透明，抖动） ----
geom_point(aes(y = y_1), shape = 1, alpha = 0.6, size = 2.5,
           position = position_jitter(width = 0.15, height = 0)) +
  # ---- 颜色映射：WT=紫色，AKR1C3=橙色 ----
scale_color_manual(values = c("WT" = "purple", "AKR1C3" = "orange")) +
  # ---- x、y 轴范围 ----
scale_x_continuous(limits = c(-15, 15), n.breaks = 10) +
  scale_y_continuous(limits = c(0, 1)) +
  # ---- 绘制拟合曲线（颜色与分组一一对应） ----
stat_function(aes_(x = ~lx), fun = l3_model2, colour = "orange",   # AKR1C3 → 橙色
              args = list(maxc = plot_CAY10603_maxc,
                          xmid = plot_CAY10603_xmid,
                          scal = plot_CAY10603_scal,
                          ANCHOR_VIAB = plot_CAY10603_ANCHOR_VIAB)) +
  stat_function(aes_(x = ~lx), fun = l3_model2, colour = "purple",   # WT → 紫色
                args = list(maxc = plot_Cisplatin_maxc,
                            xmid = plot_Cisplatin_xmid,
                            scal = plot_Cisplatin_scal,
                            ANCHOR_VIAB = plot_Cisplatin_ANCHOR_VIAB)) +
  # ---- 标注 xmid 位置 ----
geom_point(aes_(x = ~lxmid, y = ~ANCHOR_VIAB / 2), shape = 2) +
  # ---- 主题与标签 ----
theme_classic() +
  labs(y = "Response: normalized intensity",
       x = expression(Dose/log[e]~mu*M),
       title = paste0('Patient: ', pid_cur)) +
  theme(text = element_text(size = 15))

# ==================== 添加 IC50 文本注释 ====================
temp <- data.frame(nlme_stats[c("drug", "ic50_true", "IC50")])
temp <- distinct(temp)
temp$pid <- pid_cur

p_high <- p_high +
  annotate('text', x = -7, y = 0.5,
           label = paste0('WT IC50 = ', signif(temp[temp$drug == 'WT', 'IC50'], 3))) +
  annotate('text', x = -7, y = 0.4,
           label = paste0('AKR1C3 IC50 = ', signif(temp[temp$drug == 'AKR1C3', 'IC50'], 3))) +
  annotate('text', x = -7, y = 0.3,
           label = paste0('delta IC50 = ', signif(delta_ic50, 3))) +
  annotate('text', x = -7, y = 0.2,
           label = paste0('delta Emax = ', signif(delta_Emax, 3)))

print(p_high)

ggsave("./260622ptv1_fig4D.pdf",p_high,width = 8,height = 6)
