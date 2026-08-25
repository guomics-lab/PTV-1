#### G ####
# D:/chh/2025workProject/20250506PTV1/FFPE/20240401_ptv3_gdsc_6anchor_qc/ic50_anctor_h1299A20_use_gdsc_20240604_chh.R

rm(list = ls());gc()
#####################
#20240304 ic50 curve
library(openxlsx)
library(dplyr)
library(stats)
library(reshape2)
library(ggplot2)
library(nlme)
library(gdscIC50)
library(lme4)#pchisq

# source("D:/chh/2025workProject/20250506PTV1/FFPE/20240401_ptv3_gdsc_6anchor_qc/ic_50_function_20240327.R") #dosenum:8
source('//172.16.13.136/share/members/chenghonghan/PTV1/proof_check20260608/fig5/ic_50_function_20240327.R')
# library(tidyr)
# source("ic_50_function_20240327.R")
# library(doMC)
# registerDoMC()
# library(magrittr)


dose20250928 = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/PTV1_honghan/validation/20250928/patients药物浓度敏感xlsx.xlsx", sheet = 2, colNames = F)
dose20250928 = dose20250928[1:5]
colnames(dose20250928) = c('dose', paste0('rep', 1:3), 'patient')

for (i in unique(dose20250928$patient)) {
  # i = unique(dose20250928$patient)[1]
  temp = subset(dose20250928, patient == i)
  temp[2] = temp[2]/temp[1, 2 ]
  temp[3] = temp[3]/temp[1, 3 ]
  temp[4] = temp[4]/temp[1, 4 ]
  dose20250928[dose20250928$patient %in% i, ] = temp
}
dose20250928 = subset(dose20250928, dose !='ctrl')
dose20250928$drug1 = as.character(lapply(dose20250928$dose, function(x){
  strsplit(x, ' ')[[1]][1]
}))
dose20250928$drug_dose = rep(c(0.0001, 0.005,0.05,0.5,5,50,100,200 ), 25)
dose20250928$pdd = paste0(dose20250928$patient, '_', dose20250928$drug1, '_', dose20250928$drug_dose)
dose = melt(dose20250928[c(2:4, 8)], id.vars = 'pdd')
dose$doseum = as.numeric(lapply(dose$pdd, function(x){
  strsplit(x, '_')[[1]][3]
}))
dose$drug1 = as.character(lapply(dose$pdd, function(x){
  strsplit(x, '_')[[1]][2]
}))
dose$patient = as.character(lapply(dose$pdd, function(x){
  strsplit(x, '_')[[1]][1]
}))

pid = unique(dose$patient)
pid
pid_cur = pid[1]
pid_cur
temp1 = subset(dose, patient == pid_cur)
temp1$x = temp1$doseum


##### pchisq #####
unique(temp1$drug1)
unique(temp1$x)

Cisplatin <- lmer(value ~ doseum + (1|variable), data = temp1[temp1$drug1 == 'Cisplatin', ])
JNK <- lmer(value ~ doseum + (1|variable), data = temp1[temp1$drug1 == 'JNK-IN-7', ])
CAY10603 <- lmer(value ~ doseum + (1|variable), data = temp1[temp1$drug1 == 'CAY10603', ])
Citarinostat <- lmer(value ~ doseum + (1|variable), data = temp1[temp1$drug1 == 'Citarinostat', ])
THZ <- lmer(value ~ doseum + (1|variable), data = temp1[temp1$drug1 == 'THZ', ])

pid_cur
1 - pchisq((-2 * (logLik(Cisplatin) - logLik(JNK))), df = 1)
1 - pchisq((-2 * (logLik(Cisplatin) - logLik(CAY10603))), df = 1)
1 - pchisq((-2 * (logLik(Cisplatin) - logLik(Citarinostat))), df = 1)
1 - pchisq((-2 * (logLik(Cisplatin) - logLik(THZ))), df = 1)


dfDat = temp1[c('drug1', 'value', 'x')]
colnames(dfDat)[1:2] = c('drug', 'y')
dfDat$CL = 'No'
dfDat$maxc = max(temp1$x)
dfDat$ANCHOR_VIAB = 1
dfDat$x = (log(dfDat$x/dfDat$maxc)/log(2))+ length(unique(temp1$x))
dfDat$y = 1-dfDat$y # important
dfDat$y[dfDat$y<0]<-0
dfDat$y[dfDat$y>1]<-1
dfDat$drug_spec<-"DRUG_ID_lib+maxc"
dfDat$DRUG_ID_lib<-dfDat$drug
unique(dfDat$drug)
# dfDat = dfDat[dfDat$drug %in% c( "Cisplatin" ), ]# ,"CAY10603" ,"THZ" ,  "Citarinostat", "JNK-IN-7"

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
temp = data.frame(nlme_stats[c("drug", "ic50_true", 'IC50')])
temp = distinct(temp)
temp$pid = pid_cur;temp

#####################################
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


drug_Cisplatin<-plot_data[plot_data$drug== "Cisplatin" , ] #"CAY10603"
drug_Cisplatin$ythat_1<-mean(drug_Cisplatin$ANCHOR_VIAB)-drug_Cisplatin$yhat
drug_Cisplatin$y_1<-mean(drug_Cisplatin$ANCHOR_VIAB)-drug_Cisplatin$y
plot_Cisplatin_xmid <- mean(drug_Cisplatin$xmid)
plot_Cisplatin_scal <- mean(drug_Cisplatin$scal)
plot_Cisplatin_maxc <- mean(drug_Cisplatin$maxc)
plot_Cisplatin_ANCHOR_VIAB<- mean(drug_Cisplatin$ANCHOR_VIAB)
# plot_Cisplatin_x <- 1 - plot_scal * log((1 - 1e-3) / 1e-3) + plot_xmid
# plot_Cisplatin_x <- log(getConcFromX(plot_Cisplatin_x, plot_maxc))
# plot_Cisplatin_x <- min(c(plotmat_bliss$lx, plot_Cisplatin_x))

drug_CAY10603<-plot_data[plot_data$drug== "CAY10603", ] #"CAY10603"
drug_CAY10603$ythat_1<-mean(drug_CAY10603$ANCHOR_VIAB)-drug_CAY10603$yhat
drug_CAY10603$y_1<-mean(drug_CAY10603$ANCHOR_VIAB)-drug_CAY10603$y
plot_CAY10603_xmid <- mean(drug_CAY10603$xmid)
plot_CAY10603_scal <- mean(drug_CAY10603$scal)
plot_CAY10603_maxc <- mean(drug_CAY10603$maxc)
plot_CAY10603_ANCHOR_VIAB<- mean(drug_CAY10603$ANCHOR_VIAB)
# plot_CAY10603_x <- 1 - plot_scal * log((1 - 1e-3) / 1e-3) + plot_xmid
# plot_CAY10603_x <- log(getConcFromX(plot_CAY10603_x, plot_maxc))
# plot_CAY10603_x <- min(c(plotmat_bliss$lx, plot_CAY10603_x))

drug_THZ<-plot_data[plot_data$drug== "THZ", ] #"CAY10603"
drug_THZ$ythat_1<-mean(drug_THZ$ANCHOR_VIAB)-drug_THZ$yhat
drug_THZ$y_1<-mean(drug_THZ$ANCHOR_VIAB)-drug_THZ$y
plot_THZ_xmid <- mean(drug_THZ$xmid)
plot_THZ_scal <- mean(drug_THZ$scal)
plot_THZ_maxc <- mean(drug_THZ$maxc)
plot_THZ_ANCHOR_VIAB<- mean(drug_THZ$ANCHOR_VIAB)
# plot_THZ_x <- 1 - plot_scal * log((1 - 1e-3) / 1e-3) + plot_xmid
# plot_THZ_x <- log(getConcFromX(plot_THZ_x, plot_maxc))
# plot_THZ_x <- min(c(plotmat_bliss$lx, plot_THZ_x))

drug_Citarinostat<-plot_data[plot_data$drug== "Citarinostat", ] #"CAY10603"
drug_Citarinostat$ythat_1<-mean(drug_Citarinostat$ANCHOR_VIAB)-drug_Citarinostat$yhat
drug_Citarinostat$y_1<-mean(drug_Citarinostat$ANCHOR_VIAB)-drug_Citarinostat$y
plot_Citarinostat_xmid <- mean(drug_Citarinostat$xmid)
plot_Citarinostat_scal <- mean(drug_Citarinostat$scal)
plot_Citarinostat_maxc <- mean(drug_Citarinostat$maxc)
plot_Citarinostat_ANCHOR_VIAB<- mean(drug_Citarinostat$ANCHOR_VIAB)
# plot_Citarinostat_x <- 1 - plot_scal * log((1 - 1e-3) / 1e-3) + plot_xmid
# plot_Citarinostat_x <- log(getConcFromX(plot_Citarinostat_x, plot_maxc))
# plot_Citarinostat_x <- min(c(plotmat_bliss$lx, plot_Citarinostat_x))

drug_JNK<-plot_data[plot_data$drug== "JNK-IN-7", ] #"CAY10603"
drug_JNK$ythat_1<-mean(drug_JNK$ANCHOR_VIAB)-drug_JNK$yhat
drug_JNK$y_1<-mean(drug_JNK$ANCHOR_VIAB)-drug_JNK$y
plot_JNK_xmid <- mean(drug_JNK$xmid)
plot_JNK_scal <- mean(drug_JNK$scal)
plot_JNK_maxc <- mean(drug_JNK$maxc)
plot_JNK_ANCHOR_VIAB<- mean(drug_JNK$ANCHOR_VIAB)



plotmat_bliss<-rbind(drug_Cisplatin[,-11], drug_THZ[,-11], drug_Citarinostat[-11],drug_CAY10603[,-11])#  drug_JNK[-11], 

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


#### 20250928 ####
p_high<-ggplot(plotmat_bliss,aes(x = lx, y = ythat_1, group=drug, col=drug))+
  geom_point()+
  scale_color_manual(values=c("purple","orange","green","red", 'lightblue')) + # 
  scale_x_continuous(limits =c(-20, 20) , n.breaks = 10)+# c(plot_low_x, plot_high_x)
  scale_y_continuous(limits = c(0, 1)) + 
  stat_function(aes_(x =~lx), fun = l3_model2, colour="orange" ,
                args = list(maxc = plot_Cisplatin_maxc , xmid = plot_Cisplatin_xmid,
                            scal = plot_Cisplatin_scal,
                            ANCHOR_VIAB = plot_Cisplatin_ANCHOR_VIAB)) +
  stat_function(aes_(x =~lx), fun = l3_model2, colour="purple" ,
                args = list(maxc = plot_CAY10603_maxc, xmid = plot_CAY10603_xmid,
                            scal = plot_CAY10603_scal,
                            ANCHOR_VIAB=plot_CAY10603_ANCHOR_VIAB)) +
  stat_function(aes_(x =~lx), fun = l3_model2, colour="green" ,
                args = list(maxc = plot_Citarinostat_maxc , xmid = plot_Citarinostat_xmid,
                            scal = plot_Citarinostat_scal,
                            ANCHOR_VIAB=plot_Citarinostat_ANCHOR_VIAB)) +
  stat_function(aes_(x =~lx), fun = l3_model2, colour="red" ,
                args = list(maxc = plot_THZ_maxc , xmid = plot_THZ_xmid,
                            scal = plot_THZ_scal,
                            ANCHOR_VIAB=plot_THZ_ANCHOR_VIAB)) +
  # stat_function(aes_(x =~lx), fun = l3_model2, colour="lightblue" ,
  #               args = list(maxc = plot_JNK_maxc , xmid = plot_JNK_xmid,
  #                           scal = plot_JNK_scal,
  #                           ANCHOR_VIAB=plot_JNK_ANCHOR_VIAB)) +
  # # geom_point(aes_(x = plotmat_bliss$lx, y = plotmat_bliss$y_1, group=plotmat_bliss$drug), shape = 1)+
  theme_classic()+
  labs(y = "Cell survival rate", #x =  expression(Dose/log[e]~mu*M), 
       x = 'ln(conc) μM',
       title = paste0('Patient: ', pid_cur ))+
  theme(text = element_text(size = 15), axis.text = element_text(color='black'))+
  geom_point(aes_(x = plotmat_bliss$lxmid, y = plotmat_bliss$ANCHOR_VIAB/2), shape = 2)#, color = "black"
p_high#  #+annotate("label", x = plotmat_bliss$IC50 + 1, y = 0.5, hjust = "left",
#label = sprintf("IC50==%.3f~log[e]~mu*M", plotmat_bliss$IC50), parse = T)
#label = sprintf("IC50==%.3f~mu*M", plotmat_bliss$ic50_true), parse = T)

temp = data.frame(nlme_stats[c("drug", "ic50_true", 'IC50')])
temp = distinct(temp)
temp$pid = pid_cur
temp

p_high+annotate('text', x = -10, y = 0.5, label = paste0('CAY10603 IC50=', signif(temp[temp$drug == 'CAY10603', 'IC50'], 3))) +
  annotate('text', x = -10, y = 0.4, label = paste0('Cisplatin IC50=', signif(temp[temp$drug == 'Cisplatin', 'IC50'], 3))) # +
# annotate('text', x = -10, y = 0.3, label = paste0('Citarinostat IC50=', signif(temp[temp$drug == 'Citarinostat', 'IC50'], 3)))+
# annotate('text', x = -10, y = 0.2, label = paste0('JNK-IN-7 IC50=', signif(temp[temp$drug == 'JNK-IN-7', 'IC50'], 3))) +
annotate('text', x = -10, y = 0.1, label = paste0('THZ IC50=', signif(temp[temp$drug == 'THZ', 'IC50'], 3)))

############## 20260613 ##################
unique( plotmat_bliss$drug)
p_high <- ggplot(plotmat_bliss[plotmat_bliss$drug %in% c("Cisplatin", "CAY10603" ), ] ,aes(x = lx, y = y_1, color=drug))+
  # geom_boxplot(outlier.color = NA)+
  geom_point( size = 0.5)+
  scale_color_manual(values=c("purple","orange","green","red", 'lightblue')) + # 
  scale_x_continuous(limits =c(-20, 20) )+# c(plot_low_x, plot_high_x)
  scale_y_continuous(limits = c(0, 1)) + 
  stat_function(aes_(x =~lx), fun = l3_model2, colour="orange" ,
                args = list(maxc = plot_Cisplatin_maxc , xmid = plot_Cisplatin_xmid,
                            scal = plot_Cisplatin_scal,
                            ANCHOR_VIAB = plot_Cisplatin_ANCHOR_VIAB)) +
  stat_function(aes_(x =~lx), fun = l3_model2, colour="purple" ,
                args = list(maxc = plot_CAY10603_maxc, xmid = plot_CAY10603_xmid,
                            scal = plot_CAY10603_scal,
                            ANCHOR_VIAB=plot_CAY10603_ANCHOR_VIAB)) +
  # stat_function(aes_(x =~lx), fun = l3_model2, colour="green" ,
  #               args = list(maxc = plot_Citarinostat_maxc , xmid = plot_Citarinostat_xmid,
  #                           scal = plot_Citarinostat_scal,
  #                           ANCHOR_VIAB=plot_Citarinostat_ANCHOR_VIAB)) +
  # stat_function(aes_(x =~lx), fun = l3_model2, colour="red" ,
  #               args = list(maxc = plot_THZ_maxc , xmid = plot_THZ_xmid,
  #                           scal = plot_THZ_scal,
  #                           ANCHOR_VIAB=plot_THZ_ANCHOR_VIAB)) +
  # stat_function(aes_(x =~lx), fun = l3_model2, colour="lightblue" ,
  #               args = list(maxc = plot_JNK_maxc , xmid = plot_JNK_xmid,
  #                           scal = plot_JNK_scal,
  #                           ANCHOR_VIAB=plot_JNK_ANCHOR_VIAB)) +
  # # geom_point(aes_(x = plotmat_bliss$lx, y = plotmat_bliss$y_1, group=plotmat_bliss$drug), shape = 1)+
  theme_classic()+
  labs(y = "Cell survival rate", #x =  expression(Dose/log[e]~mu*M), 
       x = 'ln(conc) μM',
       title = paste0('Patient: ', pid_cur ))+
  theme(text = element_text(size = 15), axis.text = element_text(color='black')) +
  geom_point(aes (x = lxmid, y = ANCHOR_VIAB/2), shape = 2)#, color = "black"
p_high

ggsave("//172.16.13.136/share/members/chenghonghan/PTV1/proof_check20260608/fig5/P865_20260614.pdf", p_high , width = 6, height = 4)
