rm(list = ls());gc()
library(openxlsx)
library(ggplot2)

tables1 = read.xlsx("D:/chh/2025workProject/20250506PTV1/TableS1_drug_cell_20240620.xlsx", sheet = 4)
druginfo = read.xlsx('D:/chh/2025workProject/20250506PTV1/20250121_ptv3_all_cell_drugid_prism_label_check.xlsx')
E_sinfo = read.xlsx("D:/chh/2025workProject/20250506PTV1/E/PTV3_E送样表20230914.xlsx")
E_sinfo = E_sinfo[c(1,2,3,8,11)]
G_sinfo = read.xlsx("D:/chh/2025workProject/20250506PTV1/PTV3_G送样表20230914.xlsx", sheet =2)
G_sinfo = G_sinfo[c(1,2,3,8,11)]
colnames(G_sinfo)[1] = 'sample_id'

E1 = read.csv("D:/chh/2023workProject/PTV3/database/20250507E/E_20240906report.pg_matrix.tsv", sep = '\t', row.names = 1)
E1[1:2, 1:5]
E1 = E1[5:ncol(E1)]
colnames(E1) = gsub('.mzML', '', colnames(E1), fixed = T)
colnames(E1) = gsub('.wiff', '', colnames(E1), fixed = T)

E_matinfo = data.frame(filename = colnames(E1))
E_matinfo$sid = as.character(lapply(E_matinfo$filename, function(x){
  x = gsub('_rep', 'rep',x, ignore.case = T)
  x = gsub('.mzML', '',x, fixed = T)
  x = strsplit(x, '_')[[1]]
  n = length(x)
  x[n]
}))
E_matinfo$sample_id = gsub('rep', '', E_matinfo$sid)


files = list.files('D:/chh/2025workProject/20250506PTV1/EGdiannreport/')
Gmat = read.csv(paste0('D:/chh/2025workProject/20250506PTV1/EGdiannreport/', files[1]), sep = '\t', row.names = 1)
Gmat = Gmat[5:ncol(Gmat)]
for( f in files[2:length(files)]){
  df = read.csv(paste0('D:/chh/2025workProject/20250506PTV1/EGdiannreport/', f ), sep = '\t', row.names = 1)
  df = df[5:ncol(df)]
  Gmat[rownames(df), colnames(df)] = df
}
Gmat[1:4, 1:2]
colnames(Gmat) = gsub('.mzML', '', colnames(Gmat), fixed = T)
colnames(Gmat) = gsub('.wiff', '', colnames(Gmat), fixed = T)

G_matinfo = data.frame(filename = colnames(Gmat))
G_matinfo$sid = as.character(lapply(G_matinfo$filename, function(x){
  x = gsub('_rep', 'rep',x, ignore.case = T)
  x = gsub('.mzML', '',x, fixed = T)
  x = strsplit(x, '_')[[1]]
  n = length(x)
  x[n]
}))
G_matinfo$sample_id = gsub('rep', '', G_matinfo$sid)
G_matinfo = merge(G_matinfo, G_sinfo, by = c('sample_id'), all.x = T)
G_matinfo = subset(G_matinfo, !(filename %in% E_matinfo$filename))
unique(G_matinfo$Cell)

E_matinfo =  rbind(E_matinfo[1:3], subset(G_matinfo[1:3],  grepl('^E', sample_id)))
E_matinfo = distinct(E_matinfo)
E_matinfo = merge(E_matinfo, E_sinfo, by = c('sample_id'), all.x = T)
E_matinfo = subset(E_matinfo, Cell %in% c("NCI-H1975", "Panc1005", "NCI-H2170", "A375", 'HCT116'))
unique(E_matinfo$Cell)
G_matinfo = subset(G_matinfo, Cell %in% c( 'HCT116'))


controlexp = read.csv("//172.16.13.136/share/members/sunr/PTV1/PTV1_honghan/2025_EGraw/controlreport.pg_matrix.tsv", sep = '\t', row.names = 1)
colnames(controlexp) = gsub('X..members.sunr.PTV1.2025_EGraw.', '', colnames(controlexp), fixed = T)
controlexp = controlexp[5:ncol(controlexp)]
controlexp[1:3, 1:2]
colnames(controlexp) = gsub('.mzML', '', colnames(controlexp), fixed = T)
colnames(controlexp) = gsub('.wiff', '', colnames(controlexp), fixed = T)

control_matinfo = data.frame(filename = colnames(controlexp))
control_matinfo$sid = as.character(lapply(control_matinfo$filename, function(x){
  x = gsub('_rep', 'rep',x, ignore.case = T)
  x = gsub('.wiff', '',x, fixed = T)
  x = strsplit(x, '_')[[1]]
  n = length(x)
  x[n]
}))
control_matinfo$sample_id = gsub('rep', '', control_matinfo$sid)
control_matinfo = merge(control_matinfo, E_sinfo, by = c('sample_id'), all.x = T)
for (i in c('G274', 'G6255')) {
  control_matinfo[control_matinfo$sample_id == i, c("Cell","Cell_plate", "pert_id","pert_time")] = G_sinfo[G_sinfo$sample_id == i, c("Cell","Cell_plate", "pert_id","pert_time")]
}
control_matinfo = subset(control_matinfo, !(filename %in% E_matinfo$filename))
dim(control_matinfo)

allinfo = rbind(control_matinfo, G_matinfo, E_matinfo)
allinfo = distinct(allinfo)
allinfo = allinfo[order(allinfo$Cell_plate, allinfo$pert_id, allinfo$pert_time), ]

EGcontrol = E1[intersect(E_matinfo$filename, colnames(E1))]
inter = intersect(c(E_matinfo$filename, G_matinfo$filename), colnames(Gmat))
EGcontrol[rownames(Gmat), inter] = Gmat[inter]
EGcontrol[rownames(controlexp), control_matinfo$filename] = controlexp[control_matinfo$filename]
EGcontrol = EGcontrol[allinfo$filename]

narow = colSums(!is.na(EGcontrol))
EGcontrol = EGcontrol[narow!=0 ]
narow = rowSums(!is.na(EGcontrol))
EGcontrol = EGcontrol[narow!=0, ]

allinfo = subset(allinfo, filename %in% colnames(EGcontrol))

#### AB ####
techrep = allinfo[grepl('rep', allinfo$sid), 'sid']
techrep = gsub('rep', '', techrep)
techrep[1]

techrep_cor = c()
techrep_cv = c()
for (i in unique(techrep)) {
  # i = techrep[1]
  ss = allinfo[allinfo$sample_id %in% i, 'filename'] #techrep[techrep$Cell_plate %in% i, 'filename']
  if(length(ss)<2){
    print(ss)
    next
  }
  temp = EGcontrol[ss]
  temp = cor(temp,  use = "pairwise.complete.obs")
  diag(temp) = NA
  temp[upper.tri(temp)] = NA
  temp = melt(temp)
  temp = subset(temp, !is.na(value))
  temp$bid = i
  techrep_cor = rbind(techrep_cor, temp)
  
  temp = EGcontrol[ss]
  temp = data.frame(apply(as.matrix(log2(temp)), 1, function(x){# log2(temp)
    sd(x, na.rm = T)/mean(x, na.rm = T)
  }))
  temp$bid= ss[1]
  colnames(temp)[1] = 'value'
  temp$bid = i
  temp = subset(temp, !is.na(value))
  techrep_cv = rbind(techrep_cv, temp)
}

p = ggplot(techrep_cv, aes(x = 'techrep cv', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.05 )+
  #ylim(0,1)+
  annotate('text', x= 'techrep cv' , y  = median(techrep_cv$value, na.rm = T), label = paste0('Median=',round(median(techrep_cv$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+ # poolms_log2_cv
  theme_classic()+
  theme(text = element_text(size = 15),
        # axis.text.x = element_text(size = 15),
        # axis.text.y = element_text(size = 15)
  )
p
ggsave("D:/chh/2025workProject/20250506PTV1/ptv3G/ptv1_eg_ptds5_techrep_log2_cv.pdf", p)


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
p
ggsave("D:/chh/2025workProject/20250506PTV1/ptv3G/ptv1_eg_ptds5_techrep_cor.pdf", p)

#### CD ####
poolmat = read.csv("Z:/members/sunr/others/PTV1/2025_EGraw/pool/EG_pool_report.pg_matrix.tsv", sep = '\t', row.names = 1)
poolmat = poolmat[5:ncol(poolmat)]


pool_cor = cor(poolmat,  use = "pairwise.complete.obs")
diag(pool_cor) = NA
pool_cor[upper.tri(pool_cor)] = NA
pool_cor = melt(pool_cor)
head(pool_cor)
pool_cor = subset(pool_cor, !is.na(value))

pool_cv = data.frame(apply(as.matrix(log2(poolmat)), 1, function(x){# log2(temp)
  sd(x, na.rm = T)/mean(x, na.rm = T)
}))
colnames(pool_cv)[1] = 'value'
p = ggplot(techrep_cv, aes(x = 'pool cv', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.05 )+
  #ylim(0,1)+
  annotate('text', x= 'pool cv' , y  = median(techrep_cv$value, na.rm = T), label = paste0('Median=',round(median(pool_cv$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+ # poolms_log2_cv
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black') )
p
ggsave("D:/chh/2025workProject/20250506PTV1/ptv3G/ptv1_eg_ptds5_pool_log2_cv.pdf", p)

p = ggplot(pool_cor, aes(x = 'pool cor', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.1)+
  ylim(0,1)+
  annotate('text', x= 'pool cor' , y  = median(pool_cor$value, na.rm = T), label = paste0('Median=',round(median(pool_cor$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text  = element_text(size = 15, color = 'black') )
p
ggsave("D:/chh/2025workProject/20250506PTV1/ptv3G/ptv1_eg_ptds5_pool_cor.pdf", p)
