library(openxlsx)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(reshape2)
library(stringi)
library(dplyr)
#### AB ####
pg_matrix = read.csv("//172.16.13.136/share/members/sunr/PTV1/2025_EGraw/L/mzMLreport20250725.pg_matrix.tsv" , sep = '\t' )
rownames(pg_matrix) = pg_matrix$Protein.Group
colnames(pg_matrix)[1:10]
pg_matrix = pg_matrix[6:ncol(pg_matrix)]


source code:D:/chh/2025workProject/20250506PTV1/L/PTV1_L20250722.R

linfo = read.xlsx("D:/chh/2025workProject/20250506PTV1/L/Lsampleinfo20250725.xlsx");rownames(linfo) = linfo$filename

poolinfo = subset(linfo, !is.na(pool))

temp = pg_matrix[poolinfo$filename]
temp = cor(temp,  use = "pairwise.complete.obs")
diag(temp) = NA
temp[upper.tri(temp)] = NA
temp = melt(temp)
temp = subset(temp, !is.na(value))
poolcor = data.frame(temp)
head(poolcor)
# dir.create('D:/chh/2025workProject/20250506PTV1/L/20250725/')
write.xlsx(poolcor, "D:/chh/2025workProject/20250506PTV1/L/20250725/poolcor.xlsx", rowNames = T)


temp = pg_matrix[poolinfo$filename]
temp = data.frame(apply(as.matrix(temp), 1, function(x){# log2(temp)
  sd(x, na.rm = T)/mean(x, na.rm = T)
}))
poolcv = temp;colnames(poolcv) = 'temp'
poolcv = subset(poolcv, !is.na(poolcv$temp))
head(poolcv)
write.xlsx(poolcv, "D:/chh/2025workProject/20250506PTV1/L/20250725/poolcv_log2.xlsx", rowNames = T)


p = ggplot(poolcv, aes(x = 'pool cv', y = temp))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.05 )+
  #ylim(0,1)+
  annotate('text', x= 'pool cv' , y  = median(poolcv$temp, na.rm = T), label = paste0('Median=',round(median(poolcv$temp, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+ # poolms_log2_cv
  theme_classic()+
  theme(text = element_text(size = 15),
        # axis.text.x = element_text(size = 15),
        # axis.text.y = element_text(size = 15)
  )
ggsave('D:/chh/2025workProject/20250506PTV1/L/20250725/poolcv_log2.pdf', p)

p = ggplot(poolcor, aes(x = 'pool cor', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.1)+
  ylim(0,1)+
  annotate('text', x= 'pool cor' , y  = median(poolcor$value, na.rm = T), label = paste0('Median=',round(median(poolcor$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15))
ggsave('D:/chh/2025workProject/20250506PTV1/L/20250725/poolcor.pdf', p)

for (i in unique(poolinfo$machine)) {
  temp = pg_matrix[poolinfo[poolinfo$machine %in% i, 'filename']]
  temp = data.frame(apply(as.matrix(log2(temp)), 1, function(x){# log2(temp)
    sd(x, na.rm = T)/mean(x, na.rm = T)
  }))
  poolcv = temp;colnames(poolcv) = 'temp'
  # write.xlsx(poolcv, "D:/chh/2025workProject/20250506PTV1/L/poolcv.xlsx", rowNames = T)
  
  p = ggplot(poolcv, aes(x = 'pool cv', y = temp))+#combat 
    geom_violin()+
    geom_boxplot(width = 0.05 )+
    #ylim(0,1)+
    annotate('text', x= 'pool cv' , y  = median(poolcv$temp, na.rm = T), label = paste0('Median=',round(median(poolcv$temp, na.rm = T), 5)))+
    labs(x = ' ', y = ' ', title = i)+ # poolms_log2_cv
    theme_classic()+
    theme(text = element_text(size = 15) 
    )
  ggsave(paste0('D:/chh/2025workProject/20250506PTV1/L/20250725/poolcv_',i,'_log2.pdf'), p)
  print(c(i, round(median(poolcv$temp, na.rm = T), 5) ))
}

techrep = linfo[grepl('rep', linfo$sid), 'sid']
biorep = linfo[linfo$pert_id %in% c('DMSO', "combo20", "combo40", "combo60") & !grepl('rep', linfo$sid), ]

######## biorep cv cor ########
biorep_cor = c()
biorep_cv = c()
for (i in unique(biorep$Cell_plate)) {
  ss = biorep[biorep$Cell_plate %in% i, 'filename']
  temp = pg_matrix[ss]
  temp = cor(temp,  use = "pairwise.complete.obs")
  diag(temp) = NA
  temp[upper.tri(temp)] = NA
  temp = melt(temp)
  temp = subset(temp, !is.na(value))
  biorep_cor = rbind(biorep_cor, temp)
  
  temp = pg_matrix[ss]
  temp = data.frame(apply(as.matrix(temp), 1, function(x){# log2(temp)
    sd(x, na.rm = T)/mean(x, na.rm = T)
  }))
  colnames(temp) = 'value'
  biorep_cv = rbind(biorep_cv, temp)
}
write.xlsx(biorep_cv, "D:/chh/2025workProject/20250506PTV1/L/20250725/biorep_cv.xlsx", rowNames = T)
write.xlsx(biorep_cor, "D:/chh/2025workProject/20250506PTV1/L/20250725/biorep_cor.xlsx", rowNames = T)


p = ggplot(biorep_cv, aes(x = 'biorep cv', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.05 )+
  #ylim(0,1)+
  annotate('text', x= 'biorep cv' , y  = median(biorep_cv$value, na.rm = T), label = paste0('Median=',round(median(biorep_cv$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+ # poolms_log2_cv
  theme_classic()+
  theme(text = element_text(size = 15),
        # axis.text.x = element_text(size = 15),
        # axis.text.y = element_text(size = 15)
  )
ggsave("D:/chh/2025workProject/20250506PTV1/L/20250725/biorep_cv.pdf", p)
p = ggplot(biorep_cor, aes(x = 'biorep cor', y = value))+#combat 
  geom_violin()+
  geom_boxplot(width = 0.1)+
  ylim(0,1)+
  annotate('text', x= 'biorep cor' , y  = median(biorep_cor$value, na.rm = T), label = paste0('Median=',round(median(biorep_cor$value, na.rm = T), 5)))+
  labs(x = ' ', y = ' ')+
  theme_classic()+
  theme(text = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15))
ggsave("D:/chh/2025workProject/20250506PTV1/L/20250725/biorep_cor.pdf", p)
######## techrep cv cor ########
techrep = gsub('rep', '', techrep)

techrep_cor = c()
techrep_cv = c()
for (i in unique(techrep)) {
  ss = linfo[linfo$sid2 %in% i, 'filename'] #techrep[techrep$Cell_plate %in% i, 'filename']
  if(length(ss)<2){
    print(ss)
    next
  }
  temp = pg_matrix[ss]
  temp = cor(temp,  use = "pairwise.complete.obs")
  diag(temp) = NA
  temp[upper.tri(temp)] = NA
  temp = melt(temp)
  temp = subset(temp, !is.na(value))
  techrep_cor = rbind(techrep_cor, temp)
  
  temp = pg_matrix[ss]
  temp = data.frame(apply(as.matrix(temp), 1, function(x){# log2(temp)
    sd(x, na.rm = T)/mean(x, na.rm = T)
  }))
  colnames(temp) = 'value'
  techrep_cv = rbind(techrep_cv, temp)
}
write.xlsx(techrep_cv, "D:/chh/2025workProject/20250506PTV1/L/techrep_cv.xlsx", rowNames = T)
write.xlsx(techrep_cor, "D:/chh/2025workProject/20250506PTV1/L/techrep_cor.xlsx", rowNames = T)

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
ggsave("D:/chh/2025workProject/20250506PTV1/L/20250725/techrep_cv.pdf", p)

techrep_cor = read.xlsx( "D:/chh/2025workProject/20250506PTV1/L/techrep_cor.xlsx", rowNames = T)
techrep_cor = subset(techrep_cor, value>=0.9)
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
# ggsave("D:/chh/2025workProject/20250506PTV1/L/20250725/techrep_cor.pdf", p)
ggsave("D:/chh/2025workProject/20250506PTV1/L/20250725/techrep_cor0.9.pdf", p)

#### CDEF ####
library(ggnewscale)
scrna = list.files("D:/chh/2025workProject/20250506PTV1/L/scrUMAP/20250725/" , pattern = '*csv', full.names = T)
scrna
for (f in scrna ) {#1:4, 9:12
  # f = scrna[7]
  df = read.csv(f, row.names = 1)
  df$drugNum = ifelse(grepl('pool',  rownames(df), ignore.case = T), 'pool' , df$drugNum)#
  df$cell = ifelse(grepl('pool',  rownames(df), ignore.case = T), 'pool', df$cell)
  df = subset(df, cell !='pool')#
  df = subset(df, drugNum !='blk')#
  # df = subset(df , !(drugNum %in%  c( "blk", "DMSO",  "combo20", "combo40", "combo60")))
  # df = subset(df , (cell %in%  c( "HCC1806", "HCC1143", "M453(ATCC)")))
  df$pert_time = paste0(df$pert_time, 'h')
  df$pert_id = linfo[rownames(df), 'pert_id']
  df$pert_id = ifelse(df$pert_id %in% c("combo20", "combo40" , "combo60"), 'DMSO', df$pert_id)
  # unique(df$pert_time)
  df1 = df
  df1 = subset(df1, !(cell == "M453(ATCC)" & UMAP1<10 & UMAP2 > 5))
  df1 = subset(df1, !(cell == "M453(ATCC)" & UMAP1>7 & UMAP1< 12.5 &UMAP2< -3))
  df1 = subset(df1, !(cell == "HCC1143" & UMAP1>8))
  df1 = subset(df1, !(cell == "HCC1143" & UMAP1>5 &UMAP2 < -3))
  df1 = subset(df1, !(cell == "HCC1806" & UMAP1>8 &UMAP2 < 0 &UMAP2 > -4))
  
  df1[df1$drugNum %in% 'DMSO', 'pert_time'] = '0h'
  
  df1$pert_time = factor(df1$pert_time, levels = c('0h', "2h", "4h", "6h","8h", "10h", "12h","24h","36h","48h", "60h") )
  df1$drugNum = factor(df1$drugNum, levels = c("single","combo", "DMSO"))#, "blk"
  
  p = ggplot(df1, aes(UMAP1, UMAP2, color = machine )) +
    geom_point(size=0.8)+
    scale_color_d3()+
    theme_classic()+
    theme(text = element_text(size = 15),
          axis.text = element_text(size = 15) )+
    scale_x_continuous(n.breaks = 6)+ scale_y_continuous(n.breaks = 6) # + labs(title = paste0(unique(df1$drugNum)[1]) )
  p#;gsub('csv', 'time.pdf', f)
  ggsave(gsub('csv', paste0( 'machine20250708.pdf'), f), p)
  
  p = ggplot(df1, aes(UMAP1, UMAP2, color = cell )) +
    geom_point()+
    scale_color_d3()+ # theme_bw()+
    theme_classic()+
    theme(text = element_text(size = 15),
          axis.text = element_text(size = 15) )+
    scale_x_continuous(n.breaks = 6)+ scale_y_continuous(n.breaks = 6) # + labs(title = paste0(unique(df1$drugNum)[1]) )
  p#;gsub('csv', 'time.pdf', f)
  ggsave(gsub('csv', paste0( 'cell20250708.pdf'), f), p)
  
  temp1 = subset(df1, pert_time %in% c('0h', "2h", "4h", "8h", "10h", "12h", "36h", "60h"))# 6 24 48 换成五角星吧
  temp1$pert_time = factor(temp1$pert_time, levels = c('0h', "2h", "4h", "8h", "10h", "12h", "36h", "60h") )
  temp2 = subset(df1, pert_time %in% c('6h', "24h", "48h"))# 6 24 48 换成五角星吧
  temp2$pert_time = factor(temp2$pert_time, levels = c('6h', "24h", "48h") )
  
  p = ggplot() +
    # scale_color_manual(values = c("#FADCC9", "#F9C3AB", "#F7A994", "#F18B8B", "#DF7695", "#C6759C", "#AC789A", "#9377AB", "#7C7284", "#676975", "#4F515A"))+
    # scale_fill_manual(values = c("#FADCC9", "#F9C3AB", "#F7A994", "#F18B8B", "#DF7695", "#C6759C", "#AC789A", "#9377AB", "#7C7284", "#676975", "#4F515A"))+
    
    geom_point(aes(UMAP1, UMAP2, color = pert_time), data = temp1, size=1 )+
    scale_color_manual(values = c("#FADCC9", "#F9C3AB", "#F7A994",  "#DF7695", "#C6759C", "#AC789A",  "#7C7284",  "#4F515A"))+
    new_scale_color() +
    geom_point(aes(UMAP1, UMAP2, fill = pert_time), data = temp2, size=2 , shape=22, colour = "black", stroke = 0.2)+
    scale_fill_manual(values = c("#F18B8B","#9377AB","#676975"))+
    # geom_point(data = subset(df1, is.na(pert_time) ), color = 'grey') +#4F4F4F
    # theme_bw() +
    theme_classic() +
    theme(text = element_text(size = 15),
          axis.text = element_text(size = 15, color = 'black'), axis.ticks.length = unit(0.1, "cm"))  +
    # scale_x_continuous(n.breaks = 6)+
    scale_y_continuous(n.breaks = 6) # + labs(title = paste0(unique(df1$drugNum)[1]) )
  p#;gsub('csv', 'time.pdf', f)
  ggsave(gsub('csv', paste0( 'time20250708.pdf'), f), p)
  
  p = ggplot(df1, aes(UMAP1, UMAP2, color = drugNum )) +
    geom_point()+
    scale_color_d3()+
    theme_classic()+
    theme(text = element_text(size = 15),
          axis.text = element_text(size = 15))+#, axis.ticks.length.x = unit(0.3, 'cm')
    scale_x_continuous(n.breaks = 6)+ scale_y_continuous(n.breaks = 6) # + labs(title = paste0(unique(df1$drugNum)[1]) )
  p#;gsub('csv', 'time.pdf', f)
  ggsave(gsub('csv', paste0( 'drugNum20250708.pdf'), f), p)
}
