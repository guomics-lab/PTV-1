

#### B ####
df1 = read.csv("//172.16.13.136/share/members/sunr/others/PPTV2_KD/B20231222sunr_makp_PTV2_kd_60minDIA/20231222PTV2_KDreport.pg_matrix.tsv", sep = '\t')
rownames(df1) = paste0(df1$Protein.Group, '_', df1$Genes)
df1[df1$Genes %in% c('TYMS', 'AKR1C3', 'CMPK1'), c(1,4)]
#               Protein.Group  Genes
# P04818_TYMS          P04818   TYMS
# P30085_CMPK1         P30085  CMPK1
# P42330_AKR1C3        P42330 AKR1C3
df1 = df1[6:ncol(df1)]
df1 = df1[!grepl('blk', colnames(df1)) ]
colnames(df1)
colnames(df1) = sapply(colnames(df1),  function(x){
  # x = "X..members.sunr.PPTV2_KD.B20231222sunr_makp_PTV2_kd_60minDIA.B20231222sunr_makp_PTV2_kd_60minDIA_BT20_AKRIC3.raw"
  x = strsplit(x, '_')[[1]]
  paste0(x[11:length(x)], collapse = '_')
})
colnames(df1)
df1 = log2(df1)



df1.1 = df1[c('P04818_TYMS', 'P30085_CMPK1', 'P42330_AKR1C3'), ]
df1.1$Proteins = rownames(df1.1)
df1.1 = melt(df1.1, id.vars = 'Proteins')
head(df1.1)
df1.1$group = gsub('.raw', '', df1.1$variable, fixed = T)
df1.1$group = gsub('_Rep', '', df1.1$group, ignore.case = T)
df1.1$cell = gsub('_.*', '', df1.1$group)
df1.1$group = gsub('.*_', '', df1.1$group)
head(df1.1)

plotlist = list()
for (pro in unique(df1.1$Proteins)) {
  # break
  p1 = ggplot(df1.1[df1.1$Proteins %in%pro, ], aes(cell, value, color = group))+
    geom_boxplot()+
    geom_point(position = position_jitterdodge())+
    scale_color_d3()+
    labs(title = pro)+
    theme_classic()+
    scale_y_continuous(n.breaks = 8)+
    theme(text = element_text(size = 15),
          axis.text  = element_text(size = 15, color = 'black'), axis.title.x = element_blank() )
  plotlist[[pro]] = p1
}
plotlist[1]
plotlist[2]
plotlist[3]
df1.1[df1.1$Proteins %in%'P42330_AKR1C3', ]
df1.1[df1.1$Proteins %in%'P30085_CMPK1', ]
df1.1[df1.1$Proteins %in%'P04818_TYMS', ]

