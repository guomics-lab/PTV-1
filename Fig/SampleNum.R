library(openxlsx)
library(reshape2)
library(dplyr)

rm(list = ls());gc()
#### PTV1-EDF1 ####
"D:\chh\2023workProject\prottalk\database\unpool_unMedium_16311sampleinfo.xlsx"
"D:\chh\2023workProject\prottalk\database\unpool_sampleinfo.xlsx"
"D:\chh\2023workProject\prottalk\database\unpool_unMedium_16311matrix.xlsx"

df_info2 = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/PTV1/03_16311sampleinfo.xlsx")
df_info3 = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/PTV1/03_16311sampleinfo.xlsx", sheet = 2)

head(df_info2)
df_info2$cell = df_info2$cell_line

head(df_info3)
df_info3$cell = df_info3$cell_line

#### PTV1-L ####
rm(list = ls());gc()
# df1_info250722 = read.xlsx("D:/chh/2025workProject/20250506PTV1/L/Lsampleinfo20250722.xlsx")

drug_cell_drugs = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/00_check/PTV1_check/04_sTable/TableS1_drug_cell_20251030-2120.xlsx", sheet = 3)
rownames(drug_cell_drugs) = drug_cell_drugs$Pert_ID
drug_cell_drugs = drug_cell_drugs[c("Pert_ID", "Drug_name")]
colnames(drug_cell_drugs)[1] = 'pert_id'

df1 = read.xlsx("D:/chh/2025workProject/20250506PTV1/L/L_matrix20250725.xlsx", rowNames = T)
df1_info = read.xlsx("D:/chh/2025workProject/20250506PTV1/L/Linfo20250725.xlsx")
rownames(df1_info) = df1_info$filename
unique(df1_info[grepl('#', df1_info$pert_id), 'pert_id'])
unique(df1_info[grepl('combo', df1_info$pert_id), 'pert_id'])

df1[1:3, 1:3]
unique(df1_info$cell)
setdiff( df1_info$filename, colnames(df1))
pool = grep('poo', df1_info$filename, ignore.case = T)
techrep = df1_info[grepl('rep', df1_info$sid), 'sid']
df1_info[grepl('rep', df1_info$sid), 'pool'] = 'techrep'

biorep = df1_info[df1_info$pert_id %in% c('DMSO', "combo20", "combo40", "combo60") & !grepl('rep', df1_info$sid), ]
biorep$cp_cell = paste0(biorep$Cell_plate, '_', biorep$cell)

for (i in unique(biorep$Cell_plate)) {
  tmp = subset(biorep, Cell_plate %in% i)
  if(nrow(tmp)<2) next
  # print(c(i, nrow(tmp)))
  # tmp[2:nrow(tmp), 'pool'] = 'biorep'
  df1_info[rownames(tmp)[2:nrow(tmp)], 'pool'] = 'biorep'
}
table(df1_info$pool)

df1_info$drugname = drug_cell_drugs[df1_info$pert_id, 2]

combo = read.xlsx("D:/chh/2025workProject/20250506PTV1/L/PTV1_L20250704.xlsx", sheet = 3)
combo$pert_id = combo$type
rownames(combo) = combo$type
head(combo)
colnames(combo)
combo = combo[c("Anchor_id", "Anchor_iname", "Anchor_dose", "Library_id", "Library_iname", "Library_dose" )]
df1_info[colnames(combo)] = combo[df1_info$pert_id, ]


unique(df1_info$drugNum)
temp = subset(df1_info, is.na(pool))
temp = subset(temp, !grepl('rep', temp$filename) & !(pert_id %in% c('DMSO', "combo20", "combo40", "combo60") ))
temp = temp[order(temp$pert_id, temp$cell, temp$pert_time), ]
table(temp$drugNum)

temp = cbind(df1_info, data.frame(t(df1[df1_info$filename])))

# write.xlsx(temp, '//172.16.13.136/share/members/sunr/PTV1/PTV1_honghan/matrix_combine/PTV1_L_sinfo_matrix20260512-1005.xlsx')
write.xlsx(temp, '//172.16.13.136/share/members/sunr/PTV1/PTV1_honghan/matrix_combine/PTV1_L_sinfo_matrix20260616-1037.xlsx')


#### PTDS-5 ####
rm(list = ls());gc()
ptds5_mat = read.xlsx("D:/chh/2025workProject/20250506PTV1/ptv3G/EG_matrix.xlsx", rowNames = T)
ptds5_mat[1:3, 1:3]
ptds5_info = read.xlsx("D:/chh/2025workProject/20250506PTV1/ptv3G/EGinfo.xlsx")
rownames(ptds5_info) = ptds5_info$filename

ptds5_info = ptds5_info[order(ptds5_info$pert_id, ptds5_info$Cell, ptds5_info$pert_time),]
sort(unique(ptds5_info[!is.na(ptds5_info$pert_id), 'pert_id']))
sort(unique(ptds5_info[!is.na(ptds5_info$drug_name), 'drug_name']))

temp = read.xlsx('D:/chh/2025workProject/20250506PTV1/EGinfo20260128.xlsx')
setdiff(temp$sample_id, ptds5_info$sample_id)
temp[temp$sample_id %in% setdiff(temp$sample_id, ptds5_info$sample_id), ]

techrep = ptds5_info[grepl('rep', ptds5_info$sid), ]

ptds5_info[grepl('rep', ptds5_info$filename, ignore.case = T), 'techrep_biorep'] = 'techrep'

temp = ptds5_info
temp = subset(temp, sample_id %in% temp$control & !grepl('rep', filename, ignore.case = T))# 
temp$dct = paste0(temp$pert_id, '_', temp$Cell, '_', temp$pert_time)
for(i in unique(temp$dct)){
  tmp = subset(temp, dct %in% i)
  if(nrow(tmp)<2 ) next
  print(nrow(tmp))
  # break
  ptds5_info[rownames(tmp)[2:nrow(tmp)], 'techrep_biorep'] = 'biorep'
}
table(ptds5_info$techrep_biorep)

poolmat = read.csv("Z:/members/sunr/PTV1/PTV1_honghan/2025_EGraw/pool/EG_pool_report.pg_matrix.tsv", sep = '\t', row.names = 1)
poolmat = poolmat[5:ncol(poolmat)]
dim(poolmat)


df1 = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/00_check/PTV1_check/04_sTable/TableS8_PTDS-5_info_20251030_20260615-1105.xlsx", sheet=2)
df1[1:4, 1:3]
# df1$rep12 = 1:nrow(df1)
nrow(df1)
freq =  data.frame(table(df1$Sample_ID))
freq = freq[freq$Freq>1 ,]
# techrep = df1[grepl('rep', df1$Sample_rep), ]
techrep = subset(df1, Sample_ID %in% freq$Var1 | grepl('rep', Sample_ID))
nrow(techrep)/2# techrep num

for (i in unique(freq$Var1)) {
  tmp = subset(df1, Sample_ID %in% i)
  df1[rownames(tmp)[2], 'rep'] = 'techrep'
}
table(df1$rep)

biorep = subset(df1, is.na(rep))
nrow(biorep)
biorep$pct = paste0(biorep$Drug_name, '_', biorep$Cell, '_', biorep$Pert_time)
biorep = biorep %>% distinct(pct, .keep_all = T)
nrow(biorep)# unique sample
df1[df1$Sample_ID %in%biorep$Sample_ID,  ]

tmp = df1[df1$Sample_ID %in% biorep$Sample_ID  & is.na(df1$rep),  ]

df1[df1$Sample_ID %in% biorep$Sample_ID  & is.na(df1$rep),  'rep'] = 'unique sample'
# df1 = df1[order(df1$Drug_name, df1$Cell, df1$Pert_time), ]
df1[is.na(df1$rep), 'rep'] = 'biorep'
table(df1$rep)

write.xlsx(df1, "//172.16.13.136/share/members/sunr/PTV1/00_check/PTV1_check/04_sTable/TableS8_PTDS-5_info_20251030_20260616-1520.xlsx")


ptds5_mat = ptds5_mat[ptds5_info$filename]
colnames(ptds5_mat) = ptds5_info$sid
freq = data.frame(table(ptds5_info$sid))
ptds5_mat[1:3, 1:3]

# human_gene = read.xlsx("D:/chh/2025workProject/uniprotkb_Human_AND_reviewed_true_AND_m_2025_10_22.xlsx", rowNames = T)

na.omit(unique(df1$Drug_name))

temp = data.frame(t(ptds5_mat))
temp[1:3, 1:3]

write.xlsx(cbind(df1, cbind(rownames(temp), temp)), "//172.16.13.136/share/members/sunr/PTV1/00_check/PTV1_check/04_sTable/TableS8_PTDS-5_mat_info_20251030_20260616-1639.xlsx" )

#### FFPE ####
rm(list = ls());gc()
# D:/chh/2025workProject/20250506PTV1/Prognosis/PTV1_Prognosis_unicox_liutongFFPE.R
# df_ffpe = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/PTV1_rebuttal/00_PTV1_revised_202602/504FFPE_model/ptv1_ffpe_501patient_matinfo_20260410-1102.xlsx", sheet = 1)
# colnames(df_ffpe)[1:10]
# df_info = df_ffpe[1:9]

df_ffpe = read.xlsx("D:/chh/2025workProject/20250506PTV1/FFPE/ptv1_ffpe_human_mat20250912.xlsx", rowNames = T)
df_info = read.xlsx("D:/chh/2025workProject/20250506PTV1/FFPE/ptv1_ffpe_human_info20250912.xlsx")

unique(df_info[grepl('#', df_info$Pert_ID), 'Pert_ID'])
dim(df_ffpe)


ptv1_ffpe_human = read.csv('Z:/members/sunr/PTV1/PTV1_honghan/2025_EGraw/FFPE/PTV1_FFPE_mzML_human_report.pg_matrix.tsv', sep = '\t')
rownames(ptv1_ffpe_human) = ptv1_ffpe_human$Protein.Group
ptv1_ffpe_human = ptv1_ffpe_human[6:ncol(ptv1_ffpe_human)]
ptv1_ffpe_human[1:6, 1:2]
dim(ptv1_ffpe_human)
write.xlsx(ptv1_ffpe_human, 'Z:/members/sunr/PTV1/PTV1_honghan/2025_EGraw/FFPE/PTV1_FFPE_mzML_human_report.pg_matrix.xlsx', rowNames = T)

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

ffpe_info = read.xlsx("D:/chh/2025workProject/20250506PTV1/FFPE/datas/PTV1_FFPEinfo20250731.xlsx")
ffpe_info$batch.ID = gsub('-', '_', ffpe_info$batch.ID)


ptv1_ffpe_humaninfo = merge(ptv1_ffpe_humaninfo, ffpe_info[c("病理号", "batch.ID" )], by = c('batch.ID'), all.x = T)
rownames(ptv1_ffpe_humaninfo) = ptv1_ffpe_humaninfo$filename


pool = subset(ptv1_ffpe_humaninfo, grepl('pool', ptv1_ffpe_humaninfo$filename, ignore.case = T))


techrep = ptv1_ffpe_humaninfo[grepl('rep', ptv1_ffpe_humaninfo$sid), 'sid']
biorep = data.frame(table(ptv1_ffpe_humaninfo[!grepl('rep', ptv1_ffpe_humaninfo$sid), '病理号'] ))
biorep = subset(biorep, Freq>1)
biorep
biorep = subset(ptv1_ffpe_humaninfo, 病理号 %in% biorep$Var1 & !grepl('rep', sid))
#### FFPE 501 ####
source:D:/chh/2025workProject/20250506PTV1/Prognosis/PTV1_Prognosis_unicox_liutongFFPE.R

ts13_res = read.xlsx( "//172.16.13.136/share/members/sunr/PTV1/PTV1_rebuttal/00_PTV1_revised_202602/504FFPE_model/ptv1_ffpe_501patient_matinfo_202600413-1057.xlsx", sheet = 2)

#### 找到PDO: P855, P831, P865 ####
PDO = read.csv("//172.16.13.136/share/members/sunr/PTV1/PTV1_honghan/2025_EGraw/PDO/PDO_O_report.pg_matrix.tsv", sep = '\t', row.names = 1)
PDO = PDO[5:ncol(PDO)]
colnames(PDO)
PDO = PDO[,c(11, 6, 13)]
colnames(PDO) = c('P855', 'P831', 'P865')

write.xlsx(PDO, "//172.16.13.136/share/members/sunr/PTV1/PTV1_honghan/2025_EGraw/PDO/P855_P831_P865.pg_matrix.xlsx", rowNames = T)



#### 2015nm_PDX ####
df1 = read.xlsx("//172.16.13.136/share/members/sunr/PTV1/2015nm_PDX/41591_2015_loolabel(1).xlsx")
df1$cell = gsub('_.*', '', df1$label)
df1$cd = gsub(',.*', '', df1$label)
df1$dt = gsub('.*_', '', df1$label)
df1$dd = gsub(',.*', '', df1$dt)
df1$ptime = gsub('.*,', '', df1$label)
df1 = subset(df1, ptime==0)

head(df1)
nrow(df1)
temp = df1 %>% distinct(cell, .keep_all = T)
nrow(temp)# unique sample 178
nrow(df1)-nrow(temp)# rep 2180


temp = df1 %>% distinct(cd, .keep_all = T)
nrow(temp)# unique sample 178
nrow(df1)-nrow(temp)# rep 2180
#### package version ####

for (pv in c('stats', 'ggalluvial', 'dplyr', 'ggpubr', 'ggnewscale',
             'patchwork', 'reshape2' )) {
  print( paste0(c(pv, package.version(pv)), collapse = ' ') )
}
?UpSetR::fromList()