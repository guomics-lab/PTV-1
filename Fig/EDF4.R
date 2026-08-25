library(openxlsx)
library(ggplot2)
library(ggsci)
library(ggnewscale)
#### A####
f = "D:/chh/2025workProject/20250506PTV1/L/scrUMAP/embeddingUMAP/ptv1_EGembed_euclideanSeed0_202507221453.csv"

df = read.csv(f, row.names = 1)
unique(df$pert_time)
df$pert_time = paste0(df$pert_time, 'h')
df$pert_time = factor(df$pert_time, levels = c( "2h", "4h", "6h","8h", "10h", "12h","24h","36h","48h", "60h") )

temp1 = subset(df, pert_time %in% c( "2h", "4h", "8h", "10h", "12h", "36h", "60h"))# 6 24 48 换成五角星吧
temp1$pert_time = factor(temp1$pert_time, levels = c( "2h", "4h", "8h", "10h", "12h", "36h", "60h") )
temp2 = subset(df, pert_time %in% c('6h', "24h", "48h"))# 6 24 48 换成五角星吧
temp2$pert_time = factor(temp2$pert_time, levels = c('6h', "24h", "48h") )

p = ggplot() +
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
  scale_x_continuous(n.breaks = 8)+
  scale_y_continuous(n.breaks = 8) # + labs(title = paste0(unique(df1$drugNum)[1]) )
p