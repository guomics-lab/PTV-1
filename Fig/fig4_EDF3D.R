# ============================================================
# fig4_EDF3D.R
# Based on fig4.R #### D #### section
# Draw 2 IC50 dose-response curves for IC50_EDF3D.xlsx
# ============================================================

rm(list = ls());gc()
library(openxlsx)
library(reshape2)
library(dplyr)
library(ggplot2)
library(nlme)
library(gdscIC50)
source("./figEDF3D_ic50_function.R")
library(doMC)
registerDoMC()
library(magrittr)
library(lme4)

# ==================== 1. Read & reshape data ====================
dose_raw <- read.xlsx(
  "//172.16.13.136/share/members/chenghonghan/PTV1/proof_check20260608/IC50_EDF3D.xlsx",
  sheet = 1
)

# dose_raw: 9 rows x 7 cols
# colnames: con. | Capecitabine_WT_rep1 | Capecitabine_WT_rep2 | ... | Capecitabine_TYMS_rep3

# Rename columns to match expected format
colnames(dose_raw)[1] <- "doseum"

# Melt to long format
dose <- data.frame(melt(dose_raw, id.vars = "doseum"))
dose$variable <- as.character(dose$variable)

# Parse drug (WT / TYMS) and replicate info
dose$drug1 <- as.character(sapply(as.character(dose$variable), function(x) {
  # e.g., "Capecitabine_WT_rep1" -> drug = WT or TYMS
  parts <- strsplit(x, "_")[[1]]
  if (parts[2] == "WT") {
    return("WT")
  } else {
    return("TYMS")
  }
}))

dose$variable <- as.character(sapply(dose$variable, function(x) {
  # e.g., "Capecitabine_WT_rep1" -> "rep1"
  parts <- strsplit(x, "_")[[1]]
  return(paste0("rep", parts[length(parts)]))
}))

wt   <- "WT"
kd   <- "TYMS"

# ==================== 2. pchisq test ====================
pid_cur <- "EDF3D"
temp1 <- dose
temp1$x <- temp1$doseum

WT_model    <- lmer(value ~ doseum + (1 | variable), data = temp1[temp1$drug1 == "WT", ])
TYMS_model  <- lmer(value ~ doseum + (1 | variable), data = temp1[temp1$drug1 == "TYMS", ])

cat("Patient:", pid_cur, "\n")
cat("pchisq =", 1 - pchisq((-2 * (logLik(WT_model) - logLik(TYMS_model))), df = 1), "\n")

# ==================== 3. Prepare nlme input ====================
dfDat <- temp1[, c("drug1", "value", "x")]
colnames(dfDat)[1:2] <- c("drug", "y")
dfDat$CL          <- "No"
dfDat$maxc        <- max(dose$doseum)
dfDat$ANCHOR_VIAB <- 1
dfDat$x <- (log(dfDat$x / dfDat$maxc) / log(2)) + length(unique(dose$doseum))
dfDat$y <- 1 - dfDat$y  # convert viability -> kill
dfDat$y[dfDat$y < 0] <- 0
dfDat$y[dfDat$y > 1] <- 1

dfDat$drug_spec   <- "DRUG_ID_lib+maxc"
dfDat$DRUG_ID_lib <- dfDat$drug

gDat <- nlme::groupedData(y ~ x | drug / CL, data = dfDat, FUN = mean,
                          labels = list(x = "Concentration", y = "Viability"),
                          units = list(x = "uM/l", y = "percentage killed"))
gDat$type <- paste(gDat$CL, gDat$DRUG_ID_lib, sep = "_")

# ==================== 4. nlme model fitting ====================
nlme_stats <- c()
for (i in 1:length(unique(gDat$type))) {
  indsel <- gDat[gDat$type == unique(gDat$type)[i], ]
  fmMod1_lib <- try(fitModel(indsel, vStart = c(8.886464, 1.495953),
                             bLargeScale = FALSE, bSilent = TRUE), silent = TRUE)
  if ("try-error" %in% class(fmMod1_lib)) {
    next
  }
  nlme_stats_lib <- calcNlmeStats(fmMod1_lib, indsel)
  nlme_stats_lib$ic50_true <- exp(nlme_stats_lib$IC50)
  nlme_stats <- rbind(nlme_stats, nlme_stats_lib)
}

cat("\nIC50 results:\n")
print(unique(nlme_stats$ic50_true))
print(nlme_stats)

# ==================== 5. Build plot data ====================
nlme_stats1 <- nlme_stats
plot_data <- nlme_stats1 %>%
  mutate_(lx    = ~log(getConcFromX(x, maxc)),
          lxmid = ~log(getConcFromX(xmid, maxc)))

plot_data$drug1 <- plot_data$drug
plot_data$drug1 <- sapply(strsplit(plot_data$drug1, "_"), function(e) { e[length(e)] })

plot_data <- plot_data[plot_data$IC50 != Inf, ]
plot_data <- plot_data[plot_data$IC50 != -Inf, ]

# ---- WT ----
drug_WT <- plot_data[plot_data$drug == "WT", ]
drug_WT$ythat_1 <- mean(drug_WT$ANCHOR_VIAB) - drug_WT$yhat
drug_WT$y_1     <- mean(drug_WT$ANCHOR_VIAB) - drug_WT$y
plot_WT_xmid         <- mean(drug_WT$xmid)
plot_WT_scal         <- mean(drug_WT$scal)
plot_WT_maxc         <- mean(drug_WT$maxc)
plot_WT_ANCHOR_VIAB  <- mean(drug_WT$ANCHOR_VIAB)

# ---- TYMS ----
drug_TYMS <- plot_data[plot_data$drug == "TYMS", ]
drug_TYMS$ythat_1 <- mean(drug_TYMS$ANCHOR_VIAB) - drug_TYMS$yhat
drug_TYMS$y_1     <- mean(drug_TYMS$ANCHOR_VIAB) - drug_TYMS$y
plot_TYMS_xmid         <- mean(drug_TYMS$xmid)
plot_TYMS_scal         <- mean(drug_TYMS$scal)
plot_TYMS_maxc         <- mean(drug_TYMS$maxc)
plot_TYMS_ANCHOR_VIAB  <- mean(drug_TYMS$ANCHOR_VIAB)

# ---- pooled plotting matrix ----
plotmat_bliss <- rbind(drug_WT[, -11], drug_TYMS[, -11])
mean(plotmat_bliss$yhat[plotmat_bliss$x==max(plotmat_bliss$x,na.rm=T)&plotmat_bliss$drug=="WT"])
mean(plotmat_bliss$yhat[plotmat_bliss$x==max(plotmat_bliss$x,na.rm=T)&plotmat_bliss$drug=="TYMS"])
delta_Emax=mean(plotmat_bliss$yhat[plotmat_bliss$x==max(plotmat_bliss$x,na.rm=T)&plotmat_bliss$drug=="WT"])-mean(plotmat_bliss$yhat[plotmat_bliss$x==max(plotmat_bliss$x,na.rm=T)&plotmat_bliss$drug=="TYMS"])

plot_xmid       <- mean(plotmat_bliss$xmid)
plot_scal       <- mean(plotmat_bliss$scal)
plot_maxc       <- mean(plotmat_bliss$maxc)
plot_ANCHOR_VIAB <- mean(plotmat_bliss$ANCHOR_VIAB)

plot_low_x <- 1 - plot_scal * log((1 - 1e-3) / 1e-3) + plot_xmid
plot_low_x <- log(getConcFromX(plot_low_x, plot_maxc))
plot_low_x <- min(c(plotmat_bliss$lx, plot_low_x))

plot_high_x <- 1 - plot_scal * log(1e-3 / (1 - 1e-3)) + plot_xmid
plot_high_x <- log(getConcFromX(plot_high_x, plot_maxc))
plot_high_x <- max(c(plotmat_bliss$lx, plot_high_x))

# ==================== 6. Plot 1 — Simple version (one point per dose) ====================
p_high_simple <- ggplot(plotmat_bliss, aes(x = lx, y = ythat_1, group = drug, col = drug)) +
  geom_point() +
  scale_color_manual(values = c("orange", "purple", "green", "lightblue", "red")) +
  scale_x_continuous(limits = c(-15, 15), n.breaks = 10) +
  scale_y_continuous(limits = c(0, 1)) +
  stat_function(aes_(x = ~lx), fun = l3_model2, colour = "orange",
                args = list(maxc = plot_TYMS_maxc,
                            xmid = plot_TYMS_xmid,
                            scal = plot_TYMS_scal,
                            ANCHOR_VIAB = plot_TYMS_ANCHOR_VIAB)) +
  stat_function(aes_(x = ~lx), fun = l3_model2, colour = "purple",
                args = list(maxc = plot_WT_maxc,
                            xmid = plot_WT_xmid,
                            scal = plot_WT_scal,
                            ANCHOR_VIAB = plot_WT_ANCHOR_VIAB)) +
  theme_classic() +
  labs(y = "Response: normalized intensity",
       x = expression(Dose / log[e] ~ mu * M),
       title = paste0("Patient: ", pid_cur)) +
  theme(text = element_text(size = 15)) +
  geom_point(aes_(x = plotmat_bliss$lxmid, y = plotmat_bliss$ANCHOR_VIAB / 2), shape = 2)

# Add IC50 annotation
temp_ic50 <- data.frame(nlme_stats[c("drug", "ic50_true", "IC50")])
temp_ic50 <- distinct(temp_ic50)
temp_ic50$pid <- pid_cur
print(temp_ic50)
delta_ic50<-temp_ic50$IC50[temp_ic50$drug=="WT"]-temp_ic50$IC50[temp_ic50$drug=="TYMS"]

p_high_simple <- p_high_simple +
  annotate("text", x = -7, y = 0.5,
           label = paste0("WT IC50 = ", signif(temp_ic50[temp_ic50$drug == "WT", "IC50"], 3))) +
  annotate("text", x = -7, y = 0.4,
           label = paste0("TYMS IC50 = ", signif(temp_ic50[temp_ic50$drug == "TYMS", "IC50"], 3))) +
  annotate("text", x = -7, y = 0.3,
           label = paste0("delta IC50 = ", signif(delta_ic50, 3))) +
  annotate("text", x = -7, y = 0.2,
           label = paste0("delta Emax = ", signif(delta_Emax, 3)))

p_high_simple
ggsave("./260622ptv1_fig4D_EDF3D_one_point.pdf", p_high_simple, width = 8, height = 6)
cat("\nSaved: 260622ptv1_fig4D_EDF3D_one_point.pdf\n")

# ==================== 7. Plot 2 — Full version (with jittered raw points) ====================
p_high_full <- ggplot(plotmat_bliss, aes(x = lx, y = ythat_1, group = drug, col = drug)) +
  # Fitted points (solid circles)
  geom_point() +
  # Raw biological replicate points (open circles, semi-transparent, jittered)
  geom_point(aes(y = y_1), shape = 1, alpha = 0.6, size = 2.5,
             position = position_jitter(width = 0.15, height = 0)) +
  # Color mapping: WT = purple, TYMS = orange
  scale_color_manual(values = c("WT" = "purple", "TYMS" = "orange")) +
  # Axis limits
  scale_x_continuous(limits = c(-15, 15), n.breaks = 10) +
  scale_y_continuous(limits = c(0, 1)) +
  # Fitted curves
  stat_function(aes_(x = ~lx), fun = l3_model2, colour = "orange",
                args = list(maxc = plot_TYMS_maxc,
                            xmid = plot_TYMS_xmid,
                            scal = plot_TYMS_scal,
                            ANCHOR_VIAB = plot_TYMS_ANCHOR_VIAB)) +
  stat_function(aes_(x = ~lx), fun = l3_model2, colour = "purple",
                args = list(maxc = plot_WT_maxc,
                            xmid = plot_WT_xmid,
                            scal = plot_WT_scal,
                            ANCHOR_VIAB = plot_WT_ANCHOR_VIAB)) +
  # Mark xmid positions
  geom_point(aes_(x = ~lxmid, y = ~ANCHOR_VIAB / 2), shape = 2) +
  # Theme & labels
  theme_classic() +
  labs(y = "Response: normalized intensity",
       x = expression(Dose / log[e] ~ mu * M),
       title = paste0("Patient: ", pid_cur)) +
  theme(text = element_text(size = 15))

# Add IC50 annotation
p_high_full <- p_high_full +
  annotate("text", x = -7, y = 0.5,
           label = paste0("WT IC50 = ", signif(temp_ic50[temp_ic50$drug == "WT", "IC50"], 3))) +
  annotate("text", x = -7, y = 0.4,
           label = paste0("TYMS IC50 = ", signif(temp_ic50[temp_ic50$drug == "TYMS", "IC50"], 3))) +
  annotate("text", x = -7, y = 0.3,
           label = paste0("delta IC50 = ", signif(delta_ic50, 3))) +
  annotate("text", x = -7, y = 0.2,
           label = paste0("delta Emax = ", signif(delta_Emax, 3)))

print(p_high_full)

ggsave("./260622ptv1_fig4D_EDF3D.pdf", p_high_full, width = 8, height = 6)
cat("\nSaved: 260622ptv1_fig4D_EDF3D.pdf\n")
cat("Done!\n")
