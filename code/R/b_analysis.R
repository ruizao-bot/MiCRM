#!/usr/bin/env Rscript
# b_analysis.R
# Simplified analysis: plot Facilitation vs CUE and Competition vs CUE,
# fit linear regressions for each community and print summaries to console.
# Read data and set output directory (note: space in "Desktop/ micrm")
df <- read.csv("/Users/jiayi/Desktop/micrm/master_project/data/b_analysis_data_0.2.csv")
results_dir <- "/Users/jiayi/Desktop/micrm/master_project/results"

cat("Data loaded. Rows:", nrow(df), ", Columns:", ncol(df), "\n\n")

# Define color palette
pal_rgb <- c("1" = "#E74C3C",   # 红
             "2" = "#2ECC71",   # 绿
             "3" = "#3498DB")   # 蓝

# --- Plot 1: Facilitation vs CUE ---
png(filename=file.path(results_dir, 'CUE_vs_Facilitation.png'), width=1000, height=600)
par(mfrow=c(1,1), family="serif", cex.main=14/12, cex.lab=14/12, cex.axis=14/12)
plot(df$C_feed1, df$CUE1, col=pal_rgb["1"], pch=16, xlab='Facilitation', ylab='CUE',
     main='CUE vs Facilitation', xlim=range(c(df$C_feed1,df$C_feed2,df$C_feed3)),
     ylim=range(c(df$CUE1,df$CUE2,df$CUE3)))
points(df$C_feed2, df$CUE2, col=pal_rgb["2"], pch=15)
points(df$C_feed3, df$CUE3, col=pal_rgb["3"], pch=17)
legend('topleft', legend=c('Community1','Community2','Community3'),
       col=c(pal_rgb["1"], pal_rgb["2"], pal_rgb["3"]), pch=c(16,15,17), cex=14/12)

# Fit linear models (but don't plot regression lines)
lm_f1 <- lm(CUE1 ~ C_feed1, data=df)
lm_f2 <- lm(CUE2 ~ C_feed2, data=df)
lm_f3 <- lm(CUE3 ~ C_feed3, data=df)
dev.off()
cat("Saved to:", file.path(results_dir, 'CUE_vs_Facilitation.png'), "\n\n")

# --- Plot 2: Competition vs CUE ---
png(filename=file.path(results_dir, 'CUE_vs_Competition.png'), width=1000, height=600)
par(family="serif", cex.main=14/12, cex.lab=14/12, cex.axis=14/12)
plot(df$Competition1, df$CUE1, col=pal_rgb["1"], pch=16, xlab='Competition', ylab='CUE',
     main='CUE vs Competition', xlim=range(c(df$Competition1,df$Competition2,df$Competition3)),
     ylim=range(c(df$CUE1,df$CUE2,df$CUE3)))
points(df$Competition2, df$CUE2, col=pal_rgb["2"], pch=15)
points(df$Competition3, df$CUE3, col=pal_rgb["3"], pch=17)
legend('topleft', legend=c('Community1','Community2','Community3'),
       col=c(pal_rgb["1"], pal_rgb["2"], pal_rgb["3"]), pch=c(16,15,17), cex=14/12)

lm_c1 <- lm(CUE1 ~ Competition1, data=df)
lm_c2 <- lm(CUE2 ~ Competition2, data=df)
lm_c3 <- lm(CUE3 ~ Competition3, data=df)
dev.off()
cat("Saved to:", file.path(results_dir, 'CUE_vs_Competition.png'), "\n\n")

# --- Print regression summaries to console ---
cat("Community 1:\n")
print(summary(lm_f1))


cat("Community 2:\n")
print(summary(lm_f2))


cat("Community 3:\n")
print(summary(lm_f3))


cat("Community 1:\n")
print(summary(lm_c1))

cat("Community 2:\n")
print(summary(lm_c2))

cat("Community 3:\n")
print(summary(lm_c3))


