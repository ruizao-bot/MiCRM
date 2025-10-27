#!/usr/bin/env Rscript
# b_analysis.R (simplified, ggplot2-based output like thesis.R)
library(ggplot2)
library(dplyr)
library(Cairo)
# Read data and set output directory
df <- read.csv("/Users/jiayi/Desktop/micrm/master_project/data/b_analysis_data_0.2.csv")
results_dir <- "/Users/jiayi/Desktop/micrm/master_project/results"

cat("Data loaded. Rows:", nrow(df), ", Columns:", ncol(df), "\n\n")

# Palette and theme (match thesis.R styling)
pal_rgb <- c("1" = "#E74C3C",   # 红
                               "2" = "#2ECC71",   # 绿
                               "3" = "#3498DB")   # 蓝

base_theme <- theme_minimal(base_size = 14) +
     theme(
          text       = element_text(family = "Times New Roman"),
          axis.text  = element_text(family = "Times New Roman", size = 14),
          axis.title = element_text(family = "Times New Roman", size = 14),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border     = element_rect(color = "black", fill = NA, size = 0.5),
          axis.ticks       = element_line(color = "black", size = 0.3)
     )

# Helper: reshape wide columns (1/2/3) to long for plotting
make_long <- function(df, prefix_x, prefix_y, names = c("1","2","3")){
     n <- nrow(df)
     data.frame(
          Community = factor(rep(names, each = n), levels = names),
          x = c(df[[paste0(prefix_x, names[1])]], df[[paste0(prefix_x, names[2])]], df[[paste0(prefix_x, names[3])]]),
          y = c(df[[paste0(prefix_y, names[1])]], df[[paste0(prefix_y, names[2])]], df[[paste0(prefix_y, names[3])]])
     )
}

# 1) Facilitation vs CUE (ggplot) ------------------------------------------
feed_df <- make_long(df, prefix_x = "C_feed", prefix_y = "CUE")

p_feed <- ggplot(feed_df, aes(x = x, y = y, color = Community, shape = Community)) +
     geom_point(alpha = 0.8, size = 1.8) +
     scale_color_manual(values = pal_rgb) +
     scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
     labs(x = "Facilitation", y = "CUE",color = "Community", shape = "Community") +
     base_theme

ggsave(filename = file.path(results_dir, "CUE_vs_Facilitation.pdf"), plot = p_feed,
                device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 2) Competition vs CUE (ggplot) --------------------------------------------
comp_df <- make_long(df, prefix_x = "Competition", prefix_y = "CUE")

p_comp <- ggplot(comp_df, aes(x = x, y = y, color = Community, shape = Community)) +
     geom_point(alpha = 0.8, size = 1.8) +
     scale_color_manual(values = pal_rgb) +
     scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
     labs(x = "Competition", y = "CUE", color = "Community", shape = "Community") +
     base_theme

ggsave(filename = file.path(results_dir, "CUE_vs_Competition.pdf"), plot = p_comp,
                device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")
# Keep regression summaries (computed but not plotted)
lm_f1 <- lm(CUE1 ~ C_feed1, data=df)
lm_f2 <- lm(CUE2 ~ C_feed2, data=df)
lm_f3 <- lm(CUE3 ~ C_feed3, data=df)

lm_c1 <- lm(CUE1 ~ Competition1, data=df)
lm_c2 <- lm(CUE2 ~ Competition2, data=df)
lm_c3 <- lm(CUE3 ~ Competition3, data=df)

cat("--- Facilitation regressions ---\n")
cat("Community 1:\n"); print(summary(lm_f1))
cat("Community 2:\n"); print(summary(lm_f2))
cat("Community 3:\n"); print(summary(lm_f3))

cat("--- Competition regressions ---\n")
cat("Community 1:\n"); print(summary(lm_c1))
cat("Community 2:\n"); print(summary(lm_c2))
cat("Community 3:\n"); print(summary(lm_c3))


