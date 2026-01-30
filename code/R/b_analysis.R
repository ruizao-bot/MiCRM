#!/usr/bin/env Rscript
# b_analysis.R - Generate plots for b, CUE, L_eff, C_feed, and Competition
library(ggplot2)
library(dplyr)
library(Cairo)
library(patchwork)

# Read data and set output directory
df <- read.csv("/Users/jiayi/Desktop/micrm/master_project/results/b_analysis.csv")
results_dir <- "/Users/jiayi/Desktop/micrm/master_project/results"

cat("Data loaded. Rows:", nrow(df), ", Columns:", ncol(df), "\n\n")

# Palette and theme
pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")

base_theme <- theme_minimal(base_size = 12) +
     theme(
          text       = element_text(family = "Times New Roman", size = 12),
          axis.text  = element_text(family = "Times New Roman", size = 12),
          axis.title = element_text(family = "Times New Roman", size = 12),
          legend.text = element_text(family = "Times New Roman", size = 12),
          legend.title = element_text(family = "Times New Roman", size = 12),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border     = element_rect(color = "black", fill = NA, size = 0.5),
          axis.ticks       = element_line(color = "black", size = 0.3)
     )

# ============================================================================
# SECTION 1: Prepare data frames for plotting
# ============================================================================

# Prepare long format data for b vs Y plots
b_cue_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  b = rep(df$b, 3),
  CUE = c(df$CUE1, df$CUE2, df$CUE3)
)

b_leff_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  b = rep(df$b, 3),
  L_eff = c(df$L_eff1_sum, df$L_eff2_sum, df$L_eff3_sum)
)

b_cfeed_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  b = rep(df$b, 3),
  C_feed = c(df$C_feed1, df$C_feed2, df$C_feed3)
)

b_comp_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  b = rep(df$b, 3),
  Competition = c(df$Competition1, df$Competition2, df$Competition3)
)

# Prepare long format data for X vs CUE plots
leff_cue_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  L_eff = c(df$L_eff1_sum, df$L_eff2_sum, df$L_eff3_sum),
  CUE = c(df$CUE1, df$CUE2, df$CUE3)
)

cfeed_cue_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  C_feed = c(df$C_feed1, df$C_feed2, df$C_feed3),
  CUE = c(df$CUE1, df$CUE2, df$CUE3)
)

comp_cue_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  Competition = c(df$Competition1, df$Competition2, df$Competition3),
  CUE = c(df$CUE1, df$CUE2, df$CUE3)
)

# ============================================================================
# SECTION 2: b vs Y plots (4 plots)
# ============================================================================

cat("Creating b vs Y plots...\n")

# 1) b vs CUE
p_b_cue <- ggplot(b_cue_df, aes(x = b, y = CUE, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 2.0) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "b", y = "CUE", color = "Community", shape = "Community") +
  base_theme

print(p_b_cue)

ggsave(filename = file.path(results_dir, "b_vs_CUE.pdf"), plot = p_b_cue,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 2) b vs L_eff
p_b_leff <- ggplot(b_leff_df, aes(x = b, y = L_eff, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 2.0) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "b", y = "L_eff", color = "Community", shape = "Community") +
  base_theme

print(p_b_leff)

ggsave(filename = file.path(results_dir, "b_vs_Leff.pdf"), plot = p_b_leff,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 3) b vs C_feed
p_b_cfeed <- ggplot(b_cfeed_df, aes(x = b, y = C_feed, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 2.0) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "b", y = "C_feed", color = "Community", shape = "Community") +
  base_theme

print(p_b_cfeed)

ggsave(filename = file.path(results_dir, "b_vs_Cfeed.pdf"), plot = p_b_cfeed,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 4) b vs Competition
p_b_comp <- ggplot(b_comp_df, aes(x = b, y = Competition, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 2.0) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "b", y = "Competition", color = "Community", shape = "Community") +
  base_theme

print(p_b_comp)

ggsave(filename = file.path(results_dir, "b_vs_Competition.pdf"), plot = p_b_comp,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# ============================================================================
# SECTION 3: X vs CUE plots (3 plots)
# ============================================================================

cat("Creating X vs CUE plots...\n")

# 5) L_eff vs CUE
p_leff_cue <- ggplot(leff_cue_df, aes(x = L_eff, y = CUE, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 2.0) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "L_eff", y = "CUE", color = "Community", shape = "Community") +
  base_theme

print(p_leff_cue)

ggsave(filename = file.path(results_dir, "Leff_vs_CUE.pdf"), plot = p_leff_cue,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 6) C_feed vs CUE
p_cfeed_cue <- ggplot(cfeed_cue_df, aes(x = C_feed, y = CUE, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 2.0) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "C_feed", y = "CUE", color = "Community", shape = "Community") +
  base_theme

print(p_cfeed_cue)

ggsave(filename = file.path(results_dir, "Cfeed_vs_CUE.pdf"), plot = p_cfeed_cue,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 7) Competition vs CUE
p_comp_cue <- ggplot(comp_cue_df, aes(x = Competition, y = CUE, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 2.0) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "Competition", y = "CUE", color = "Community", shape = "Community") +
  base_theme

print(p_comp_cue)

ggsave(filename = file.path(results_dir, "Competition_vs_CUE.pdf"), plot = p_comp_cue,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# ============================================================================
# SECTION 4: Combined plots
# ============================================================================

cat("Creating combined plots...\n")

# Combined: All 4 b vs Y plots in 2x2 grid
p_b_combined_4panel <- (p_b_cue + p_b_leff) / (p_b_cfeed + p_b_comp) +
  plot_layout(guides = "collect") &
  theme(legend.position = "right")

print(p_b_combined_4panel)

ggsave(filename = file.path(results_dir, "b_vs_all_4panel.pdf"), 
       plot = p_b_combined_4panel,
       device = cairo_pdf, width = 28, height = 24, units = "cm", dpi = 600, bg = "white")

# Combined: All 3 X vs CUE plots in 1x3 grid
p_cue_combined_3panel <- (p_leff_cue + p_cfeed_cue + p_comp_cue) +
  plot_layout(ncol = 3, guides = "collect") &
  theme(legend.position = "bottom")

print(p_cue_combined_3panel)

ggsave(filename = file.path(results_dir, "CUE_vs_all_3panel.pdf"), 
       plot = p_cue_combined_3panel,
       device = cairo_pdf, width = 32, height = 12, units = "cm", dpi = 600, bg = "white")


