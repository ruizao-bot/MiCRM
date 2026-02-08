# ============================================================================
# THESIS ANALYSIS - MAIN FIGURES
# ============================================================================
# This script generates key figures for thesis:
# 1. final_plot: Species CUE vs Abundance
# 2. p_comp_cue: Competition vs Community CUE
# 3. p_fac_cue: Facilitation vs Community CUE
# 4. p_cue_sim_domin: CUE vs Bray-Curtis Similarity (with dominance)
# ============================================================================

setwd("/Users/jiayi/Desktop/micrm/master_project")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(stringr)
library(patchwork)
library(vegan)
# library(fitdistrplus)
# library(minpack.lm)
# library(purrr)
# library(jsonlite)

# Palette and theme setup
pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")
base_theme <- theme_minimal(base_size = 12) +
  theme(
    text       = element_text(family = "Times New Roman"),
    axis.text  = element_text(family = "Times New Roman", size = 12),
    axis.title = element_text(family = "Times New Roman", size = 12),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, size = 0.3),
    axis.ticks       = element_line(color = "black", size = 0.3),
    axis.ticks.length = unit(0.15, "cm")
  )

# Load data
df <- read.csv("data/coal.csv")
df_select <- df %>% mutate(Status = ifelse(Abundance < 1e-5, "Extinction", "Survival"))
df_surv <- df_select %>% filter(Abundance > 1e-5)

# ============================================================================
# COMMENTED OUT: Abundance Histogram
# ============================================================================
# # 1. Abundance Histogram
# p_hist <- ggplot(df, aes(x = Abundance, fill = factor(Community), color = factor(Community))) +
#   geom_histogram(position = "identity", bins = 50, alpha = 0.3) +
#   geom_vline(xintercept = 1e-5, linetype = "dashed", linewidth = 0.7, colour = "black") +
#   scale_x_log10() +
#   facet_wrap(~ Community, ncol = 1, scales = "free_y") +
#   base_theme+
#   labs(x = "Abundance", y = "Frequency", fill = "Community", color = "Community") +
#   scale_fill_manual(values = pal_rgb) +
#   scale_color_manual(values = pal_rgb)
# print(p_hist)
# 
# ggsave("results/presentation/abund_hist.png", plot = p_hist,
#        width = 21, height = 18, units = "cm", dpi = 600, bg = "white")
# 
# ggsave("results/abundance_histogram.pdf",
#        plot = p_hist,
#        device = cairo_pdf,
#        width = 21,
#        height = 18,
#        units = "cm",
#        dpi = 600,
#        bg = "white")

# ============================================================================
# FIGURE 1: Species CUE vs Abundance (final_plot)
# ============================================================================
# Create log-transformed abundance for linear modeling
df_surv <- df_surv %>% mutate(log10_Abundance = log10(Abundance))

# Fit linear models and output diagnostics
fit_diagnostics <- data.frame()
cat("\n=== Linear Fit: log10(Abundance) ~ Species_CUE ===\n")
for (comm in c("1", "2", "3")) {
  dat <- df_surv %>% filter(Community == comm)
  model_fit <- lm(log10_Abundance ~ Species_CUE, data = dat)
  fit_diagnostics <- rbind(fit_diagnostics, data.frame(
    Community = comm,
    R2 = summary(model_fit)$r.squared,
    Slope = coef(model_fit)[2],
    Intercept = coef(model_fit)[1],
    P_value = summary(model_fit)$coefficients[2, 4],
    N = nrow(dat)
  ))
}
print(fit_diagnostics)

# Set uniform y-axis range across communities
y_min <- min(df_surv$log10_Abundance, na.rm = TRUE)
y_max <- max(df_surv$log10_Abundance, na.rm = TRUE)

# Generate plots for each community
plots <- list()

for (comm in c("1", "2", "3")) {
  df_i <- df_surv %>% filter(Community == comm)

  # Main plot: CUE vs log10(Abundance)
  p_main <- ggplot(df_i, aes(x = Species_CUE, y = log10_Abundance)) +
    geom_point(color = pal_rgb[comm], alpha = 0.3) +
    # geom_smooth(method = "lm", se = TRUE, color = pal_rgb[comm], linewidth = 1) +
    ylim(y_min, y_max) +
    labs(x = "Species-level CUE", y = expression(log[10](Abundance))) +
    base_theme

  # Side histogram
  p_hist <- ggplot(df_i, aes(x = log10_Abundance)) +
    geom_histogram(bins = 50, fill = pal_rgb[comm], alpha = 0.3, color = pal_rgb[comm]) +
    xlim(y_min, y_max) +
    coord_flip() +
    scale_y_reverse() +
    labs(x = "Frequency", y = NULL) +
    base_theme +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin = margin(5, 5, 5, 5)
    )

  p_combo <- p_hist + p_main + plot_layout(widths = c(1.2, 3))
  plots[[comm]] <- p_combo
}

# Combine plots vertically for 3 communities
final_plot <- wrap_plots(plots, ncol = 1) & base_theme
final_plot

# Export high-resolution images
ggsave("results/cue_abund.png", plot = final_plot, width = 16, height = 20, units = "cm", dpi = 600, bg = "white")
ggsave("results/cue_abund.pdf",
       plot = final_plot,
       device = cairo_pdf,
       width = 16,
       height = 20,
       units = "cm",
       dpi = 600,
       bg = "white")

# ============================================================================
# FIGURE 2: Competition vs Community CUE - Separate by Community
# ============================================================================
# Prepare aggregated data with community-level metrics
df_comm_agg <- df %>%
  group_by(Seed, Community, Competition, Community_CUE, Facilitation) %>%
  summarise(
    Species_CUE_Var = var(Species_CUE, na.rm = TRUE),
    .groups = "drop"
  )

cat("\n=== Competition vs CUE Analysis ===\n")

# Solid color points with fit lines
comp_plots <- list()
for (comm in c(1, 2, 3)) {
  df_comm <- df_comm_agg %>% filter(Community == comm)
  model <- lm(Community_CUE ~ Competition, data = df_comm)
  
  r2 <- round(summary(model)$r.squared, 3)
  pval <- summary(model)$coefficients[2, 4]
  cat("\n--- Community", comm, "---\n")
  cat("R-squared:", r2, "\n")
  cat("P-value:", format.pval(pval, digits = 4), "\n")
  
  p <- ggplot(df_comm, aes(x = Competition, y = Community_CUE)) +
    geom_point(size = 1.8, alpha = 0.6, color = pal_rgb[as.character(comm)]) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed", linewidth = 0.8) +
    scale_x_continuous(
      breaks = function(x) {
        rng <- range(x)
        mid <- (rng[1] + rng[2]) / 2
        c(rng[1], mid, rng[2])
      },
      labels = function(x) sprintf("%.2f", x * 1e3)
    ) +
    labs(
      x = expression(Competition~("*"~10^{-3})),
      y = "Community CUE"
    ) +
    base_theme
  
  comp_plots[[comm]] <- p
}

# Combine into single row
ggsave("results/Competition_vs_communityCUE.png")
p_comp_cue_combined <- wrap_plots(comp_plots, ncol = 3) & theme(plot.margin = margin(5, 5, 5, 5))
print(p_comp_cue_combined)
ggsave("results/Competition_vs_communityCUE.png",
  plot = p_comp_cue_combined,
  width = 18,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")
ggsave("results/Competition_vs_communityCUE.pdf",
  plot = p_comp_cue_combined,
  device = cairo_pdf,
  width = 18,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

# ============================================================================
# FIGURE 3: Facilitation vs Community CUE - Separate by Community
# ============================================================================
cat("\n=== Facilitation vs CUE Analysis ===\n")


# ---- Community-level aggregation ----
df_comm_fac <- df_surv %>%
  group_by(Seed, Community) %>%
  summarise(
    Community_CUE = unique(Community_CUE),
    Facilitation = mean(Facilitation),
    .groups = "drop"
  )

# Plot: Facilitation vs Community CUE (community-level)
fac_plots <- list()
## Statistical summary table
fac_stats <- data.frame(Community=character(), Coefficient=numeric(), R2=numeric(), P_value=numeric(), N=integer())
for (comm in c(1, 2, 3)) {
  df_comm <- df_comm_fac %>% filter(Community == comm)
  model <- lm(Community_CUE ~ Facilitation, data = df_comm)
  coef_val <- coef(model)[2]
  r2 <- summary(model)$r.squared
  pval <- summary(model)$coefficients[2, 4]
  n <- nrow(df_comm)
  fac_stats <- rbind(fac_stats, data.frame(Community=comm, Coefficient=coef_val, R2=r2, P_value=pval, N=n))
  cat("\n--- Community", comm, "(Facilitation vs Community CUE) ---\n")
  cat("Coefficient:", round(coef_val, 4), "\n")
  cat("R-squared:", round(r2, 4), "\n")
  cat("P-value:", format.pval(pval, digits = 4), "\n")
  p <- ggplot(df_comm, aes(x = Facilitation, y = Community_CUE)) +
    geom_point(size = 1.8, alpha = 0.6, color = pal_rgb[as.character(comm)]) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed", linewidth = 0.8) +
    scale_x_continuous(
      breaks = function(x) {
        rng <- range(x)
        c(rng[1], (rng[1] + rng[2]) / 2, rng[2])
      },
      labels = function(x) sprintf("%.2f", x * 1e3)
    ) +
    labs(
      x = expression(Facilitation~("*"~10^{-3})),
      y = "Community CUE"
    ) +
    base_theme
  fac_plots[[comm]] <- p
}
cat("\nFacilitation vs Community CUE statistics summary:\n")
print(fac_stats)
## Horizontal layout: nrow = 1 (one row), ncol = NA lets patchwork auto-arrange
p_fac_cue_combined <- wrap_plots(fac_plots, nrow = 1) & theme(plot.margin = margin(5, 5, 5, 5))
print(p_fac_cue_combined)
ggsave("results/Facilitation_vs_communityCUE.png",
  plot = p_fac_cue_combined,
  width = 24,
  height = 7,
  units = "cm",
  dpi = 600,
  bg = "white")
ggsave("results/Facilitation_vs_communityCUE.pdf",
  plot = p_fac_cue_combined,
  device = cairo_pdf,
  width = 24,
  height = 7,
  units = "cm",
  dpi = 600,
  bg = "white")

# ============================================================================
# FIGURE 3B: Species-level Competition Analysis
# ============================================================================
cat("\n=== Species-level Competition Analysis ===\n")

# Plot 1: Species Competition vs Species CUE
species_comp_cue_plots <- list()
for (comm in c(1, 2, 3)) {
  df_comm <- df_surv %>% filter(Community == comm)
  model <- lm(Species_CUE ~ Species_Competition, data = df_comm)
  
  r2 <- round(summary(model)$r.squared, 3)
  pval <- summary(model)$coefficients[2, 4]
  cat("\n--- Community", comm, "CUE vs Species Competition ---\n")
  cat("R-squared:", r2, "\n")
  cat("P-value:", format.pval(pval, digits = 4), "\n")
  
  p <- ggplot(df_comm, aes(x = Species_Competition, y = Species_CUE)) +
    geom_point(size = 1.5, alpha = 0.4, color = pal_rgb[as.character(comm)]) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed", linewidth = 0.8) +
    scale_x_continuous(
      breaks = function(x) {
        rng <- range(x)
        c(rng[1], (rng[1] + rng[2]) / 2, rng[2])
      },
      labels = function(x) sprintf("%.2f", x * 1e3)
    ) +
    labs(
      x = expression(Species~Competition~("*"~10^{-3})),
      y = "Species-level CUE"
    ) +
    base_theme
  
  species_comp_cue_plots[[comm]] <- p
}

p_species_comp_cue <- wrap_plots(species_comp_cue_plots, ncol = 3) & theme(plot.margin = margin(5, 5, 5, 5))
print(p_species_comp_cue)

ggsave("results/Species_Competition_vs_CUE.png",
  plot = p_species_comp_cue,
  width = 18,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

ggsave("results/Species_Competition_vs_CUE.pdf",
  plot = p_species_comp_cue,
  device = cairo_pdf,
  width = 18,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

# Plot 2: Species Competition vs Abundance
species_comp_abund_plots <- list()
for (comm in c(1, 2, 3)) {
  df_comm <- df_surv %>% filter(Community == comm) %>%
    mutate(log10_Abundance = log10(Abundance))
  
  model <- lm(log10_Abundance ~ Species_Competition, data = df_comm)
  
  r2 <- round(summary(model)$r.squared, 3)
  pval <- summary(model)$coefficients[2, 4]
  cat("\n--- Community", comm, "Abundance vs Species Competition ---\n")
  cat("R-squared:", r2, "\n")
  cat("P-value:", format.pval(pval, digits = 4), "\n")
  
  p <- ggplot(df_comm, aes(x = Species_Competition, y = log10_Abundance)) +
    geom_point(size = 1.5, alpha = 0.4, color = pal_rgb[as.character(comm)]) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed", linewidth = 0.8) +
    scale_x_continuous(
      breaks = function(x) {
        rng <- range(x)
        c(rng[1], (rng[1] + rng[2]) / 2, rng[2])
      },
      labels = function(x) sprintf("%.2f", x * 1e3)
    ) +
    labs(
      x = expression(Species~Competition~("*"~10^{-3})),
      y = expression(log[10](Abundance))
    ) +
    base_theme
  
  species_comp_abund_plots[[comm]] <- p
}

p_species_comp_abund <- wrap_plots(species_comp_abund_plots, ncol = 3) & theme(plot.margin = margin(5, 5, 5, 5))
print(p_species_comp_abund)

ggsave("results/Species_Competition_vs_Abundance.png",
  plot = p_species_comp_abund,
  width = 18,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

ggsave("results/Species_Competition_vs_Abundance.pdf",
  plot = p_species_comp_abund,
  device = cairo_pdf,
  width = 18,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

# ============================================================================
# FIGURE 3C: CUE Distribution by Competition Level - Polarization Analysis
# ============================================================================
cat("\n=== CUE Distribution by Competition Level ===\n")

# Create competition bins for each community
df_surv_binned <- df %>%
  group_by(Community) %>%
  mutate(
    Comp_Quantile = ntile(Species_Competition, 3),
    Comp_Level = case_when(
      Comp_Quantile == 1 ~ "Low",
      Comp_Quantile == 2 ~ "Medium",
      Comp_Quantile == 3 ~ "High"
    ),
    Comp_Level = factor(Comp_Level, levels = c("Low", "Medium", "High"))
  ) %>%
  ungroup()

# Calculate variance and other statistics for each bin
comp_stats <- df_surv_binned %>%
  group_by(Community, Comp_Level) %>%
  summarise(
    Mean_CUE = mean(Species_CUE),
    Var_CUE = var(Species_CUE),
    SD_CUE = sd(Species_CUE),
    CV_CUE = sd(Species_CUE) / mean(Species_CUE),
    N = n(),
    .groups = "drop"
  )

cat("\n--- CUE Statistics by Competition Level ---\n")
print(comp_stats)

# Plot 1: Density plots showing CUE distribution by competition level
cue_dist_plots <- list()
for (comm in c(1, 2, 3)) {
  df_comm <- df_surv_binned %>% filter(Community == comm)
  
  p <- ggplot(df_comm, aes(x = Species_CUE, fill = Comp_Level, color = Comp_Level)) +
    geom_density(alpha = 0.3, linewidth = 0.8) +
    scale_fill_manual(
      values = c("Low" = "#3498DB", "Medium" = "#F39C12", "High" = "#E74C3C"),
      name = "Competition"
    ) +
    scale_color_manual(
      values = c("Low" = "#3498DB", "Medium" = "#F39C12", "High" = "#E74C3C"),
      name = "Competition"
    ) +
    labs(
      x = "Species-level CUE",
      y = "Density",
      title = paste("Community", comm)
    ) +
    base_theme +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, size = 12)
    )
  
  cue_dist_plots[[comm]] <- p
}

p_cue_dist <- wrap_plots(cue_dist_plots, ncol = 3, guides = "collect") & 
  theme(plot.margin = margin(5, 5, 5, 5))
print(p_cue_dist)

ggsave("results/CUE_Distribution_by_Competition.png",
  plot = p_cue_dist,
  width = 24,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

ggsave("results/CUE_Distribution_by_Competition.pdf",
  plot = p_cue_dist,
  device = cairo_pdf,
  width = 24,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

# Plot 2: Violin plots showing CUE distribution by competition level
cue_violin_plots <- list()
for (comm in c(1, 2, 3)) {
  df_comm <- df_surv_binned %>% filter(Community == comm)
  
  p <- ggplot(df_comm, aes(x = Comp_Level, y = Species_CUE, fill = Comp_Level)) +
    geom_violin(alpha = 0.6, trim = FALSE) +
    geom_boxplot(width = 0.2, alpha = 0.8, outlier.alpha = 0.3, outlier.size = 0.8) +
    scale_fill_manual(
      values = c("Low" = "#3498DB", "Medium" = "#F39C12", "High" = "#E74C3C"),
      name = "Competition"
    ) +
    labs(
      x = "Competition Level",
      y = "Species-level CUE",
      title = paste("Community", comm)
    ) +
    base_theme +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, size = 12)
    )
  
  cue_violin_plots[[comm]] <- p
}

p_cue_violin <- wrap_plots(cue_violin_plots, ncol = 3) & 
  theme(plot.margin = margin(5, 5, 5, 5))
print(p_cue_violin)

ggsave("results/CUE_Violin_by_Competition.png",
  plot = p_cue_violin,
  width = 20,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

ggsave("results/CUE_Violin_by_Competition.pdf",
  plot = p_cue_violin,
  device = cairo_pdf,
  width = 20,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

# Plot 3: Variance trend - shows increasing variance with competition
p_var_trend <- ggplot(comp_stats, aes(x = Comp_Level, y = Var_CUE, 
                                       color = factor(Community), 
                                       group = Community)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(
    values = pal_rgb,
    name = "Community"
  ) +
  labs(
    x = "Competition Level",
    y = "CUE Variance",
    title = "CUE Variance vs Competition Level"
  ) +
  base_theme +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12),
    legend.position = "right"
  )

print(p_var_trend)

ggsave("results/CUE_Variance_Trend.png",
  plot = p_var_trend,
  width = 14,
  height = 10,
  units = "cm",
  dpi = 600,
  bg = "white")

ggsave("results/CUE_Variance_Trend.pdf",
  plot = p_var_trend,
  device = cairo_pdf,
  width = 14,
  height = 10,
  units = "cm",
  dpi = 600,
  bg = "white")

# ============================================================================
# FIGURE 3D: Community-level Competition Effect on Species CUE Distribution
# ============================================================================
cat("\n=== Community Competition Effect on Species CUE ===\n")

# Create community competition bins
df_surv_comm_comp <- df %>%
  group_by(Community) %>%
  mutate(
    Comm_Comp_Quantile = ntile(Competition, 3),
    Comm_Comp_Level = case_when(
      Comm_Comp_Quantile == 1 ~ "Low",
      Comm_Comp_Quantile == 2 ~ "Medium",
      Comm_Comp_Quantile == 3 ~ "High"
    ),
    Comm_Comp_Level = factor(Comm_Comp_Level, levels = c("Low", "Medium", "High"))
  ) %>%
  ungroup()

# Calculate statistics by community competition level
comm_comp_stats <- df_surv_comm_comp %>%
  group_by(Community, Comm_Comp_Level) %>%
  summarise(
    Mean_CUE = mean(Species_CUE),
    Var_CUE = var(Species_CUE),
    SD_CUE = sd(Species_CUE),
    CV_CUE = sd(Species_CUE) / mean(Species_CUE),
    Mean_Competition = mean(Competition),
    N = n(),
    .groups = "drop"
  )

cat("\n--- CUE Statistics by Community Competition Level ---\n")
print(comm_comp_stats)

# Plot 1: Density plots by community competition level
comm_comp_density_plots <- list()
for (comm in c(1, 2, 3)) {
  df_comm <- df_surv_comm_comp %>% filter(Community == comm)
  
  p <- ggplot(df_comm, aes(x = Species_CUE, fill = Comm_Comp_Level, color = Comm_Comp_Level)) +
    geom_density(alpha = 0.3, linewidth = 0.8) +
    scale_fill_manual(
      values = c("Low" = "#3498DB", "Medium" = "#F39C12", "High" = "#E74C3C"),
      name = "Community\nCompetition"
    ) +
    scale_color_manual(
      values = c("Low" = "#3498DB", "Medium" = "#F39C12", "High" = "#E74C3C"),
      name = "Community\nCompetition"
    ) +
    labs(
      x = "Species-level CUE",
      y = "Density",
      title = paste("Community", comm)
    ) +
    base_theme +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, size = 12)
    )
  
  comm_comp_density_plots[[comm]] <- p
}

p_comm_comp_density <- wrap_plots(comm_comp_density_plots, ncol = 3, guides = "collect") & 
  theme(plot.margin = margin(5, 5, 5, 5))
print(p_comm_comp_density)

ggsave("results/CUE_Distribution_by_Community_Competition.png",
  plot = p_comm_comp_density,
  width = 24,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

ggsave("results/CUE_Distribution_by_Community_Competition.pdf",
  plot = p_comm_comp_density,
  device = cairo_pdf,
  width = 24,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

# Plot 2: Variance trend by community competition - with all scatter points
# Prepare individual point data
df_comm_comp_points <- df_surv_comm_comp %>%
  group_by(Seed, Community, Competition) %>%
  summarise(
    SD_CUE = sd(Species_CUE),
    .groups = "drop"
  )

# Add values for group means
comm_comp_stats <- comm_comp_stats %>%
  mutate(
    SD_CUE = sqrt(Var_CUE)
  )

p_comm_comp_var <- ggplot() +
  geom_point(data = df_comm_comp_points, 
             aes(x = Competition, y = SD_CUE, color = factor(Community)),
             size = 1.5, alpha = 0.4, shape = 16) +
  facet_wrap(~ Community, ncol = 3) +
  scale_color_manual(
    values = pal_rgb,
    name = "Community"
  ) +
  scale_x_continuous(
    breaks = function(x) {
      rng <- range(x)
      c(rng[1], (rng[1] + rng[2]) / 2, rng[2])
    },
    labels = function(x) sprintf("%.2f", x * 1e3)
  ) +
  labs(
    x = expression(Community~Competition~("*"~10^{-3})),
    y = "Species CUE Standard Deviation"
  ) +
  base_theme +
  theme(
    legend.position = "none",
    strip.text = element_text(size = 12, family = "Times New Roman")
  )

print(p_comm_comp_var)

ggsave("results/CUE_Variance_by_Community_Competition.png",
  plot = p_comm_comp_var,
  width = 14,
  height = 10,
  units = "cm",
  dpi = 600,
  bg = "white")

ggsave("results/CUE_Variance_by_Community_Competition.pdf",
  plot = p_comm_comp_var,
  device = cairo_pdf,
  width = 14,
  height = 10,
  units = "cm",
  dpi = 600,
  bg = "white")

# Plot 3: Scatter plot with continuous competition values
comm_comp_scatter_plots <- list()
for (comm in c(1, 2, 3)) {
  df_comm <- df_surv_comm_comp %>% filter(Community == comm)
  
  p <- ggplot(df_comm, aes(x = Competition, y = Species_CUE)) +
    geom_point(size = 1.5, alpha = 0.4, color = pal_rgb[as.character(comm)], shape = 16) +
    geom_smooth(method = "lm", se = TRUE, color = "black", linetype = "dashed", linewidth = 0.8, alpha = 0.2) +
    scale_x_continuous(
      breaks = function(x) {
        rng <- range(x)
        c(rng[1], (rng[1] + rng[2]) / 2, rng[2])
      },
      labels = function(x) sprintf("%.2f", x * 1e3)
    ) +
    labs(
      x = expression(Community~Competition~("*"~10^{-3})),
      y = "Species-level CUE",
      title = paste("Community", comm)
    ) +
    base_theme +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12)
    )
  
  comm_comp_scatter_plots[[comm]] <- p
}

p_comm_comp_box <- wrap_plots(comm_comp_scatter_plots, ncol = 3) & 
  theme(plot.margin = margin(5, 5, 5, 5))
print(p_comm_comp_box)

ggsave("results/CUE_Scatter_by_Community_Competition.png",
  plot = p_comm_comp_box,
  width = 20,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

ggsave("results/CUE_Scatter_by_Community_Competition.pdf",
  plot = p_comm_comp_box,
  device = cairo_pdf,
  width = 20,
  height = 8,
  units = "cm",
  dpi = 600,
  bg = "white")

# ============================================================================
# FIGURE 4: CUE vs Bray-Curtis Similarity with Dominance (p_cue_sim_domin)
# ============================================================================
# Prepare data: Assign global species IDs to avoid overlap between communities
df_mut <- df_surv %>%
  mutate(Global_Species_ID = case_when(
    Community == 2 ~ Species_ID + 100,
    TRUE ~ Species_ID
  ))

# Calculate Bray-Curtis dissimilarity between communities
bray_results <- data.frame()

for (s in unique(df_mut$Seed)) {
  df_seed <- df_mut %>% filter(Seed == s) %>% as.data.frame()
  if (!all(c(1, 2, 3) %in% unique(df_seed$Community))) next
  
  # Create community matrix for Bray-Curtis calculation
  comm_mat <- df_seed %>%
    dplyr::select(Community, Global_Species_ID, Abundance) %>%
    pivot_wider(
      names_from = Global_Species_ID,
      values_from = Abundance,
      values_fill = list(Abundance = 0)
    )
  rownames(comm_mat) <- comm_mat$Community
  comm_mat$Community <- NULL
  if (nrow(comm_mat) != 3) next
  
  # Calculate Bray-Curtis dissimilarity
  bc <- vegdist(comm_mat, method = "bray")
  bc_mat <- as.matrix(bc)
  if (!all(c("3", "1", "2") %in% rownames(bc_mat))) next
  
  d31 <- bc_mat["3", "1"]
  d32 <- bc_mat["3", "2"]
  cue1 <- unique(df_seed$Community_CUE[df_seed$Community == 1])
  cue2 <- unique(df_seed$Community_CUE[df_seed$Community == 2])
  
  bray_results <- rbind(bray_results, data.frame(
    Seed = s,
    Bray_3vs1 = d31,
    Bray_3vs2 = d32,
    CUE_1 = cue1,
    CUE_2 = cue2,
    Sim_3vs1 = 1 - d31,
    Sim_3vs2 = 1 - d32
  ))
}

# Statistical analysis: Fit linear models
# mod_1 <- lm(Sim_3vs1 ~ CUE_1, data = bray_results)
# mod_2 <- lm(Sim_3vs2 ~ CUE_2, data = bray_results)
# summary(mod_1)
# summary(mod_2)

# Prepare data for plotting with dominance information
df_comm <- df_select %>%
  filter(Community %in% c(1, 2)) %>%
  group_by(Seed, Community) %>%
  summarise(
    Community_CUE = unique(Community_CUE),
    Dominant_Community = unique(Dominant_Community),
    .groups = "drop"
  ) %>%
  mutate(
    Dominance = ifelse(paste0("Community ", Community) == Dominant_Community, 1, 0)
  )

# Join with similarity data
df_comm_sim <- df_comm %>%
  left_join(
    bray_results %>% dplyr::select(Seed, Sim_3vs1, Sim_3vs2),
    by = "Seed"
  ) %>%
  mutate(
    Similarity = ifelse(Community == 1, Sim_3vs1, Sim_3vs2),
    Community = factor(Community)
  )

# Define color scheme for dominance groups
dominance_colors <- c(
  "0" = "grey60",
  "1_1" = as.vector(pal_rgb["1"]),
  "1_2" = as.vector(pal_rgb["2"])
)

# Create dominance group variable
df_comm_sim <- df_comm_sim %>%
  mutate(
    DominanceGroup = case_when(
      Dominance == 0 ~ "0",
      Dominance == 1 & Community == 1 ~ "1_1",
      Dominance == 1 & Community == 2 ~ "1_2"
    )
  )

# Generate plot
p_cue_sim_domin <- ggplot(df_comm_sim, aes(x = Community_CUE, y = Similarity, color = factor(DominanceGroup), shape = factor(Community))) +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_color_manual(
    values = dominance_colors,
    labels = c("0" = "Not Dominant", "1_1" = "Dominant", "1_2" = "Dominant"),
    name = "Dominance"
  ) +
  scale_shape_manual(
    values = c("1" = 16, "2" = 17),
    name = "Community",
    labels = c("1" = "Community 1", "2" = "Community 2")
  ) +
  labs(
    x = "Community-level CUE",
    y = "Bray–Curtis Similarity to Community 3"
  ) +
  base_theme+
  theme(
    legend.title = element_text(size = 12, family = "Times New Roman"),
    legend.text  = element_text(size = 11, family = "Times New Roman")
  )

print(p_cue_sim_domin)

ggsave("results/dom_sim.pdf",
       plot = p_cue_sim_domin,
       device = cairo_pdf,
       width = 16,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("results/dom_sim.png",
       plot = p_cue_sim_domin,
       width = 16,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

# ============================================================================
# COMMENTED OUT: Additional analyses and plots
# ============================================================================

# # Bray-Curtis similarity plot (without dominance coloring)
# p_sim <- ggplot() +
#   geom_point(data = bray_results, aes(x = CUE_1, y = Sim_3vs1, color = "1"), alpha = 0.4, size = 2) +
#   geom_smooth(data = bray_results, aes(x = CUE_1, y = Sim_3vs1, color = "1"), method = "lm", se = TRUE, alpha = 0.7) +
#   geom_point(data = bray_results, aes(x = CUE_2, y = Sim_3vs2, color = "2"), alpha = 0.4, size = 2) +
#   geom_smooth(data = bray_results, aes(x = CUE_2, y = Sim_3vs2, color = "2"), method = "lm", se = TRUE, alpha = 0.7) +
#   scale_color_manual(values = pal_rgb, name = "Community") +
#   labs(x = "Community CUE", y = "Bray–Curtis similarity to Community 3", color = "Community") +
#   base_theme +
#   theme(legend.position = "right")

# # Dominance probability plot
# model_all <- glm(Dominance ~ Community_CUE, data = df_comm, family = binomial)
# cue_seq <- seq(min(df_comm$Community_CUE), max(df_comm$Community_CUE), length.out = 200)
# df_pred_all <- data.frame(
#   Community_CUE = cue_seq,
#   Dominance = predict(model_all, newdata = data.frame(Community_CUE = cue_seq), type = "response")
# )
# 
# p_domin <- ggplot(df_comm, aes(x = Community_CUE, y = Dominance, color = factor(Community))) +
#   geom_jitter(width = 0.0005, height = 0.05, alpha = 0.6, size = 2) +
#   geom_line(data = df_pred_all, aes(x = Community_CUE, y = Dominance), color = "grey", linewidth = 0.9, inherit.aes = FALSE) +
#   scale_color_manual(values = pal_rgb) +
#   labs(x = "Community-level CUE", y = "Probability of Dominance (1 = Dominant)", color = "Community") +
#   base_theme

# # Richness analysis
# df_stats <- df_surv %>%
#   group_by(Seed, Community, Community_CUE) %>%
#   summarise(
#     Richness = n_distinct(Species_ID),
#     CUE.Var = var(Species_CUE, na.rm = TRUE),
#     .groups = "drop"
#   )
# 
# p_rich <- ggplot(df_stats, aes(x = CUE.Var, y = Richness, color = factor(Community), shape = factor(Community))) +
#   geom_point(size = 2, alpha = 0.8) +
#   geom_smooth(method = "lm", se = TRUE, linetype = "solid", size = 1) +
#   scale_color_manual(values = pal_rgb, name = "Community") +
#   facet_wrap(~ Community, scales = "free_x") +
#   labs(x = expression("Species CUE Variance"), y = "Species Richness") +
#   scale_x_log10(
#     breaks = function(lims) 10^seq(log10(lims[1]), log10(lims[2]), length.out = 3),
#     labels = scales::label_scientific(digits = 1),
#     expand = expansion(mult = c(0.05, 0.05)),
#     minor_breaks = NULL
#   ) +
#   base_theme +
#   scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") +
#   theme(
#     axis.text.x = element_text(angle = 45, hjust = 1),
#     panel.spacing = unit(1.2, "lines")
#   )

# # Uptake variance vs Species CUE
# p_uv <- ggplot(df_surv, aes(x = Species_CUE, y = UptakeVar, color = factor(Community))) +
#   geom_point(alpha = 0.5, size = 1.2) +
#   geom_smooth(method = "lm", se = TRUE, linetype = "solid", size = 1) +
#   scale_color_manual(values = pal_rgb, name = "Community") +
#   scale_x_log10(
#     breaks = function(lims) 10^seq(log10(lims[1]), log10(lims[2]), length.out = 3),
#     labels = scales::label_scientific(digits = 1),
#     expand = expansion(mult = c(0.05, 0.05)),
#     minor_breaks = NULL
#   ) +
#   labs(x = "Uptake Variance", y = "Species-level CUE", color = "Community") +
#   facet_wrap(~Community) +
#   base_theme +
#   scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") +
#   theme(
#     axis.text.x = element_text(angle = 45, hjust = 1),
#     panel.spacing = unit(1.2, "lines")
#   )

# # Depletion analysis

df_depletion <- df %>%
  group_by(Seed, Community) %>%
  summarise(
    Community_CUE = unique(Community_CUE),
    Niche_Overlap = unique(Competition),
    Depletion = unique(Depletion),
    .groups = "drop"
  )

p_depletion_cue <- ggplot(df_depletion, aes(x = Community_CUE, y = Depletion, color = factor(Community), shape = factor(Community))) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, linetype = "solid", size = 1) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") +
  labs(x = "Community CUE", y = "Resource Depletion (Residual Sum)", color = "Community") +
  base_theme
ggsave("results/depletion_vs_CUE.pdf",
       plot = p_depletion_cue,
       device = cairo_pdf,
       width = 16,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")
# ============================================================================
# FIGURE 5: Rare Species Invasion - CUE vs Survival Probability (p_s)
# ============================================================================
# Load rare species invasion data
df_rare <- read.csv("data/rare.csv")

# Create survival labels
df_rare <- df_rare %>%
  mutate(survival     = ifelse(C_final > 1e-5, "Survived", "Extinct"),
         survived_bin = ifelse(survival == "Survived", 1, 0))

df_rare_surv <- df_rare %>% filter(survival == "Survived")

# Check available dilution rates
cat("\n=== Rare Species Analysis ===\n")
cat("Available dilution rates:", unique(df_rare$DilutionRate), "\n")

# Plot: CUE vs survival probability
p_s <- ggplot(df_rare, aes(x = CUE, y = survived_bin, colour = factor(DilutionRate))) +
  geom_jitter(height = 0.05, width = 0, alpha = 0.4, size = 1.2, shape = 16) +
  stat_smooth(method = "glm",
              method.args = list(family = "binomial"),
              se = FALSE, size = 1.2, linetype = "dashed") +
  labs(
    x = "Species-level CUE",
    y = "Probability of Survival",
    colour = "Dilution Rate"
  ) +
  scale_color_manual(
    values = c("0.01" = "#E74C3C", "0.05" = "#2ECC71", "0.1" = "#3498DB"),
    labels = c("0.01", "0.05", "0.1")
  ) +
  theme_minimal(base_size = 12) +
  theme(
    text = element_text(family = "Times New Roman"),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 11),
    panel.grid.major = element_line(color = "grey80", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks = element_line(color = "black", size = 0.5),
    axis.ticks.length = unit(0.15, "cm")
  )

print(p_s)

# Statistical analysis: logistic regression
cat("\n=== Logistic Regression: Survival ~ CUE * DilutionRate ===\n")
library(broom)
glm(survived_bin ~ CUE * DilutionRate, data = df_rare, family = "binomial") %>%
  tidy() %>%
  print()

ggsave("results/rare_survival.pdf",
       plot = p_s,
       device = cairo_pdf,
       width = 16,
       height = 8,
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("results/survival_by_dilution.png",
       plot = p_s,
       width = 16,
       height = 8,
       units = "cm",
       dpi = 600,
       bg = "white")
