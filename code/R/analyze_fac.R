#!/usr/bin/env Rscript
# analyze_fac.R - Competition, Facilitation, and Net Competition vs CUE Analysis
setwd("/Users/jiayi/Desktop/micrm/master_project")
library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
library(scales)
library(Cairo)
library(patchwork)

# Palette and theme
pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")
base_theme <- theme_minimal(base_size = 12) +
  theme(
    text       = element_text(family = "Times New Roman"),
    axis.text  = element_text(family = "Times New Roman", size = 12),
    axis.title = element_text(family = "Times New Roman", size = 12),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks       = element_line(color = "black", size = 0.3),
    axis.ticks.length = unit(0.15, "cm")
  )

results_dir <- "data"


# Load data
cat("Loading data...\n")
df_wide <- read.csv("results/fac.csv")

cat("Data dimensions:", nrow(df_wide), "rows,", ncol(df_wide), "columns\n\n")

# Reshape data from wide to long format
cat("Reshaping data...\n")
df_long <- bind_rows(
    df_wide %>% 
      mutate(Community = 1,
        Richness = Richness1,
        L_eff = L_eff1,
        Community_CUE = CUE1,
        Competition = Competition1) %>%
      select(match_pct, N_modules, Community, Richness, L_eff, Community_CUE, Competition),
  df_wide %>% 
      mutate(Community = 2,
        Richness = Richness2,
        L_eff = L_eff2,
        Community_CUE = CUE2,
        Competition = Competition2) %>%
      select(match_pct, N_modules, Community, Richness, L_eff, Community_CUE, Competition),
  df_wide %>% 
      mutate(Community = 3,
        Richness = Richness3,
        L_eff = L_eff3,
        Community_CUE = CUE3,
        Competition = Competition3) %>%
      select(match_pct, N_modules, Community, Richness, L_eff, Community_CUE, Competition)
)

# Use raw Competition and Facilitation values (no scaling)

cat("\nUsing raw Competition and L_eff values (no scaling)...\n")

df_long <- df_long %>%
  mutate(
    Competition_scaled = Competition,
    C_feed_scaled = L_eff
  )

# Calculate Net Competition = Competition - L_eff (raw values)
df_long <- df_long %>%
  mutate(Net_Competition_scaled = Competition_scaled - C_feed_scaled)

# Get data splits
df_12 <- df_long %>% filter(Community %in% c(1, 2))
df_3 <- df_long %>% filter(Community == 3)

# ============================================================
# COMBINED PLOT: Facilitation & Competition vs CUE
# ============================================================

cat("Creating combined Facilitation & Competition vs CUE plot...\n")

# Prepare Facilitation data
l_eff_range_12 <- range(df_12$C_feed_scaled, na.rm = TRUE)
l_eff_range_3 <- range(df_3$C_feed_scaled, na.rm = TRUE)

transform_3_to_12_l_eff <- function(x) {
  (x - l_eff_range_3[1]) / (l_eff_range_3[2] - l_eff_range_3[1]) * 
    (l_eff_range_12[2] - l_eff_range_12[1]) + l_eff_range_12[1]
}

transform_12_to_3_l_eff <- function(x) {
  (x - l_eff_range_12[1]) / (l_eff_range_12[2] - l_eff_range_12[1]) * 
    (l_eff_range_3[2] - l_eff_range_3[1]) + l_eff_range_3[1]
}

df_3_transformed_l_eff <- df_3 %>%
  mutate(L_eff_transformed = transform_3_to_12_l_eff(C_feed_scaled))

df_plot_cfeed <- bind_rows(
  df_12 %>% mutate(Value = C_feed_scaled),
  df_3_transformed_l_eff %>% mutate(Value = L_eff_transformed)
)

# Create Facilitation plot with secondary axis
p_facilitation <- ggplot(df_plot_cfeed, 
                         aes(x = Value, y = Community_CUE,
                             color = factor(Community), shape = factor(Community))) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, aes(group = factor(Community))) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") +
  scale_x_continuous(
    name = expression(Facilitation ~ ("*" ~ 10^-4)),
    sec.axis = sec_axis(
      trans = ~ transform_12_to_3_l_eff(.),
      name = expression(Facilitation ~ "for" ~ Community ~ 3 ~ ("*" ~ 10^-4)),
      breaks = scales::pretty_breaks(n = 5),
      labels = scales::label_number(scale = 1e4)
    ),
    labels = scales::label_number(scale = 1e4)
  ) +
  labs(y = "Community CUE") +
  base_theme +
  theme(
    axis.title.x.bottom = element_text(color = "black"),
    axis.title.x.top = element_text(color = "black", size = 12, margin = margin(b = 10))
  )

# Prepare Competition data
comp_range_12 <- range(df_12$Competition_scaled, na.rm = TRUE)
comp_range_3 <- range(df_3$Competition_scaled, na.rm = TRUE)

transform_3_to_12_comp <- function(x) {
  (x - comp_range_3[1]) / (comp_range_3[2] - comp_range_3[1]) * 
    (comp_range_12[2] - comp_range_12[1]) + comp_range_12[1]
}

transform_12_to_3_comp <- function(x) {
  (x - comp_range_12[1]) / (comp_range_12[2] - comp_range_12[1]) * 
    (comp_range_3[2] - comp_range_3[1]) + comp_range_3[1]
}

df_3_transformed_comp <- df_3 %>%
  mutate(Competition_transformed = transform_3_to_12_comp(Competition_scaled))

df_plot_comp <- bind_rows(
  df_12 %>% mutate(Value = Competition_scaled),
  df_3_transformed_comp %>% mutate(Value = Competition_transformed)
)

# Create Competition plot with secondary axis
p_competition <- ggplot(df_plot_comp, 
                        aes(x = Value, y = Community_CUE,
                            color = factor(Community), shape = factor(Community))) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, aes(group = factor(Community))) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") +
  scale_x_continuous(
    name = expression(Competition ~ ("*" ~ 10^-3)),
    sec.axis = sec_axis(
      trans = ~ transform_12_to_3_comp(.),
      name = expression(Competition ~ "for" ~ Community ~ 3 ~ ("*" ~ 10^-3)),
      breaks = scales::pretty_breaks(n = 5),
      labels = scales::label_number(scale = 1e3)
    ),
    labels = scales::label_number(scale = 1e3)
  ) +
  labs(y = "Community CUE") +
  base_theme +
  theme(
    axis.title.x.bottom = element_text(color = "black"),
    axis.title.x.top = element_text(color = "black", size = 12, margin = margin(b = 10))
  )

# Combine plots using patchwork
p_combined <- p_facilitation + p_competition + 
  plot_layout(guides = "collect") &
  theme(legend.position = "right")

p_combined

ggsave(file.path(results_dir, "fac_combined_facilitation_competition_cue.pdf"),
       plot = p_combined,
       device = cairo_pdf,
       width = 18,
       height = 9,
       units = "cm",
       dpi = 600,
       bg = "white")

cat("Combined Facilitation & Competition plot saved\n\n")

# ============================================================
# Community 3 Facilitation-CUE Analysis
# ============================================================

cat("\n=== Community 3 Facilitation-CUE Analysis ===\n")

# Extract Community 3 data
df_comm3 <- df_long %>% filter(Community == 3)

# Correlation analysis
cor_l_eff_cue <- cor(df_comm3$L_eff, df_comm3$Community_CUE, method = "pearson")
cor_test_result <- cor.test(df_comm3$L_eff, df_comm3$Community_CUE, method = "pearson")

cat(sprintf("Community 3: L_eff vs CUE\n"))
cat(sprintf("  Pearson correlation: r = %.4f\n", cor_l_eff_cue))
cat(sprintf("  p-value: %.4e\n", cor_test_result$p.value))
cat(sprintf("  95%% CI: [%.4f, %.4f]\n", cor_test_result$conf.int[1], cor_test_result$conf.int[2]))

# Linear regression
lm_comm3 <- lm(Community_CUE ~ L_eff, data = df_comm3)
lm_summary <- summary(lm_comm3)

cat(sprintf("\nLinear regression: CUE ~ L_eff\n"))
cat(sprintf("  Intercept: %.4f (p = %.4e)\n", 
            coef(lm_summary)[1, 1], coef(lm_summary)[1, 4]))
cat(sprintf("  Slope: %.4f (p = %.4e)\n", 
            coef(lm_summary)[2, 1], coef(lm_summary)[2, 4]))
cat(sprintf("  R-squared: %.4f\n", lm_summary$r.squared))
cat(sprintf("  Adjusted R-squared: %.4f\n", lm_summary$adj.r.squared))
cat(sprintf("  F-statistic: %.4f (p = %.4e)\n", 
            lm_summary$fstatistic[1], 
            pf(lm_summary$fstatistic[1], lm_summary$fstatistic[2], 
               lm_summary$fstatistic[3], lower.tail = FALSE)))

# Save Community 3 analysis results
comm3_analysis <- data.frame(
  Metric = c("Pearson_r", "p_value", "CI_lower", "CI_upper", 
             "Intercept", "Slope", "R_squared", "Adj_R_squared"),
  Value = c(cor_l_eff_cue, cor_test_result$p.value, 
            cor_test_result$conf.int[1], cor_test_result$conf.int[2],
            coef(lm_summary)[1, 1], coef(lm_summary)[2, 1],
            lm_summary$r.squared, lm_summary$adj.r.squared)
)

write.csv(comm3_analysis, file.path(results_dir, "fac_community3_analysis.csv"), row.names = FALSE)
cat("\nCommunity 3 analysis saved to results/fac_community3_analysis.csv\n")

# ============================================================
# Summary table
# ============================================================

cat("\n=== Summary Statistics ===\n")
summary_table <- df_long %>%
  group_by(match_pct, Community) %>%
  summarise(
    CUE = mean(Community_CUE, na.rm = TRUE),
    Competition = mean(Competition, na.rm = TRUE),
    L_eff = mean(L_eff, na.rm = TRUE),
    Net_Competition = mean(Net_Competition_scaled, na.rm = TRUE),
    Richness = mean(Richness, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(match_pct, Community)

print(summary_table, n = Inf)

write.csv(summary_table, file.path(results_dir, "fac_summary_table.csv"), row.names = FALSE)
cat("\nSummary table saved to results/fac_summary_table.csv\n")

cat("\n============================================================\n")
cat("DONE\n")
cat("============================================================\n")

