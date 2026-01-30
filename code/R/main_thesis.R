#!/usr/bin/env Rscript
# main_thesis.R - Net Competition vs CUE Plot
setwd("/Users/jiayi/Desktop/micrm/master_project")
library(ggplot2)
library(readr)
library(dplyr)
library(scales)
library(Cairo)

# Palette and theme
pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")
base_theme <- theme_minimal(base_size = 14) +
  theme(
    text       = element_text(family = "Times New Roman"),
    axis.text  = element_text(family = "Times New Roman", size = 14),
    axis.title = element_text(family = "Times New Roman", size = 14),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks       = element_line(color = "black", size = 0.3),
    axis.ticks.length = unit(0.15, "cm")
  )

results_dir <- "results"

cat("============================================================\n")
cat("FIGURE 5: Net Competition vs CUE\n")
cat("============================================================\n\n")

# Load and prepare data
cat("Loading data...\n")
df <- read.csv("data/coal.csv")

# Check if L_eff column exists
if ("L_eff" %in% colnames(df)) {
  
  # Calculate net competitive effect: Competition - Facilitation
  df <- df %>%
    mutate(Net_Competition = Competition - L_eff)
  
  # Get Net_Competition ranges for different communities
  df_12_net <- df %>% filter(Community %in% c(1, 2))
  df_3_net <- df %>% filter(Community == 3)
  
  net_range_12 <- range(df_12_net$Net_Competition, na.rm = TRUE)
  net_range_3 <- range(df_3_net$Net_Competition, na.rm = TRUE)
  
  # Create transformation functions to map Community 3's Net_Competition to Community 1&2's range
  transform_3_to_12_net <- function(x) {
    (x - net_range_3[1]) / (net_range_3[2] - net_range_3[1]) * 
      (net_range_12[2] - net_range_12[1]) + net_range_12[1]
  }
  
  transform_12_to_3_net <- function(x) {
    (x - net_range_12[1]) / (net_range_12[2] - net_range_12[1]) * 
      (net_range_3[2] - net_range_3[1]) + net_range_3[1]
  }
  
  # Transform Community 3 data to fit Community 1&2 scale
  df_3_transformed_net <- df_3_net %>%
    mutate(Net_Competition_transformed = transform_3_to_12_net(Net_Competition))
  
  # Combine all data with transformed values for plotting
  df_plot_net <- bind_rows(
    df_12_net %>% mutate(Net_Competition_plot = Net_Competition),
    df_3_transformed_net %>% mutate(Net_Competition_plot = Net_Competition_transformed)
  )
  
  # Create plot with dual x-axes
  p_net_comp_cue <- ggplot(df_plot_net, 
                           aes(x = Net_Competition_plot, y = Community_CUE,
                               color = factor(Community), shape = factor(Community))) +
    geom_point(size = 2, alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE, aes(group = factor(Community))) +
    scale_color_manual(values = pal_rgb, name = "Community") +
    scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") +
    scale_x_continuous(
      name = "Net Competition (Competition - Facilitation)",
      sec.axis = sec_axis(
        trans = ~ transform_12_to_3_net(.),
        name = "Net Competition for Community 3",
        breaks = scales::pretty_breaks(n = 5)
      )
    ) +
    labs(y = "Community CUE") +
    base_theme +
    theme(
      axis.title.x.bottom = element_text(color = "black"),
      axis.title.x.top = element_text(color = pal_rgb["3"], size = 14)
    )
  
  print(p_net_comp_cue)
  
  # Statistical analysis
  cat("\n=== Statistical Analysis: Net Competition vs Community CUE ===\n")
  for (comm in c("1", "2", "3")) {
    df_comm <- df %>% filter(Community == comm)
    model <- lm(Community_CUE ~ Net_Competition, data = df_comm)
    
    cat("\n--- Community", comm, "---\n")
    cat("Sample size:", nrow(df_comm), "\n")
    cat("Net Competition range:", round(min(df_comm$Net_Competition, na.rm = TRUE), 6),
        "to", round(max(df_comm$Net_Competition, na.rm = TRUE), 6), "\n")
    cat("R-squared:", round(summary(model)$r.squared, 4), "\n")
    cat("Adjusted R-squared:", round(summary(model)$adj.r.squared, 4), "\n")
    cat("Slope:", round(coef(model)[2], 6), "\n")
    cat("P-value:", format.pval(summary(model)$coefficients[2, 4], digits = 4), "\n")
    
    if (summary(model)$coefficients[2, 4] < 0.05) {
      cat("*** Significant relationship (p < 0.05) ***\n")
    } else {
      cat("Not significant (p >= 0.05)\n")
    }
  }
  
  ggsave(file.path(results_dir, "main_net_competition_cue.pdf"),
         plot = p_net_comp_cue,
         device = cairo_pdf,
         width = 21,
         height = 12,
         units = "cm",
         dpi = 600,
         bg = "white")
  
  cat("\nFigure 5 saved to results/main_net_competition_cue.pdf\n")
  
} else {
  cat("Error: L_eff column not found in coal.csv\n")
  cat("Cannot calculate Net Competition = Competition - Facilitation\n")
}

cat("\n============================================================\n")
cat("DONE\n")
cat("============================================================\n")
