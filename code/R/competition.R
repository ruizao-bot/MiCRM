setwd("/Users/jiayi/Desktop/micrm/master_project")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(patchwork)

# Palette and theme setup
pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")
base_theme <- theme_minimal(base_size = 12) +
  theme(
    text = element_text(family = "Times New Roman"),
    axis.text = element_text(family = "Times New Roman", size = 12),
    axis.title = element_text(family = "Times New Roman", size = 12),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 0.3),
    axis.ticks = element_line(color = "black", size = 0.3),
    axis.ticks.length = unit(0.15, "cm")
  )

# Load data (all species)
df <- read.csv("data/coal.csv")

# ============================================================================
# 1. Species-level Competition vs CUE
# ============================================================================
cat("\n=== Species-level Competition Analysis ===\n")

species_comp_cue_plots <- list()
for (comm in c(1, 2, 3)) {
  df_comm <- df %>% filter(Community == comm)
  model <- lm(Species_CUE ~ Species_Competition, data = df_comm)
  
  r2 <- round(summary(model)$r.squared, 3)
  pval <- summary(model)$coefficients[2, 4]
  cat("\n--- Community", comm, "CUE vs Species Competition ---\n")
  cat("R-squared:", r2, "\n")
  cat("P-value:", format.pval(pval, digits = 4), "\n")
  
  p <- ggplot(df_comm, aes(x = Species_Competition, y = Species_CUE)) +
    geom_point(size = 1.5, alpha = 0.4, color = pal_rgb[as.character(comm)], shape = 16) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed", linewidth = 0.8) +
    scale_x_continuous(labels = function(x) sprintf("%.2f", x)) +
    labs(
      x = "Species Competition",
      y = "Species-level CUE"
    ) +
    base_theme
  
  species_comp_cue_plots[[comm]] <- p
}

p_species_comp_cue <- wrap_plots(species_comp_cue_plots, ncol = 3) & 
  theme(plot.margin = margin(5, 5, 5, 5))
print(p_species_comp_cue)

ggsave("results/Species_Competition_vs_CUE.png",
  plot = p_species_comp_cue, width = 18, height = 8, units = "cm", dpi = 600, bg = "white")

