setwd("/Users/jiayi/Desktop/micrm/master_project")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(patchwork)

# Palette and theme setup
pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")
community_labels <- c("1" = "Parent 1", "2" = "Parent 2", "3" = "Coalesced")
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
df_select <- df %>% mutate(Status = ifelse(Abundance < 1e-5, "Extinction", "Survival"))
df_surv <- df_select %>% filter(Abundance > 1e-5)
# ============================================================================
# 1. Species-level Competition vs CUE
# ============================================================================
cat("\n=== Species-level Competition Analysis ===\n")

p_species_comp_cue <- ggplot(df_surv, aes(x = Species_Competition2, y = Species_CUE, color = factor(Community))) +
  geom_point(size = 1.5, alpha = 0.4, shape = 16) +
  scale_color_manual(values = pal_rgb) +
  scale_x_continuous(labels = scales::label_scientific()) +
  facet_wrap(~ Community, ncol = 1, strip.position = "right", labeller = labeller(Community = community_labels)) +
  labs(
    x = "Species Competition",
    y = "Species-level CUE"
  ) +
  base_theme +
  theme(plot.margin = margin(5, 5, 5, 5), legend.position = "none")+
  theme(
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_line(color = "grey92", linewidth = 0.15),
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.5),
    strip.text = element_text(family = "Times New Roman", size = 12),
    legend.position = "none"
  )

print(p_species_comp_cue)
ggsave("results/species_competition.pdf",
       plot = p_species_comp_cue,
       device = cairo_pdf,
       width = 21,
       height = 18,
       units = "cm",
       dpi = 600,
       bg = "white")

