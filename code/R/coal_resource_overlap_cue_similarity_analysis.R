# ============================================================================
# Analysis of CUE vs Dominance relationship under different resource overlap
# Required R packages: tidyverse
# ============================================================================

library(tidyverse)

# Theme setup
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
df <- read_csv('data/coal_resource.csv')
df$Overlap <- as.factor(df$Overlap)

# ============================================================
# CUE vs Dominance Analysis
# ============================================================
cat("\n========== CUE vs Dominance Analysis ==========\n\n")

# Combine data from community 1 and 2
df_combined <- df %>%
  select(Seed, Overlap, CUE1, CUE2, Dominant_Community, Total_Abundance_1, Total_Abundance_2) %>%
  pivot_longer(
    cols = c(CUE1, CUE2),
    names_to = "Community",
    names_prefix = "CUE",
    values_to = "CUE"
  ) %>%
  mutate(
    Community_ID = paste0("Community ", Community),
    Dominance = ifelse(Community_ID == Dominant_Community, 1, 0),
    Dominance_factor = factor(Dominance, levels = c(0, 1), labels = c("Not Dominant", "Dominant"))
  ) %>%
  select(Seed, Overlap, Community, CUE, Dominant_Community, Dominance, Dominance_factor)

# ============================================================
# Relationship difference across overlap levels
# ============================================================

# Interaction model: does the CUE effect differ by Overlap?
df_combined <- df_combined %>%
  mutate(Overlap = factor(Overlap))

glm_full <- glm(Dominance ~ CUE * Overlap, data = df_combined, family = binomial)
glm_no_interaction <- glm(Dominance ~ CUE + Overlap, data = df_combined, family = binomial)

cat("\n--- Likelihood ratio test (interaction) ---\n")
print(anova(glm_no_interaction, glm_full, test = "Chisq"))

cat("\n--- Slope of CUE by Overlap (log-odds) ---\n")
cue_slopes <- df_combined %>%
  group_by(Overlap) %>%
  summarise(
    n = n(),
    slope = coef(glm(Dominance ~ CUE, data = cur_data(), family = binomial))["CUE"],
    p_value = summary(glm(Dominance ~ CUE, data = cur_data(), family = binomial))$coefficients["CUE", 4],
    .groups = "drop"
  )

print(cue_slopes)

# ============================================================
# Visualization: CUE Difference vs Similarity Difference — faceted by overlap
# ============================================================

# Compute CUE difference and similarity difference
df_diff <- df %>%
  mutate(
    CUE_diff = CUE1 - CUE2,
    Sim_diff = Sim_3vs1 - Sim_3vs2
  )

# Bin CUE difference into 7 equal-width intervals
n_bins <- 7
cue_range <- range(df_diff$CUE_diff)
breaks <- seq(cue_range[1], cue_range[2], length.out = n_bins + 1)
df_diff <- df_diff %>%
  mutate(CUE_bin = cut(CUE_diff, breaks = breaks, include.lowest = TRUE, dig.lab = 2))

# Overlap color palette
overlap_colors <- c("0.25" = "#E74C3C", "0.5" = "#F39C12", "0.75" = "#3498DB")

p_scatter <- ggplot(df_diff, aes(x = CUE_bin, y = Sim_diff, fill = Overlap)) +
  geom_boxplot(position = position_dodge(0.8), width = 0.7,
               alpha = 0.7, outlier.size = 1, outlier.alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  scale_fill_manual(values = overlap_colors, name = "Resource\nOverlap") +
  coord_cartesian(ylim = c(-1, 1)) +
  labs(
    x = expression(Delta*"CUE (Parent 1 "~-~" Parent 2)"),
    y = expression(Delta*"Similarity (Parent 1 "~-~" Parent 2)")
  ) +
  base_theme +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
    legend.title = element_text(size = 12, family = "Times New Roman"),
    legend.text  = element_text(size = 11, family = "Times New Roman")
  )

print(p_scatter)

# Save plot as PDF
ggsave("results/cue_dominance_overlap.pdf",
       plot = p_scatter,
       device = cairo_pdf,
       width = 21,
       height = 12,
       units = "cm",
       dpi = 600,
       bg = "white")

