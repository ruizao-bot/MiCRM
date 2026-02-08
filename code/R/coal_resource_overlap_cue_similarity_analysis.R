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
# Visualization: CUE vs Dominance scatter plot
# ============================================================

# Add jitter to better display overlapping points


# Make the plot longer (wider), move the legend inside, add y=0.5 dashed line, reduce jitter, and expand y-axis
p_scatter <- ggplot(df_combined, aes(x = CUE, y = Dominance, color = Overlap)) +
  geom_jitter(alpha = 0.5, width = 0, height = 0.025, size = 2) + # reduce jitter height
  geom_smooth(method = "glm", method.args = list(family = "binomial"), 
              se = TRUE, alpha = 0.2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", size = 0.7) + # add dividing line
  scale_y_continuous(
    breaks = c(0, 0.5, 1),
    labels = c("0", "0.5", "1"),
    expand = expansion(mult = c(0.08, 0.08))
  ) +
  base_theme +
  labs(title = '',
       x = 'Community CUE',
       y = 'Dominance',
       color = 'Resource Overlap') +
  theme(
    legend.position = c(0.98, 0.08), # inside plot, bottom right
    legend.justification = c(1, 0),
    legend.direction = "vertical",
    legend.background = element_blank(), # no border
    legend.key = element_rect(fill = NA)
  )

print(p_scatter)

# Save plot as PDF
ggsave("results/cue_dominance_overlap.pdf",
       plot = p_scatter,
       device = cairo_pdf,
       width = 21,   # make the plot longer (wider)
       height = 12,
       units = "cm",
       dpi = 600,
       bg = "white")

