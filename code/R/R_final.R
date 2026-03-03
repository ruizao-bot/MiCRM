# ============================================================================
# ANALYSIS: Community CUE Difference vs Similarity Difference
# ============================================================================
# This script analyzes the effect of community CUE difference (CUE1 - CUE2)
# on similarity difference (Sim_3vs1 - Sim_3vs2) between parent communities
# where Sim_3vs1 = Bray-Curtis similarity between coalesced and community 1
# and Sim_3vs2 = Bray-Curtis similarity between coalesced and community 2

library(ggplot2)
library(dplyr)
library(scales)

# Set working directory
setwd("/Users/jiayi/Desktop/micrm/master_project")

# Load coalescence metrics data
metrics_data <- read.csv("data/coalescence_metrics.csv")

# Calculate CUE difference and Similarity difference (matching thesis.R)
df_diff <- metrics_data %>%
  mutate(
    CUE_Diff = CUE_1 - CUE_2,
    Sim_Diff = Sim_3vs1 - Sim_3vs2  # Difference in Bray-Curtis similarity
  )

# Print summary statistics
cat("\n=== Summary Statistics ===\n")
cat("CUE Difference (CUE1 - CUE2):\n")
cat("  Mean:", mean(df_diff$CUE_Diff), "\n")
cat("  SD:", sd(df_diff$CUE_Diff), "\n")
cat("  Range:", range(df_diff$CUE_Diff), "\n")

cat("\nSimilarity Difference (Sim_3vs1 - Sim_3vs2):\n")
cat("  Mean:", mean(df_diff$Sim_Diff), "\n")
cat("  SD:", sd(df_diff$Sim_Diff), "\n")
cat("  Range:", range(df_diff$Sim_Diff), "\n")

cat("\nBray-Curtis Similarity with Coalesced Community:\n")
cat("  Sim_3vs1 (with Community 1): Mean =", mean(df_diff$Sim_3vs1), ", SD =", sd(df_diff$Sim_3vs1), "\n")
cat("  Sim_3vs2 (with Community 2): Mean =", mean(df_diff$Sim_3vs2), ", SD =", sd(df_diff$Sim_3vs2), "\n")

# Fit linear model
model <- lm(Sim_Diff ~ CUE_Diff, data = df_diff)
cat("\n=== Linear Regression: Sim_Diff ~ CUE_Diff ===\n")
print(summary(model))

# Define color scheme for dominance (matching thesis.R)
dom_colors <- c(
  "Community 1" = "#E74C3C",
  "Community 2" = "#2ECC71"
)

# Base theme (matching thesis.R style)
base_theme <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Plot: CUE Difference vs Similarity Difference with dominance coloring
p_cue_sim_domin <- ggplot(df_diff, aes(x = CUE_Diff, y = Sim_Diff, color = Dominant_Community)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  scale_color_manual(values = dom_colors, name = "Dominant Community") +
  labs(
    title = "Community CUE Difference vs Bray-Curtis Similarity Difference",
    x = expression(paste(Delta, "CUE (CUE"[1], " - CUE"[2], ")")),
    y = expression(paste(Delta, "Similarity (Sim"[paste("3", symbol("\256"), "1")], " - Sim"[paste("3", symbol("\256"), "2")], ")"))
  ) +
  base_theme

print(p_cue_sim_domin)

# Create results directory if it doesn't exist
dir.create("results", showWarnings = FALSE)

# Save plot
ggsave("results/cue_similarity_dominance.png",
       plot = p_cue_sim_domin,
       width = 21, height = 14, units = "cm", dpi = 600, bg = "white")

ggsave("results/cue_similarity_dominance.pdf",
       plot = p_cue_sim_domin,
       device = cairo_pdf,
       width = 21, height = 14, units = "cm", dpi = 600, bg = "white")

# Plot 2: Simple scatter plot without dominance coloring
p_simple <- ggplot(df_diff, aes(x = CUE_Diff, y = Sim_Diff)) +
  geom_point(color = "#3498DB", size = 3, alpha = 0.6) +
  geom_smooth(method = "lm", color = "#E74C3C", se = TRUE, linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  labs(
    title = "Community CUE Difference vs Bray-Curtis Similarity Difference",
    x = expression(paste(Delta, "CUE (CUE"[1], " - CUE"[2], ")")),
    y = expression(paste(Delta, "Similarity (Sim"[paste("3", symbol("\256"), "1")], " - Sim"[paste("3", symbol("\256"), "2")], ")"))
  ) +
  base_theme

print(p_simple)

ggsave("results/cue_similarity_simple.png",
       plot = p_simple,
       width = 18, height = 14, units = "cm", dpi = 600, bg = "white")
