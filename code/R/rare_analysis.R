library(tidyr)
library(broom)

# Load data
setwd("/Users/jiayi/Desktop/micrm/master_project")
df <- read.csv("data/rare.csv")

# Check dilution rates
cat("Available dilution rates:", unique(df$DilutionRate), "\n")

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


# Survival labels
df <- df %>%
  mutate(survival     = ifelse(C_final > 1e-5, "Survived", "Extinct"),
         survived_bin = ifelse(survival == "Survived", 1, 0))

df_surv <- df %>% filter(survival == "Survived")

# 1) CUE vs survival probability
# Sample data
# Use only 50% of data points
df_sampled <- df %>% sample_frac(0.5)

p_s <- ggplot(df, aes(x = CUE, y = survived_bin, colour = factor(DilutionRate))) +
  geom_jitter(height = 0.05, width = 0, alpha = 0.4, size = 1.2, shape = 16) +  # Set point shape
  stat_smooth(method = "glm",
              method.args = list(family = "binomial"),
              se = FALSE, size = 1.2, linetype = "dashed") +  # Set line style
  labs(
    x = "Species-level CUE",
    y = "Probability of Survival",
    colour = "Dilution Rate"
  ) +
  scale_color_manual(
    values = c("0.01" = "#E74C3C", "0.05" = "#2ECC71", "0.1" = "#3498DB"),
    labels = c("0.01", "0.05", "0.1")
  ) +
  theme_minimal(base_size = 14) +
  theme(
    text = element_text(family = "Times New Roman"),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    panel.grid.major = element_line(color = "grey80", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks = element_line(color = "black", size = 0.5),
    axis.ticks.length = unit(0.15, "cm")
  )

p_s
glm(survived_bin ~ CUE * DilutionRate, data = df, family = "binomial") %>%
  tidy()

ggsave("results/survival_by_dilution.pdf",
       plot = p_s,
       device = cairo_pdf,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("results/presentation/survival_by_dilution.png",
       plot = p_s,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

# Compare predictive power of dilution rate vs CUE on final abundance
cat("\n=== Model Comparison ===\n")

# Fit models
m1 <- lm(log10(C_final) ~ CUE, data = df_surv)
m2 <- lm(log10(C_final) ~ DilutionRate, data = df_surv)
m3 <- lm(log10(C_final) ~ CUE + DilutionRate, data = df_surv)
m4 <- lm(log10(C_final) ~ CUE * DilutionRate, data = df_surv)

# Model comparison table
model_summary <- data.frame(
  Model = c("CUE", "DilutionRate", "CUE + DilutionRate", "CUE * DilutionRate"),
  R2 = c(summary(m1)$r.squared, summary(m2)$r.squared, 
         summary(m3)$r.squared, summary(m4)$r.squared),
  Adj_R2 = c(summary(m1)$adj.r.squared, summary(m2)$adj.r.squared,
             summary(m3)$adj.r.squared, summary(m4)$adj.r.squared),
  AIC = c(AIC(m1), AIC(m2), AIC(m3), AIC(m4))
)
print(model_summary)

# Best model details
cat("\n=== Best Model (CUE * DilutionRate) ===\n")
print(summary(m4))

# ANOVA
cat("\n=== ANOVA ===\n")
print(anova(m1, m3, m4))

# Visualization
p_comparison <- ggplot(df_surv, aes(x = CUE, y = C_final, colour = factor(DilutionRate))) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = "lm", se = TRUE, size = 1) +
  scale_y_log10() +
  scale_color_manual(values = c("0.01" = "#E74C3C", "0.05" = "#2ECC71", "0.1" = "#3498DB"),
                     labels = c("0.01", "0.05", "0.1"),
                     name = "Dilution Rate") +
  labs(x = "Species-level CUE",
       y = "Final Abundance") +
  base_theme +
  facet_wrap(~ DilutionRate, nrow = 1)

p_comparison

ggsave("results/rare_predictive_comparison.pdf",
       plot = p_comparison,
       device = cairo_pdf,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")
