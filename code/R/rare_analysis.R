setwd("/home/jiayi-chen/Documents/MiCRM/code")
# Load necessary libraries
library(dplyr)
library(ggplot2)

# Read the data (replace with your actual file path if needed)
df <- read.csv("../data/rare_invade_hpc.csv")

# Filter for survival (C_final > 1e-5)
df <- df %>%
  mutate(survival = ifelse(C_final > 1e-5, "Survived", "Extinct"))
df_surv <- df %>%
  filter(C_final > 1e-5)
# Summarize survival/extinction by CUE and invading status
df$survived_bin <- ifelse(df$survival == "Survived", 1, 0)
# survival or not by dilusion rate
ggplot(df, aes(x = CUE, y = survived_bin, color = factor(DilutionRate))) +
  geom_jitter(height = 0.05, width = 0, alpha = 0.3) +  # 离散点
  stat_smooth(method = "glm", method.args = list(family = "binomial"),
              se = FALSE) +  # 可改 se=TRUE 加置信区间
  theme_minimal() +
  labs(title = "",
       x = "CUE", y = "Probability of Survival",
       color = "Dilution Rate")

ggplot(df_surv, aes(x = CUE, y = C_final, color = factor(DilutionRate))) +
  geom_point(alpha = 0.5) +
  scale_y_log10() +  # 建议 log 变换，C_final 跨度大
  theme_minimal() +
  labs(title = "",
       x = "CUE", y = "C_final (log scale)", color = "Dilution Rate")

df2 <- read.csv("../data/rare_invade_hpc.csv")
