setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)

df <- read.csv("../data/elv5.csv")
library(tidyr)
library(dplyr)

df_long <- df %>%
  pivot_longer(
    cols = starts_with("α_"),
    names_to = "Target_Species",
    names_prefix = "α_",
    values_to = "Alpha_ij"
  )

df_clean <- df_long %>% drop_na()
df_select <- df_long %>% filter(CUE > 0 )

for (community in unique(df_select$Community)) {
  df_sub <- df_select %>% filter(Community == community)
  model <- lm(r ~ CUE, data = df_sub)
  cat("\n---", community, "---\n")
  print(summary(model))
}

ggplot(df_select, aes(x = CUE, y = r)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  facet_wrap(~ Community) +
  labs(title = "Growth Rate (r) vs Species CUE by Community",
       x = "Species CUE",
       y = "Growth Rate (r)") +
  theme_minimal()

ggplot(df, aes(x = factor(Community), y = CUE)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "CUE by Community",
       x = "Community", y = "CUE") +
  theme_minimal()
#############################################################################
setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)

df50 <- read.csv("../data/elv50.csv")
df_clean <- df50 %>% drop_na()
df_select <- df50 %>% filter(CUE > 0 )

for (community in unique(df_select$Community)) {
  df_sub <- df_select %>% filter(Community == community)
  model <- lm(r ~ CUE, data = df_sub)
  cat("\n---", community, "---\n")
  print(summary(model))
}

ggplot(df_select, aes(x = CUE, y = r)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  facet_wrap(~ Community) +
  labs(title = "Growth Rate (r) vs Species CUE by Community",
       x = "Species CUE",
       y = "Growth Rate (r)") +
  theme_minimal()

ggplot(df50, aes(x = factor(Community), y = CUE, fill = Community)) +
  geom_boxplot() +
  labs(title = "CUE by Community",
       x = "Community", y = "CUE") +
  theme_minimal()


df50$Community <- as.factor(df50$Community)

ggplot(df50, aes(x = Community, y = r, fill = Community)) +
  geom_boxplot() +
  labs(title = "Intrinsic Growth Rate (r) by Community",
       x = "Community",
       y = "Intrinsic Growth Rate (r)") +
  theme_minimal()


