setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)

df <- read.csv("../data/coal_single.csv")
df_survival <- df %>%filter( Max_Growth_Rate > 0)
ggplot(df_survival, aes(x = CUE, y = Max_Growth_Rate)) +
  geom_point(alpha = 0.6) +
  labs(title = "Growth Rate (r) vs Species CUE by Community",
       x = "Species CUE",
       y = "Growth Rate (r)") +
  theme_minimal()

ggplot(df, aes(x = factor(Community), y = CUE, fill = Community)) +
  geom_boxplot() +
  labs(title = "",
       x = "Community", y = "CUE") +
  theme_minimal()

ggplot(df, aes(x = Status, y = CUE, fill = Status)) +
  geom_boxplot() +
  facet_wrap(~ Community) +
  labs(title = "CUE by Survival Status in Each Community",
       x = "Status", y = "CUE") +
  theme_minimal()
