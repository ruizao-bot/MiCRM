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
