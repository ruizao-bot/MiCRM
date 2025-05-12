setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)

df <- read.csv("../data/elv_recursive.csv")
df_long <- df %>%
  pivot_longer(
    cols = -Seed,
    names_to = c(".value", "Community", "Species"),
    names_pattern = "(r|CUE|Cfinal)_Comm(\\d+)_Sp(\\d+)"
  ) %>%
  mutate(
    Community = paste0("Community ", Community),
    Species = paste0("Species ", Species)
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
########## One community ##############
df_single <- read.csv("../data/df_elv.csv")
df_single_select <- df_single %>%filter(Species.CUE >0 )

ggplot(df_single_select, aes(x = Species.CUE, y = r)) +
  geom_point(alpha = 0.6) +
  labs(title = "Growth Rate (r) vs Species CUE by Community",
       x = "Species CUE",
       y = "Growth Rate (r)") +
  theme_minimal()

