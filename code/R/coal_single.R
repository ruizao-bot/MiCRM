setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)

df <- read.csv("../data/coal_single.csv")
df_surv <- df %>% filter(Status == "Survival")
ggplot(df, aes(x = CUE, y = Max_Growth_Rate)) +
  geom_point(alpha = 0.6) +
  labs(title = "Growth Rate (r) vs Species CUE by Community",
       x = "Species CUE",
       y = "Growth Rate (r)") +
  theme_minimal()

ggplot(df, 
       aes(x = CUE, fill = factor(Community), color = factor(Community))) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "",
       x = "CUE", y = "Frequency",
       fill = "Community", color = "Community") +
  scale_fill_manual(values = c("red", "#2ca02c", "blue")) +
  scale_color_manual(values = c("red", "#2ca02c", "blue"))


df_ext <- df %>%
  mutate(StatusGroup = ifelse(Status == "Extinction", "Extinction",
                              paste0("Survival_", Community)))

ggplot(df_ext, aes(x = factor(Community), y = CUE, fill = StatusGroup)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.3, position = position_dodge(width = 0.8)) +  
  geom_jitter(aes(color = StatusGroup), 
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 0.1, alpha = 0.3) +
  scale_fill_manual(values = c(
    "Extinction" = "grey60",
    "Survival_1" = "red",  # red
    "Survival_2" = "chartreuse3",  # chartreuse3
    "Survival_3" = "blue"   # blue
  )) +
  scale_color_manual(values = c(
    "Extinction" = "grey60",
    "Survival_1" = "red",
    "Survival_2" = "chartreuse3",
    "Survival_3" = "blue"
  )) +
  labs(x = "Community", y = "CUE", fill = "Status", color = "Status") +
  theme_minimal()
#### abundance and frequnecy####

df_line <- df_surv %>%
  mutate(bin = cut(C_final, breaks = seq(0, max(C_final), length.out = 51), include.lowest = TRUE)) %>%
  group_by(Community, bin) %>%
  summarise(count = n(), .groups = "drop") %>%
  mutate(
    bin_low = as.numeric(gsub("\\[|\\(|,.*", "", bin)),
    bin_high = as.numeric(gsub(".*,", "", gsub("]", "", bin))),
    bin_mid = (bin_low + bin_high) / 2
  )

ggplot(df_surv, aes(x = C_final, fill = factor(Community))) +
  geom_histogram(binwidth = 0.5, position = "identity", alpha = 0.3) +
  geom_line(data = df_line, aes(x = bin_mid, y = count, color = factor(Community)), size = 1.2) +
  theme_minimal() +
  labs(
    x = "Abundance (C_final, survivors only)",
    y = "Frequency (Species)",
    fill = "Community",
    color = "Community"
  ) +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))
#################################################
ggplot(df_surv, aes(x = CUE, y = C_final, color = factor(Community))) +
  geom_point(alpha = 0.7, size = 2) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_minimal() +
  labs(
    x = "CUE",
    y = "Abundance",
    color = "Community"
  ) +
  scale_color_manual(values = c(
    "1" = "#E41A1C",  # red
    "2" = "#4DAF4A",  # green
    "3" = "#377EB8"   # blue
  ))
# CUE abundance
df_binned <- df_surv %>%
  mutate(CUE_bin = cut(CUE, breaks = seq(0, 1, by = 0.05), include.lowest = TRUE))

df_bar <- df_binned %>%
  group_by(CUE_bin, Community) %>%
  summarise(TotalC_final = sum(C_final), .groups = "drop")  # 使用 Cfinal 而不是 RelAbun


  ggplot(df_bar, aes(x = CUE_bin, y = TotalC_final, fill = factor(Community))) +
  geom_col(position = "stack") +
  theme_minimal() +
  labs(
    x = "Species CUE",
    y = "Abundance",
    fill = "Community"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


