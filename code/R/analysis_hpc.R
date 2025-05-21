setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(stringr)
library(patchwork)

df <- read.csv("../data/coal_recursive_hpc.csv")

df_long <- df %>%
  pivot_longer(
    cols = matches("^(RelAbun|CUE|Abundance)_Comm\\d+_Sp\\d+"),
    names_to = c(".value", "Community", "Species"),
    names_pattern = "(RelAbun|CUE|Abundance)_Comm(\\d+)_Sp(\\d+)"
  ) %>%
  mutate(
    Community = paste0("Community ", Community),
    Species = paste0("Species ", Species),
    Community.CUE = case_when(
      Community == "Community 1" ~ Community.CUE.1,
      Community == "Community 2" ~ Community.CUE.2,
      Community == "Community 3" ~ Community.CUE.3
    )
  )
df_select <- df %>%
  mutate(Status = ifelse(Abundance < 1e-5, "Extinction", "Survival"))
df_surv <- df_select %>%
  filter(Abundance > 1e-5)
# standrdise species number
species_counts <- df_select %>%
  group_by(Community) %>%
  summarise(SpeciesCount = n_distinct(Species_ID))
############ Density of relative abundance ################
df_surv <- df_select %>% filter(Status == "Survival")

#ggplot(df_surv, aes(x = Abundance, fill = Community)) +
#  geom_histogram(binwidth = 0.01, position = "identity", alpha = 0.3) +
#  geom_line(data = df_line, aes(x = bin_mid, y = count, color = Community), size = 1.2) +
#  theme_minimal() +
#  labs(x = "Abundance", y = "Frequency (Samples)") +
#  scale_fill_manual(values = c("red", "green", "blue")) +
#  scale_color_manual(values = c("red", "green", "blue"))

##### frequency #########
ggplot(df_surv, 
       aes(x = Abundance, fill = factor(Community), color = factor(Community))) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "",
       x = "Abundance (log-scaled axis)", y = "Frequency",
       fill = "Community", color = "Community") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))
##### kernel density #########
ggplot(df_surv, aes(x = Abundance, fill = factor(Community), color = factor(Community))) +
  geom_density(alpha = 0.4, adjust = 1.5) +
  scale_x_log10() +
  labs(
    x = "Abundance (log-scaled axis)",
    y = "Density",
    fill = "Community", color = "Community"
  ) +
  theme_minimal() +
  theme(
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )
############## CUE distribution and separation between survival and extinction #############
ggplot(df_surv, 
       aes(x = Species_CUE, fill = factor(Community), color = factor(Community))) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "",
       x = "CUE", y = "Frequency",
       fill = "Community", color = "Community") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))

# Survival
df_ext <- df_select %>%
  mutate(StatusGroup = ifelse(Status == "Extinction", "Extinction",
                              paste0("Survival_", Community)))

ggplot(df_ext, aes(x = factor(Community), y = Species_CUE, fill = StatusGroup)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.3, position = position_dodge(width = 0.8)) +  
  geom_jitter(aes(color = StatusGroup), 
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 0.1, alpha = 0.3) +
  scale_fill_manual(values = c(
    "Extinction" = "grey60",
    "Survival_1" = "red",  # red
    "Survival_2" = "green",  # green
    "Survival_3" = "blue"   # blue
  )) +
  scale_color_manual(values = c(
    "Extinction" = "grey60",
    "Survival_1" = "red",
    "Survival_2" = "green",
    "Survival_3" = "blue"
  )) +
  labs(x = "Community", y = "CUE", fill = "Status", color = "Status") +
  theme_minimal()
########### linear regression between CUE and abundance#######
fit_lines <- data.frame()

for (comm in unique(df_surv$Community)) {
  # 用 df_surv 中当前群落的数据
  model <- lm(log10(Abundance) ~ Species_CUE,
              data = df_surv %>% filter(Community == comm, Abundance > 1e-5))
  
  # 打印模型摘要
  cat("=== Summary for Community", comm, "===\n")
  print(summary(model))
  cat("\n")
  
  # 用当前群落生成预测线
  cue_seq <- seq(
    min(df_surv$Species_CUE[df_surv$Community == comm & df_surv$Abundance > 1e-5]),
    max(df_surv$Species_CUE[df_surv$Community == comm & df_surv$Abundance > 1e-5]),
    length.out = 200
  )
  
  pred <- data.frame(
    Species_CUE = cue_seq,
    log_Abundance = predict(model, newdata = data.frame(Species_CUE = cue_seq)),
    Community = as.factor(comm)
  )
  
  fit_lines <- bind_rows(fit_lines, pred)
}

ggplot(df_surv %>% filter(Abundance > 1e-5), aes(x = Species_CUE, y = log10(Abundance + 1e-8), color = factor(Community))) +
  geom_point(alpha = 0.7, size = 2) +
  geom_line(data = fit_lines, aes(x = Species_CUE, y = log_Abundance, color = factor(Community)), size = 1.2) +
  theme_minimal() +
  labs(
    x = "CUE",
    y = "log10(Abundance)",
    color = "Community"
  ) 
########################### logistic #####################################
library(minpack.lm)
library(dplyr)

fit_lines <- data.frame()

for (comm in c("1", "2", "3")) {
  dat <- df_surv[df_surv$Community == comm, ]
  
  if (nrow(dat) > 0) {
    model <- nlsLM(
      Abundance ~ K / (1 + exp(-r * (Species_CUE - x0))),
      data = dat,
      start = list(K = max(dat$Abundance), r = 5, x0 = median(dat$Species_CUE)),
      control = nls.control(maxiter = 200, warnOnly = TRUE)
    )
    
    cat("=== Summary for", comm, "===\n")
    print(summary(model))
    cat("\n")
    
    cue_seq <- seq(min(dat$Species_CUE), max(dat$Species_CUE), length.out = 200)
    
    pred <- data.frame(
      Species_CUE = cue_seq,
      Abundance = predict(model, newdata = data.frame(Species_CUE = cue_seq)),
      Community = comm
    )
    
    fit_lines <- bind_rows(fit_lines, pred)
  } else {
    cat("No valid data for", comm, "\n\n")
  }
}

# 绘图
ggplot(df_surv, aes(x = Species_CUE, y = Abundance, color = factor(Community)) )+
  geom_point(alpha = 0.7, size = 2) +
  geom_line(data = fit_lines, aes(x = Species_CUE, y = Abundance, color = Community), size = 1.2) +
  theme_minimal() +
  labs(
    x = "Species-level CUE",
    y = "Abundance",
    color = "Community"
  )

########## richness and species CUE Variance############
df_stats <- df_surv%>%
  group_by(Seed, Community, Community_CUE) %>%
  summarise(
    Richness = n_distinct(Species_ID),
    CUE.Var = var(Species_CUE),
    .groups = "drop"
  )

ggplot(df_stats, aes(x = CUE.Var, y = Richness, color = factor(Community))) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~ Community, scales = "free_x") +
  labs(x = "CUE Variation", y = "Species Richness") +
  theme_minimal()+
  labs(title = "",
       x = "Abundance (log-scaled axis)", y = "Frequency") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))


for (comm in c(1, 2,3)) {
  model_var <- lm(Richness ~ CUE.Var, data = subset(df_stats, Community == comm))
  cat("\n---", comm, "---\n")
  print(summary(model_var))
}
############## Survival ##################




