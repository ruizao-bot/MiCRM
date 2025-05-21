setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(stringr)
library(patchwork)

df <- read.csv("../data/coal_recursive.csv")

df_long <- df %>%
  pivot_longer(
    cols = matches("^(RelAbun|CUE|Cfinal)_Comm\\d+_Sp\\d+"),
    names_to = c(".value", "Community", "Species"),
    names_pattern = "(RelAbun|CUE|Cfinal)_Comm(\\d+)_Sp(\\d+)"
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
df_select <- df_long %>%
  select(-Community.CUE.1, -Community.CUE.2, -Community.CUE.3) %>%
  mutate(Status = ifelse(Cfinal < 1e-5, "Extinction", "Survival"))
df_surv <- df_select %>%
  filter(Cfinal > 1e-5)
# standrdise species number
species_counts <- df_select %>%
  group_by(Community) %>%
  summarise(SpeciesCount = n_distinct(Species))
############ Density of relative abundance ################
df_surv <- df_select %>% filter(Status == "Survival")

#ggplot(df_surv, aes(x = Cfinal, fill = Community)) +
#  geom_histogram(binwidth = 0.01, position = "identity", alpha = 0.3) +
#  geom_line(data = df_line, aes(x = bin_mid, y = count, color = Community), size = 1.2) +
#  theme_minimal() +
#  labs(x = "Abundance", y = "Frequency (Samples)") +
#  scale_fill_manual(values = c("red", "green", "blue")) +
#  scale_color_manual(values = c("red", "green", "blue"))

##### frequency #########
ggplot(df_surv, 
       aes(x = Cfinal, fill = Community, color = Community)) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "",
       x = "Abundance (log-scaled axis)", y = "Frequency") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))
##### kernel density #########
ggplot(df_surv, aes(x = Cfinal, fill = Community, color = Community)) +
  geom_density(alpha = 0.4, adjust = 1.5) +
  scale_x_log10() +
  labs(
    x = "Abundance (log-scaled axis)",
    y = "Density"
  ) +
  theme_minimal() +
  theme(
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )


################### Shannon and evenness ###################
df_division <- df_long %>%
  mutate(Status = ifelse(Cfinal < 1e-5, "Extinction", "Survival"))

# shannon evenness analysis
shannon_evenness <- df_division %>%
  group_by(Seed, Community) %>%
  summarize(
    Shannon = -sum(RelAbun[Status == "Survival"] * log(RelAbun[Status == "Survival"])),
    Evenness = Shannon / log(sum(Status == "Survival"))
  )
# shannon
shannon <- shannon_evenness %>%
  group_by(Community) %>%
  mutate(mean_Shannon = mean(Shannon),
         distance_to_mean = abs(Shannon - mean_Shannon),
         norm_dist = rescale(-distance_to_mean))

base_colors <- c("Community1" = "red", "Community2" = "blue", "Community3" = "green")

shannon <- shannon %>%
  mutate(fill_color = mapply(function(comm, intensity) {
    col <- col2rgb(base_colors[comm]) / 255
    rgb(1 - (1 - col[1]) * intensity,
        1 - (1 - col[2]) * intensity,
        1 - (1 - col[3]) * intensity)
  }, Community, norm_dist))

ggplot(shannon, aes(x = Community, y = Shannon)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA, fill = base_colors ) + 
  geom_jitter(aes(fill = fill_color), color = base_colors , shape = 21, 
              size = 2.5, width = 0.15, alpha = 0.9) +  
  scale_fill_identity() +
  labs(x = "Community", y = "Shannon") +
  theme_minimal()


# evenness
shannon_evenness <- shannon_evenness %>%
  group_by(Community) %>%
  mutate(mean_evenness = mean(Evenness),
         distance_to_mean = abs(Evenness - mean_evenness),
         norm_dist = rescale(-distance_to_mean)) 

base_colors <- c("Community1" = "red", "Community2" = "blue", "Community3" = "green") 

shannon_evenness <- shannon_evenness %>%
  mutate(fill_color = mapply(function(comm, intensity) {
    col <- col2rgb(base_colors[comm]) / 255
    rgb(1 - (1 - col[1]) * intensity,
        1 - (1 - col[2]) * intensity,
        1 - (1 - col[3]) * intensity)
  }, Community, norm_dist))

ggplot(shannon_evenness, aes(x = Community, y = Evenness)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA, fill = "gray90") + 
  geom_jitter(aes(fill = fill_color), color = "black", shape = 21, 
              size = 2.5, width = 0.15, alpha = 0.9) +  
  scale_fill_identity() +
  labs(x = "Community", y = "Evenness") +
  theme_minimal()

# anova of Shannon
anova_result <- aov(Shannon ~ Community, data = shannon_evenness)
summary(anova_result)
TukeyHSD(anova_result)

# anova of evenness
anova_result <- aov(Evenness ~ Community, data = shannon_evenness)
summary(anova_result)
TukeyHSD(anova_result)

############ CUE and Abundance ############
df_binned <- df_surv %>%
  mutate(CUE_bin = cut(CUE, breaks = seq(0, 1, by = 0.05), include.lowest = TRUE))

df_bar <- df_binned %>%
  group_by(CUE_bin, Community) %>%
  summarise(TotalAbundance = sum(Cfinal), .groups = "drop")

ggplot(df_bar, aes(x = CUE_bin, y = TotalAbundance, fill = Community)) +
  geom_col(position = "stack") +
  theme_minimal() +
  labs(
       x = "Species CUE", y = "Total Abundance") +
  scale_fill_manual(values = c("Community 1" = "#F8766D",  # red
                               "Community 2" = "#7CAE00",  # green
                               "Community 3" = "#00BFC4")) +  # blue
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(df_surv, 
       aes(x = CUE, fill = Community, color = Community)) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "",
       x = "CUE", y = "Frequency") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))
########### linear regression between CUE and abundance#######
fit_lines <- data.frame()

for (comm in unique(df_surv$Community)) {
  # 用 df_surv 中当前群落的数据
  model <- lm(log10(Cfinal) ~ CUE,
              data = df_surv %>% filter(Community == comm, Cfinal > 1e-5))
  
  # 打印模型摘要
  cat("=== Summary for Community", comm, "===\n")
  print(summary(model))
  cat("\n")
  
  # 用当前群落生成预测线
  cue_seq <- seq(
    min(df_surv$CUE[df_surv$Community == comm & df_surv$Cfinal > 1e-5]),
    max(df_surv$CUE[df_surv$Community == comm & df_surv$Cfinal > 1e-5]),
    length.out = 200
  )
  
  pred <- data.frame(
    CUE = cue_seq,
    log_Cfinal = predict(model, newdata = data.frame(CUE = cue_seq)),
    Community = as.factor(comm)
  )
  
  fit_lines <- bind_rows(fit_lines, pred)
}

ggplot(df_surv %>% filter(Cfinal > 1e-5), aes(x = CUE, y = log10(Cfinal + 1e-8), color = factor(Community))) +
  geom_point(alpha = 0.7, size = 2) +
  geom_line(data = fit_lines, aes(x = CUE, y = log_Cfinal, color = Community), size = 1.2) +
  theme_minimal() +
  labs(
    x = "CUE",
    y = "log10(Abundance)",
    color = "Community"
  ) 
# logistic
library(minpack.lm)
library(dplyr)

fit_lines <- data.frame()


for (comm in c("Community 1", "Community 2", "Community 3")) {
  dat <- df_surv[df_surv$Community == comm, ]
  
  if (nrow(dat) > 0) {
    model <- nlsLM(
      Cfinal ~ K / (1 + exp(-r * (CUE - x0))),
      data = dat,
      start = list(K = max(dat$Cfinal), r = 5, x0 = median(dat$CUE)),
      control = nls.control(maxiter = 200, warnOnly = TRUE)
    )
    
    cat("=== Summary for", comm, "===\n")
    print(summary(model))
    cat("\n")
    
    cue_seq <- seq(min(dat$CUE), max(dat$CUE), length.out = 200)
    
    pred <- data.frame(
      CUE = cue_seq,
      Cfinal = predict(model, newdata = data.frame(CUE = cue_seq)),
      Community = comm
    )
    
    fit_lines <- bind_rows(fit_lines, pred)
  } else {
    cat("No valid data for", comm, "\n\n")
  }
}

ggplot(df_surv, aes(x = CUE, y = Cfinal, color = Community)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_line(data = fit_lines, aes(x = CUE, y = Cfinal, color = Community), size = 1.2) +
  theme_minimal() +
  labs(
    x = "CUE",
    y = "Abundance (Cfinal)",
    color = "Community"
  )


########### CUE comparison between three comunities###########
ggplot(df_select, aes(x = Community, y = Community.CUE, fill = Community)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA) + 
  geom_jitter(aes(fill = Community), color = "black", shape = 21, 
              size = 1.5, width = 0.15, alpha = 0.3) +  
  scale_fill_manual(values = c(
    "Community 1" = "red",  # red
    "Community 2" = "green",  # green
    "Community 3" = "blue"   # blue
  )) +
  labs(x = "Community", y = "CUE") +
  theme_minimal() +
  theme(legend.position = "none")


########## richness and species CUE Variance############
df_stats <- df_surv%>%
  group_by(Seed, Community, Community.CUE) %>%
  summarise(
    Richness = n_distinct(Species),
    CUE.Var = var(CUE),
    .groups = "drop"
  )

ggplot(df_stats, aes(x = CUE.Var, y = Richness, color = Community)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~ Community, scales = "free_x") +
  labs(x = "CUE Variation", y = "Species Richness") +
  theme_minimal()


for (comm in c("Community 1", "Community 2", "Community 3")) {
  model_var <- lm(Richness ~ CUE.Var, data = subset(df_stats, Community == comm))
  cat("\n---", comm, "---\n")
  print(summary(model_var))
}
############## Survival ##################
df_survival <- df_long %>%
  group_by(Community, Seed) %>%
  summarise(
    SurvivalRate = mean(RelAbun > 1e-5), 
    .groups = "drop"
  )

ggplot(df_survival, aes(x = Community, y = SurvivalRate, fill = Community)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA) + 
  geom_jitter(aes(fill = Community), color = "black", shape = 21, 
              size = 2.5, width = 0.15, alpha = 0.8) +  
  scale_fill_manual(values = c(
    "Community 1" = "red",  # red
    "Community 2" = "green",  # green
    "Community 3" = "blue"   # blue
  )) +
  labs(x = "Community", y = "Survival Rate") +
  theme_minimal() +
  theme(legend.position = "none")

df_division_filtered <- df_division %>%
  filter(CUE > -1)
ggplot(df_select, aes(x = factor(Community), y = CUE, fill = Status)) +
  geom_boxplot(outlier.shape = NA, position = position_dodge(width = 0.8)) +  
  geom_jitter(aes(color = Status), 
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 0.1, alpha = 0.7) +
  scale_fill_manual(values = c("Extinction" = "#F8766D", "Survival" = "#00BFC4")) +
  scale_color_manual(values = c("Extinction" = "#F8766D", "Survival" = "#00BFC4")) +
  labs(x = "Community", y = "CUE") +
  theme_minimal()


df_ext <- df_select %>%
  mutate(StatusGroup = ifelse(Status == "Extinction", "Extinction",
                              paste0("Survival_", Community)))

ggplot(df_ext, aes(x = factor(Community), y = CUE, fill = StatusGroup)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.3, position = position_dodge(width = 0.8)) +  
  geom_jitter(aes(color = StatusGroup), 
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 0.1, alpha = 0.3) +
  scale_fill_manual(values = c(
    "Extinction" = "grey60",
    "Survival_Community 1" = "red",  # red
    "Survival_Community 2" = "green",  # green
    "Survival_Community 3" = "blue"   # blue
  )) +
  scale_color_manual(values = c(
    "Extinction" = "grey60",
    "Survival_Community 1" = "red",
    "Survival_Community 2" = "green",
    "Survival_Community 3" = "blue"
  )) +
  labs(x = "Community", y = "CUE", fill = "Status", color = "Status") +
  theme_minimal()


