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
df_select <- df_long %>% filter(Cfinal > 1e-5)
# standrdise species number
species_counts <- df_select %>%
  group_by(Community) %>%
  summarise(SpeciesCount = n_distinct(Species))
############ Density of relative abundance ################
df_a <- df_select %>% 
  mutate(RelAbunPercent = RelAbun * 100)

df_line <- df_a %>%
  mutate(bin = cut(RelAbunPercent, breaks = seq(0, 100, by = 2), include.lowest = TRUE)) %>%
  group_by(Community, bin) %>%
  summarise(count = n(), .groups = "drop") %>%
  mutate(bin_mid = as.numeric(gsub(".*, *(.*)]", "\\1", bin)))

ggplot(df_a, aes(x = RelAbunPercent, fill = Community)) +
  geom_histogram(stat = "bin", bins = 50, position = "identity", alpha = 0.3) +
  geom_line(data = df_line, aes(x = bin_mid, y = count, color = Community), size = 1.2) +
  theme_minimal() +
  labs(x = "Relative Abundance (%)", y = "Frequency (Samples)") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))
##### frequency #########
ggplot(df_select %>% filter(RelAbun > 0), 
       aes(x = RelAbun, fill = Community, color = Community)) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "Frequency Relative Abundance by Community (log scale)",
       x = "Relative abundance", y = "Frequency") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))
##### relative frequency #########
ggplot(df_select %>% filter(RelAbun > 0), 
       aes(x = RelAbun, fill = Community, color = Community)) +
  geom_histogram(aes(y = after_stat(count / sum(count))), 
                 binwidth = 0.1, position = "identity", alpha = 0.4) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "Frequency of Relative Abundance by Community (log scale)",
       x = "Relative Abundance (log10 scale)", y = "Relative Frequency") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))


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

base_colors <- c("Community1" = "#E41A1C", "Community2" = "#377EB8", "Community3" = "#4DAF4A")

shannon <- shannon %>%
  mutate(fill_color = mapply(function(comm, intensity) {
    col <- col2rgb(base_colors[comm]) / 255
    rgb(1 - (1 - col[1]) * intensity,
        1 - (1 - col[2]) * intensity,
        1 - (1 - col[3]) * intensity)
  }, Community, norm_dist))

ggplot(shannon, aes(x = Community, y = Shannon)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA, fill = "gray90") + 
  geom_jitter(aes(fill = fill_color), color = "black", shape = 21, 
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

base_colors <- c("Community1" = "#E41A1C", "Community2" = "#377EB8", "Community3" = "#4DAF4A") 

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
df_binned <- df_select %>%
  mutate(CUE_bin = cut(CUE, breaks = seq(0, 1, by = 0.05), include.lowest = TRUE))

df_bar <- df_binned %>%
  group_by(CUE_bin, Community) %>%
  summarise(TotalAbundance = sum(RelAbun), .groups = "drop")

ggplot(df_bar, aes(x = CUE_bin, y = TotalAbundance, fill = Community)) +
  geom_col(position = "stack") +
  theme_minimal() +
  labs(
       x = "Species CUE", y = "Total Relative Abundance") +
  scale_fill_manual(values = c("Community 1" = "#E41A1C",  # red
                               "Community 2" = "#4DAF4A",  # green
                               "Community 3" = "#377EB8")) +  # blue
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
########### linear regression between CUE and relative abundance#######
fit_lines <- data.frame()  # Initialize dataframe to store predicted values

for (comm in c("Community 1", "Community 2", "Community 3")) {
  df_sub <- df_select %>%
    filter(CUE > 0, Community == comm)
  
  # Generate a sequence of Species.CUE values for prediction
  cue_seq <- seq(min(df_sub$CUE), max(df_sub$CUE), length.out = 200)
  
  if (comm %in% c("Community 1", "Community 2")) {
    # Fit logistic model for Community1 and Community2
    model <- nls(RelAbun ~ K / (1 + exp(-r * (CUE - x0))),
                 data = df_sub,
                 start = list(K = 1, r = 10, x0 = 0.3),
                 control = nls.control(maxiter = 200, warnOnly = TRUE))
    
    # Print model summary
    print(summary(model))
    
    # Calculate R² manually
    pred_vals <- predict(model)
    obs_vals <- df_sub$RelAbun
    rss <- sum((obs_vals - pred_vals)^2)
    tss <- sum((obs_vals - mean(obs_vals))^2)
    r2 <- 1 - rss / tss
    cat("R² =", round(r2, 4), "\n")
    
    # Create prediction dataframe
    pred <- data.frame(
      CUE = cue_seq,
      RelAbun = predict(model, newdata = data.frame(CUE = cue_seq)),
      Community = comm
    )
    
  } else {
    # Fit linear model for Community3
    model <- lm(RelAbun ~ CUE, data = df_sub)
    summary_out <- summary(model)
    
    # Print summary and R²
    print(summary_out)
    cat("R² =", round(summary_out$r.squared, 4), "\n")
    
    # Create prediction dataframe
    pred <- data.frame(
      CUE = cue_seq,
      RelAbun = predict(model, newdata = data.frame(CUE = cue_seq)),
      Community = comm
    )
  }
  
  # Combine predictions for all communities
  fit_lines <- bind_rows(fit_lines, pred)
}

ggplot(df_select, aes(x = CUE, y = RelAbun, color = Community)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_line(data = fit_lines, aes(x = CUE, y = RelAbun, color = Community), size = 1.2) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_minimal() +
  labs(
    x = "CUE",
    y = "Relative Abundance"
  ) +
  scale_color_manual(values = c(
    "Community 1" = "#E41A1C",  # red
    "Community 2" = "#4DAF4A",  # green
    "Community 3" = "#377EB8"   # blue
  ))
########### CUE comparison between three comunities###########
ggplot(df_select, aes(x = Community, y = Community.CUE, fill = Community)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA) + 
  geom_jitter(aes(fill = Community), color = "black", shape = 21, 
              size = 1.5, width = 0.15, alpha = 0.8) +  
  scale_fill_manual(values = c(
    "Community 1" = "#E41A1C",  # red
    "Community 2" = "#4DAF4A",  # green
    "Community 3" = "#377EB8"   # blue
  )) +
  labs(x = "Community", y = "CUE") +
  theme_minimal() +
  theme(legend.position = "none")
########## richness and species CUE Variance############
df_stats <- df_select %>%
  group_by(Seed, Community, Community.CUE) %>%
  summarise(
    Richness = n_distinct(Species),
    CUE.Var = var(CUE),
    .groups = "drop"
  )

ggplot(df_stats, aes(x = CUE.Var, y = Richness)) +
    geom_point() +
    geom_smooth(method = "lm", se = TRUE) +
    facet_wrap(~ Community, scales = "free_x")+
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
    "Community 1" = "#E41A1C",  # red
    "Community 2" = "#4DAF4A",  # green
    "Community 3" = "#377EB8"   # blue
  )) +
  labs(x = "Community", y = "Survival Rate") +
  theme_minimal() +
  theme(legend.position = "none")

df_division_filtered <- df_division %>%
  filter(CUE > -1)
ggplot(df_division_filtered, aes(x = factor(Community), y = CUE, fill = Status)) +
  geom_boxplot(outlier.shape = NA, position = position_dodge(width = 0.8)) +  
  geom_jitter(aes(color = Status), 
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 1, alpha = 0.7) +
  scale_fill_manual(values = c("Extinction" = "#F8766D", "Survival" = "#00BFC4")) +
  scale_color_manual(values = c("Extinction" = "#F8766D", "Survival" = "#00BFC4")) +
  labs(x = "Community", y = "CUE") +
  theme_minimal()

