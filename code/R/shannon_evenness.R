setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(stringr)
library(patchwork)

df <- read.csv("../data/df_results.csv")

df <- pivot_longer(
  df,
  cols = starts_with("RelAbun_Comm"), 
  names_to = c("Community", "Species"),
  names_pattern = "RelAbun_(Comm\\d+)_Sp(\\d+)", 
  values_to = "RelativeAbundance"
  ) %>%
  rowwise() %>%
  mutate(
    Community.CUE = get(paste0("Community.CUE.", str_remove(Community, "Comm"))),
    Species.CUE = get(paste0("CUE_Comm", str_remove(Community, "Comm"), "_Sp", Species))
   ) %>%
  ungroup() %>%
  select(-starts_with("Community.CUE."), -matches("^CUE_Comm")) 
df_select <- df %>% filter(Species.CUE > 0 & RelativeAbundance > 1e-5)
# standrdise species number
species_counts <- df_select %>%
  group_by(Community) %>%
  summarise(SpeciesCount = n_distinct(Species))
############ Density of relative abundance ################
ggplot(df_select %>% filter(RelativeAbundance > 0), 
       aes(x = RelativeAbundance, fill = Community, color = Community)) +
  geom_density(alpha = 0.3, adjust = 2) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "Density of Relative Abundance by Community (log scale)",
       x = "Relative Abundance (log10 scale)", y = "Density") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))
##### frequency #########
ggplot(df_select %>% filter(RelativeAbundance > 0), 
       aes(x = RelativeAbundance, fill = Community, color = Community)) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "Frequency Relative Abundance by Community (log scale)",
       x = "Relative abundance", y = "Frequency") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))
##### relative frequency #########
ggplot(df_select %>% filter(RelativeAbundance > 0), 
       aes(x = RelativeAbundance, fill = Community, color = Community)) +
  geom_histogram(aes(y = after_stat(count / sum(count))), 
                 binwidth = 0.1, position = "identity", alpha = 0.4) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "Frequency of Relative Abundance by Community (log scale)",
       x = "Relative Abundance (log10 scale)", y = "Relative Frequency") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))


################### Shannon and evenness ###################
df_division <- df_select %>%
  mutate(Status = ifelse(RelativeAbundance < 1e-5, "Extinction", "Survival"))

# shannon evenness analysis
shannon_evenness <- df_division %>%
  group_by(Seed, Community) %>%
  summarize(
    Shannon = -sum(RelativeAbundance[Status == "Survival"] * log(RelativeAbundance[Status == "Survival"])),
    Evenness = Shannon / log(sum(Status == "Survival"))
  )
# shannon
shannon <- shannon_evenness %>%
  group_by(Community) %>%
  mutate(mean_Shannon = mean(Shannon),
         distance_to_mean = abs(Shannon - mean_Shannon),
         norm_dist = rescale(-distance_to_mean))

base_colors <- c("Comm1" = "#E41A1C", "Comm2" = "#377EB8", "Comm3" = "#4DAF4A")

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

base_colors <- c("Comm1" = "#E41A1C", "Comm2" = "#377EB8", "Comm3" = "#4DAF4A") 

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
  filter(RelativeAbundance > 1e-5) %>%
  mutate(CUE_bin = cut(Species.CUE, breaks = seq(0, 1, by = 0.05), include.lowest = TRUE))

df_bar <- df_binned %>%
  group_by(CUE_bin, Community) %>%
  summarise(TotalAbundance = sum(RelativeAbundance), .groups = "drop")

ggplot(df_bar, aes(x = CUE_bin, y = TotalAbundance, fill = Community)) +
  geom_col(position = "stack") +
  theme_minimal() +
  labs(
       x = "Species CUE", y = "Total Relative Abundance") +
  scale_fill_manual(values = c("Comm1" = "#E41A1C",  # red
                               "Comm2" = "#4DAF4A",  # green
                               "Comm3" = "#377EB8")) +  # blue
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
########### linear regression between CUE and relative abundance#######
df_comm3 <- df_select %>% filter(Community == "Comm3")
ggplot(df_comm3, aes(x = Species.CUE, y = RelativeAbundance)) +
  geom_point(color = "#984EA3", alpha = 0.7, size = 2) +  # 紫色点
  geom_smooth(method = "lm", se = TRUE, color = "black") +  # 回归线
  scale_x_log10() +
  theme_minimal() +
  labs(
    x = "CUE",
    y = "Relative Abundance (log10 scale)",
    title = "Linear Relationship between CUE and Abundance in Comm3"
  )
ggplot(df_select, aes(x = Species.CUE, y = RelativeAbundance, color = Community)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", se = TRUE)+
  scale_x_log10() +
  coord_cartesian(ylim = c(0, 1)) +
  theme_minimal() +
  labs(
    x = "CUE",
    y = "Relative Abundance (log10 scale)"
  ) +
  scale_color_manual(values = c(
    "Comm1" = "#E41A1C",  # red
    "Comm2" = "#4DAF4A",   # green
    "Comm3" = "#377EB8"
  ))

lm_comm1 <- lm(RelativeAbundance ~ Species.CUE, data = subset(df_select, Community == "Comm1"))
summary(lm_comm1)

lm_comm2 <- lm(RelativeAbundance ~ Species.CUE, data = subset(df_select, Community == "Comm2"))
summary(lm_comm2)

lm_comm3 <- lm(RelativeAbundance ~ Species.CUE, data = subset(df_select, Community == "Comm3"))
summary(lm_comm3)

########### CUE comparison between three comunities###########
ggplot(df_select, aes(x = Community, y = Community.CUE, fill = Community)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA) + 
  geom_jitter(aes(fill = Community), color = "black", shape = 21, 
              size = 1.5, width = 0.15, alpha = 0.8) +  
  scale_fill_manual(values = c(
    "Comm1" = "#E41A1C",  # red
    "Comm2" = "#4DAF4A",  # green
    "Comm3" = "#377EB8"   # blue
  )) +
  labs(x = "Community", y = "CUE") +
  theme_minimal() +
  theme(legend.position = "none")
########## richness and species CUE ############
df_stats <- df_select %>%
  group_by(Seed, Community, Community.CUE) %>%
  summarise(
    Richness = n_distinct(Species),
    CUE_Var = var(Species.CUE),
    .groups = "drop"
  )

ggplot(df_stats, aes(x = Community.CUE, y = Richness)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(x = "CUE Variation", y = "Species Richness") +
  theme_minimal()

model_var <- lm(Richness ~ CUE_Var, data = df_stats)
summary(model_var)
