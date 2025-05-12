setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)
library(patchwork)

df <- read.csv("../data/df_results.csv")

long_data <- pivot_longer(
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
# standrdise species number
species_counts <- long_data %>%
  group_by(Community) %>%
  summarise(SpeciesCount = n_distinct(Species))
############ Density of relative abundance ################
ggplot(long_data %>% filter(RelativeAbundance > 0), 
       aes(x = RelativeAbundance, fill = Community, color = Community)) +
  geom_density(alpha = 0.3, adjust = 2) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "Density of Relative Abundance by Community (log scale)",
       x = "Relative Abundance (log10 scale)", y = "Density") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))

################### calculate the species richness ###################
long_data_d <- long_data %>%
  mutate(Status = ifelse(RelativeAbundance < 1e-5, "Extinction", "Survival"))

richness <- long_data_d %>%
  group_by(Seed, Community) %>%
  summarise(
   richness = sum(Status == "Survival"),
    TotalSpecies = n(),
    SurvivalRate =richness / TotalSpecies,
    .groups = "drop"
  )

# plot survival rate
# shannon evenness analysis
shannon_evenness <- long_data_d %>%
  group_by(Seed, Community) %>%
  summarize(
  Shannon = -sum(RelativeAbundance[RelativeAbundance >0 ] * log(RelativeAbundance[RelativeAbundance >0 ])),
  Evenness = Shannon/log( sum(Status == "Survival")),
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
  labs(x = "Community", y = "Evenness") +
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

# plot relative richness
ggplot(richness, aes(x = Community, y = SurvivalRate, fill = Community)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA) + 
  geom_jitter(aes(fill = Community), color = "black", shape = 21, 
              size = 2.5, width = 0.15, alpha = 0.8) +  
  scale_fill_brewer(palette = "Set2") +  
  labs(x = "Community", y = "Evenness") +
  theme_minimal() +
  theme(legend.position = "none")

# anova of Shannon
anova_result <- aov(Shannon ~ Community, data = shannon_evenness)
summary(anova_result)
TukeyHSD(anova_result)

# anova of evenness
anova_result <- aov(Evenness ~ Community, data = shannon_evenness)
summary(anova_result)
TukeyHSD(anova_result)

