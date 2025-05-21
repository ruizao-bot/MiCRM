setwd("/home/jiayi-chen/Documents/MiCRM/code")
# Read the CSV file into R
df <- read.csv("../data/coal_recursive_hpc.csv")
df_select <- df %>%
  filter(Abundance > 1e-5)

library(dplyr)

df_comm <- df_select %>%
  filter(Community %in% c(1, 2)) %>%
  group_by(Seed, Community) %>%
  summarise(
    Community_CUE = unique(Community_CUE),
    Dominant_Community = unique(Dominant_Community),
    .groups = "drop"
  ) %>%
  mutate(
    Dominance = ifelse(paste0("Community ", Community) == Dominant_Community, 1, 0)
  )
model <- glm(Dominance ~ Community_CUE, data = df_comm, family = binomial)
summary(model)

library(ggplot2)

cue_seq <- seq(min(df_comm$Community_CUE), max(df_comm$Community_CUE), length.out = 300)

predicted <- predict(model, newdata = data.frame(Community_CUE = cue_seq), type = "response")

df_pred <- data.frame(
  Community_CUE = cue_seq,
  Probability = predicted
)

ggplot(df_comm, aes(x = Community_CUE, y = Dominance, color = factor(Community))) +
  geom_jitter(width = 0.0005, height = 0.05, alpha = 0.6, size = 2) +
  geom_line(data = df_pred, aes(x = Community_CUE, y = Probability), 
            color = "black", linewidth = 1.2, inherit.aes = FALSE) +
  labs(title = "Logistic Regression",
       x = "CUE Value",
       y = "Probability of Dominance (1 = Dominant)",
       color = "Community") +
  theme_minimal()
