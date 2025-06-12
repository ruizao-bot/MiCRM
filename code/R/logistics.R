setwd("/home/jiayi-chen/Documents/MiCRM/code")
# Read the CSV file into R
<<<<<<< HEAD
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
  geom_jitter(width = 0.0005, height = 0.05, alpha = 1, size = 2) +
  geom_line(data = df_pred, aes(x = Community_CUE, y = Probability), 
            color = "grey", linewidth = 1, alpha = 1, inherit.aes = FALSE) +
  scale_color_manual(values = c("1" = "red", "2" = "#2ca02c")) +
  labs(title = "",
       x = "CUE Value",
       y = "Probability of Dominance (1 = Dominant)",
       color = "Community") +
  theme_minimal()
=======
df_combined <- read.csv("../data/df_combined.csv")

# Fit a logistic regression model
model <- glm(Dominance ~ CUE, data = df_combined, family = binomial(link = "logit"))

# Display the summary of the model
summary(model)


# Create a sequence of CUE values for predictions
cue_seq <- seq(min(df_combined$CUE), max(df_combined$CUE), length.out = 300)

# Predict the probability of Dominance = 1 using the model
predicted <- predict(model, newdata = data.frame(CUE = cue_seq), type = "response")
colors <- c("blue", "darkred")[as.factor(df_combined$Group)]
# Plot the original data (scatter plot)
plot(df_combined$CUE, df_combined$Dominance, 
     xlab = "CUE Value", 
     ylab = "Probability of Dominance (1 = Dominant)", 
     main = "Logistic Regression", 
     pch = 19,
     col = colors)

# Add the fitted logistic regression curve (red line)
lines(cue_seq, predicted, col ="black", lwd = 2)
>>>>>>> origin/main
