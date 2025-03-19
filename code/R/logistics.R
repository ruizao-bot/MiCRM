setwd("/home/jiayi-chen/Documents/MiCRM/code")
# Read the CSV file into R
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
