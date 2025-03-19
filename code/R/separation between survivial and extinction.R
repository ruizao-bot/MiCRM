setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
# Read the CSV file into R
df_combined <- read.csv("../data/CUE_distribution.csv")
# Create a boxplot of CUE by Status
boxplot(CUE ~ Status, 
        data = df_combined , 
        main = "Boxplot of CUE by Status",
        xlab = "Status (Survival vs. Extinction)",
        ylab = "CUE",
        col = c("gray", "green"))
# Create a boxplot of CUE by Community and Status using ggplot2
ggplot(df_combined, aes(x = factor(Community), y = CUE, fill = Status)) +
  geom_jitter(aes(color = Status), width = 0.2, alpha = 0.6, size = 2) + 
  geom_boxplot() +
  labs(title = "CUE Across Communities by Status",
       x = "Community",
       y = "CUE") +
  theme_minimal()
