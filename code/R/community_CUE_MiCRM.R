setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(tidyr)
# Read the CSV file into R
df <- read.csv("../data/df_results.csv")
df_long <- pivot_longer(df, 
                        cols = starts_with("Community.CUE"), 
                        names_to = "Community", 
                        values_to = "CUE")
df_long$Community <- gsub("Community\\.CUE\\.", "Community ", df_long$Community)

ggplot(df_long, aes(x = Community, y = CUE, fill = Community)) +
  geom_boxplot() +
  labs(xlab("Community"),
       ylab("CUE") ,
       color = "Community Type")
  theme_minimal() +
  theme(legend.position = "none")
  