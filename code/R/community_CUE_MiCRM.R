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
<<<<<<< HEAD
  geom_boxplot(alpha = 0.4, outlier.shape = NA) + 
  geom_jitter(aes(fill = Community), color = "black", shape = 21, 
              size = 2.5, width = 0.15, alpha = 0.8) +  
  scale_fill_manual(values = c(
    "Comm1" = "#E41A1C",  # red
    "Comm2" = "#4DAF4A",  # green
    "Comm3" = "#377EB8"   # blue
  )) +
  labs(x = "Community", y = "CUE") +
  theme_minimal() +
  theme(legend.position = "none")


=======
  geom_boxplot() +
  labs(xlab("Community"),
       ylab("CUE") ,
       color = "Community Type")
  theme_minimal() +
  theme(legend.position = "none")
  
>>>>>>> origin/main
