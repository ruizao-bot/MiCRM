setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(tidyr)
# Read the CSV file into R
df <- read.csv("../data/growth rate length vs. CUE.csv")

df_filtered1 <- subset(df, Community.CUE.1 > 0)
df_filtered2 <- subset(df, Community.CUE.2 > 0)
df_filtered3 <- subset(df, Community.CUE.3 > 0)

model1<- lm(log(Community.CUE.1) ~ log(Length.r1), data = df_filtered1)
model2<- lm(log(Community.CUE.2) ~ log(Length.r2), data = df_filtered2)
model3<- lm(log(Community.CUE.3) ~ log(Length.r3), data = df_filtered3)
summary(model1)
summary(model2)
summary(model3)

p <- ggplot(df) +
  geom_point(aes(x = log(Length.r1), y = log(Community.CUE.1), color = "Community 1"), alpha = 0.6, size = 2) +
  geom_smooth(aes(x = log(Length.r1), y = log(Community.CUE.1), color = "Community 1"), method = "lm", se = FALSE) +  
  
  geom_point(aes(x = log(Length.r2), y = log(Community.CUE.2), color = "Community 2"), alpha = 0.6, size = 2) +
  geom_smooth(aes(x = log(Length.r2), y = log(Community.CUE.2), color = "Community 2"), method = "lm", se = FALSE) +
  
  geom_point(aes(x = log(Length.r3), y = log(Community.CUE.3), color = "Community 3"), alpha = 0.6, size = 2) +
  geom_smooth(aes(x = log(Length.r3), y = log(Community.CUE.3), color = "Community 3"), method = "lm", se = FALSE) +

  labs(title = "Length of Growth Rate vs. CUE",
       x = "Log of Length_r",
       y = "Log of Community CUE",
       color = "Community Type") +  
  theme_minimal()

print(p)

#####
model1<- lm(Length.r1 ~ Community.CUE.1, data = df_filtered1)
model2<- lm(Length.r2 ~ Community.CUE.2, data = df_filtered2)
model3<- lm(Length.r3 ~ Community.CUE.3, data = df_filtered3)
summary(model1)
summary(model2)
summary(model3)

p1 <- ggplot(df) +
  geom_point(aes(x = log(Community.CUE.1), y = log(Length.r1), color = "Community 1"), alpha = 0.6, size = 2) +
  geom_smooth(aes(x = log(Community.CUE.1), y = log(Length.r1), color = "Community 1"), method = "lm", se = FALSE) +  
  
  geom_point(aes(x = log(Community.CUE.2), y = log(Length.r2), color = "Community 2"), alpha = 0.6, size = 2) +
  geom_smooth(aes(x = log(Community.CUE.2), y = log(Length.r2), color = "Community 2"), method = "lm", se = FALSE) +
  
  geom_point(aes(x = log(Community.CUE.3), y = log(Length.r3), color = "Community 3"), alpha = 0.6, size = 2) +
  geom_smooth(aes(x = log(Community.CUE.3), y = log(Length.r3), color = "Community 3"), method = "lm", se = FALSE) +
  
  labs(title = "CUE vs. Length of Growth Rate ",
       x = "Log of Community CUE",
       y = "Log of Length_r",
       color = "Community Type") +  
  theme_minimal()

print(p1)
####### community CUE ##################
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
####### community growth rate ##################
df_growth_rate <- pivot_longer(df, 
                        cols = starts_with("Length.r"), 
                        names_to = "Community", 
                        values_to = "r")
df_growth_rate$Community <- gsub("Length\\.r", "Community ", df_growth_rate$Community)
ggplot(df_growth_rate, aes(x = Community, y = r, fill = Community)) +
  geom_boxplot() +
  labs(xlab("Community"),
       ylab("Growth rate (r)") ,
       color = "Community Type")
  theme_minimal() +
  theme(legend.position = "none")