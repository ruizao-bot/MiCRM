# 分析community1/2的CUE与其在community3中的similarity关系
# 需要R包: tidyverse

library(tidyverse)

df <- read_csv('data/coal_resource.csv')
df$Overlap <- as.factor(df$Overlap)

# 将群落1和2的数据混合在一起
df_combined <- df %>%
  select(Seed, Overlap, CUE1, CUE2, Similarity1, Similarity2) %>%
  pivot_longer(
    cols = c(CUE1, CUE2),
    names_to = "Community",
    names_prefix = "CUE",
    values_to = "CUE"
  ) %>%
  mutate(
    Similarity = ifelse(Community == "1", Similarity1, Similarity2)
  ) %>%
  select(Seed, Overlap, CUE, Similarity)

# 显著性分析
cat("\n========== 显著性分析 ==========\n\n")

# 1. 整体模型：CUE对Similarity的主效应和与Overlap的交互作用
model_full <- lm(Similarity ~ CUE * Overlap, data = df_combined)
cat("模型1: Similarity ~ CUE * Overlap\n")
print(summary(model_full))
cat("\n")

# 2. 简化模型：仅主效应
model_simple <- lm(Similarity ~ CUE + Overlap, data = df_combined)
cat("模型2: Similarity ~ CUE + Overlap\n")
print(summary(model_simple))
cat("\n")

# 3. 模型比较
cat("模型比较（是否需要交互项）:\n")
print(anova(model_simple, model_full))
cat("\n")

# 4. 分层分析：每个Overlap水平下CUE的效应
cat("分层分析（按Overlap水平）:\n")
for (ovlp in c("0.25", "0.5", "0.75")) {
  df_sub <- df_combined %>% filter(Overlap == ovlp)
  model_sub <- lm(Similarity ~ CUE, data = df_sub)
  cat(sprintf("\nOverlap = %s:\n", ovlp))
  print(summary(model_sub)$coefficients)
  cat(sprintf("R² = %.4f, p-value = %.4e\n", 
              summary(model_sub)$r.squared, 
              summary(model_sub)$coefficients[2, 4]))
}

# 可视化：CUE vs Similarity
ggplot(df_combined, aes(x = CUE, y = Similarity, color = Overlap)) +
  geom_point(alpha=0.5) +
  geom_smooth(method = 'lm', se = TRUE) +
  theme_bw() +
  labs(title = 'CUE vs Similarity',
       x = 'Community CUE',
       y = 'Similarity')
