pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")
# competition_analysis.R
# 分析 competition 对 communityCUE 和 communityCUE2 的影响，并比较不同竞争强度下的物种组成

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

setwd("/Users/jiayi/Desktop/micrm/master_project")
df <- read.csv("data/coal.csv")
df_surv <- df %>% filter(Abundance > 1e-5)

# 1. 按 community 聚合 competition, CUE, CUE2
comm_df <- df_surv %>%
  group_by(Seed, Community) %>%
  summarise(
    Competition = unique(Competition),
    Community_CUE = unique(Community_CUE),
    .groups = "drop"
  )
# 2.1 community level: Competition vs Community CUE
p1 <- ggplot(comm_df, aes(x = Competition, y = Community_CUE, color = factor(Community))) +
  geom_point(alpha = 0.9, shape = 16) +
  labs(x = "Community Competition", y = "Community CUE", color = "Community") +
  scale_color_manual(values = pal_rgb) +
  theme_minimal() +
  facet_grid(rows = vars(Community))
print(p1)
ggsave("results/community_competition_vs_CUE.pdf", plot = p1, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 2.2 species level: Competition vs Species CUE (by Community)
species_df <- df_surv %>%
  group_by(Community, Species_ID) %>%
  summarise(
    Competition =Species_Competition,
    Species_CUE = Species_CUE,
    Abundance = Abundance,
    .groups = "drop"
  )

p2 <- ggplot(species_df, aes(x = Competition, y = Species_CUE, color = factor(Community))) +
  geom_point(alpha = 0.6, shape = 16) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
  labs(x = "Species Competition", y = "Species CUE", color = "Community") +
  scale_color_manual(values = pal_rgb) +
  theme_minimal() +
  facet_grid(rows = vars(Community))
print(p2)
ggsave("results/species_competition_vs_CUE.pdf", plot = p2, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 2.3 species level: Competition vs Species Abundance
p3 <- ggplot(species_df, aes(x = Competition, y = Abundance, color = factor(Community))) +
  geom_point(alpha = 0.6, shape = 16) +
  labs(x = "Species Competition", y = "Species Abundance", color = "Community") +
  scale_color_manual(values = pal_rgb) +
  theme_minimal() +
  facet_grid(rows = vars(Community))
print(p3)
ggsave("results/species_competition_vs_abundance.pdf", plot = p3, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# # 3. 按 competition quantile 分组
# comm_df <- comm_df %>%
#   group_by(Community) %>%
#   mutate(
#     Comp_Level = ntile(Competition, 3),
#     Comp_Level = factor(Comp_Level, labels = c("Low", "Medium", "High"))
#   ) %>%
#   ungroup()
# 
# # 4. 合并 competition 分组回原数据
# species_comm <- df_surv %>%
#   left_join(comm_df %>% select(Seed, Community, Comp_Level), by = c("Seed", "Community"))
# 
# # 5. 统计每组下的物种组成
# species_comp <- species_comm %>%
#   group_by(Comp_Level, Species_ID) %>%
#   summarise(Freq = n(), .groups = "drop")
# 
# # 6. 可视化：不同竞争强度下物种出现频率
# p_comp_species <- ggplot(species_comp, aes(x = Species_ID, y = Freq, fill = Comp_Level)) +
#   geom_bar(stat = "identity", position = "dodge") +
#   labs(x = "Species ID", y = "Frequency", fill = "Competition Level") +
#   theme_minimal()
# 
# print(p_comp_species)
# 
# ggsave("results/species_composition_by_competition.png", plot = p_comp_species, width = 18, height = 8, units = "cm", dpi = 600, bg = "white")
# 
# # 7. 计算每个物种的 CUE_fixed（全局平均）
# species_cue_fixed <- df_surv %>%
#   group_by(Species_ID) %>%
#   summarise(CUE_fixed = mean(Species_CUE), .groups = "drop")
# 
# # 8. 计算每个物种在 High/Low 竞争下的出现频率
# species_freq_high <- species_comm %>%
#   filter(Comp_Level == "High") %>%
#   group_by(Species_ID) %>%
#   summarise(Freq_High = n(), .groups = "drop")
# species_freq_low <- species_comm %>%
#   filter(Comp_Level == "Low") %>%
#   group_by(Species_ID) %>%
#   summarise(Freq_Low = n(), .groups = "drop")
# 
# # 9. 合并并计算 Delta_Freq
# species_delta <- species_cue_fixed %>%
#   left_join(species_freq_high, by = "Species_ID") %>%
#   left_join(species_freq_low, by = "Species_ID") %>%
#   mutate(
#     Freq_High = ifelse(is.na(Freq_High), 0, Freq_High),
#     Freq_Low = ifelse(is.na(Freq_Low), 0, Freq_Low),
#     Delta_Freq = Freq_High - Freq_Low
#   )
# 
# # 10. 相关性分析
# cor_test <- cor.test(species_delta$CUE_fixed, species_delta$Delta_Freq)
# cat("\nCorrelation between CUE_fixed and Delta_Freq (High-Low):\n")
# print(cor_test)
# 
# # 11. 可视化
# p_cue_delta <- ggplot(species_delta, aes(x = CUE_fixed, y = Delta_Freq)) +
#   geom_point(size = 2, color = "#34495E", alpha = 0.7) +
#   geom_smooth(method = "lm", se = TRUE, color = "#E74C3C", linetype = "dashed") +
#   labs(x = expression(CUE[fixed]), y = expression(Delta~Frequency~(High-Low)),
#        title = "CUE_fixed vs Delta Frequency (High-Low Competition)") +
#   theme_minimal()
# print(p_cue_delta)
# ggsave("results/CUE_fixed_vs_DeltaFreq.png", plot = p_cue_delta, width = 14, height = 8, units = "cm", dpi = 600, bg = "white")
# 
# # 12. 分解 community CUE 的变化量为优势物种和稀有物种贡献
# # 计算每个 community 内丰度前 10% 的物种（优势物种）和其余物种（稀有物种）
# 
# # 先为每个 community 标记优势/稀有物种
# species_comm <- species_comm %>%
#   group_by(Seed, Community) %>%
#   mutate(
#     AbundRank = rank(-Abundance, ties.method = "first"),
#     N = n(),
#     Top10_cut = ceiling(N * 0.1),
#     Group = ifelse(AbundRank <= Top10_cut, "Dominant", "Rare")
#   ) %>%
#   ungroup()
# 
# # 计算每个 community 在不同竞争强度下，优势/稀有物种的 CUE 平均值
# cue_decomp <- species_comm %>%
#   group_by(Seed, Community, Comp_Level, Group) %>%
#   summarise(
#     Mean_CUE = mean(Species_CUE),
#     .groups = "drop"
#   )
# 
# # 计算每个 community 的整体 CUE
# cue_total <- species_comm %>%
#   group_by(Seed, Community, Comp_Level) %>%
#   summarise(
#     Community_CUE = unique(Community_CUE),
#     .groups = "drop"
#   )
# 
# # 合并 dominant/rare 贡献
# cue_wide <- cue_decomp %>%
#   tidyr::pivot_wider(names_from = Group, values_from = Mean_CUE)
# 
# cue_wide <- cue_wide %>%
#   left_join(cue_total, by = c("Seed", "Community", "Comp_Level"))
# 
# # 画图：不同竞争强度下，优势/稀有物种 CUE 及群落 CUE
# library(reshape2)
# plot_data <- cue_wide %>%
#   select(Comp_Level, Dominant, Rare, Community_CUE) %>%
#   melt(id.vars = "Comp_Level", variable.name = "Type", value.name = "CUE")
# 
# p_cue_decomp <- ggplot(plot_data, aes(x = Comp_Level, y = CUE, fill = Type)) +
#   geom_boxplot(position = position_dodge(0.8), width = 0.7, outlier.alpha = 0.2) +
#   labs(x = "Competition Level", y = "CUE", fill = "Type",
#        title = "Decomposition of Community CUE by Dominant and Rare Species") +
#   scale_fill_manual(values = c("Dominant" = "#E74C3C", "Rare" = "#3498DB", "Community_CUE" = "#2ECC71")) +
#   theme_minimal()
# print(p_cue_decomp)
# ggsave("results/CUE_decomposition_by_competition.png", plot = p_cue_decomp, width = 16, height = 8, units = "cm", dpi = 600, bg = "white")
# 
# # 相关性分析：dominant CUE vs competition
# cor_dom <- cor.test(cue_wide$Dominant, as.numeric(cue_wide$Comp_Level))
# cat("\nCorrelation between Dominant CUE and Competition Level:\n")
# print(cor_dom)
