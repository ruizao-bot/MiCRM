setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(stringr)
library(patchwork)

df <- read.csv("../data/coal_sameR0_lsoda.csv")
df_select <- df %>%mutate(Status = ifelse(Abundance < 1e-5, "Extinction", "Survival"))
df_surv <- df_select %>%
  filter(Abundance > 1e-5)
# standrdise species number
species_counts <- df_surv %>%
  distinct(Seed, Community, Species_ID) %>%  # 每个种群中独立物种去重
  group_by(Seed, Community) %>%
  summarise(SpeciesCount = n(), .groups = "drop")

###TEST lsoda####
#df1 <- read.csv("../data/coal_sameR0_lsoda.csv")
#df_select1 <- df1 %>%
#  mutate(Status = ifelse(Abundance < 1e-5, "Extinction", "Survival"))
#df_surv1 <- df_select1 %>%
#  filter(Abundance > 1e-5)
## standrdise species number
#species_counts1 <- df_surv1 %>%
#  distinct(Seed, Community, Species_ID) %>%  # 每个种群中独立物种去重
#  group_by(Seed, Community) %>%
#  summarise(SpeciesCount = n(), .groups = "drop")
#write.csv(species_counts1, "../data/species_count_LSODA.csv", row.names = FALSE)
#df_stats1 <- df_surv1%>%
#  group_by(Seed, Community, Community_CUE) %>%
#  summarise(
#    Richness = n_distinct(Species_ID),
#    CUE.Var = var(Species_CUE, na.rm = TRUE),
#    .groups = "drop"
#  )

############ Rank abundance ################
# Calculate rank for each species within each Seed and Community (1 = most abundant)
df_surv <- df_surv %>%
  group_by(Seed, Community) %>%
  mutate(Rank = rank(-Abundance, ties.method = "first")) %>%
  ungroup()

library(ggplot2)

# 自定义 RGB 唯美配色
pal_rgb <- c("1" = "#E74C3C",   # 红
             "2" = "#2ECC71",   # 绿
             "3" = "#3498DB")   # 蓝

# Rank abundance
p <- ggplot(df_surv,
            aes(x     = Rank,
                y     = Abundance,
                color = factor(Community),
                group = interaction(Seed, Community))) +
  geom_line(alpha = 0.3) +
  scale_y_log10() +
  facet_wrap(~ Community, nrow = 1, scales = "free_y") +
  theme_minimal() +
  labs(x = "Rank",
       y = "Abundance (log10)",
       color = "Community") +
  scale_color_manual(values = pal_rgb)

ggsave(
  filename = "../results/rank_abund_wide.png",
  plot     = p,
  width    = 20,   # ← 适度拉宽
  height   = 4,
  dpi      = 300,
  bg       = "white"
)


##### frequency #########
df_sad <- df_surv %>%
  group_by(Community) %>%
  mutate(RelAbund = Abundance / sum(Abundance)) %>%
  ungroup()

ggplot(df, aes(x = Abundance,
               fill  = factor(Community),
               color = factor(Community))) +
  geom_histogram(position = "identity",
                 bins = 50,
                 alpha = 0.3) +
  # 生存阈值：1e-5
  geom_vline(xintercept = 1e-5,
             linetype   = "dashed",
             linewidth  = 0.7,
             colour     = "black") +
  scale_x_log10() +
  facet_wrap(~ Community, ncol = 1, scales = "free_y") +  # ← 分 3 个 panel
  theme_minimal() +
  labs(x = "relabundance",
       y = "Frequency (Histogram)",
       fill  = "Community",
       color = "Community") +
  scale_fill_manual(values  = pal_rgb) +
  scale_color_manual(values = pal_rgb)

# CUE distribution
ggplot(df_surv, 
       aes(x = Species_CUE, fill = factor(Community), color = factor(Community))) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "",
       x = "CUE", y = "Frequency",
       fill = "Community", color = "Community") +
  scale_fill_manual(values = pal_rgb) +
  scale_color_manual(values = pal_rgb)

# Survival boxplot
ggplot(df_ext, aes(x = factor(Community), y = Species_CUE, fill = StatusGroup)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.3, position = position_dodge(width = 0.8)) +  
  geom_jitter(aes(color = StatusGroup), 
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 0.1, alpha = 0.3) +
  scale_fill_manual(values = c(
    "Extinction" = "grey60",
    "Survival_1" = pal_rgb["1"],
    "Survival_2" = pal_rgb["2"],
    "Survival_3" = pal_rgb["3"]
  )) +
  scale_color_manual(values = c(
    "Extinction" = "grey60",
    "Survival_1" = pal_rgb["1"],
    "Survival_2" = pal_rgb["2"],
    "Survival_3" = pal_rgb["3"]
  )) +
  labs(x = "Community", y = "CUE", fill = "Status", color = "Status") +
  theme_minimal()

########################### logistic #####################################
library(minpack.lm)
library(dplyr)

fit_lines <- data.frame()

for (comm in c("1", "2", "3")) {
  dat <- df_surv[df_surv$Community == comm, ]
  
  if (nrow(dat) > 0) {
    model <- nlsLM(
      Abundance ~ K / (1 + exp(-r * (Species_CUE - x0))),
      data = dat,
      start = list(K = max(dat$Abundance), r = 5, x0 = median(dat$Species_CUE)),
      control = nls.control(maxiter = 200, warnOnly = TRUE)
    )
    
    cat("=== Summary for", comm, "===\n")
    print(summary(model))
    cat("\n")
    
    cue_seq <- seq(min(dat$Species_CUE), max(dat$Species_CUE), length.out = 200)
    
    pred <- data.frame(
      Species_CUE = cue_seq,
      Abundance = predict(model, newdata = data.frame(Species_CUE = cue_seq)),
      Community = comm
    )
    
    fit_lines <- bind_rows(fit_lines, pred)
  } else {
    cat("No valid data for", comm, "\n\n")
  }
}

# Logistic fit
ggplot(df_surv,
       aes(x = Species_CUE,
           y = Abundance,
           colour = factor(Community))) +
  geom_point(alpha = 0.3, size = 2) +
  
  # 拟合曲线（fit_lines 必须含 Species_CUE、Abundance、Community）
  geom_line(data = fit_lines,
            aes(x = Species_CUE,
                y = Abundance,
                colour = factor(Community)),
            size = 1.2, alpha = 0.7) +
  facet_wrap(~ Community, nrow = 1, scales = "free_y") +  # ← 按 Community 分 3 面板
  theme_minimal() +
  labs(
    x     = "Species-level CUE (log scale)",
    y     = "Abundance (log scale)",
    color = "Community"
  ) 

# log-log plot
library(dplyr)
library(ggplot2)

# 1. 数据取 log
df_log  <- df_surv  %>%
  filter(Abundance > 0, Species_CUE > 0) %>%
  mutate(logCUE = log10(Species_CUE),
         logAb  = log10(Abundance))

fit_log <- fit_lines %>%
  mutate(logCUE = log10(Species_CUE),
         logAb  = log10(Abundance))

# 2. 画图：只用 log 列，绝不再加 scale_x_log10()
ggplot(df_log,
       aes(x = logCUE, y = logAb,
           color = factor(Community))) +
  geom_point(alpha = 0.25, size = 1.8) +
  scale_color_manual(values = pal_rgb) +
  labs(x = "log₁₀ Species-level CUE",
       y = "log₁₀ Abundance",
       color = "Community") +
  theme_minimal(base_size = 13)

########## richness and species CUE Variance############
df_stats <- df_surv%>%
  group_by(Seed, Community, Community_CUE) %>%
  summarise(
    Richness = n_distinct(Species_ID),
    CUE.Var = var(Species_CUE, na.rm = TRUE),
    .groups = "drop"
  )

comm_colors <- c("1" = "red", "2" = "chartreuse3", "3" = "blue")

ggplot(df_stats,
       aes(x = CUE.Var, y = Richness,
           color = factor(Community), shape = factor(Community))) +
  geom_point(size = 2, alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, linetype = "solid", size = 1) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  scale_shape_manual(values = c(16, 17, 15), name = "Community") +
  facet_wrap(~ Community, scales = "free_x") +
  labs(
    x = expression("Species CUE Variance"),
    y = "Species Richness"
  ) +
  scale_x_log10(
    breaks  = function(lims) log_breaks(n = 4)(lims),
    minor_breaks = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title    = element_text(size = 13),
    legend.text     = element_text(size = 12),
    axis.title      = element_text(size = 15),
    axis.text       = element_text(size = 13),
    strip.text      = element_text(size = 14)
  )
for (comm in c(1, 2,3)) {
  model_var <- lm(Richness ~ CUE.Var, data = subset(df_stats, Community == comm))
  cat("\n---", comm, "---\n")
  print(summary(model_var))
}
############## Similarity ##################
library(dplyr)   # 确保 dplyr 在搜索路径前面
library(tidyr)   # pivot_wider()
library(vegan)   # vegdist()

bray_results <- data.frame()     # 若已存在则可省略

for (s in unique(df_mut$Seed)) {
  
  df_seed <- df_mut %>% 
    filter(Seed == s) %>% 
    as.data.frame()
  
  if (!all(c(1, 2, 3) %in% unique(df_seed$Community))) next
  
  # —— 宽格式矩阵，每行一个 Community ——
  comm_mat <- df_seed %>%
    dplyr::select(Community, Global_Species_ID, Abundance) %>%   # ← 唯一改动
    pivot_wider(
      names_from  = Global_Species_ID,
      values_from = Abundance,
      values_fill = list(Abundance = 0)
    )
  rownames(comm_mat) <- comm_mat$Community
  comm_mat$Community <- NULL
  
  if (nrow(comm_mat) != 3) next
  
  # —— Bray–Curtis 距离 ——
  bc     <- vegdist(comm_mat, method = "bray")
  bc_mat <- as.matrix(bc)
  
  if (!all(c("3", "1", "2") %in% rownames(bc_mat))) next
  
  d31  <- bc_mat["3", "1"]
  d32  <- bc_mat["3", "2"]
  cue1 <- unique(df_seed$Community_CUE[df_seed$Community == 1])
  cue2 <- unique(df_seed$Community_CUE[df_seed$Community == 2])
  
  bray_results <- rbind(
    bray_results,
    data.frame(
      Seed      = s,
      Bray_3vs1 = d31,
      Bray_3vs2 = d32,
      CUE_1     = cue1,
      CUE_2     = cue2,
      Sim_3vs1  = 1 - d31,
      Sim_3vs2  = 1 - d32
    )
  )
}

# 拟合线性模型
mod_1 <- lm(Sim_3vs1 ~ CUE_1, data = bray_results)
mod_2 <- lm(Sim_3vs2 ~ CUE_2, data = bray_results)

# 输出摘要
summary(mod_1)
summary(mod_2)

# Similarity plot
ggplot() +
  geom_point(data = bray_results, aes(x = CUE_1, y = Sim_3vs1), color = pal_rgb["1"], alpha = 0.4, size = 2) +
  geom_smooth(data = bray_results, aes(x = CUE_1, y = Sim_3vs1), method = "lm", se = FALSE, color = pal_rgb["1"], alpha = 0.7) +
  geom_point(data = bray_results, aes(x = CUE_2, y = Sim_3vs2), color = pal_rgb["2"], alpha = 0.4, size = 2) +
  geom_smooth(data = bray_results, aes(x = CUE_2, y = Sim_3vs2), method = "lm", se = FALSE, color = pal_rgb["2"], alpha = 0.7) +
  labs(
    x = "Community CUE",
    y = "Bray–Curtis similarity to Community 3",
    title = "",
    color = "Community"
  ) +
  theme_minimal(base_size = 14)

# Dominance plot
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
  labs(title = "",
       x = "CUE Value",
       y = "Probability of Dominance (1 = Dominant)",
       color = "Community") +
  theme_minimal()


