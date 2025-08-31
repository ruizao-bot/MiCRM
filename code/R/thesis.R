setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(stringr)
library(patchwork)
library(vegan)
library(fitdistrplus)
library(minpack.lm)
library(purrr)
library(jsonlite)

# Palette
pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")
base_theme <- theme_minimal(base_size = 14) +
  theme(
    text       = element_text(family = "Times New Roman"),
    axis.text  = element_text(family = "Times New Roman", size = 14),
    axis.title = element_text(family = "Times New Roman", size = 14),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks       = element_line(color = "black", size = 0.3),
    axis.ticks.length = unit(0.15, "cm")
  )

# Data
df <- read.csv("../data/coal.csv")
df_select <- df %>% mutate(Status = ifelse(Abundance < 1e-5, "Extinction", "Survival"))
df_surv <- df_select %>% filter(Abundance > 1e-5)

# 1. Abundance Histogram
p_hist <- ggplot(df, aes(x = Abundance, fill = factor(Community), color = factor(Community))) +
  geom_histogram(position = "identity", bins = 50, alpha = 0.3) +
  geom_vline(xintercept = 1e-5, linetype = "dashed", linewidth = 0.7, colour = "black") +
  scale_x_log10() +
  facet_wrap(~ Community, ncol = 1, scales = "free_y") +
  base_theme+
  labs(x = "Abundance", y = "Frequency", fill = "Community", color = "Community") +
  scale_fill_manual(values = pal_rgb) +
  scale_color_manual(values = pal_rgb)
print(p_hist)

ggsave("/home/jiayi-chen/Documents/presentation/abund_hist.png", plot = p_hist,
       width = 21, height = 18, units = "cm", dpi = 600, bg = "white")

ggsave("../results/abundance_histogram.pdf",
       plot = p_hist,
       device = cairo_pdf,
       width = 21,
       height = 18,
       units = "cm",
       dpi = 600,
       bg = "white")

# 2. Survival Boxplot
df_ext <- df %>% 
  mutate(
    StatusGroup = ifelse(Abundance < 1e-5,
                         "Extinction",
                         paste0("Survival_", Community))
  ) %>% 
  dplyr::select(Community, Species_CUE, StatusGroup) %>% 
  mutate(
    Community   = factor(Community, levels = c(1, 2, 3)),
    StatusGroup = factor(StatusGroup,
                         levels = c("Extinction",
                                    "Survival_1",
                                    "Survival_2",
                                    "Survival_3"))
  )
p_box <- ggplot(df_ext, aes(x = factor(Community), y = Species_CUE, fill = StatusGroup)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.3, position = position_dodge(width = 0.8)) +
  base_theme +
  geom_jitter(aes(color = StatusGroup), 
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8), 
              size = 0.1, alpha = 0.3) +
  scale_fill_manual(values = c(
    Extinction  = "grey60",
    Survival_1  = "#E74C3C",
    Survival_2  = "#2ECC71",
    Survival_3  = "#3498DB"
  )) +
  scale_color_manual(values = c(
    Extinction  = "grey60",
    Survival_1  = "#E74C3C",
    Survival_2  = "#2ECC71",
    Survival_3  = "#3498DB"
  )) +
  # 添加星号标注：x = 3 表示 Community 3，y 值根据你的数据稍作调整
  annotate("text", x = 3, y = 0.29, label = "*", size = 6) +
  labs(x = "Community", y = "CUE", fill = "Status", color = "Status") +
  theme_minimal()+
  base_theme
print(p_box)
ggsave("/home/jiayi-chen/Documents/presentation/survival_boxplot.png", plot = p_box,
       width = 21, height = 10, units = "cm", dpi = 600, bg = "white")

ggsave("../results/survival_boxplot.pdf", plot = p_box, device = cairo_pdf, width = 21, height = 10, units = "cm", dpi = 600, bg = "white")

# 3. Logistic Fit (CUE vs Abundance)
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
    cue_seq <- seq(min(dat$Species_CUE), max(dat$Species_CUE), length.out = 200)
    pred <- data.frame(
      Species_CUE = cue_seq,
      Abundance = predict(model, newdata = data.frame(Species_CUE = cue_seq)),
      Community = comm
    )
    fit_lines <- bind_rows(fit_lines, pred)
  }
}
p_logistic <- ggplot(df_surv, aes(x = Species_CUE, y = Abundance, colour = factor(Community))) +
  geom_point(alpha = 0.3, size = 2) +
  geom_line(data = fit_lines, aes(x = Species_CUE, y = Abundance, colour = factor(Community)), size = 1.2, alpha = 0.7) +
  facet_wrap(~ Community, nrow = 1) +
  base_theme+
  labs(x = "Species-level CUE", y = "Abundance", color = "Community") +
  scale_color_manual(values = pal_rgb)+
  scale_x_continuous(breaks = seq(0.2, 0.4, length.out = 6))
print(p_logistic)
ggsave("../results/logistic_fit.pdf", plot = p_logistic, device = cairo_pdf, width = 21, height = 15, units = "cm", dpi = 600, bg = "white")



# 4. CUE-abundance Plot
# 设置统一 y 轴范围（log10）
y_min <- min(df_surv$Abundance[df_surv$Abundance > 0], na.rm = TRUE)
y_max <- max(df_surv$Abundance, na.rm = TRUE)

# 保存拼图结果
plots <- list()

for (comm in c("1", "2", "3")) {
  df_i <- df_surv %>% filter(Community == comm)
  fit_i <- fit_lines %>% filter(Community == comm)
  
  # 主图：CUE vs Abundance（log10 Y 轴）
  p_main <- ggplot(df_i, aes(x = Species_CUE, y = Abundance)) +
    geom_point(color = pal_rgb[comm], alpha = 0.3)  +
    scale_y_log10(limits = c(y_min, y_max)) +
    labs(x = "Species-level CUE", y = "Abundance") +
    base_theme
  
  p_hist <- ggplot(df_i, aes(x = Abundance)) +
    geom_histogram(bins = 50,
                   fill = pal_rgb[comm],
                   alpha = 0.3,
                   color = pal_rgb[comm]) +
    geom_vline(xintercept = 1e-5, linetype = "dashed") +
    scale_x_log10(limits = c(y_min, y_max)) +
    coord_flip() +  # 横过来：Abundance 成为 y 轴
    scale_y_reverse() +
    labs(x = "Frequency", y = NULL) +
    base_theme +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin = margin(5, 5, 5, 5)
    )
  
  p_combo <- p_hist + p_main + plot_layout(widths = c(1.2, 3))
  plots[[comm]] <- p_combo
}
# 纵向排列 3 个社区图
final_plot <- wrap_plots(plots, ncol = 1) & base_theme
final_plot
# 导出高清图像（A4 宽，论文可用
ggsave("/home/jiayi-chen/Documents/presentation/cue_abund.png", plot = final_plot, width = 21, height = 24, units = "cm", dpi = 600, bg = "white")
ggsave("../results/cue_abund.pdf",
       plot = final_plot,
       device = cairo_pdf,
       width = 21,      # A4 宽
       height = 24,     # 每个子图约 8cm
       units = "cm",
       dpi = 600,
       bg = "white")
############5. niche overlap vs CUE ############
p_cosine_cue <- ggplot(df, aes(x = Niche_Overlap_surv, y = Community_CUE, color = factor(Community),shape = factor(Community))) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, aes(group = factor(Community), color = factor(Community))) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  
  labs(
    x = "Uptake Cosine Similarity",
    y = "Community CUE",
    color = "Community"
  ) +
  base_theme+
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") 

print(p_cosine_cue)

ggsave("../results/cosine_similarity_vs_communityCUE.png",
       plot = p_cosine_cue,
       width = 21,
       height = 12,
       units = "cm",
       dpi = 600,
       bg = "white")



# Linear regression for each community
lm_results <- df %>%
  group_by(Community) %>%
  group_map(~ broom::tidy(lm(Community_CUE ~ Niche_Overlap_surv, data = .x)), .keep = TRUE) %>%
  bind_rows(.id = "Community")

print(lm_results)

df_survival <- df %>%
  group_by(Seed, Community) %>%
  summarise(
    Survival_Rate = mean(Abundance > 1e-5),
    Community_CUE = unique(Community_CUE),
    Niche_Overlap = unique(Niche_Overlap_surv),
    .groups = "drop"
  )
models <- df_survival %>%
  group_by(Community) %>%
  do(model = lm(Survival_Rate ~ Niche_Overlap, data = .))

# view results
lapply(models$model, summary)

p_cue_niche_surv <- ggplot(df_survival, aes(
  x = Niche_Overlap, y = Community_CUE,
  color = Survival_Rate
)) +
  geom_point(size = 1.7, alpha = 0.9) +
  geom_smooth(method = "lm", se = TRUE, color = "black", size = 0.6) +
  scale_color_viridis_c(option = "plasma", name = "Survival Rate", limits = c(0.03, 0.15)) +
  facet_wrap(~ Community, labeller = as_labeller(c(
    "1" = "Community 1", "2" = "Community 2", "3" = "Community 3"
  )), scales = "fixed") +
  labs(
    x = "Niche Overlap (Survivors)",
    y = "Community CUE"
  ) +
  base_theme +
  theme(panel.spacing = unit(1.2, "lines"))+
  scale_x_continuous(
    limits = c(0.78, 0.84),
    breaks = seq(0.78, 0.84, by = 0.02)
  )

print(p_cue_niche_surv)

ggsave("../results/communityCUE_vs_nicheOverlap_survival.png",
       plot = p_cue_niche_surv,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

############ 6. community dominance and similarity ############
# 4. Similarity Plot
df_mut <- df_surv %>%
  mutate(Global_Species_ID = case_when(
    Community == 2 ~ Species_ID + 100,
    TRUE ~ Species_ID
  ))

bray_results <- data.frame()

for (s in unique(df_mut$Seed)) {
  df_seed <- df_mut %>% filter(Seed == s) %>% as.data.frame()
  if (!all(c(1, 2, 3) %in% unique(df_seed$Community))) next
  
  comm_mat <- df_seed %>%
    dplyr::select(Community, Global_Species_ID, Abundance) %>%
    pivot_wider(
      names_from = Global_Species_ID,
      values_from = Abundance,
      values_fill = list(Abundance = 0)
    )
  rownames(comm_mat) <- comm_mat$Community
  comm_mat$Community <- NULL
  if (nrow(comm_mat) != 3) next
  
  bc <- vegdist(comm_mat, method = "bray")
  bc_mat <- as.matrix(bc)
  if (!all(c("3", "1", "2") %in% rownames(bc_mat))) next
  
  d31 <- bc_mat["3", "1"]
  d32 <- bc_mat["3", "2"]
  cue1 <- unique(df_seed$Community_CUE[df_seed$Community == 1])
  cue2 <- unique(df_seed$Community_CUE[df_seed$Community == 2])
  
  bray_results <- rbind(bray_results, data.frame(
    Seed = s,
    Bray_3vs1 = d31,
    Bray_3vs2 = d32,
    CUE_1 = cue1,
    CUE_2 = cue2,
    Sim_3vs1 = 1 - d31,
    Sim_3vs2 = 1 - d32
  ))
  
}

# 拟合线性模型
mod_1 <- lm(Sim_3vs1 ~ CUE_1, data = bray_results)
mod_2 <- lm(Sim_3vs2 ~ CUE_2, data = bray_results)

# 输出摘要
summary(mod_1)
summary(mod_2)

# (Assume bray_results is already calculated as in your code)
p_sim <- ggplot() +
  geom_point(data = bray_results, aes(x = CUE_1, y = Sim_3vs1, color = "1"), alpha = 0.4, size = 2) +
  geom_smooth(data = bray_results, aes(x = CUE_1, y = Sim_3vs1, color = "1"), method = "lm", se = TRUE, alpha = 0.7) +
  geom_point(data = bray_results, aes(x = CUE_2, y = Sim_3vs2, color = "2"), alpha = 0.4, size = 2) +
  geom_smooth(data = bray_results, aes(x = CUE_2, y = Sim_3vs2, color = "2"), method = "lm", se = TRUE, alpha = 0.7) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  labs(x = "Community CUE", y = "Bray–Curtis similarity to Community 3", color = "Community") +
  base_theme +
  theme(legend.position = "right")

print(p_sim)

ggsave("../results/pre/similarity.png", plot = p_sim, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")
ggsave("../results/similarity.pdf", plot = p_sim, device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")
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

model_all <- glm(Dominance ~ Community_CUE, data = df_comm, family = binomial)

# 构造预测输入
cue_seq <- seq(min(df_comm$Community_CUE),
               max(df_comm$Community_CUE),
               length.out = 200)

# 生成预测数据框
df_pred_all <- data.frame(
  Community_CUE = cue_seq,
  Dominance = predict(model_all,
                      newdata = data.frame(Community_CUE = cue_seq),
                      type = "response")
)

p_domin <- ggplot(df_comm, aes(x = Community_CUE, y = Dominance, color = factor(Community))) +
  geom_jitter(width = 0.0005, height = 0.05, alpha = 0.6, size = 2) +
  geom_line(data = df_pred_all,
            aes(x = Community_CUE, y = Dominance),
            color = "grey", linewidth = 0.9, inherit.aes = FALSE) +
  scale_color_manual(values = pal_rgb) +
  labs(
    x = "Community-level CUE",
    y = "Probability of Dominance (1 = Dominant)",
    color = "Community"
  ) +
  base_theme
p_domin
ggsave("../results/Dominance.pdf",
       plot = p_domin,
       device = cairo_pdf,
       width = 21,    # A4 宽
       height = 12,   # 自定义高度
       units = "cm",
       dpi = 600,
       bg = "white")

df_comm_sim <- df_comm %>%
  left_join(
    bray_results %>% dplyr::select(Seed, Sim_3vs1, Sim_3vs2),
    by = "Seed"
  ) %>%
  mutate(
    Similarity = ifelse(Community == 1, Sim_3vs1, Sim_3vs2),
    Community = factor(Community)
  )

dominance_colors <- c(
  "0" = "grey60",
  "1_1" = as.vector(pal_rgb["1"]),
  "1_2" = as.vector(pal_rgb["2"])
)


df_comm_sim <- df_comm_sim %>%
  mutate(
    DominanceGroup = case_when(
      Dominance == 0 ~ "0",
      Dominance == 1 & Community == 1 ~ "1_1",
      Dominance == 1 & Community == 2 ~ "1_2"
    )
  )

p_cue_sim_domin <- ggplot(df_comm_sim, aes(x = Community_CUE, y = Similarity, color = factor(DominanceGroup), shape = factor(Community))) +
  geom_point(size = 3, alpha = 0.8) +
  scale_color_manual(
    values = dominance_colors,
    labels = c("0" = "Not Dominant", "1_1" = "Dominant", "1_2" = "Dominant"),
    name = "Dominance"
  ) +
  scale_shape_manual(
    values = c("1" = 16, "2" = 17),
    name = "Community",
    labels = c("1" = "Community 1", "2" = "Community 2")
  ) +
  labs(
    x = "Community-level CUE",
    y = "Bray–Curtis Similarity to Community 3"
  ) +
  base_theme+
  theme(
    legend.title = element_text(size = 14, family = "Times New Roman"),
    legend.text  = element_text(size = 14, family = "Times New Roman")
  )

print(p_cue_sim_domin)
ggsave("../results/dom_sim.pdf",
       plot = p_cue_sim_domin,
       device = cairo_pdf,
       width = 21,    # A4 宽
       height = 12,   # 自定义高度
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("/home/jiayi-chen/Documents/presentation/dom_sim.png",
       plot = p_cue_sim_domin,
       width = 21,    # A4 宽
       height = 12,   # 自定义高度
       units = "cm",
       dpi = 600,
       bg = "white")

#########################8. richness ################
df_stats <- df_surv%>%
  group_by(Seed, Community, Community_CUE) %>%
  summarise(
    Richness = n_distinct(Species_ID),
    CUE.Var = var(Species_CUE, na.rm = TRUE),
    .groups = "drop"
  )

comm_colors <- c("1" = "red", "2" = "chartreuse3", "3" = "blue")


p_rich <- ggplot(df_stats, aes(x = CUE.Var, y = Richness, color = factor(Community), shape = factor(Community))) +
  geom_point(size = 2, alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, linetype = "solid", size = 1) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  facet_wrap(~ Community, scales = "free_x") +
  labs(
    x = expression("Species CUE Variance"),
    y = "Species Richness"
  ) +
  scale_x_log10(
    breaks = function(lims) 10^seq(log10(lims[1]), log10(lims[2]), length.out = 3),
    labels = scales::label_scientific(digits = 1),
    expand = expansion(mult = c(0.05, 0.05)),
    minor_breaks = NULL
  )+
  base_theme +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.spacing = unit(1.2, "lines")
  )


print(p_rich)

for (comm in c(1, 2,3)) {
  model_var <- lm(Richness ~ CUE.Var, data = subset(df_stats, Community == comm))
  cat("\n---", comm, "---\n")
  print(summary(model_var))
}
ggsave("../results/richness.pdf",
       plot = p_rich,
       device = cairo_pdf,
       width = 21,    # A4 宽
       height = 12,   # 自定义高度
       units = "cm",
       dpi = 600,
       bg = "white")


ggsave("/home/jiayi-chen/Documents/presentation/richness.png",
       plot = p_rich,
       width = 21,    # A4 宽
       height = 12,   # 自定义高度
       units = "cm",
       dpi = 600,
       bg = "white")

# Plot: Uptake variance vs Species CUE
p_uv <- ggplot(df_surv, aes(x = Species_CUE, y = UptakeVar, color = factor(Community))) +
  geom_point(alpha = 0.5, size = 1.2) +
  geom_smooth(method = "lm", se = TRUE, linetype = "solid", size = 1) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  scale_x_log10(
    breaks = function(lims) 10^seq(log10(lims[1]), log10(lims[2]), length.out = 3),
    labels = scales::label_scientific(digits = 1),
    expand = expansion(mult = c(0.05, 0.05)),
    minor_breaks = NULL
  ) +
  labs(
    x = "Uptake Variance",
    y = "Species-level CUE",
    color = "Community"
  ) +
  facet_wrap(~Community) +
  base_theme +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), name = "Community") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.spacing = unit(1.2, "lines")
  )
ggsave("../results/uptake variance.pdf",
       plot = p_uv,
       device = cairo_pdf,
       width = 21,    # A4 宽
       height = 12,   # 自定义高度
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("/home/jiayi-chen/Documents/presentation/uptake variance.png",
       plot = p_uv,
       width = 21,    # A4 宽
       height = 12,   # 自定义高度
       units = "cm",
       dpi = 600,
       bg = "white")

################### 9. community CUE vs depletion ##################

df_depletion <- df %>%
  group_by(Seed, Community) %>%
  summarise(
    Niche_Overlap = unique(Niche_Overlap_surv),
    Depletion = unique(Depletion),
    .groups = "drop"
  )

p_cue_depletion <- ggplot(df_depletion, aes(x = Niche_Overlap, y = Depletion, color = factor(Community))) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, linetype = "solid", size = 1) +
  scale_color_manual(values = pal_rgb, name = "Community") +
  labs(
    x = "Niche Overlap",
    y = "Resource Depletion",
    color = "Community"
  ) +
  base_theme

print(p_cue_depletion)

ggsave("../results/communityCUE_vs_depletion.png",
       plot = p_cue_depletion,
       width = 21,
       height = 12,
       units = "cm",
       dpi = 600,
       bg = "white")
