setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(stringr)
library(patchwork)

df <- read.csv("../data/coal_recursive_hpc.csv")

df_select <- df3 %>%
  mutate(Status = ifelse(Abundance < 1e-10, "Extinction", "Survival"))
df_surv <- df_select %>%
  filter(Abundance > 1e-10)
# standrdise species number
species_counts <- df_surv %>%
  distinct(Seed, Community, Species_ID) %>%  # 每个种群中独立物种去重
  group_by(Seed, Community) %>%
  summarise(SpeciesCount = n(), .groups = "drop")

############ Density of relative abundance ################
#ggplot(df_surv, aes(x = Abundance, fill = Community)) +
#  geom_histogram(binwidth = 0.01, position = "identity", alpha = 0.3) +
#  geom_line(data = df_line, aes(x = bin_mid, y = count, color = Community), size = 1.2) +
#  theme_minimal() +
#  labs(x = "Abundance", y = "Frequency (Samples)") +
#  scale_fill_manual(values = c("red", "chartreuse3", "blue")) +
#  scale_color_manual(values = c("red", "chartreuse3", "blue"))

##### frequency #########
df_sad <- df_surv %>%
  group_by(Community) %>%
  mutate(RelAbund = Abundance / sum(Abundance)) %>%
  ungroup()

ggplot(df, aes(x = Abundance, fill = factor(Community), color = factor(Community))) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  scale_x_log10() +
  theme_minimal() +
  labs(x = "Abundance (log-scaled axis)",
       y = "Frequency (Histogram) / Density (Curve)",
       fill = "Community", color = "Community") +
  scale_fill_manual(values = c("red", "chartreuse3", "blue")) +
  scale_color_manual(values = c("red", "chartreuse3", "blue"))


ggplot(df_surv, 
       aes(x = Abundance, fill = factor(Community), color = factor(Community))) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "",
       x = "Abundance (log-scaled axis)", y = "Frequency",
       fill = "Community", color = "Community") +
  scale_fill_manual(values = c("red", "chartreuse3", "blue")) +
  scale_color_manual(values = c("red", "chartreuse3", "blue"))
##### kernel density #########
ggplot(df_surv, aes(x = Abundance, fill = factor(Community), color = factor(Community))) +
  geom_density(alpha = 0.4, adjust = 1.5) +
  scale_x_log10() +
  labs(
    x = "Abundance (log-scaled axis)",
    y = "Density",
    fill = "Community", color = "Community"
  ) +
  theme_minimal() +
  theme(
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )
############## CUE distribution and separation between survival and extinction #############
ggplot(df_surv, 
       aes(x = Species_CUE, fill = factor(Community), color = factor(Community))) +
  geom_histogram(position = "identity", alpha = 0.3, bins = 50) +
  theme_minimal() +
  scale_x_log10() +
  labs(title = "",
       x = "CUE", y = "Frequency",
       fill = "Community", color = "Community") +
  scale_fill_manual(values = c("red", "#2ca02c", "blue")) +
  scale_color_manual(values = c("red", "#2ca02c", "blue"))

# Survival
df_ext <- df_select %>%
  mutate(StatusGroup = ifelse(Status == "Extinction", "Extinction",
                              paste0("Survival_", Community)))

ggplot(df_ext, aes(x = factor(Community), y = Species_CUE, fill = StatusGroup)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.3, position = position_dodge(width = 0.8)) +  
  geom_jitter(aes(color = StatusGroup), 
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 0.1, alpha = 0.3) +
  scale_fill_manual(values = c(
    "Extinction" = "grey60",
    "Survival_1" = "red",  # red
    "Survival_2" = "chartreuse3",  # chartreuse3
    "Survival_3" = "blue"   # blue
  )) +
  scale_color_manual(values = c(
    "Extinction" = "grey60",
    "Survival_1" = "red",
    "Survival_2" = "chartreuse3",
    "Survival_3" = "blue"
  )) +
  labs(x = "Community", y = "CUE", fill = "Status", color = "Status") +
  theme_minimal()
########### nls between CUE and abundance#######
communities <- unique(df_surv$Community)
fit_summary <- data.frame()
fit_lines <- data.frame()

for (comm in communities) {
  dat <- df_surv %>% filter(Community == comm)
  if (nrow(dat) < 10) next 

  nls_mod <- tryCatch({
    nlsLM(Abundance ~ K / (1 + exp(-r * (Species_CUE - x0))),
          data = dat,
          start = list(K = max(dat$Abundance), r = 5, x0 = median(dat$Species_CUE)),
          control = nls.control(maxiter = 200, warnOnly = TRUE))
  }, error = function(e) NULL)
  
  if (!is.null(nls_mod)) {
    # 计算 R² 和 AIC
    y <- dat$Abundance
    yhat <- predict(nls_mod)
    rss <- sum((y - yhat)^2)
    tss <- sum((y - mean(y))^2)
    pseudo_r2 <- 1 - rss / tss
    aic_val <- AIC(nls_mod)
    
    fit_summary <- bind_rows(fit_summary, data.frame(
      Community = comm,
      AIC = aic_val,
      R2 = pseudo_r2
    ))
    cue_seq <- seq(min(dat$Species_CUE), max(dat$Species_CUE), length.out = 200)
    fit_lines <- bind_rows(fit_lines, data.frame(
      Species_CUE = cue_seq,
      Abundance = predict(nls_mod, newdata = data.frame(Species_CUE = cue_seq)),
      Community = comm
    ))
  }
}

ggplot(df_surv, aes(x = Species_CUE, y = Abundance)) +
  geom_point(aes(color = factor(Community)), alpha = 0.4) +
  geom_line(data = fit_lines, aes(x = Species_CUE, y = Abundance, color = factor(Community)), size = 1.2) +
  facet_wrap(~ Community, scales = "free_x") +
  scale_color_manual(
    values = c("1" = "red", "2" = "#2ca02c", "3" = "blue")
  ) +
  theme_minimal() +
  labs(
    title = "",
    x = "Species-level CUE",
    y = "Abundance",
    color = "Community"
  )

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

# 绘图
ggplot(df_surv, aes(x = Species_CUE, y = Abundance, color = factor(Community)) )+
  geom_point(alpha = 0.7, size = 2) +
  geom_line(data = fit_lines, aes(x = Species_CUE, y = Abundance, color = Community), size = 1.2) +
  theme_minimal() +
  labs(
    x = "Species-level CUE",
    y = "Abundance",
    color = "Community"
  )
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
  scale_color_manual(values = comm_colors, name = "Community") +
  scale_shape_manual(values = c(16, 17, 15), name = "Community") +
  facet_wrap(~ Community, scales = "free_x") +
  labs(
    x = expression("Species CUE Variance"),
    y = "Species Richness"
  ) +
  scale_x_log10(
    breaks  = function(lims) log_breaks(n = 4)(lims),   # ⬅️ 每面板 4 个刻度
    minor_breaks = NULL           # 关掉小刻度（可选）
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
library(vegan)
library(tibble)
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
    select(Community, Global_Species_ID, Abundance) %>%
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


ggplot() +
  geom_point(data = bray_results, aes(x = CUE_1, y = Sim_3vs1), color = "red", alpha = 1, size = 2) +
  geom_smooth(data = bray_results, aes(x = CUE_1, y = Sim_3vs1), method = "lm", se = FALSE, color = "red") +
  
  geom_point(data = bray_results, aes(x = CUE_2, y = Sim_3vs2), color = "#2ca02c", alpha = 1, size = 2) +
  geom_smooth(data = bray_results, aes(x = CUE_2, y = Sim_3vs2), method = "lm", se = FALSE, color = "#2ca02c") +
  
  labs(
    x = "Community CUE",
    y = "Bray–Curtis similarity to Community 3",
    title = "",
    color = "Community"
  ) +
  theme_minimal(base_size = 14)

df2 <- read.csv("../data/coal_recursive.csv")
df3 <- read.csv("../data/coal_recursive_1.csv")
