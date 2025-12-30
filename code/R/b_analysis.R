#!/usr/bin/env Rscript
# b_analysis.R (simplified, ggplot2-based output like thesis.R)
library(ggplot2)
library(dplyr)
library(Cairo)
library(patchwork)
# Read data and set output directory
df <- read.csv("/Users/jiayi/Desktop/micrm/master_project/results/b_analysis.csv")
results_dir <- "/Users/jiayi/Desktop/micrm/master_project/results"

cat("Data loaded. Rows:", nrow(df), ", Columns:", ncol(df), "\n\n")

# Palette and theme (match thesis.R styling)
pal_rgb <- c("1" = "#E74C3C",   # 红
                               "2" = "#2ECC71",   # 绿
                               "3" = "#3498DB")   # 蓝

base_theme <- theme_minimal(base_size = 12) +
     theme(
          text       = element_text(family = "Times New Roman", size = 12),
          axis.text  = element_text(family = "Times New Roman", size = 12),
          axis.title = element_text(family = "Times New Roman", size = 12),
          legend.text = element_text(family = "Times New Roman", size = 12),
          legend.title = element_text(family = "Times New Roman", size = 12),
          plot.title = element_text(family = "Times New Roman", size = 12),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border     = element_rect(color = "black", fill = NA, size = 0.5),
          axis.ticks       = element_line(color = "black", size = 0.3)
     )

# Helper: reshape wide columns (1/2/3) to long for plotting
make_long <- function(df, prefix_x, prefix_y, names = c("1","2","3")){
     n <- nrow(df)
     data.frame(
          Community = factor(rep(names, each = n), levels = names),
          x = c(df[[paste0(prefix_x, names[1])]], df[[paste0(prefix_x, names[2])]], df[[paste0(prefix_x, names[3])]]),
          y = c(df[[paste0(prefix_y, names[1])]], df[[paste0(prefix_y, names[2])]], df[[paste0(prefix_y, names[3])]])
     )
}

# 1) Facilitation vs CUE (ggplot) ------------------------------------------
feed_df <- make_long(df, prefix_x = "C_feed", prefix_y = "CUE")

p_feed <- ggplot(feed_df, aes(x = x, y = y, color = Community, shape = Community)) +
     geom_point(alpha = 0.8, size = 1.8) +
     scale_color_manual(values = pal_rgb) +
     scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
     labs(x = "Facilitation", y = "CUE",color = "Community", shape = "Community") +
     base_theme

ggsave(filename = file.path(results_dir, "CUE_vs_Facilitation.pdf"), plot = p_feed,
                device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 2) Competition vs CUE (ggplot) --------------------------------------------
comp_df <- make_long(df, prefix_x = "Competition", prefix_y = "CUE")

p_comp <- ggplot(comp_df, aes(x = x, y = y, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1"=16,"2"=17,"3"=15)) +
  scale_x_continuous(
    breaks = pretty(range(comp_df$x, na.rm = TRUE), n = 4),
    limits = range(comp_df$x, na.rm = TRUE)
  ) +
  labs(x = "Competition", y = "CUE", color = "Community", shape = "Community") +
  base_theme

p_comp
ggsave(filename = file.path(results_dir, "CUE_vs_Competition.pdf"), plot = p_comp,
                device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")


p_feed2 <- p_feed + labs(title = "")
p_comp2 <- p_comp + 
  scale_x_continuous(breaks = pretty(comp_df$x, n = 3))

p_combined <- (p_feed2 + p_comp2) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(
    legend.position = "right",          # 图例放右侧
  )

p_combined


ggsave(
  filename = file.path(results_dir, "CUE_vs_Facilitation_Competition.pdf"),
  plot = p_combined,
  device = cairo_pdf, width = 21, height = 10, units = "cm",
  dpi = 600, bg = "white"
)


# Keep regression summaries (computed but not plotted)
lm_f1 <- lm(CUE1 ~ C_feed1, data=df)
lm_f2 <- lm(CUE2 ~ C_feed2, data=df)
lm_f3 <- lm(CUE3 ~ C_feed3, data=df)

lm_c1 <- lm(CUE1 ~ Competition1, data=df)
lm_c2 <- lm(CUE2 ~ Competition2, data=df)
lm_c3 <- lm(CUE3 ~ Competition3, data=df)

cat("--- Facilitation regressions ---\n")
cat("Community 1:\n"); print(summary(lm_f1))
cat("Community 2:\n"); print(summary(lm_f2))
cat("Community 3:\n"); print(summary(lm_f3))

cat("--- Competition regressions ---\n")
cat("Community 1:\n"); print(summary(lm_c1))
cat("Community 2:\n"); print(summary(lm_c2))
cat("Community 3:\n"); print(summary(lm_c3))

# Check column names to debug
cat("\nColumn names in df:\n")
print(colnames(df))

# 3) b vs Competition (ggplot) --------------------------------------------
# Create long format manually since b column doesn't have 1/2/3 suffix
comp_b_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  b = rep(df$b, 3),
  Competition = c(df$Competition1, df$Competition2, df$Competition3)
)

p_b_comp <- ggplot(comp_b_df, aes(x = b, y = Competition, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "b", y = "Competition", color = "Community", shape = "Community") +
  base_theme

print(p_b_comp)

ggsave(filename = file.path(results_dir, "b_vs_Competition.pdf"), plot = p_b_comp,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 4) b vs Cooperation/Facilitation (ggplot) -------------------------------
coop_b_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  b = rep(df$b, 3),
  C_feed = c(df$C_feed1, df$C_feed2, df$C_feed3)
)

p_b_coop <- ggplot(coop_b_df, aes(x = b, y = C_feed, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "b", y = "Cooperation", color = "Community", shape = "Community") +
  base_theme

print(p_b_coop)

ggsave(filename = file.path(results_dir, "b_vs_Facilitation.pdf"), plot = p_b_coop,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# 5) Combined b plots (competition and cooperation side by side) ----------
p_b_combined <- (p_b_comp + p_b_coop) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

print(p_b_combined)

ggsave(filename = file.path(results_dir, "b_vs_Competition_Facilitation.pdf"), 
       plot = p_b_combined,
       device = cairo_pdf, width = 28, height = 12, units = "cm", dpi = 600, bg = "white")


# 6) Four-panel combined figure (b vs competition, b vs facilitation, competition vs CUE, facilitation vs CUE)
# Arrange as 2x2 and collect a single legend on the right
p_top_left <- p_b_comp + labs(title = "")
p_top_right <- p_b_coop + labs(title = "")
p_bottom_left <- p_comp + labs(title = "")
p_bottom_right <- p_feed + labs(title = "")

p_4panel <- (p_top_left | p_top_right) / (p_bottom_left | p_bottom_right) +
  plot_layout(guides = "collect") &
  theme(legend.position = "right")

print(p_4panel)

ggsave(filename = file.path(results_dir, "b_comp_coop_CUE.pdf"),
  plot = p_4panel, device = cairo_pdf, width = 28, height = 24, units = "cm",
  dpi = 600, bg = "white")

# 7) b vs Community CUE ----------------------------------------------------
# Create long format for b vs CUE
cue_b_df <- data.frame(
  Community = factor(rep(c("1", "2", "3"), each = nrow(df)), levels = c("1", "2", "3")),
  b = rep(df$b, 3),
  CUE = c(df$CUE1, df$CUE2, df$CUE3)
)

p_b_cue <- ggplot(cue_b_df, aes(x = b, y = CUE, color = Community, shape = Community)) +
  geom_point(alpha = 0.8, size = 1.8) +
  scale_color_manual(values = pal_rgb) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15)) +
  labs(x = "b", y = "Community CUE", 
       color = "Community", shape = "Community") +
  base_theme

print(p_b_cue)

ggsave(filename = file.path(results_dir, "b_vs_CUE.pdf"), plot = p_b_cue,
       device = cairo_pdf, width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

ggsave(filename = file.path(results_dir, "presentation/b_vs_CUE.png"), plot = p_b_cue,
       width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# Regression analysis for b vs CUE
cat("\n--- b vs CUE regressions ---\n")
lm_b_cue1 <- lm(b ~ CUE1, data = df)
lm_b_cue2 <- lm(b ~ CUE2, data = df)
lm_b_cue3 <- lm(b ~ CUE3, data = df)

cat("Community 1:\n"); print(summary(lm_b_cue1))
cat("Community 2:\n"); print(summary(lm_b_cue2))
cat("Community 3:\n"); print(summary(lm_b_cue3))

