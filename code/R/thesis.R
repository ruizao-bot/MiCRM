setwd("/Users/jiayi/Desktop/micrm/master_project")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(scales)
library(stringr)
library(patchwork)
library(vegan)
# library(fitdistrplus)
# library(minpack.lm)
# library(purrr)
# library(jsonlite)

# Palette and theme setup
pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")
community_labels <- c("1" = "Parent 1", "2" = "Parent 2", "3" = "Coalesced")
base_theme <- theme_minimal(base_size = 12) +
  theme(
    text       = element_text(family = "Times New Roman"),
    axis.text  = element_text(family = "Times New Roman", size = 12),
    axis.title = element_text(family = "Times New Roman", size = 12),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, size = 0.3),
    axis.ticks       = element_line(color = "black", size = 0.3),
    axis.ticks.length = unit(0.15, "cm")
  )

# Load data
df <- read.csv("data/coal.csv")
df_select <- df %>% mutate(Status = ifelse(Abundance < 1e-5, "Extinction", "Survival"))
df_surv <- df_select %>% filter(Abundance > 1e-5)

# # 1. Abundance Histogram
# p_hist <- ggplot(df, aes(x = Abundance, fill = factor(Community), color = factor(Community))) +
#   geom_histogram(position = "identity", bins = 50, alpha = 0.3) +
#   geom_vline(xintercept = 1e-5, linetype = "dashed", linewidth = 0.7, colour = "black") +
#   scale_x_log10() +
#   facet_wrap(~ Community, ncol = 1, scales = "free_y") +
#   base_theme+
#   labs(x = "Abundance", y = "Frequency", fill = "Community", color = "Community") +
#   scale_fill_manual(values = pal_rgb) +
#   scale_color_manual(values = pal_rgb)
# print(p_hist)
# 
# ggsave("results/presentation/abund_hist.png", plot = p_hist,
#        width = 21, height = 18, units = "cm", dpi = 600, bg = "white")
# 
# ggsave("results/abundance_histogram.pdf",
#        plot = p_hist,
#        device = cairo_pdf,
#        width = 21,
#        height = 18,
#        units = "cm",
#        dpi = 600,
#        bg = "white")

# FIGURE 1: Species CUE vs Abundance (final_plot)
# Use all species (df), not just survivors (df_surv)
df_surv <- df_surv %>% mutate(log10_Abundance = log10(Abundance))
df <- df %>% mutate(log10_Abundance = log10(Abundance))
# Fit linear models and output diagnostics
fit_diagnostics <- data.frame()
cat("\n=== Linear Fit: log10(Abundance) ~ Species_CUE ===\n")
for (comm in c("1", "2", "3")) {
  dat <- df %>% filter(Community == comm)
  model_fit <- lm(log10_Abundance ~ Species_CUE, data = dat)
  fit_diagnostics <- rbind(fit_diagnostics, data.frame(
    Community = comm,
    R2 = summary(model_fit)$r.squared,
    Slope = coef(model_fit)[2],
    Intercept = coef(model_fit)[1],
    P_value = summary(model_fit)$coefficients[2, 4],
    N = nrow(dat)
  ))
}
print(fit_diagnostics)

# Set uniform y-axis range across communities (using all species)
y_min <- min(df$log10_Abundance, na.rm = TRUE)
y_max <- max(df$log10_Abundance, na.rm = TRUE)

# Generate plots for each community
plots <- list()

for (comm in c("1", "2", "3")) {
  df_i <- df %>% filter(Community == comm)

  # Main plot: CUE vs log10(Abundance)
  p_main <- ggplot(df_i, aes(x = Species_CUE, y = log10_Abundance)) +
    geom_point(color = pal_rgb[comm], alpha = 0.3) +
    ylim(y_min, y_max) +
    labs(x = "Species-level CUE", y = expression(log[10](Abundance))) +
    base_theme

  # Side histogram
  p_hist <- ggplot(df_i, aes(x = log10_Abundance)) +
    geom_histogram(bins = 50, fill = pal_rgb[comm], alpha = 0.3, color = pal_rgb[comm]) +
    xlim(y_min, y_max) +
    coord_flip() +
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

# Combine plots vertically for 3 communities
final_plot <- wrap_plots(plots, ncol = 1) & base_theme
final_plot

# Export high-resolution images
ggsave("results/cue_abund.png", plot = final_plot, width = 16, height = 20, units = "cm", dpi = 600, bg = "white")
ggsave("results/cue_abund.pdf",
       plot = final_plot,
       device = cairo_pdf,
       width = 21,
       height = 20,
       units = "cm",
       dpi = 600,
       bg = "white")


# FIGURE 2: Competition vs Community CUE - Separate by Community
df_comm_agg <- df_surv %>%
  group_by(Seed, Community, Competition, Community_CUE, Facilitation) %>%
  summarise(
    Species_CUE_Var = var(Species_CUE, na.rm = TRUE),
    .groups = "drop"
  )

p_comp_cue_combined <- ggplot(df_comm_agg, aes(x = Competition, y = Community_CUE,
                                                color = factor(Community))) +
  geom_point(size = 1.5, alpha = 0.4, shape = 16) +
  scale_color_manual(values = pal_rgb, labels = community_labels, name = "") +
  scale_x_continuous(
    breaks = function(x) {
      rng <- range(x)
      mid <- (rng[1] + rng[2]) / 2
      c(rng[1], mid, rng[2])
    },
    labels = function(x) sprintf("%.2f", x * 1e3)
  ) +
  facet_wrap(~ Community, ncol = 1, strip.position = "right",
             labeller = labeller(Community = community_labels)) +
  labs(
    x = expression(Competition~("*"~10^{-3})),
    y = "Community CUE"
  ) +
  base_theme +
  theme(
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_line(color = "grey92", linewidth = 0.15),
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.5),
    strip.text = element_text(family = "Times New Roman", size = 12),
    legend.position = "none"
  )
print(p_comp_cue_combined)
ggsave("results/Competition_vs_communityCUE.png",
  plot = p_comp_cue_combined,
  width = 8,
  height = 18,
  units = "cm",
  dpi = 600,
  bg = "white")
ggsave("results/Competition_vs_communityCUE.pdf",
  plot = p_comp_cue_combined,
  device = cairo_pdf,
  width = 21,
  height = 18,
  units = "cm",
  dpi = 600,
  bg = "white")
#Species competition
p_species_comp2_cue <- ggplot(df_surv, aes(x = Species_Competition2, y = Species_CUE,
                                           color = factor(Community))) +
  geom_point(size = 1.5, shape = 16, alpha = 0.6) +
  facet_wrap(~ Community, ncol = 1, strip.position = "right",
             labeller = labeller(Community = community_labels)) +
  labs(
    x = "Species Competition2",
    y = "Species CUE"
  ) +
  base_theme +
  theme(
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_line(color = "grey92", linewidth = 0.15),
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.5),
    strip.text = element_text(family = "Times New Roman", size = 12),
    legend.position = "none"
  )

print(p_species_comp2_cue)

ggsave("results/Species_Competition2_vs_CUE.png",
       plot = p_species_comp2_cue,
       width = 21,
       height = 18,
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("results/Species_Competition2_vs_CUE.pdf",
       plot = p_species_comp2_cue,
       device = cairo_pdf,
       width = 21,
       height = 18,
       units = "cm",
       dpi = 600,
       bg = "white")

# FIGURE 3: Facilitation vs Community CUE
df_comm_fac <- df_surv %>%
  group_by(Seed, Community) %>%
  summarise(
    Community_CUE = unique(Community_CUE),
    Facilitation = mean(Facilitation),
    .groups = "drop"
  )

p_fac_cue_combined <- ggplot(df_comm_fac, aes(x = Facilitation, y = Community_CUE, color = as.character(Community))) +
  geom_point(size = 2, alpha = 0.7, shape = 16) +
  facet_wrap(~Community, nrow = 1, labeller = labeller(Community = community_labels)) +
  scale_color_manual(values = pal_rgb) +
  scale_x_continuous(labels = function(x) sprintf("%.2f", x * 1e3)) +
  labs(
    x = expression(Facilitation~(x10^{-3})),
    y = "Community CUE"
  ) +
  base_theme +
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text = element_text(face = "plain", family = "Times New Roman", size = 12),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_line(color = "grey95", linewidth = 0.15)
  )

print(p_fac_cue_combined)
ggsave("results/Facilitation_vs_communityCUE.png",
  plot = p_fac_cue_combined,
  width = 21,
  height = 12,
  units = "cm",
  dpi = 600,
  bg = "white")
ggsave("results/Facilitation_vs_communityCUE.pdf",
  plot = p_fac_cue_combined,
  device = cairo_pdf,
  width = 21,
  height = 12,
  units = "cm",
  dpi = 600,
  bg = "white")

# FIGURE 4: CUE vs Bray-Curtis Similarity with Dominance (p_cue_sim_domin)
df_mut <- df_surv %>%
  mutate(Global_Species_ID = case_when(
    Community == 2 ~ Species_ID + 100,
    TRUE ~ Species_ID
  ))

# Calculate Bray-Curtis dissimilarity between communities
bray_results <- data.frame()

for (s in unique(df_mut$Seed)) {
  df_seed <- df_mut %>% filter(Seed == s) %>% as.data.frame()
  if (!all(c(1, 2, 3) %in% unique(df_seed$Community))) next
  
  # Create community matrix for Bray-Curtis calculation
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
  
  # Calculate Bray-Curtis dissimilarity
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

# Statistical analysis: Fit linear models
# mod_1 <- lm(Sim_3vs1 ~ CUE_1, data = bray_results)
# mod_2 <- lm(Sim_3vs2 ~ CUE_2, data = bray_results)
# summary(mod_1)
# summary(mod_2)

# Prepare data for plotting with dominance information
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

df_comm_sim <- df_comm %>%
  left_join(
    bray_results %>% dplyr::select(Seed, Sim_3vs1, Sim_3vs2, CUE_1, CUE_2),
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

df_diff <- bray_results %>%
  mutate(
    CUE_Diff = CUE_1 - CUE_2,
    Sim_Diff = Sim_3vs1 - Sim_3vs2
  ) %>%
  left_join(
    df_comm %>%
      filter(Community == 1) %>%
      dplyr::select(Seed, Dominant_Community),
    by = "Seed"
  ) %>%
  mutate(
    DomGroup = case_when(
      Dominant_Community == "Community 1" ~ community_labels["1"],
      Dominant_Community == "Community 2" ~ community_labels["2"],
      TRUE ~ "Neither"
    )
  )

dom_colors <- c(
  "Parent 1"  = as.vector(pal_rgb["1"]),
  "Parent 2"  = as.vector(pal_rgb["2"]),
  "Neither"   = "grey60"
)

p_cue_sim_domin <- ggplot(df_diff, aes(x = CUE_Diff, y = Sim_Diff, color = DomGroup)) +
  geom_point(size = 2.5, alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  scale_color_manual(values = dom_colors, name = "Dominant") +
  labs(
    x = expression(Delta*"CUE (Parent 1 "~-~" Parent 2)"),
    y = expression(Delta*"Similarity (Parent 1 "~-~" Parent 2)")
  ) +
  base_theme +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 12, family = "Times New Roman"),
    legend.text  = element_text(size = 11, family = "Times New Roman")
  )

print(p_cue_sim_domin)

ggsave("results/dom_sim.pdf",
       plot = p_cue_sim_domin,
       device = cairo_pdf,
       width = 21, height = 12, units = "cm", dpi = 600, bg = "white")
ggsave("results/dom_sim.png",
       plot = p_cue_sim_domin,
       width = 16, height = 10, units = "cm", dpi = 600, bg = "white")

# FIGURE 4B: CUE vs Dominance under Different Resource Overlap

df_resource <- read_csv('data/coal_resource.csv')
df_resource$Overlap <- as.factor(df_resource$Overlap)

# Compute CUE difference and similarity difference
df_diff_resource <- df_resource %>%
  mutate(
    CUE_diff = CUE1 - CUE2,
    Sim_diff = Sim_3vs1 - Sim_3vs2
  )

n_bins <- 7
cue_range <- range(df_diff_resource$CUE_diff)
breaks <- seq(cue_range[1], cue_range[2], length.out = n_bins + 1)
df_diff_resource <- df_diff_resource %>%
  mutate(
    CUE_bin = cut(CUE_diff, breaks = breaks, include.lowest = TRUE, dig.lab = 2),
    CUE_mid = (as.numeric(sub("\\[?\\(?(.+),.*", "\\1", as.character(CUE_bin))) +
               as.numeric(sub(".*,(.+)\\]", "\\1", as.character(CUE_bin)))) / 2
  )


overlap_colors <- c("0.25" = "#E74C3C", "0.5" = "#F39C12", "0.75" = "#3498DB")

p_cue_dominance_overlap <- ggplot(df_diff_resource, aes(x = reorder(CUE_bin, CUE_mid), y = Sim_diff, fill = Overlap)) +
  geom_boxplot(position = position_dodge(0.8), width = 0.7,
               alpha = 0.7, outlier.size = 1, outlier.alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  scale_fill_manual(values = overlap_colors, name = "Resource Overlap") +
  coord_cartesian(ylim = c(-0.5, 0.5)) +
  labs(
    x = expression(Delta*"CUE (Parent 1 "~-~" Parent 2)"),
    y = expression(Delta*"Similarity (Parent 1 "~-~" Parent 2)")
  ) +
  base_theme +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "bottom",
    legend.title = element_text(size = 12, family = "Times New Roman"),
    legend.text  = element_text(size = 11, family = "Times New Roman")
  )

print(p_cue_dominance_overlap)

ggsave("results/cue_dominance_overlap.pdf",
       plot = p_cue_dominance_overlap,
       device = cairo_pdf,
       width = 21, height = 12, units = "cm", dpi = 600, bg = "white")
ggsave("results/cue_dominance_overlap.png",
       plot = p_cue_dominance_overlap,
       width = 21, height = 12, units = "cm", dpi = 600, bg = "white")

# FIGURE 5: CUE vs Depletion
df_depletion <- df %>%
  group_by(Seed, Community) %>%
  summarise(
    Community_CUE = unique(Community_CUE),
    Niche_Overlap = unique(Competition),
    Depletion = unique(Depletion),
    .groups = "drop"
  )

p_depletion_cue <- ggplot(df_depletion, aes(x = Community_CUE, y = Depletion, color = factor(Community), shape = factor(Community))) +
  geom_point(size = 2, alpha = 0.7) +
  scale_color_manual(values = pal_rgb, labels = community_labels, name = "Community") +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15), labels = community_labels, name = "Community") +
  labs(x = "Community CUE", y = "Sum of Resource Residual", color = "Community") +
  base_theme

# Boxplot of Community CUE by community type
p_cue_boxplot <- ggplot(df_depletion, aes(x = factor(Community), y = Community_CUE, fill = factor(Community))) +
  geom_boxplot(alpha = 0.6, outlier.size = 1, outlier.alpha = 0.4) +
  scale_fill_manual(values = pal_rgb, labels = community_labels, name = "Community") +
  scale_x_discrete(labels = c("1" = "P1", "2" = "P2", "3" = "Coal.")) +
  labs(x = NULL, y = "Community CUE") +
  base_theme +
  theme(legend.position = "none")

# Combine: left = depletion scatter, right = CUE boxplot
p_depletion_combined <- p_depletion_cue + p_cue_boxplot +
  plot_layout(ncol = 2, widths = c(2, 1), guides = "collect") &
  theme(plot.margin = margin(5, 5, 5, 5), legend.position = "bottom")

print(p_depletion_combined)

ggsave("results/Residual_vs_CUE.pdf",
       plot = p_depletion_combined,
       device = cairo_pdf,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")
ggsave("results/Residual_vs_CUE.png",
       plot = p_depletion_combined,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

# FIGURE 6: Rare Species Invasion 
df_rare <- read.csv("data/rare.csv")
df_rare <- df_rare %>%
  mutate(survival = ifelse(C_final > 1e-5, "Survived", "Extinct"))
df_rare_filt <- df_rare %>%
  filter(DilutionRate %in% c(0.01, 0.1))


n_bins <- 20
df_rare_filt <- df_rare_filt %>%
  mutate(CUE_bin = cut(CUE, breaks = n_bins, dig.lab = 3))

# Compute survival/extinction counts per CUE bin and dilution rate
df_rare_bar <- df_rare_filt %>%
  group_by(DilutionRate, CUE_bin, survival) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(DilutionRate, CUE_bin) %>%
  mutate(total = sum(count), freq = count / total) %>%
  ungroup()

# Extract bin midpoints for ordering
df_rare_bar <- df_rare_bar %>%
  mutate(
    CUE_mid = as.numeric(sub("\\((.+),.*", "\\1", as.character(CUE_bin))) +
              (as.numeric(sub(".*,(.+)\\]", "\\1", as.character(CUE_bin))) -
               as.numeric(sub("\\((.+),.*", "\\1", as.character(CUE_bin)))) / 2
  )

# Panel labels
dilution_labels <- c("0.01" = "Rarity Level = 0.01", "0.1" = "Rarity Level = 0.1")

# Stacked bar plot: survival/extinction counts by CUE bin
p_rare_bar <- ggplot(df_rare_bar, aes(x = reorder(CUE_bin, CUE_mid), y = count, fill = survival)) +
  geom_col(position = "stack", width = 0.85) +
  facet_wrap(~ DilutionRate, ncol = 1, labeller = labeller(DilutionRate = dilution_labels)) +
  scale_fill_manual(
    values = c("Survived" = "#2ECC71", "Extinct" = "#E74C3C"),
    name = "Outcome"
  ) +
  labs(
    x = "Species-level CUE",
    y = "Species Count"
  ) +
  base_theme +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    strip.text = element_text(size = 12, family = "Times New Roman"),
    legend.title = element_text(size = 12, family = "Times New Roman"),
    legend.text  = element_text(size = 11, family = "Times New Roman")
  )

print(p_rare_bar)

ggsave("results/rare_survival.pdf",
       plot = p_rare_bar,
       device = cairo_pdf,
       width = 21,
       height = 14,
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("results/survival_by_dilution.png",
       plot = p_rare_bar,
       width = 16,
       height = 14,
       units = "cm",
       dpi = 600,
       bg = "white")

