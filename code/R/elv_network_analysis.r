# elv_network_analysis.R
# ------------------------------------------------------------
# Translate the original Python "testnetwork.py" into R.
# The script reconstructs species‑interaction networks, derives
# summary metrics (degree, bidirectional strength, CUE, …) and
# generates the same exploratory plots.
# ------------------------------------------------------------
# Author: Jiayi (R translation via ChatGPT‑o3)
# ------------------------------------------------------------

# ==== Packages ==============================================================
# Install any missing package once
# install.packages(c("tidyverse","igraph","ggraph","fitdistrplus",
#                    "MASS","ggrepel","scales","broom"))
 
library(tidyverse)    # dplyr, ggplot2, etc.
library(igraph)       # network handling
library(ggraph)       # tidygraph/ggplot2 network plotting
library(fitdistrplus) # distribution fitting (log‑normal, gamma)
library(MASS)         # for fitdistr() fallback
library(scales)       # pretty axis helpers
library(broom)        # turn model objects into tibbles
library(gridExtra)
library(readr)
setwd("/home/jiayi-chen/Documents/MiCRM/code")
data_path   <- "../data/elv_hpc_sameR0.csv"
communities <- c("Comm1", "Comm2", "Comm3")
pal_rgb     <- c(Comm1 = "#E74C3C", Comm2 = "#2ECC71", Comm3 = "#3498DB")
seed_range  <- 51:100

df <- read_csv(data_path, show_col_types = FALSE)

# --- Helper for parsing alpha column ---
clean_alpha <- function(x) {
  x %>%
    str_replace_all("[\n]", " ") %>%
    str_replace_all("[\\[\\]]", " ") %>%
    str_squish()
}
parse_alpha_vec <- function(alpha_str) {
  vals <- str_split(clean_alpha(alpha_str), "[ ,]+", simplify = TRUE)
  as.numeric(vals[vals != ""])
}

# --- 1. Plot one representative network per community (seed 52) ---
for (comm in communities) {
  df_sub <- df %>%
    filter(Seed == 52, community_id == comm, Cfinal > 1e-10)
  if (nrow(df_sub) < 2) next
  species_ids <- df_sub$species_id
  cues        <- df_sub$CUE
  survivor_idx <- as.integer(str_remove(species_ids, "Sp")) - 1
  alpha_mat <- map(df_sub$alpha, ~{
    vec <- parse_alpha_vec(.x)
    # Defensive: only keep if length matches length(survivor_idx)
    if(length(vec) == length(survivor_idx)) {
      vec[survivor_idx]
    } else {
      rep(NA, length(survivor_idx))
    }
  }) %>% reduce(rbind)
  
  # Defensive: skip if not square
  if(nrow(alpha_mat) != ncol(alpha_mat)) next
  comb <- combn(seq_along(species_ids), 2)
  weights <- map2_dbl(comb[1, ], comb[2, ], ~{
    i <- .x; j <- .y
    (abs(alpha_mat[i, j]) + abs(alpha_mat[j, i])) / 2
  })
  if (length(weights) == 0) next
  threshold <- quantile(weights, 0.7, na.rm = TRUE)
  g <- make_empty_graph(directed = FALSE) %>%
    add_vertices(n = length(species_ids), name = species_ids, cue = cues)
  for (k in seq_len(ncol(comb))) {
    i <- comb[1, k]; j <- comb[2, k]
    w <- (abs(alpha_mat[i, j]) + abs(alpha_mat[j, i])) / 2
    if (!is.na(w) && w > threshold)
      g <- add_edges(g, c(species_ids[i], species_ids[j]), attr = list(weight = w))
  print(
    ggraph(g, layout = "fr") +
      geom_edge_link(aes(width = weight, colour = weight), show.legend = FALSE) +
      scale_edge_width(range = c(0.2, 2)) +
      geom_node_point(aes(colour = cue), size = 3) +
      scale_colour_viridis_c(option = "C") +
      theme_void() +
      ggtitle(paste0("Network of ", comm, " (Seed 52)"))
  )
  }
}

# --- 2. Degree histograms across seeds ---
deg_plots <- list()
for (comm in communities) {
  degs <- c()
  for (seed in seed_range) {
    df_sub <- df %>%
      filter(Seed == seed, community_id == comm, Cfinal > 1e-10)
    if (nrow(df_sub) < 2) next
    species_ids <- df_sub$species_id
    cues        <- df_sub$CUE
    survivor_idx <- as.integer(str_remove(species_ids, "Sp")) - 1
    alpha_mat <- map(df_sub$alpha, ~{
      vec <- parse_alpha_vec(.x)
      vec[survivor_idx]
    }) %>% reduce(rbind)
    comb <- combn(seq_along(species_ids), 2)
    weights <- map2_dbl(comb[1, ], comb[2, ], ~{
      i <- .x; j <- .y
      (abs(alpha_mat[i, j]) + abs(alpha_mat[j, i])) / 2
    })
    if (length(weights) == 0) next
    threshold <- quantile(weights, 0.7, na.rm = TRUE)
    g <- make_empty_graph(directed = FALSE) %>%
      add_vertices(n = length(species_ids), name = species_ids, cue = cues)
    for (k in seq_len(ncol(comb))) {
      i <- comb[1, k]; j <- comb[2, k]
      w <- (abs(alpha_mat[i, j]) + abs(alpha_mat[j, i])) / 2
      if (w > threshold)
        g <- add_edges(g, c(species_ids[i], species_ids[j]), attr = list(weight = w))
    }
    degs <- c(degs, degree(g))
  }
  deg_plots[[comm]] <- tibble(degree = degs) %>%
    ggplot(aes(degree)) +
    geom_histogram(bins = 30, fill = pal_rgb[comm], colour = NA) +
    scale_y_log10() +
    ggtitle(comm) +
    theme_minimal()
}
gridExtra::grid.arrange(grobs = deg_plots, nrow = 1)

# --- 3. Interaction strength histograms across seeds ---
str_plots <- list()
for (comm in communities) {
  strengths <- c()
  for (seed in seed_range) {
    df_sub <- df %>%
      filter(Seed == seed, community_id == comm, Cfinal > 1e-10)
    if (nrow(df_sub) < 2) next
    species_ids <- df_sub$species_id
    cues        <- df_sub$CUE
    survivor_idx <- as.integer(str_remove(species_ids, "Sp")) - 1
    alpha_mat <- map(df_sub$alpha, ~{
      vec <- parse_alpha_vec(.x)
      vec[survivor_idx]
    }) %>% reduce(rbind)
    comb <- combn(seq_along(species_ids), 2)
    weights <- map2_dbl(comb[1, ], comb[2, ], ~{
      i <- .x; j <- .y
      (abs(alpha_mat[i, j]) + abs(alpha_mat[j, i])) / 2
    })
    strengths <- c(strengths, weights)
  }
  fit <- fitdist(strengths, "lnorm")
  str_plots[[comm]] <- tibble(strength = strengths) %>%
    ggplot(aes(strength)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = pal_rgb[comm], alpha = .4) +
    stat_function(fun = dlnorm, args = list(meanlog = fit$estimate["meanlog"], sdlog = fit$estimate["sdlog"]),
                  colour = "black", linetype = 2) +
    scale_x_continuous() +
    scale_y_log10() +
    ggtitle(comm) +
    theme_minimal()
}
gridExtra::grid.arrange(grobs = str_plots, nrow = 1)

# --- 4. CUE vs Degree scatter per species with LM ---
cue_deg_plots <- list()
for (comm in communities) {
  cues <- c()
  degs <- c()
  for (seed in seed_range) {
    df_sub <- df %>%
      filter(Seed == seed, community_id == comm, Cfinal > 1e-10)
    if (nrow(df_sub) < 2) next
    species_ids <- df_sub$species_id
    cues_sub    <- df_sub$CUE
    survivor_idx <- as.integer(str_remove(species_ids, "Sp")) - 1
    alpha_mat <- map(df_sub$alpha, ~{
      vec <- parse_alpha_vec(.x)
      vec[survivor_idx]
    }) %>% reduce(rbind)
    comb <- combn(seq_along(species_ids), 2)
    weights <- map2_dbl(comb[1, ], comb[2, ], ~{
      i <- .x; j <- .y
      (abs(alpha_mat[i, j]) + abs(alpha_mat[j, i])) / 2
    })
    threshold <- quantile(weights, 0.7, na.rm = TRUE)
    g <- make_empty_graph(directed = FALSE) %>%
      add_vertices(n = length(species_ids), name = species_ids, cue = cues_sub)
    for (k in seq_len(ncol(comb))) {
      i <- comb[1, k]; j <- comb[2, k]
      w <- (abs(alpha_mat[i, j]) + abs(alpha_mat[j, i])) / 2
      if (w > threshold)
        g <- add_edges(g, c(species_ids[i], species_ids[j]), attr = list(weight = w))
    }
    cues <- c(cues, cues_sub)
    degs <- c(degs, degree(g))
  }
  cue_deg_plots[[comm]] <- tibble(cue = cues, degree = degs) %>%
    ggplot(aes(cue, degree)) +
    geom_point(colour = pal_rgb[comm], alpha = .5, size = 1.2) +
    geom_smooth(method = "lm", colour = "black", se = TRUE) +
    ggtitle(comm) +
    theme_minimal()
}
gridExtra::grid.arrange(grobs = cue_deg_plots, nrow = 1)
