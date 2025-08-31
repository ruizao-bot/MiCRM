setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(ggraph)
library(tidygraph)
library(igraph)
library(scales)
library(dplyr)
library(purrr)
library(stringr)
library(tibble)
library(MASS)

df <- read.csv("../data/elv_hpc_sameR0.csv")

build_network <- function(df, community, seed, quantile = 0.7) {
  # 1. 子集筛选
  df_sub <- df %>%
    filter(community_id == community, Seed == seed, Cfinal > 1e-5)
  
  if (nrow(df_sub) < 2) return(NULL)
  
  species_ids <- df_sub$species_id
  cues <- df_sub$CUE
  survivor_indices <- as.integer(str_remove(species_ids, "Sp"))  # "Sp3" → 3
  
  # 2. 将 alpha 字符串转为数值向量，并取出对应索引
  alpha_matrix <- df_sub$alpha %>%
    map(~ as.numeric(str_extract_all(.x, "-?\\d+\\.\\d+([eE][-+]?\\d+)?")[[1]])) %>%
    map(~ .x[survivor_indices]) %>%
    do.call(what = rbind)
  
  # 3. 计算相互作用权重矩阵（对称平均）
  N <- nrow(alpha_matrix)
  if (N < 2) return(NULL)
  
  weights <- matrix(0, nrow = N, ncol = N)
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      w_ij <- mean(abs(c(alpha_matrix[i, j], alpha_matrix[j, i])))
      weights[i, j] <- w_ij
      weights[j, i] <- w_ij
    }
  }
  
  # 4. 提取上三角边并筛选
  edge_idx <- which(upper.tri(weights), arr.ind = TRUE)
  edge_weights <- weights[edge_idx]
  threshold <- quantile(edge_weights, probs = quantile, na.rm = TRUE)
  valid_edges <- edge_weights > threshold
  
  # 5. 构建边数据框
  edge_list <- data.frame(
    from = species_ids[edge_idx[valid_edges, 1]],
    to   = species_ids[edge_idx[valid_edges, 2]],
    weight = edge_weights[valid_edges]
  )
  
  # 6. 构建 igraph 网络
  g <- graph_from_data_frame(edge_list, directed = FALSE,
                             vertices = data.frame(name = species_ids, cue = cues))
  
  return(g)
}

plot_network <- function(g, title = "") {
  # 将 igraph 转为 tidygraph 对象
  tg <- as_tbl_graph(g)
  
  # 转换为数据框，提取 node CUE 和 edge weight
  node_cue_range <- range(V(g)$cue, na.rm = TRUE)
  edge_weight_range <- range(E(g)$weight, na.rm = TRUE)
  
  ggraph(tg, layout = "fr") +
    geom_edge_link(aes(color = weight), width = 1) +
    scale_edge_color_gradient(low = "#D0D1E6", high = "#54278F", name = "Interaction Strength") +
    geom_node_point(aes(color = cue), size = 5) +
    scale_color_viridis_c(option = "viridis", name = "CUE", limits = node_cue_range) +
    theme_void() +
    ggtitle(title) +
    theme(legend.position = "right")
  
}

g1 <- build_network(df, community = "Comm1", seed = 50)

plot_network(g1, title = "Comm1 (Seed 50)")

########## cue and degree#############
communities <- c("Comm1", "Comm2", "Comm3")
seeds <- sort(unique(df$Seed))
# --- Analysis: CUE and Degree ---
compute_cue_and_degree <- function(df, community, seed_range) {
  cues <- c()
  degrees <- c()
  for (seed in seed_range) {
    g <- build_network(df, community, seed)
    if (!is.null(g)) {
      cues <- c(cues, as.numeric(V(g)$cue))
      degrees <- c(degrees, degree(g))
    }
  }
  tibble(CUE = cues, Degree = degrees)
}
# Extract CUE and degree for all nodes in all communities and seeds
node_stats <- map_dfr(communities, function(comm) {
  map_dfr(seeds, function(seed) {
    g <- build_network(df, community = comm, seed = seed)
    if (!is.null(g)) {
      tibble(
        Community = comm,
        Seed = seed,
        CUE = as.numeric(V(g)$cue),
        Degree = igraph::degree(g)
      )
    }
  })
})

# --- Gaussian Fit ---
gaussian <- function(x, A, mu, sigma, B) {
  A * exp(-(x - mu)^2 / (2 * sigma^2)) + B
}

# --- Main Analysis ---
df <- read_csv("../data/elv_hpc_sameR0.csv")
communities <- c("Comm1", "Comm2", "Comm3")
pal_rgb <- c("Comm1" = "#E74C3C", "Comm2" = "#2ECC71", "Comm3" = "#3498DB")
seed_range <- 51:100

# Panel plot: CUE vs Degree with Gaussian fit
library(minpack.lm)
library(ggplot2)


fits <- node_stats %>%
  filter(Degree > 0) %>%
  group_by(Community) %>%
  group_modify(~ {
    cues <- .x$CUE
    degrees <- .x$Degree
    if (length(cues) < 5) return(tibble(x = numeric(0), y = numeric(0), peak = NA))
    A0 <- max(degrees) - min(degrees)
    mu0 <- cues[which.max(degrees)]
    sig0 <- (max(cues) - min(cues)) / 6
    B0 <- min(degrees)
    tryCatch({
      fit <- nlsLM(Degree ~ gaussian(CUE, A, mu, sigma, B),
                   data = .x,
                   start = list(A = A0, mu = mu0, sigma = sig0, B = B0),
                   lower = c(0, min(cues), 1e-4, 0),
                   upper = c(Inf, max(cues), Inf, Inf),
                   control = nls.lm.control(maxiter = 5000))
      xfit <- seq(min(cues), max(cues), length.out = 200)
      yfit <- predict(fit, newdata = tibble(CUE = xfit))
      tibble(x = xfit, y = yfit, peak = coef(fit)['mu'])
    }, error = function(e) {
      message(paste("Fit failed for community", .x$Community[1]))
      return(tibble(x = numeric(0), y = numeric(0), peak = NA))
    })
  }) %>% ungroup()

peak_df <- fits %>%
  group_by(Community) %>%
  summarise(peak = first(peak), .groups = "drop")

p <- ggplot(node_stats %>% filter(Degree > 0), aes(x = CUE, y = Degree, color = Community)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_line(data = fits, aes(x = x, y = y, group = Community), color = "black", size = 1) +
  geom_vline(data = peak_df, aes(xintercept = peak), linetype = "dashed", color = "gray") +
  facet_wrap(~ Community, nrow = 1, scales = "free") +
  scale_color_manual(values = pal_rgb) +
  labs(x = "CUE", y = "Degree") +
  theme_minimal(base_size = 14) +
  theme(
    text = element_text(family = "Times New Roman"),
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 14),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA)
  )
ggsave("../results/degree.pdf",
       plot = p,
       device = cairo_pdf,
       width = 27,
       height = 8,
       units = "cm",
       dpi = 600,
       bg = "white")
