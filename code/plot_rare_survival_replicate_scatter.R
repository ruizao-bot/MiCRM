#!/usr/bin/env Rscript

# Replicate-level scatter plot of rare-species survival by CUE bin.
# Each point is the survival proportion within one Seed x DilutionRate x CUE bin.

data_file <- "data/rare.csv"
output_file <- "output/pdf/rare_survival_replicate_scatter.pdf"

survival_threshold <- 1e-5
cue_cutoff <- 0.42
cue_width <- 0.01
cue_edges <- seq(cue_cutoff, 0.48, by = cue_width)
minimum_pooled_bin_count <- 50

data <- read.csv(data_file)
data <- data[
  data$Community == 3 &
    data$Origin == "Comm2" &
    data$DilutionRate %in% c(0.01, 0.10),
]
data$survived <- data$Abundance > survival_threshold

category_labels <- c(sprintf("<%.2f", cue_cutoff), sprintf("%.2f", cue_edges[-1]))
bin_index <- rep(1L, nrow(data))
above <- data$Species_CUE >= cue_cutoff
bin_index[above] <- findInterval(
  data$Species_CUE[above],
  cue_edges,
  rightmost.closed = FALSE,
  all.inside = TRUE
) + 1L
data$cue_group <- factor(category_labels[bin_index], levels = category_labels)

pooled_counts <- aggregate(
  survived ~ DilutionRate + cue_group,
  data = data,
  FUN = length,
  drop = TRUE
)
names(pooled_counts)[3] <- "pooled_count"
shown_categories <- category_labels[
  category_labels %in% as.character(
    pooled_counts$cue_group[
      pooled_counts$pooled_count >= minimum_pooled_bin_count
    ]
  )
]

replicate_data <- aggregate(
  survived ~ Seed + DilutionRate + cue_group,
  data = data,
  FUN = mean,
  drop = TRUE
)
names(replicate_data)[4] <- "survival_proportion"
replicate_data <- replicate_data[
  as.character(replicate_data$cue_group) %in% shown_categories,
]
replicate_data$x <- match(
  as.character(replicate_data$cue_group),
  shown_categories
)

styles <- list(
  "0.01" = list(
    color = "#1F77B4",
    label = "Rarity level = 0.01",
    offset = -0.115
  ),
  "0.1" = list(
    color = "#D62728",
    label = "Rarity level = 0.10",
    offset = 0.115
  )
)

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
cairo_pdf(output_file, width = 7.8, height = 5.3, family = "serif")
par(
  mar = c(4.5, 5.1, 1.0, 1.2),
  mgp = c(3.0, 0.78, 0),
  tcl = -0.35,
  family = "serif"
)

plot(
  NA,
  xlim = c(0.5, length(shown_categories) + 0.5),
  ylim = c(-0.035, 1.035),
  axes = FALSE,
  xlab = "",
  ylab = ""
)

abline(h = seq(0, 1, by = 0.2), col = "#D9D9D9", lwd = 0.75)
axis(
  1,
  at = seq_along(shown_categories),
  labels = shown_categories,
  lwd = 0.85,
  lwd.ticks = 0.85
)
axis(
  2,
  at = seq(0, 1, by = 0.2),
  labels = paste0(seq(0, 100, by = 20), "%"),
  las = 1,
  lwd = 0.85,
  lwd.ticks = 0.85
)
box(bty = "l", lwd = 0.85)

set.seed(20260821)
for (rate in c(0.01, 0.10)) {
  style <- styles[[as.character(rate)]]
  subset <- replicate_data[
    abs(replicate_data$DilutionRate - rate) < 1e-12,
  ]
  jitter <- runif(nrow(subset), min = -0.075, max = 0.075)
  points(
    subset$x + style$offset + jitter,
    subset$survival_proportion,
    pch = 21,
    cex = 0.78,
    bg = adjustcolor(style$color, alpha.f = 0.25),
    col = adjustcolor(style$color, alpha.f = 0.55),
    lwd = 0.9
  )
}

title(
  xlab = "Species-level CUE",
  ylab = "Survival proportion",
  cex.lab = 1.07,
  line = 2.8
)

legend(
  "topleft",
  legend = vapply(styles, `[[`, character(1), "label"),
  pch = 21,
  pt.bg = vapply(
    styles,
    function(style) adjustcolor(style$color, alpha.f = 0.30),
    character(1)
  ),
  col = vapply(styles, `[[`, character(1), "color"),
  pt.cex = 1.12,
  pt.lwd = 1.0,
  bty = "o",
  bg = adjustcolor("white", alpha.f = 0.94),
  box.col = "#BDBDBD",
  inset = 0.012,
  cex = 0.91
)

dev.off()
