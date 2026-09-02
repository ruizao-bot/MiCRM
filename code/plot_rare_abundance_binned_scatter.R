#!/usr/bin/env Rscript

# Binned raw-data scatter plot of rare-invader final abundance.
# Values at or below the survival threshold are displayed at the lower limit.

data_file <- "data/rare.csv"
output_file <- "output/pdf/rare_abundance_binned_scatter.pdf"

abundance_floor <- 1e-5
cue_cutoff <- 0.42
cue_width <- 0.01
cue_edges <- seq(cue_cutoff, 0.48, by = cue_width)
minimum_pooled_bin_count <- 50

data <- read.csv(data_file)
data <- data[
  data$Community == 3 &
    data$Origin == "Comm2" &
    data$DilutionRate %in% c(0.01, 0.10) &
    is.finite(data$Species_CUE) &
    is.finite(data$Abundance),
]

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
  Abundance ~ DilutionRate + cue_group,
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

data <- data[as.character(data$cue_group) %in% shown_categories, ]
data$x <- match(as.character(data$cue_group), shown_categories)
data$display_abundance <- pmax(data$Abundance, abundance_floor * 1.05)

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
  mar = c(5.3, 5.25, 3.6, 1.2),
  mgp = c(3.15, 0.80, 0),
  tcl = -0.35,
  family = "serif"
)

plot(
  NA,
  xlim = c(0.5, length(shown_categories) + 0.5),
  ylim = c(1e-5, 1e2),
  log = "y",
  axes = FALSE,
  xlab = "",
  ylab = ""
)

y_powers <- -5:2
abline(h = 10^y_powers, col = "#E0E0E0", lwd = 0.7)
axis(
  1,
  at = seq_along(shown_categories),
  labels = shown_categories,
  lwd = 0.85,
  lwd.ticks = 0.85
)
axis(
  2,
  at = 10^y_powers,
  labels = parse(text = paste0("10^", y_powers)),
  las = 1,
  lwd = 0.85,
  lwd.ticks = 0.85
)
box(bty = "l", lwd = 0.85)

set.seed(20260821)
for (rate in c(0.01, 0.10)) {
  style <- styles[[as.character(rate)]]
  subset <- data[abs(data$DilutionRate - rate) < 1e-12, ]
  jitter <- runif(nrow(subset), min = -0.080, max = 0.080)
  points(
    subset$x + style$offset + jitter,
    subset$display_abundance,
    pch = 21,
    cex = 0.58,
    bg = adjustcolor(style$color, alpha.f = 0.10),
    col = adjustcolor(style$color, alpha.f = 0.24),
    lwd = 0.65
  )
}

title(
  main = "Rare-species abundance after community coalescence",
  xlab = "Species-level CUE",
  ylab = "Abundance",
  font.main = 1,
  cex.main = 1.22,
  cex.lab = 1.07,
  line = 2.9
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

mtext(
  "Each point is one rare invader; abundances <= 10^-5 are displayed at the lower axis limit.",
  side = 1,
  line = 4.25,
  adj = 1,
  cex = 0.68,
  col = "#555555"
)

dev.off()
