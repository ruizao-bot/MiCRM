#!/usr/bin/env Rscript

# Scatter-plot variant of rare-species survival by species-level CUE.
# Filtering, binning, and Wilson intervals match plot_rare_survival_redesigned.py.

data_file <- "data/rare.csv"
output_file <- "output/pdf/rare_survival_scatter.pdf"

survival_threshold <- 1e-5
cue_cutoff <- 0.42
cue_width <- 0.01
cue_edges <- seq(cue_cutoff, 0.48, by = cue_width)
minimum_bin_count <- 50
z_95 <- qnorm(0.975)

data <- read.csv(data_file)
data <- data[data$Community == 3 & data$Origin == "Comm2", ]
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

totals <- aggregate(
  survived ~ DilutionRate + cue_group,
  data = data,
  FUN = length,
  drop = TRUE
)
names(totals)[3] <- "total"
survivors <- aggregate(
  survived ~ DilutionRate + cue_group,
  data = data,
  FUN = sum,
  drop = TRUE
)
names(survivors)[3] <- "survivors"
summary_data <- merge(totals, survivors, by = c("DilutionRate", "cue_group"))
summary_data <- summary_data[summary_data$total >= minimum_bin_count, ]
summary_data$probability <- summary_data$survivors / summary_data$total

denominator <- 1 + z_95^2 / summary_data$total
center <- (
  summary_data$probability + z_95^2 / (2 * summary_data$total)
) / denominator
half_width <- z_95 * sqrt(
  summary_data$probability * (1 - summary_data$probability) / summary_data$total +
    z_95^2 / (4 * summary_data$total^2)
) / denominator
summary_data$ci_low <- center - half_width
summary_data$ci_high <- center + half_width

shown_categories <- category_labels[
  category_labels %in% as.character(summary_data$cue_group)
]
summary_data$x <- match(as.character(summary_data$cue_group), shown_categories)

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
cairo_pdf(output_file, width = 7.6, height = 5.2, family = "serif")
par(
  mar = c(6.4, 5.1, 3.7, 1.2),
  mgp = c(3.0, 0.75, 0),
  tcl = -0.35,
  las = 1,
  xaxs = "i",
  yaxs = "i",
  family = "serif"
)

plot(
  NA,
  xlim = c(0.5, length(shown_categories) + 0.5),
  ylim = c(-0.025, 1.025),
  axes = FALSE,
  xlab = "",
  ylab = ""
)

abline(h = seq(0, 1, by = 0.2), col = "#D9D9D9", lwd = 0.8)
axis(
  1,
  at = seq_along(shown_categories),
  labels = FALSE,
  lwd = 0.8,
  lwd.ticks = 0.8
)
text(
  x = seq_along(shown_categories),
  y = par("usr")[3] - 0.055,
  labels = shown_categories,
  srt = 25,
  adj = 1,
  xpd = TRUE,
  cex = 0.92
)
axis(
  2,
  at = seq(0, 1, by = 0.2),
  labels = paste0(seq(0, 100, by = 20), "%"),
  lwd = 0.8,
  lwd.ticks = 0.8
)
box(bty = "l", lwd = 0.8)

styles <- list(
  "0.01" = list(color = "#0072B2", pch = 21, offset = -0.07,
                label = "Rarity level = 0.01"),
  "0.1" = list(color = "#D55E00", pch = 22, offset = 0.07,
               label = "Rarity level = 0.10")
)

for (rate in c(0.01, 0.10)) {
  style <- styles[[as.character(rate)]]
  subset <- summary_data[abs(summary_data$DilutionRate - rate) < 1e-12, ]
  subset <- subset[order(subset$x), ]
  x <- subset$x + style$offset

  arrows(
    x0 = x,
    y0 = subset$ci_low,
    x1 = x,
    y1 = subset$ci_high,
    angle = 90,
    code = 3,
    length = 0.045,
    col = style$color,
    lwd = 1.1
  )
  points(
    x,
    subset$probability,
    pch = style$pch,
    cex = if (rate == 0.01) 1.25 else 1.12,
    bg = "white",
    col = style$color,
    lwd = 1.6
  )
}

title(
  main = "Rare-species survival after community coalescence",
  xlab = "Species-level CUE",
  ylab = "Survival proportion",
  line = 2.8,
  cex.main = 1.25,
  font.main = 1,
  cex.lab = 1.05
)

legend(
  "topleft",
  legend = vapply(styles, `[[`, character(1), "label"),
  pch = vapply(styles, `[[`, numeric(1), "pch"),
  pt.bg = "white",
  col = vapply(styles, `[[`, character(1), "color"),
  pt.cex = c(1.25, 1.12),
  pt.lwd = 1.6,
  bty = "o",
  bg = "white",
  box.col = "#BDBDBD",
  inset = 0.012,
  cex = 0.92
)

mtext(
  "CUE < 0.42 is pooled; remaining labels are upper bounds of 0.01-wide bins.",
  side = 1,
  line = 4.8,
  adj = 1,
  cex = 0.66,
  col = "#555555"
)
mtext(
  "Points are horizontally offset; error bars show 95% Wilson CIs (n >= 50).",
  side = 1,
  line = 5.5,
  adj = 1,
  cex = 0.66,
  col = "#555555"
)

dev.off()
