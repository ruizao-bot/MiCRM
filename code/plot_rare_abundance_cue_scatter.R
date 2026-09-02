#!/usr/bin/env Rscript

# Raw-data scatter plot of rare-invader abundance against species-level CUE.
# Each point is one surviving Comm2 species observed after coalescence.

data_file <- "data/rare.csv"
output_file <- "output/pdf/rare_abundance_cue_scatter.pdf"
survival_threshold <- 1e-5

data <- read.csv(data_file)
data <- data[
  data$Community == 3 &
    data$Origin == "Comm2" &
    data$DilutionRate %in% c(0.01, 0.10) &
    is.finite(data$Species_CUE) &
    is.finite(data$Abundance) &
    data$Abundance > survival_threshold,
]

styles <- list(
  "0.01" = list(
    color = "#5B8FF9",
    label = "Rarity level = 0.01",
    offset = 0.00045
  ),
  "0.1" = list(
    color = "#F4664A",
    label = "Rarity level = 0.10",
    offset = -0.00045
  )
)

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
cairo_pdf(output_file, width = 8.2, height = 5.4, family = "serif")
par(
  mar = c(5.0, 5.2, 3.6, 1.2),
  mgp = c(3.1, 0.78, 0),
  tcl = -0.35,
  family = "serif"
)

plot(
  NA,
  xlim = c(0.27, 0.48),
  ylim = c(1e-5, 1e2),
  log = "y",
  axes = FALSE,
  xlab = "",
  ylab = ""
)

x_ticks <- seq(0.28, 0.48, by = 0.04)
y_powers <- -5:2
axis(1, at = x_ticks, labels = sprintf("%.2f", x_ticks), lwd = 0.9, lwd.ticks = 0.9)
axis(
  2,
  at = 10^y_powers,
  labels = parse(text = paste0("10^", y_powers)),
  las = 1,
  lwd = 0.9,
  lwd.ticks = 0.9
)
box(bty = "l", lwd = 0.9)

# Draw the larger-rarity treatment first so neither group systematically
# obscures the other in dense regions.
for (rate in c(0.10, 0.01)) {
  style <- styles[[as.character(rate)]]
  subset <- data[abs(data$DilutionRate - rate) < 1e-12, ]
  points(
    subset$Species_CUE + style$offset,
    subset$Abundance,
    pch = 21,
    cex = 0.88,
    bg = adjustcolor(style$color, alpha.f = 0.20),
    col = adjustcolor(style$color, alpha.f = 0.54),
    lwd = 1.0
  )
}

title(
  main = "Rare-species abundance after community coalescence",
  xlab = "Species-level CUE",
  ylab = "Final abundance",
  font.main = 1,
  cex.main = 1.22,
  cex.lab = 1.08,
  line = 2.8
)

legend(
  "topleft",
  legend = vapply(styles, `[[`, character(1), "label"),
  pch = 21,
  pt.bg = vapply(
    styles,
    function(style) adjustcolor(style$color, alpha.f = 0.32),
    character(1)
  ),
  col = vapply(styles, `[[`, character(1), "color"),
  pt.cex = 1.15,
  pt.lwd = 1.1,
  bty = "o",
  bg = adjustcolor("white", alpha.f = 0.94),
  box.col = "#BDBDBD",
  inset = 0.012,
  cex = 0.92
)

mtext(
  "Each point is one surviving rare invader (final abundance > 10^-5); groups are slightly offset horizontally for clarity.",
  side = 1,
  line = 4.0,
  adj = 1,
  cex = 0.70,
  col = "#555555"
)

dev.off()
