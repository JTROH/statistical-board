# Reference answers from R, for the same frozen cases the tool computes.
#
#   Rscript validation/validate_r.R
#
# Writes validation/results/r.json. Uses FrF2 (fractional factorials, the
# authoritative minimum-aberration catalogue) and rsm (Box-Behnken and central
# composite designs). Both are long-established and widely used, which is the
# point: agreement with them is evidence a statistician will accept.

suppressMessages({
  library(FrF2)
  library(rsm)
  library(jsonlite)
})

results_dir <- file.path("validation", "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Fractional factorials
# ---------------------------------------------------------------------------

fractional_cases <- list(
  c(3, 1), c(4, 1), c(5, 1), c(5, 2),
  c(6, 1), c(6, 2), c(6, 3),
  c(7, 1), c(7, 2), c(7, 3), c(7, 4),
  c(8, 2), c(8, 3), c(8, 4),
  c(9, 4), c(9, 5),
  c(10, 5), c(10, 6)
)

fractional <- lapply(fractional_cases, function(case) {
  k <- case[1]; p <- case[2]
  n_runs <- 2^(k - p)
  design <- FrF2(nruns = n_runs, nfactors = k, randomize = FALSE)
  entry <- design.info(design)$catlg.entry[[1]]

  # FrF2's catalogue entry stores the word-length pattern indexed from word
  # length 1 (its print method labels the trimmed version "3plus", which is
  # easy to mistake for the raw vector). The tool reports lengths 3..7, so
  # slice to match; short vectors pad with zeros.
  wlp_raw <- as.numeric(entry$WLP)
  wlp <- wlp_raw[3:7]
  wlp[is.na(wlp)] <- 0

  list(
    case = sprintf("2^(%d-%d)", k, p),
    n_factors = k,
    n_generators = p,
    n_runs = nrow(design),
    resolution = as.numeric(entry$res),
    word_length_pattern = wlp
  )
})

# ---------------------------------------------------------------------------
# Box-Behnken
# ---------------------------------------------------------------------------

box_behnken <- lapply(c(3, 4, 5, 6, 7), function(k) {
  design <- bbd(k, n0 = 0, randomize = FALSE, block = FALSE)
  matrix_cols <- as.matrix(design[, grep("^x", names(design))])
  storage.mode(matrix_cols) <- "double"
  rows <- lapply(seq_len(nrow(matrix_cols)), function(i) as.numeric(matrix_cols[i, ]))
  list(
    case = sprintf("bbd-%d", k),
    n_factors = k,
    n_runs = nrow(matrix_cols),
    matrix = rows
  )
})

# ---------------------------------------------------------------------------
# Central composite
# ---------------------------------------------------------------------------

ccd_cases <- list(
  list(k = 2, rule = "rotatable"), list(k = 3, rule = "rotatable"),
  list(k = 4, rule = "rotatable"), list(k = 5, rule = "rotatable"),
  list(k = 2, rule = "face"), list(k = 3, rule = "face"), list(k = 4, rule = "face")
)

central_composite <- lapply(ccd_cases, function(case) {
  k <- case$k
  n_factorial <- 2^k
  alpha <- if (case$rule == "face") 1.0 else n_factorial^(1 / 4)
  list(
    case = sprintf("ccd-%d-%s", k, case$rule),
    n_factors = k,
    alpha_rule = case$rule,
    alpha = alpha,
    n_runs = n_factorial + 2 * k,
    n_factorial_points = n_factorial,
    n_axial_points = 2 * k
  )
})

payload <- list(
  source = "R",
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  packages = list(FrF2 = as.character(packageVersion("FrF2")),
                  rsm = as.character(packageVersion("rsm"))),
  fractional = fractional,
  box_behnken = box_behnken,
  central_composite = central_composite
)

write(toJSON(payload, auto_unbox = TRUE, digits = 12),
      file = file.path(results_dir, "r.json"))
cat("wrote", file.path(results_dir, "r.json"), "\n")
