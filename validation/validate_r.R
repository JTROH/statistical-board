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

# ---------------------------------------------------------------------------
# Power per kind of model term
# ---------------------------------------------------------------------------
# Same cases as POWER_CASES in cases.py. Design, model matrix and power are all
# built here, independently of the tool.

power_cases <- list(
  list(family = "full", k = 3, nc = 3, order = "interaction", effect = 2.0),
  list(family = "full", k = 4, nc = 3, order = "interaction", effect = 1.5),
  list(family = "ccd-face", k = 3, nc = 3, order = "quadratic", effect = 2.0),
  list(family = "ccd-face", k = 2, nc = 4, order = "quadratic", effect = 1.5),
  list(family = "bbd", k = 3, nc = 3, order = "quadratic", effect = 2.0),
  list(family = "bbd", k = 4, nc = 3, order = "quadratic", effect = 2.0)
)

power <- lapply(power_cases, function(case) {
  k <- case$k
  vars <- paste0("x", seq_len(k))
  pts <- if (case$family == "full") {
    g <- expand.grid(rep(list(c(-1, 1)), k)); names(g) <- vars
    rbind(g, as.data.frame(matrix(0, case$nc, k, dimnames = list(NULL, vars))))
  } else if (case$family == "ccd-face") {
    d <- ccd(k, n0 = c(case$nc, 0), alpha = "faces", randomize = FALSE, oneblock = TRUE)
    as.data.frame(d)[, vars]
  } else {
    d <- bbd(k, n0 = case$nc, randomize = FALSE, block = FALSE)
    as.data.frame(d)[, vars]
  }
  pts <- as.data.frame(lapply(pts, as.numeric))
  rhs <- paste0("(", paste(vars, collapse = " + "), ")^2")
  if (case$order == "quadratic") rhs <- paste(rhs, "+", paste0("I(", vars, "^2)", collapse = " + "))
  X <- model.matrix(as.formula(paste("~", rhs)), pts)
  V <- solve(crossprod(X))
  df <- nrow(X) - ncol(X)
  crit <- qt(0.975, df)
  pw <- function(ncp) pt(crit, df, ncp, lower.tail = FALSE) + pt(-crit, df, ncp)
  se <- sqrt(diag(V))
  cols <- colnames(X)
  kind <- ifelse(cols %in% vars, "main", ifelse(grepl("^I\\(", cols), "curvature",
                 ifelse(grepl(":", cols), "interaction", "intercept")))
  coef <- ifelse(kind == "curvature", case$effect, case$effect / 2)
  powers <- pw(coef / se)
  min_of <- function(k) if (any(kind == k)) min(powers[kind == k]) else NA
  list(
    case = sprintf("power-%s-%d-c%d-%s", case$family, k, case$nc, case$order),
    n_runs = nrow(X),
    residual_df = df,
    main = min_of("main"),
    interaction = min_of("interaction"),
    curvature = min_of("curvature")
  )
})

# ---------------------------------------------------------------------------
# Blocked full factorials
# ---------------------------------------------------------------------------
# block.gen holds Yates column numbers: bit i set = factor i in the word. The
# block words are every non-empty XOR of the generators.

blocking_cases <- list(c(3, 2), c(4, 2), c(5, 2), c(5, 4), c(6, 2), c(6, 4))

blocking <- lapply(blocking_cases, function(case) {
  k <- case[1]; b <- case[2]
  d <- FrF2(nruns = 2^k, nfactors = k, blocks = b, randomize = FALSE, alias.block.2fis = FALSE)
  gens <- as.integer(unlist(design.info(d)$block.gen))
  r <- length(gens)
  words <- integer(0)
  for (m in 1:(2^r - 1)) {
    w <- 0L
    for (i in seq_len(r)) if (bitwAnd(m, bitwShiftL(1L, i - 1L)) > 0) w <- bitwXor(w, gens[i])
    words <- c(words, w)
  }
  popcount <- function(x) sum(as.integer(intToBits(x)))
  list(
    case = sprintf("2^%d in %d blocks", k, b),
    block_word_lengths = sort(vapply(words, popcount, numeric(1)))
  )
})

payload <- list(
  source = "R",
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  packages = list(FrF2 = as.character(packageVersion("FrF2")),
                  rsm = as.character(packageVersion("rsm"))),
  fractional = fractional,
  box_behnken = box_behnken,
  central_composite = central_composite,
  power = power,
  blocking = blocking
)

write(toJSON(payload, auto_unbox = TRUE, digits = 12),
      file = file.path(results_dir, "r.json"))
cat("wrote", file.path(results_dir, "r.json"), "\n")
