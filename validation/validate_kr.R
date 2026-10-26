# Reference Kenward-Roger answers from R, for the frozen mixed-model cases.
#
#   Rscript validation/validate_kr.R
#
# Writes validation/results/r_kr.json. Uses lme4 (REML fit, random intercept per
# whole plot) and pbkrtest (Kenward & Roger's adjusted covariance, scaled F and
# denominator df) — the reference implementation of the method, by Halekoh &
# Hojsgaard. Kept separate from validate_r.R so this check runs whether or not
# FrF2 and rsm are installed.
#
# Keep KR_CASES here in step with validation/cases.py (the cases are frozen:
# add, never change).

suppressMessages({
  library(lme4)
  library(pbkrtest)
  library(jsonlite)
})

results_dir <- file.path("validation", "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

cases <- list(
  list(file = "kr_balanced.csv", fixed = "y ~ A * B", terms = c("A", "B", "A:B")),
  list(file = "kr_lost_runs.csv", fixed = "y ~ A * B", terms = c("A", "B", "A:B")),
  list(file = "kr_unequal.csv", fixed = "y ~ A * B", terms = c("A", "B", "A:B")),
  list(file = "kr_categorical.csv", fixed = "y ~ media + B", terms = c("media", "B")),
  list(file = "kr_low_wp_variance.csv", fixed = "y ~ A * B", terms = c("A", "B", "A:B")),
  list(file = "kr_tool_design.csv", fixed = "y ~ (temp + pH + feed)^2",
       terms = c("temp", "pH", "feed", "temp:pH", "temp:feed", "pH:feed"))
)

# Tight optimiser tolerances, so the comparison measures the method and not
# where two optimisers happened to stop on a flat REML surface.
control <- lmerControl(
  optimizer = "nloptwrap",
  optCtrl = list(xtol_abs = 1e-12, ftol_abs = 1e-14, xtol_rel = 1e-12, ftol_rel = 1e-14, maxeval = 1e5),
  check.conv.singular = "ignore"
)

kr <- lapply(cases, function(case) {
  d <- read.csv(file.path("validation", "datasets", case$file), stringsAsFactors = TRUE)
  d$whole_plot <- factor(d$whole_plot)
  model <- lmer(as.formula(paste(case$fixed, "+ (1 | whole_plot)")), data = d, REML = TRUE, control = control)

  x <- getME(model, "X")
  assign <- attr(x, "assign")
  labels <- attr(terms(model, fixed.only = TRUE), "term.labels")

  tests <- lapply(case$terms, function(term) {
    cols <- which(assign == match(term, labels))
    l <- matrix(0, nrow = length(cols), ncol = ncol(x))
    l[cbind(seq_along(cols), cols)] <- 1
    res <- KRmodcomp(model, l)$test
    list(
      term = term,
      F = res["Ftest", "stat"],
      num_df = res["Ftest", "ndf"],
      den_df = res["Ftest", "ddf"],
      scale = res["Ftest", "F.scaling"],
      p = res["Ftest", "p.value"]
    )
  })

  vc <- as.data.frame(VarCorr(model))
  list(
    case = case$file,
    n = nrow(d),
    var_group = vc$vcov[vc$grp == "whole_plot"],
    var_resid = vc$vcov[vc$grp == "Residual"],
    coef_names = colnames(x),
    coef = unname(fixef(model)),
    se_adjusted = unname(sqrt(diag(as.matrix(vcovAdj(model))))),
    se_plugin = unname(sqrt(diag(as.matrix(vcov(model))))),
    tests = tests
  )
})

payload <- list(
  source = "R",
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  packages = list(lme4 = as.character(packageVersion("lme4")),
                  pbkrtest = as.character(packageVersion("pbkrtest"))),
  kenward_roger = kr
)

write(toJSON(payload, auto_unbox = TRUE, digits = 15), file = file.path(results_dir, "r_kr.json"))
cat("wrote", file.path(results_dir, "r_kr.json"), "\n")
