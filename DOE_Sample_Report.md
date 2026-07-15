# DoE Optimization of Cell-Culture Titre: Best Condition and Why It Is Not Yet the True Optimum

*Read the **Conclusion** for the direct answer. **Methods** explains why a linear response-surface model was chosen and where it breaks down. The **Appendix** holds the full coefficient tables and diagnostics for anyone checking the numbers.*

## Question and decision rule
This is a response-surface optimization, not a single significance test: model titre (g/L) as a function of four medium factors — glucose, glutamine, serum, and sodium bicarbonate — and find the settings that maximize predicted titre. For each model term the null is "this factor has no effect" (coefficient = 0) versus a two-sided alternative; the overall model null is "no factor matters." Decision threshold alpha = 0.05.

## Methods — which model, and why
The four factors are continuous and the design is a 16-corner two-level full factorial plus 3 replicated center points (19 runs total), so ordinary least-squares **regression** is the correct engine (the two-group t-test / Mann-Whitney commands do not apply). We fit, in order:
- a **full model** with all four main effects and all six two-factor interactions;
- a **reduced main-effects model** (interactions dropped) — the primary model;
- a **curvature check** adding a center-vs-corner term.

The design is perfectly orthogonal — every variance-inflation factor (VIF, a measure of how much predictors overlap and destabilize each other) equals 1.0, so the four factor estimates are independent, stable, and identical across all three models. No run was influential: the largest Cook's distance (a measure of how much a single run pulls the fit) was 0.162 at the all-high corner (run_order 10), comfortably under the 0.211 flag; leverage was likewise below its 0.526 threshold. Run order showed no serial correlation (Durbin–Watson ≈ 1.96, i.e. residuals are not trending run-to-run).

Because we test ~11 coefficients (4 main effects + 6 interactions + 1 curvature term), we note a multiple-comparison concern (running many tests inflates the chance of a false positive). Under a Holm correction the four main effects (p ≈ 3×10⁻⁵) survive trivially, so it changes none of the substantive conclusions; we flag it for completeness.

**Why the linear model is the right *description* but the wrong *optimizer*:** two facts, developed in Results, force us to report a best *tested* condition rather than a mathematical peak — (a) every factor's best setting lands at the edge of the tested range, and (b) the center points sit systematically above the linear prediction, i.e. the true surface is curved and a flat plane has no peak.

## Results
**Overall fit.** The reduced main-effects model is highly significant (overall F-test p = 3.4×10⁻⁷; R² = 0.910, adjusted R² = 0.885 — the four factors together explain about 91% of the variation in titre). Adding the six interactions did not improve it (adjusted R² fell to 0.866).

**The four main effects — all large, all real.** In coded units (−1 = low, +1 = high), each factor raises titre by ~4.5 g/L per step, and each is decisive:
- glucose: +4.48 g/L (p = 3.8×10⁻⁵, 95% CI [2.85, 6.10])
- glutamine: +4.56 g/L (p = 3.1×10⁻⁵, 95% CI [2.94, 6.19])
- serum: +4.53 g/L (p = 3.4×10⁻⁵, 95% CI [2.90, 6.15])
- sodium bicarbonate: +4.51 g/L (p = 3.6×10⁻⁵, 95% CI [2.88, 6.13])

Each factor is a very large effect on its own: partial η² ≈ 0.72 (partial η² is the share of the leftover variance a single factor accounts for; ~0.72 is well above the 0.14 "large" benchmark). **Significance and importance coincide here** — these are big, robust effects, and the four coefficients are statistically indistinguishable from one another (they differ by less than 0.1 g/L against a per-coefficient standard error of 0.76), so no factor can be crowned the "strongest"; they are effectively tied.

**Robustness to non-normality (see Limitations).** The residuals are non-normal (Shapiro–Wilk W = 0.763, p = 3.5×10⁻⁴), but this does not threaten the main-effect estimates, for a design-based reason stronger than any robust-SE patch: the non-normality is entirely the three center points being under-predicted (which *is* the curvature finding), and center points sit at coded 0, so they contribute exactly zero to the main-effect slopes. If anything the unmodeled curvature inflates the residual variance, making the main-effect standard errors *conservative* — the tests understate, not overstate, significance.

**Interactions are negligible.** None of the six two-factor interactions approached significance (p = 0.32–0.58), and their estimated sizes (0.47–0.87 g/L) are 5–6× smaller than the main effects. More importantly, the data *positively favor* the additive model rather than merely failing to reject it: comparing the two models' fit-with-penalty scores (a Schwarz/BIC approximation to a Bayes factor, not a Cauchy-prior Bayes factor) gives about **137-to-1 odds in favor of the no-interaction model** (Bayes factor ≈ 137 — i.e. the data are ~137× more consistent with additivity than with interactions; on the usual scale this is decisive). We therefore treat titre as **additive** in these four factors over the tested box.

**Curvature — the surface bends; direction is clear, magnitude is imprecise.** The 3 center points averaged 53.4 g/L versus a corner average of 47.7 g/L — they sit ~5.8 g/L *above* what a flat plane predicts, and they are the three largest residuals in the model, all in the same direction (+4.7, +4.6, +5.2 g/L). The honest test uses only the pure replicate scatter of those 3 center runs (2 degrees of freedom) and gives t = −2.42, p = 0.029 with an explicit **low-statistical-power** flag. A less conservative version that borrows error from the whole model reports p = 0.0003, but with only 3 center points that overstates the precision. (A BIC-based approximation gives extreme odds for curvature, ≈5×10³-to-1, but it borrows the whole-model residual and therefore inflates evidence the design does not actually have; we do not lean on it.) The defensible reading: **the existence and direction of curvature are well-established; its exact magnitude is not.** The practical consequence is unchanged either way — a flat plane cannot describe a curved surface, so it cannot locate a peak, and a 16+3 design cannot estimate the individual squared (quadratic) terms needed to do so.

**Best condition found.** Ranking the additive model's predictions over the actually-tested settings, the highest titre is at every factor's **high (+1) edge**:

| Factor | Coded | Original units |
|---|---|---|
| glucose | +1 | 8 g/L |
| glutamine | +1 | 6 mM |
| serum | +1 | 10 |
| sodium bicarbonate | +1 | 3.7 g/L |

The additive model predicts **66.66 g/L** at this corner; the **observed** titre there was **70.11 g/L** (the excess over the additive prediction is itself the curvature signal). The optimization engine explicitly flags all four factors as **"boundary"** and states it "never claims an interior/untested optimum."

## Conclusion
**Within the ranges tested, the best condition is glucose 8 g/L, glutamine 6 mM, serum 10, and sodium bicarbonate 3.7 g/L, where the observed titre was ~70 g/L — about 21 g/L above the design average of 48.6 g/L.** All four factors help, additively and substantially — each adds roughly +4.5 g/L per coded step, a very large effect (partial η² ≈ 0.72 each), and the four are statistically tied, so none dominates.

**This is the best *observed* condition, not a proven optimum.** Every factor is still climbing at the top of its tested range (all four settings hit the +1 boundary with positive slopes), and the response surface is curved (center points run ~5.8 g/L above the flat-plane prediction). Both facts point the same way: the true maximum very likely lies **beyond the current ranges**, and this design cannot pin down where the surface peaks. Treat ~70 g/L as a floor for what these factors can reach, not a ceiling.

## Recommended Next Experiments
Grounded in the engine diagnostics — the `doe-optimum` boundary flags (all four factors = +1), the `design-coverage` curvature signal and its low-power warning, the absence of collinearity (all VIF = 1.0), and no influential runs (max Cook's D 0.162 < 0.211):
1. **Push the factor ranges upward / take a steepest-ascent step.** Because the best setting sits at the +1 edge of all four factors with positive slopes, extend each factor beyond its current high level (a steepest-ascent move along the fitted gradient) to find where titre stops rising.
2. **Augment to a central composite / response-surface design (CCD).** Add axial/"star" runs plus more replicated center points so the four squared (curvature) terms become estimable individually — the current design detects *that* the surface curves but not *how*, so a genuine interior optimum cannot be computed yet.
3. **Replicate the center point more heavily.** The curvature conclusion rests on just 3 center runs (2 df) and carries a low-power warning; more center replicates will sharpen the curvature magnitude, which is currently the least precise part of the analysis. Confirm from the run log that the 3 center runs are independent experimental runs (not analytical repeats of one culture); if they are repeats, the pure-error estimate is optimistic.
No interaction-resolving or confound-resolving runs are needed (interactions negligible; design orthogonal).

## Limitations
- **Model form / no interior optimum.** The linear surface is formally rejected for lack of fit (F(12,2) ≈ 142, p ≈ 0.007, an algebraic upper-tail value sharing the same fragile 2-df pure error as the curvature test), driven by curvature. A plane has no stationary point, so no true optimum is identifiable — hence "best tested condition," not "the optimum."
- **Non-normal residuals.** Shapiro–Wilk p < 0.05 in every model, but this is misspecification (the under-predicted center points), not heavy-tailed noise; per the Results robustness argument it does not bias the main-effect coefficients, whose CIs are valid conditional on the additive form. The requested HC3 heteroskedasticity-robust standard errors could **not** be computed — the engine exposes no such option — so that specific check is unmet; the orthogonality argument is the substitute and is arguably stronger for this design.
- **Curvature power.** The curvature evidence rests on 2 degrees of freedom; direction is firm, magnitude imprecise. The requested Cauchy-prior Bayesian curvature test (at prior widths r = 0.5 and r = 1.0) could **not** be run — the engine's `bayes-ttest` cannot accept the required group/value columns — so only the BIC approximation was available and it is explicitly down-weighted above.
- **Extrapolation.** Recommendations 1–2 move outside the tested box; confirm empirically before trusting predictions there.
- **Additivity leans on the Bayes factor.** The "no interactions" conclusion is supported by decisive Bayesian odds (≈137:1); the frequentist non-significance alone (n = 19, 6 terms) would be low-powered and only "absence of evidence."

## Appendix (technical detail)
- **Reduced (primary) model** `titre_g_l ~ glucose_coded + glutamine_coded + serum_coded + sodium_bicarb_coded`: n=19, R²=0.9104, adj R²=0.8848, overall F-p=3.42×10⁻⁷; Intercept 48.58; coefficients (coef, SE 0.7577, p, 95% CI): glucose 4.4775, p 3.808×10⁻⁵, [2.852, 6.103]; glutamine 4.56375, p 3.129×10⁻⁵, [2.939, 6.189]; serum 4.52625, p 3.407×10⁻⁵, [2.901, 6.151]; sodium_bicarb 4.5075, p 3.556×10⁻⁵, [2.882, 6.133]. Partial η² ≈ 0.714/0.722/0.718/0.717. Shapiro W=0.763, p=3.51×10⁻⁴; Durbin–Watson=1.961. (Model is orthogonal, so Type I/II/III sums of squares are identical.)
- **Full 2-way model**: n=19, R²=0.9407, adj R²=0.8665, overall F-p=7.15×10⁻⁴, AIC=104.43, BIC=114.81. Interaction p-values: glucose:glutamine 0.344, glucose:serum 0.316, glucose:sodium_bicarb 0.420, glutamine:serum 0.579, glutamine:sodium_bicarb 0.503, serum:sodium_bicarb 0.557; interaction coefficients 0.471–0.872. Shapiro W=0.574, p=2.46×10⁻⁶; DW=1.954.
- **Additive vs. 2-way (interaction) Bayes factor**: BIC(reduced)=104.977, BIC(full)=114.814, ΔBIC=9.837 → BF01 = exp(9.837/2) = 136.8 in favor of additivity (Schwarz/BIC approximation, not a Cauchy-prior Bayes factor).
- **Curvature**: center-vs-corner term. Pure-error contrast (engine `design-coverage`): corner_mean 47.671, center_mean 53.427, diff −5.7554, t=−2.4198, p=0.028566, df=2, low_power_warning=true. Df-borrowing version (`regression` with C(run_type)): coef −5.7554, p=2.80×10⁻⁴, model R²=0.9687. BIC-based curvature BF10 ≈ 5×10³ (whole-model residual; borrows precision — not relied upon). Lack-of-fit partition: SS_pure_error=0.150 (df 2), SS_lack_of_fit=128.463 (df 12), F(12,2)≈142.3, p≈0.007 (algebraic tail); curvature = 83.7 of 128.5 SS of lack-of-fit.
- **VIF**: all terms = 1.0 (orthogonal, no collinearity). **Design coverage**: 81 possible combinations, 17 tested (20.99%), 3 center replicates. **Influence** (`predict`): leverage threshold 0.526, Cook's-D threshold 0.211, 0 rows flagged, max Cook's D 0.162 at run_order 10 (all-+1 corner); center-row residuals +4.73/+4.65/+5.16.
- **doe-optimum**: best tested combo all coded +1 → predicted 66.655 (additive model), observed 70.11; all four factors flagged "boundary."
- **Unmet engine checks (documented limitations, not omissions)**: HC3 robust SEs (no `cov_type` option on `regression`); Cauchy-prior Bayesian curvature test at r=0.5/1.0 (`bayes-ttest` requires group/value columns not applicable to a regression contrast); per-coefficient Bayes factors for the main effects (engine returns no BF for slopes — an earlier draft's ">1000 each" claim was removed as unverifiable).

---

*Generated by the statistical board over 2 round(s) on model `claude-opus-4-8`. Every statistic was computed and independently reproduced with the stat_board engine.*
