# Do groups A, B, and C differ — and which pairs?

*Statistical board report · α = 0.05, two-sided · n = 30 per group · 1 round to finalize*

## Question & hypothesis

- **Omnibus** — H₀: μ_A = μ_B = μ_C vs. H₁: at least one group mean differs.
- **Pairwise** (if omnibus rejects) — which of A–B, A–C, B–C differ, with
  family-wise error controlled.

## Methods — and why these tests

The choice of test was **decided by the data, not assumed**:

- **Normality holds** for all three groups (Shapiro–Wilk p: A = 0.416, B = 0.744,
  C = 0.093), so a parametric route is defensible.
- **Variances are unequal** — Levene's test **p = 0.0046** (Bartlett agrees,
  p = 0.017), and the SDs bear it out (A = 7.77, B = 8.05, **C = 12.37**, a
  variance ratio ≈ 2.5).

Because homogeneity of variance fails, the **classic one-way ANOVA and Tukey HSD
are invalid here** (both pool a single error variance) and are *not* reported as
the result. The board used their heteroscedastic-robust replacements:

| Purpose | Test used | Why |
|---|---|---|
| Omnibus | **Welch's ANOVA** | Does not assume equal variance |
| Omnibus backup | Kruskal–Wallis | Non-parametric cross-check |
| Pairwise | **Welch t-tests + Holm correction** | Unequal-variance-safe; controls family-wise error |
| A–C equivalence | TOST | "No difference" needs a positive test, not a large p |
| Evidence strength | JZS Bayes factors (r = 0.707) | Quantifies evidence, incl. *for* the null |

## Results

**Omnibus — the groups differ.** Welch's ANOVA **F(2, 56.28) = 9.41, p = 0.00030**;
Kruskal–Wallis **H = 12.40, p = 0.0020** agrees. Omnibus effect is large
(η² = 0.152, ω² = 0.132; Cohen's f ≈ 0.42).

**Pairwise (Holm-adjusted Welch t-tests):**

| Pair | Mean diff | Adj. p | Cohen's d | Bayes factor | Verdict |
|---|---|---|---|---|---|
| **A vs B** | −7.97 | **0.00075** | −1.01 (large) | BF₁₀ = 100 (extreme) | **Differ** |
| **B vs C** | +8.97 | **0.0033** | 0.86 (large) | BF₁₀ = 21.8 (strong) | **Differ** |
| A vs C | −1.00 | 0.710 | 0.10 (negligible) | BF₀₁ = 3.6 (moderate, null) | No difference detected |

## Conclusion

**B stands apart from both A and C; A and C are statistically indistinguishable.**
This is *not* "all three groups differ." The two differences (A–B, B–C) are both
**large and decisive** — significant after Holm correction and corroborated by
extreme/strong Bayes factors — so they are not artifacts of multiplicity or the
variance imbalance.

On **A vs C**, the frequentist test (p = 0.71) and a moderate Bayes factor
(**BF₀₁ = 3.6** — the data are ~3.6× more likely under "no difference") both point
to no effect, and the observed difference is negligible (d = 0.10). But this is
"**no difference detected**," not proven equivalence: a **TOST within ±5 units
fails (p = 0.069)**, so a small-to-moderate A–C difference cannot be ruled out.

## Limitations

- **Underpowered for small effects.** At k = 3, n = 30, α = 0.05, power is only
  **≈ 0.20** for a small effect (f = 0.25) and **≈ 0.49** even at the observed
  effect (f ≈ 0.42). The design reliably catches only large effects — which is
  why the A–C null is "undetected," not "confirmed equal." Detecting an A–C
  difference the size observed would need a far larger n.
- **Bayesian A–C strength is prior-dependent.** BF₀₁ for A–C moves from 2.8
  (skeptical prior, r = 0.5 → only *anecdotal*) to 4.8 (wide prior, r = 1.0). The
  *direction* (favoring the null) is robust; the *strength* is borderline
  moderate. The A–B and B–C conclusions are prior-insensitive.
- **Independence is assumed, not verified.** The tests treat A/B/C as independent
  between-subjects samples. This cannot be checked from the numbers; if the data
  are actually paired/repeated-measures on the same units, a repeated-measures
  ANOVA / Friedman design would be required instead.

## Sources

Every statistic above was computed and independently reproduced with the project
engine, `python3 -m stat_board.engine`, against `sample_data/long.csv`. No number
was hand-computed. The classic ANOVA (F = 7.82, p = 0.00075) and Tukey HSD were
run but **excluded from the conclusion** because Levene's test invalidated their
equal-variance assumption.

*Board: 1 round · analyst → frequentist / assumptions / Bayesian critics →
verifier → judge. Verifier reproduced all 8 published claims (all supported).*
