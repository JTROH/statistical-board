# Debate transcript — "Do groups A, B, C differ?"

Data: `sample_data/long.csv` (long format, group/value), 3 groups × n=30, α=0.05,
two-sided. Board: judge-led, 1 round to finalize.

## Plan (judge)

- Omnibus H₀: μ_A = μ_B = μ_C; if rejected, pairwise A–B/A–C/B–C with FWER control.
- Fork: classic ANOVA + Tukey **if** variances homogeneous & residuals normal;
  else Welch ANOVA + heteroscedastic pairwise; Kruskal if normality fails.
- Round-1 uncertainties: does Levene flag unequal variance; is normality OK; is
  the design powered enough (esp. high-variance C).

## Round 1

### Analyst (engine outputs)
- Descriptives: A 100.17 (sd 7.77) · B 108.14 (sd 8.05) · C 99.17 (**sd 12.37**).
- Assumptions: Shapiro A 0.416 / B 0.744 / C 0.093 (all normal); **Levene
  p = 0.0046** (unequal var); Bartlett p = 0.017; 1 IQR outlier in A.
- Classic ANOVA F(2,87)=7.82, p=0.00075, η²=0.152. Welch ANOVA F(2,56.28)=9.41,
  p=0.00030. Kruskal H=12.40, p=0.0020.
- Tukey: A–B reject (0.0053), A–C no (0.915), B–C reject (0.0015).
- Per-pair Welch t: A–B p=0.00025 d=−1.008; A–C p=0.710 d=0.097; B–C p=0.00165 d=0.859.
- Bayes: A–B BF₁₀=100.4; A–C BF₁₀=0.278 (BF₀₁=3.59); B–C BF₁₀=21.8.
- Power (as first run, **k defaulted to 2 — wrong**): f=0.25→0.262, f=0.40→0.562.

### Frequentist critic
1. **Tukey invalid** — assumes the homogeneity Levene rejected; use Games–Howell
   or Holm-corrected Welch pairwise. 2. Per-pair t-tests **need correction**;
   ran Holm/BH → A–B and B–C survive, A–C not. 3. **Headline Welch ANOVA**, demote
   classic. 4. A–C null is underpowered — don't claim equality from p=0.71; needs
   TOST. 5. **Power figures don't reproduce** — engine gives 0.195 / 0.442 (k=3),
   not 0.262 / 0.562. *Survives scrutiny:* A–B & B–C are large, real effects;
   normality holds; η² correctly "large."

### Assumptions skeptic
1. Homogeneity **violated** → classic ANOVA & Tukey invalid; report Welch ANOVA
   + Kruskal, replace Tukey with Games–Howell / Holm-Welch. 2. Pairwise reported
   raw — apply Holm. 3. **A–C is "no LARGE difference," not equivalence**: power at
   observed d=0.097 ≈ 0.066; TOST ±5 → p=0.069 (fails); BF₀₁=3.59 only moderate.
   4. C's marginal normality + single A outlier: low risk, covered by robust tests.
   5. **Independence untestable** — confirm samples aren't repeated-measures.

### Bayesian critic
1. A–C: BF turns the non-significant p into a positive statement — **BF₀₁=3.59,
   moderate evidence for the null** (the unique Bayesian contribution). 2. Prior
   sweep on A–C: r=0.5→2.76 (anecdotal), 0.707→3.59, 1.0→4.83 — direction robust,
   strength borderline. 3. A–B BF₁₀≈90–102 across priors (extreme). 4. B–C
   BF₁₀≈21 across priors (strong). 5. Correct summary: "B differs from A and C;
   A and C do not," with A=C a *supported* leg, not merely absent.

### Judge adjudication (pre-verify)
- Resolved the **power discrepancy**: analyst omitted `--k-groups 3`, so the CLI
  defaulted to k=2. Correct values: f=0.25→0.195, f=0.40→0.442, observed
  f≈0.42→0.49. Frequentist critic correct. *(Flagged as an engine follow-up: the
  anova power default k=2 is a trap.)*
- Adopted critics' corrections: drop classic ANOVA + Tukey; Holm-corrected Welch
  pairwise (A–B 0.00075, A–C 0.710, B–C 0.0033); A–C framed as undetected +
  moderate-null, **not** equivalent (TOST ±5 → p=0.069).

### Verifier
Independently reproduced **all 8 published claims — every one supported, none
contradicted**: Levene 0.00463; Welch ANOVA F=9.413 p=0.000297; Kruskal H=12.40
p=0.00203; Shapiro values; Holm-adjusted pairwise (0.000751 / 0.710 / 0.003293)
with d = −1.008 / 0.097 / 0.859; A–C TOST p=0.0693 (not equivalent); BF₁₀ 100.4 /
21.8 and BF₀₁ 3.59; and confirmed the k=3 power values (0.49 observed, 0.195 at
f=0.25) are correct while the default-k=2 0.262 was the erroneous one.

### Decision: **FINALIZE**
Every conclusion is backed by a valid, reproduced test; all critic objections
incorporated; the only residual uncertainty (independence) is stated in the
report. No cosmetic reason to iterate.

## What the board caught and changed
- Rejected the invalid **classic ANOVA + Tukey HSD** (Levene) → Welch ANOVA +
  Holm-corrected Welch pairwise.
- Corrected the pairwise family for **multiplicity** (Holm).
- Reframed **A–C** from "no difference" to "undetected difference + moderate
  Bayesian support for the null, equivalence NOT established (TOST fails)."
- Fixed a **power miscalculation** (k=2 default → correct k=3), reframing power
  around the observed effect.
