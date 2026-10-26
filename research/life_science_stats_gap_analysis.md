# Statistical Board: gap analysis against how life-science teams use JMP, R and similar tools

*Prepared 2026-10-18 against branch `merge-doe-advisor` @ `d314849`.*

## 1. What the tool does today (from the code)

| Area | Present |
|---|---|
| **Design (`doe_advisor`)** | Full/fractional factorial (min-aberration, resolution from the defining relation), DSD, CCD (rotatable / face / spherical), Box-Behnken; power per term with noise stress and measured-noise CI; CV / fold-change noise model; aliasing; I/D/A/G efficiency + FDS; robustness to run loss; D-optimal blocking; hard ranges; Sf9 / CHO / E. coli presets; randomised run sheet; augmentation from analysis diagnostics. Cross-checked against R `FrF2`/`rsm`. |
| **Analysis (`stat_board`)** | t / Welch / Mann-Whitney / ANOVA / Welch ANOVA / Kruskal / Tukey / TOST / Bayes t; correlation; chi-square; power; p-value correction; OLS regression, two-way ANOVA, ANCOVA, lack-of-fit, Box-Cox, VIF, leverage/Cook's D, design coverage, tested-point optimum, JZS Bayes factors, stationary point (canonical analysis); Poisson / NB regression. |
| **Vetting + reporting** | Six-role LLM board with a Verifier re-running numbers; no-unsupported-numbers check on narration; deterministic PDF appendix; data-prep recipes; web app, desktop .app, Claude Code skills. |

**What already sets it apart:** the closed plan → analyse → augment loop, run count deliberately excluded from scoring, robustness-to-run-loss, and computation-grounded adversarial review. JMP, Design-Expert, MODDE and R do none of these as one guided flow.

**Hard limits found in the code**
- `Factor` is continuous only (`low`/`high`). The design side has no categorical factors (cell line, media vendor, resin), no discrete numeric levels, and no mixture components.
- Classical designs only. There is no optimal/custom design generator (coordinate exchange) with constraints, disallowed combinations or fixed prior runs. Only the block assignment step is D-optimal.
- Power, prediction and robustness use only `spec.primary_response` (`properties.py:284,337,471`). Titer, viability, glycan and aggregate are rarely all served by one design, but only the first one gets scored.
- There are no random effects anywhere: no mixed models, split-plot or nested variance components.
- There is no nonlinear curve fitting: no 4PL/5PL, EC50 or growth curves.

## 2. What the research says practitioners do and struggle with

1. **Non-statisticians cannot pick a design or read the output.** A Unilever R&D statistician names three problems: too many possible designs, too much detail to specify them, and output that is hard to interpret. Unilever's answer was a guided tool rather than more training ([JMP Discovery abstract](https://community.jmp.com/t5/Abstracts/DOE-Approached-Differently-Making-Every-Experiment-Count/ev-p/849775)). Synthace's pitch rests on the same claim: JMP and Design-Expert "can be bewildering" to biologists ([Synthace](https://synthace.com/blog/introducing-synthace-doe)). Biomanufacturing hires come from mixed backgrounds with little DoE exposure. → *Statistical Board's guided shortlist fits this need well.*
2. **R is flexible but weak for DoE.** Practitioners report caps such as Box-Behnken limited to 7 factors in `rsm` and DSDs limited to 4–7 factors in `daewr`, and they return to JMP for DoE even after moving their analysis to R ([Posit forum](https://forum.posit.co/t/recommended-packages-for-design-of-experiments-and-definitive-screening-designs/101372)).
3. **Hard-to-change factors are routine in bioprocess work.** Temperature on a BioLector or Ambr run, or plate-level factors in 24-well plates, force split-plot designs. Analysing them as if fully randomised biases the p-values for those factors ([JMP: 24-well plates](https://community.jmp.com/t5/Discussions/24-deep-well-plates-with-hard-to-change-factors/m-p/908738), [JMP: BioLector split-plot](https://community.jmp.com/t5/Discussions/Custom-Split-Plot-Design-Replicates/m-p/911713), [JMP QbD column](https://community.jmp.com/t5/JMP-Blog/The-QbD-Column-Split-plot-experiments/ba-p/30716)).
4. **Augmentation pain points in JMP itself.** Users want to add a *new factor* to completed runs; that wish-list item has been archived and reopened ([JMP wish list](https://community.jmp.com/t5/JMP-Wish-List/Augment-DOE-with-new-factor/m-p/251800)). Replicates are greyed out when augmenting split-plot or Plackett-Burman designs, so users copy and paste rows by hand ([thread](https://community.jmp.com/t5/Discussions/Augmenting-a-Plackett-Burman-design-to-a-Full-Factorial-Design/m-p/746898)). Novices have asked for "easy augment" ([wish](https://community.jmp.com/t5/JMP-Wish-List/Easy-DOE-augment-with-existing-design/idc-p/836809)). → *Augmentation is a strength of this repo, so these are direct openings.*
5. **Pseudoreplication and confusion between biological and technical replicates** remain common in published biology ([bioRxiv 2025 survey](https://www.biorxiv.org/content/10.1101/2025.02.04.636422.full.pdf), [J Gen Physiol](https://rupress.org/jgp/article/153/2/e202012826/211691/Pseudoreplication-in-physiology-More-means)). Cell work rarely makes clear which unit (well, coverslip or cell) is the replicate ([FEBS](https://network.febs.org/posts/31433-seeing-triple-a-short-guide-to-experiment-reproducibility)).
6. **Bioassay potency is a standing pain point.** USP <1032>/<1034> call for equivalence-based parallelism on 4PL fits, which labs find hard to implement without custom templates ([PDA J 2012](https://pubmed.ncbi.nlm.nih.gov/22634591/), [BioProcess Intl](https://www.bioprocessintl.com/assays/assessing-similarity-with-parallel-line-and-parallel-curve-models-implementing-the-usp-development-validation-approach-to-a-relative-potency-assay)). Teams buy dedicated tools such as PLA or SoftMax, or build JMP scripts.
7. **QbD design space is moving to probabilistic definitions.** Here the design space is the region where the predicted probability of meeting all CQA specs exceeds a threshold, usually with a random batch term ([Peterson, Bayesian ICH Q8](https://www.researchgate.net/publication/23247262_A_Bayesian_Approach_to_the_ICH_Q8_Definition_of_Design_Space), [fermentation PAR case](https://www.researchgate.net/publication/285165843_Finding_Design_Space_and_a_Reliable_Operating_Region_Using_a_Multivariate_Bayesian_Approach_with_Experimental_Design), [nested sampling](https://arxiv.org/pdf/2008.05917)). JMP's profiler simulator and desirability functions are the everyday stand-ins.
8. **Regulated work has validation overhead.** JMP will not certify Part 11 compliance, so validation falls on the user ([JMP pharma users group](https://community.jmp.com/t5/Pharma-JMP-Users-Group/Hello-and-Welcome-to-the-Virtual-Pharma-Users-Group/m-p/826508)). The cost of validation delays upgrades for years ([LNS Research](https://blog.lnsresearch.com/state-of-the-market-software-validation-in-life-sciences)). R is usable under GxP with risk-based package validation, but the sign-off burden is real ([R Validation Hub](https://pharmar.org/overview/), [Appsilon](https://www.appsilon.com/post/introducing-axon-r-appsilons-framework-for-r-package-validation-in-pharma)). A GxP bioassay workflow abstract flags manual operator steps as a risk.
9. **Downstream lifecycle statistics are separate products today.** Continued process verification (CPV) needs control charts, Cpk/Ppk and out-of-trend rules ([Merck CPV](https://www.sigmaaldrich.com/US/en/technical-documents/technical-article/pharmaceutical-and-biopharmaceutical-manufacturing/monoclonal-antibody-manufacturing/continued-process-verification)). ICH Q1E stability work needs ANCOVA poolability at α = 0.25 and shelf life where the 95% bound crosses the spec ([R `expirest`](https://cran.r-project.org/web/packages/expirest/refman/expirest.html), [PharmTech](https://www.pharmtech.com/view/pooling-of-batches-for-stability-data-analysis)).
10. **High-throughput platforms produce data faster than teams can analyse it.** Ambr systems run up to 48 bioreactors and ship a bundled MODDE licence. FUJIFILM Diosynth hired data scientists to build analysis workflows for that volume ([Sartorius Ambr 15](https://www.sartorius.com/en/products/fermentation-bioreactors/ambr-multi-parallel-bioreactors/ambr-15-cell-culture), [JMP/FDB story](https://www.jmp.com/content/dam/jmp/documents/en/customer-stories/fujifilm-diosynthbio.pdf)). Plate-based DoE also needs layout and liquid-handler output ([Synthace media DoE](https://www.synthace.com/applications/design-of-experiments/doe-for-media-optimization)), and randomised plate layouts are an open method question ([bioRxiv](https://www.biorxiv.org/content/10.1101/2022.03.31.486595v1.full.pdf)).
11. **LLM analyses vary from run to run.** Reproducibility-focused setups improve accuracy ([AIRepr, EMNLP 2025](https://arxiv.org/pdf/2502.16395)). → *The repo's design (engine as the single source of numbers, a Verifier role, a guard against unsupported numbers) is the right answer. Make this visible to users as a trust feature.*

## 3. Gaps, ranked by value to bioprocess / life-science users

### Tier 1: blocks real studies today
| # | Gap | Why it matters | Fits existing architecture |
|---|---|---|---|
| G1 | **Categorical and discrete factors in design** | Cell line, media, resin and vendor appear in almost every screening study. | `Factor` → add `levels`; coded columns via effects coding in `model.py`. |
| G2 | **Optimal / custom designs** (coordinate exchange: D/I-optimal, constraints, disallowed combinations, fixed prior runs, odd run budgets) | This is JMP's core advantage over R. Classical designs break as soon as a budget or constraint is imposed. Also enables "augment with prior runs" and "add a new factor to old runs" (JMP wish-list items). | New generator in `designs/`; the existing scorer and the four axes stay as they are. |
| G3 | **Split-plot / hard-to-change factors + mixed-model analysis** (REML, random block/batch/plate, Kenward-Roger df) | Bioreactor temperature, plate-level factors and day/lot effects are routine. The repo already blocks runs but analyses blocks as fixed effects with OLS. | Design: whole-plot constraint in coordinate exchange. Engine: `statsmodels` MixedLM. Board: assumptions skeptic checks the error strata. |
| G4 | **Multi-response design and optimisation** (power across all responses; desirability; overlay / sweet-spot plots) | Every bioprocess DoE trades titer against quality. Today only the primary response is scored. | `properties.py` loops over responses (report the minimum power); engine gains `desirability` and an `overlay` figure. |

### Tier 2: strong pull, differentiating
| # | Gap | Notes |
|---|---|---|
| G5 | **Probabilistic design space / PAR** (Monte Carlo through the fitted model + parameter uncertainty → P(meet all specs) map; Bayesian posterior predictive) | The modern ICH Q8 framing that reviewers increasingly expect. Builds directly on `stationary_point` plus the existing Bayes work. |
| G6 | **Replicate-structure guard** (declare the experimental unit vs. technical replicates; auto-aggregate or nest) | Addresses pseudoreplication, the most common error in the literature. A natural check for `prep` and the assumptions-skeptic role. |
| G7 | **Mixture and mixture-process designs** (simplex lattice/centroid, constrained mixtures, Scheffé models) | Media and feed formulation and buffer optimisation. |
| G8 | **Dose-response / bioassay module** (4PL/5PL, EC50 with CI, USP <1032>/<1034> equivalence parallelism, relative potency) | Large unmet need. Nonlinear fits would also give growth-curve and kinetic models. |
| G9 | **Sequential / adaptive DoE** (Bayesian optimisation or model-based next-batch suggestion for Ambr-style 24–48 runs) | Extends the existing augment loop from "one more design" to "the next batch, given everything so far". |

### Tier 3: lifecycle and adoption
| # | Gap | Notes |
|---|---|---|
| G10 | **Stability (ICH Q1E)**: poolability ANCOVA at α = 0.25, shelf life from the confidence bound, mixed-model option | Mostly reuses the existing `ancova` + `predict`. Cross-validate against R `expirest`. |
| G11 | **SPC / CPV**: I-MR and X̄-R charts, Western Electric/Nelson rules, Cpk/Ppk with CIs, non-normal capability, tolerance intervals | Tolerance intervals are also needed for setting specs. |
| G12 | **Multivariate**: PCA / PLS for batch-trajectory and process-analytical-technology (PAT) data | This is what SIMCA and JMP Pro offer; less core to the current scope. |
| G13 | **Compliance / audit pack**: hashed inputs, engine version, environment lockfile, signed run log, validation report generated from `validation/` + tests | Turns the existing R/JMP cross-validation into an IQ/OQ-style package. Lowers the validation burden that keeps teams on old JMP versions. |
| G14 | **Integration and export**: JMP `.jsl` / Design-Expert / MODDE run-sheet export, Ambr/BioLector import, plate-map layout with edge-effect randomisation, liquid-handler CSV | Meets users where they already work. Plate maps reuse the D-optimal blocking code. |
| G15 | **Interactive profiler** in the web app (prediction profiler with CI, contour sliders for >2 factors) | The one visual feature scientists repeatedly cite as the reason they stay on JMP. |

## 4. Suggested order

1. **G1 + G2 together.** One coordinate-exchange generator that handles categorical factors, constraints and prior runs removes the biggest reason to fall back to JMP. Validate it against R `skpr`/`AlgDesign` in `validation/`.
2. **G3** (split-plot + MixedLM). It reuses the blocking work and needs G2's exchange engine.
3. **G4 → G5.** Multi-response first, then a probabilistic design space on top. Together they turn "Recommended Next Experiments" into a defensible PAR / design-space statement.
4. **G6** is small and cheap and fits the board's mission. It could ship any time.
5. Then pick a vertical: **G8** (bioassay) or **G10 + G11** (CMC lifecycle), depending on who the first users are.

## 5. Caveats on the evidence

Most accessible sources are vendor pages or vendor-hosted communities (JMP Community, Synthace, Sartorius, Merck). I found no independent survey that measures how common each pain point is across bioprocess teams. The ranking above combines those qualitative signals with what the code is currently missing. It should be tested against 5–10 interviews with target users (upstream/downstream process development, analytical development, and QC) before committing to a roadmap.
