# Cross-validation

This suite exists because "does it match JMP?" is the first question any
statistician will ask about a tool that generates experimental designs, and
because a design engine written in-house has no credibility until it agrees with
the ones people already trust.

It is not optional polish. It is the adoption strategy.

## Running it

```bash
python3 -m validation.run_tool     # this tool's answers  -> results/tool.json
Rscript validation/validate_r.R    # R's design answers   -> results/r.json
Rscript validation/validate_kr.R   # R's mixed-model / Kenward-Roger answers -> results/r_kr.json
python3 -m validation.compare      # compare; non-zero exit on any disagreement
```

The comparison also runs as part of `pytest` (`tests/test_validation.py`), which
skips the R leg automatically when R is not installed.

## What is checked

| Family | Reference | Compared on |
|---|---|---|
| Fractional factorials | R, `FrF2` | run count, resolution, **word-length pattern** |
| Box-Behnken | R, `rsm::bbd` | run count, full set of design points |
| Central composite | R, `rsm::ccd` | run count, factorial/axial point counts, alpha |
| Definitive screening | published properties | 2m+1 runs, three levels, main effects orthogonal to every 2FI |
| Power | `scipy` directly, in `tests/designs/test_properties.py` | non-central *t* against a closed form |
| Mixed model (split-plot analysis) | R, `lme4` + `pbkrtest` | REML variance components, coefficients, plug-in and Kenward–Roger-adjusted SEs, and each term's KR F, denominator df, F scaling and p |

The **word-length pattern** is the strict test for fractional factorials. It
counts defining words of length 3, 4, 5, 6, 7, and it is a design's aliasing
fingerprint — two designs with the same pattern confound things the same way.
Resolution is only its first non-zero entry, so comparing patterns catches
errors that comparing resolutions would miss.

## Requirements

R with `FrF2`, `rsm` and `jsonlite`:

```r
install.packages(c("FrF2", "rsm", "jsonlite"))
```

The Kenward–Roger leg needs only `lme4`, `pbkrtest` and `jsonlite`, and runs (and
is skipped) independently of the design leg:

```r
install.packages(c("lme4", "pbkrtest", "jsonlite"))
```

Its six cases are frozen CSVs in `datasets/kr_*.csv`, read identically by both
sides (`kr_datasets.py` records how they were made). They stress different parts
of the method: a balanced split-plot, lost runs, unequal whole plots with a
continuous subplot factor, a three-level categorical whole-plot factor, almost no
whole-plot variance (a boundary fit), and a design made by this tool's own
split-plot search. R tests each term with `KRmodcomp` and an explicit restriction
matrix on that term's columns, the same hypothesis the tool tests.

Without R the suite still runs and still checks the definitive screening designs
against their published properties; it just reports the R legs as skipped.

## JMP

JMP has no scriptable interface here, so that leg is manual. See
[`JMP_INSTRUCTIONS.md`](JMP_INSTRUCTIONS.md), and record the outcome in
`results/jmp.md`.

## Current status

As of the last run of the design leg: **all 112 automated checks agree** with R
4.5.3 (`FrF2` 2.3.4, `rsm` 2.10.6). The JMP leg has not yet been run.

Kenward–Roger leg, 2026-10-25: **all 205 checks agree** with R 4.3.1 (`lme4`
1.1.35.5, `pbkrtest` 0.5.3) at a relative tolerance of 1e-6. Largest differences:
F 3e-8, denominator df 2e-10, F scaling 2e-13, p 8e-8 (relative). The tolerance
is set by the two REML optimisers, not by the method; with statsmodels' default
L-BFGS stopping rule the gap was ~1e-4, which is why `mixed_model` now fits with
tight `pgtol`/`factr`.

## Two findings worth remembering

The suite earned its keep on its first run:

1. **`FrF2`'s stored word-length pattern is indexed from word length 1**, while
   its print method labels the trimmed vector "3plus". Reading the raw vector as
   though it started at length 3 makes every design look wrong. That was a bug
   in `validate_r.R`, not in the engine.

2. **The generators forced at least one centre point**, so asking for zero
   silently gave you one. The design *points* matched `rsm` exactly; only the run
   count was off by one. The floor now lives in `doe_advisor/candidates.py`,
   where the policy belongs, and the generators honour what they are asked for.

Neither would have been caught by unit tests written against the same
assumptions as the code.
