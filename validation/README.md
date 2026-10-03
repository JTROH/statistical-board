# Cross-validation

This suite exists because "does it match JMP?" is the first question any
statistician will ask about a tool that generates experimental designs, and
because a design engine written in-house has no credibility until it agrees with
the ones people already trust.

It is not optional polish. It is the adoption strategy.

## Running it

```bash
python3 -m validation.run_tool     # this tool's answers  -> results/tool.json
Rscript validation/validate_r.R    # R's answers          -> results/r.json
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

Without R the suite still runs and still checks the definitive screening designs
against their published properties; it just reports the R legs as skipped.

## JMP

JMP has no scriptable interface here, so that leg is manual. See
[`JMP_INSTRUCTIONS.md`](JMP_INSTRUCTIONS.md), and record the outcome in
`results/jmp.md`.

## Current status

As of the last run: **all 112 automated checks agree** with R 4.5.3
(`FrF2` 2.3.4, `rsm` 2.10.6). The JMP leg has not yet been run.

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
