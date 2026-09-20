# Cross-checking against JMP

The R leg of this suite runs automatically. JMP has no scriptable interface here,
so this leg is manual — but it is the one that matters most politically, because
"does it match JMP?" is the first question a statistician in a pharma process
development group will ask.

Run this before anyone relies on the tool for a real campaign, and again after
any change to `doe_advisor/designs/`.

Record what you find in `validation/results/jmp.md` and commit it.

---

## Before you start

```bash
python3 -m validation.run_tool     # writes validation/results/tool.json
```

Open that file alongside JMP. Every number quoted below comes from it.

---

## 1. Fractional factorials — resolution and aliasing

**In JMP:** `DOE → Classical → Screening Design`, choose "Choose from a list of
fractional factorial designs".

For each case in `tool.json → fractional`:

1. Enter the stated number of continuous factors.
2. Pick the design with the matching run count (`n_runs`).
3. Read off the resolution JMP reports.
4. Open `Design Evaluation → Alias Matrix` and `Color Map on Correlations`.

**Check:**

| What | Where in tool.json | Must match |
|---|---|---|
| Run count | `n_runs` | JMP's run count |
| Resolution | `resolution` | JMP's stated resolution (III/IV/V/…) |
| Aliasing | `word_length_pattern` | The alias matrix's structure |

The word-length pattern is the strict test. It counts defining words of length
3, 4, 5, 6, 7. JMP does not print it directly, but a design with pattern
`[0, 7, 0, 0, 0]` must show exactly seven length-4 words in its defining
relation — i.e. seven rows of the alias structure pairing two-factor
interactions with each other, and none pairing a main effect with a 2FI.

> **Note.** JMP may offer a *different* design of the same size and resolution.
> Different minimum-aberration catalogues make different (equally valid) choices
> of generator. If the run count and word-length pattern agree, the designs are
> equivalent for our purposes even if the columns are permuted. Only escalate if
> the **pattern** differs.

---

## 2. Box-Behnken

**In JMP:** `DOE → Classical → Response Surface Design → Box-Behnken`.

For each case in `tool.json → box_behnken`:

- Run count must equal `n_runs` (which excludes centre points — JMP adds its own
  default centre points, so subtract them before comparing).
- The set of design points must match `matrix`, up to row order and column
  permutation.

---

## 3. Central composite

**In JMP:** `DOE → Classical → Response Surface Design → Central Composite`.

For each case in `tool.json → central_composite`:

- `n_factorial_points` and `n_axial_points` must match.
- `alpha` must match JMP's axial value. For the rotatable rule this is
  `(number of factorial points)^(1/4)`; for the face-centred rule it is exactly 1.

---

## 4. Power

**In JMP:** on any of the designs above, `Design Evaluation → Power Analysis`.

Enter an anticipated coefficient of half the target effect, in standard-deviation
units (the tool's convention is that `target_effect` is the change across a
factor's whole low-to-high range, so the coefficient is half of it), with
significance level 0.05.

JMP's power for each main effect must match the tool's, which you can read from:

```bash
python3 -c "
from doe_advisor.designs import classical as C
from doe_advisor.designs.spec import DesignSpec, Factor, Response, ModelOrder
from doe_advisor.designs.properties import power_report
spec = DesignSpec(
    factors=[Factor(f'x{i}', -1, 1) for i in range(4)],
    responses=[Response('y', target_effect=0.5, noise_sd=0.25)],
    model_order=ModelOrder.MAIN)
print(power_report(C.full_factorial(4, n_center=3), spec).per_term)
"
```

Agreement to three decimal places is expected; both use the non-central *t*.

---

## 5. Prediction variance

**In JMP:** `Design Evaluation → Prediction Variance Profile` and
`Fraction of Design Space Plot`.

JMP plots *relative* prediction variance (unscaled); the tool reports *scaled*
prediction variance, which is `n` times larger. Compare shapes and the ratio
between designs rather than absolute values, and check that the tool's
`g_efficiency` matches JMP's reported G-efficiency where JMP offers it.

---

## Recording the result

Create `validation/results/jmp.md` with:

- JMP version and the date.
- One line per case: agree / disagree, and the numbers if they disagree.
- Any case where JMP offered a different but equivalent design.

A disagreement in run count or word-length pattern is a bug in this tool until
proven otherwise. Open an issue and do not ship.
