"""Compare this tool's answers with R's, and report every disagreement.

    python3 -m validation.run_tool
    Rscript validation/validate_r.R
    python3 -m validation.compare

Exits non-zero when anything disagrees, so it can gate a release. "Does it match
JMP?" is the first question any statistician will ask; this is the machine-
checkable half of the answer, and ``JMP_INSTRUCTIONS.md`` is the manual half.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOL = 1e-9
# Mixed models are fitted by two different optimisers on the same REML surface
# (statsmodels here, lme4 in R), so they agree to optimiser precision, not to
# machine precision — about 1e-8 with the tight settings both sides use.
# Everything downstream inherits that, so one relative tolerance covers
# variance components, standard errors, F and df alike.
KR_RTOL = 1e-6


class Comparison:
    """Accumulates checks so that one mismatch does not hide the others."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, bool]] = []

    def check(self, case: str, field: str, ours, theirs, tol: float = TOL) -> None:
        if isinstance(ours, float) or isinstance(theirs, float):
            ok = abs(float(ours) - float(theirs)) <= tol
        else:
            ok = ours == theirs
        self.rows.append((case, field, _fmt(ours), _fmt(theirs), ok))

    @property
    def failures(self) -> list[tuple]:
        return [r for r in self.rows if not r[4]]

    def report(self, title: str) -> None:
        total = len(self.rows)
        bad = len(self.failures)
        mark = "OK" if bad == 0 else "MISMATCH"
        print(f"\n{title}: {total - bad}/{total} agree  [{mark}]")
        for case, field, ours, theirs, ok in self.rows:
            if not ok:
                print(f"    {case:22s} {field:24s} tool={ours}  R={theirs}")


def _fmt(value) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_fmt(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _index(entries: list[dict]) -> dict[str, dict]:
    return {e["case"]: e for e in entries}


def compare_fractional(ours: list[dict], theirs: list[dict], comparison: Comparison) -> None:
    mine, reference = _index(ours), _index(theirs)
    for case in sorted(set(mine) & set(reference)):
        a, b = mine[case], reference[case]
        comparison.check(case, "n_runs", a["n_runs"], b["n_runs"])
        comparison.check(case, "resolution", float(a["resolution"]), float(b["resolution"]))
        comparison.check(
            case,
            "word_length_pattern",
            [float(v) for v in a["word_length_pattern"][:5]],
            [float(v) for v in b["word_length_pattern"][:5]],
        )


def compare_box_behnken(ours: list[dict], theirs: list[dict], comparison: Comparison) -> None:
    mine, reference = _index(ours), _index(theirs)
    for case in sorted(set(mine) & set(reference)):
        a, b = mine[case], reference[case]
        comparison.check(case, "n_runs", a["n_runs"], b["n_runs"])
        # Row order is arbitrary, so both sides are canonicalised before
        # comparing the point set.
        comparison.check(
            case,
            "design points",
            sorted(tuple(row) for row in a["matrix"]),
            sorted(tuple(float(v) for v in row) for row in b["matrix"]),
        )


def compare_central_composite(ours: list[dict], theirs: list[dict], comparison: Comparison) -> None:
    mine, reference = _index(ours), _index(theirs)
    for case in sorted(set(mine) & set(reference)):
        a, b = mine[case], reference[case]
        for field in ("n_runs", "n_factorial_points", "n_axial_points"):
            comparison.check(case, field, a[field], b[field])
        comparison.check(case, "alpha", float(a["alpha"]), float(b["alpha"]), tol=1e-9)


def compare_power(ours: list[dict], theirs: list[dict], comparison: Comparison) -> None:
    mine, reference = _index(ours), _index(theirs)
    for case in sorted(set(mine) & set(reference)):
        a, b = mine[case], reference[case]
        comparison.check(case, "n_runs", a["n_runs"], b["n_runs"])
        comparison.check(case, "residual_df", a["residual_df"], b["residual_df"])
        for kind in ("main", "interaction", "curvature"):
            # R writes NA as null; the tool writes None when the model has no such term.
            if a[kind] is None or b.get(kind) is None:
                comparison.check(case, f"power {kind} present", a[kind] is None, b.get(kind) is None)
            else:
                comparison.check(case, f"power {kind}", float(a[kind]), float(b[kind]), tol=1e-6)


def compare_blocking(ours: list[dict], theirs: list[dict], comparison: Comparison) -> None:
    mine, reference = _index(ours), _index(theirs)
    for case in sorted(set(mine) & set(reference)):
        comparison.check(
            case,
            "block word lengths",
            [float(v) for v in mine[case]["block_word_lengths"]],
            # jsonlite unboxes a one-element vector to a scalar.
            [float(v) for v in np.atleast_1d(reference[case]["block_word_lengths"])],
        )


def compare_kenward_roger(ours: list[dict], theirs: list[dict], comparison: Comparison) -> None:
    """Mixed-model fits and Kenward-Roger tests against lme4 + pbkrtest."""

    def close(case: str, field: str, a: float, b: float) -> None:
        tol = KR_RTOL * max(1.0, abs(float(b)))
        comparison.check(case, field, float(a), float(b), tol=tol)

    mine, reference = _index(ours), _index(theirs)
    for case in sorted(set(mine) & set(reference)):
        a, b = mine[case], reference[case]
        comparison.check(case, "method", a["df_method"], "Kenward-Roger")
        comparison.check(case, "n", a["n"], b["n"])
        # A whole-plot variance on the boundary is zero on both sides; scale by the residual one.
        comparison.check(case, "var_group", float(a["var_group"]), float(b["var_group"]),
                         tol=KR_RTOL * max(1.0, float(b["var_resid"])))
        close(case, "var_resid", a["var_resid"], b["var_resid"])
        for name, x, y in zip(a["coef_names"], a["coef"], np.atleast_1d(b["coef"]), strict=True):
            close(case, f"coef {name}", x, y)
        for name, x, y in zip(a["coef_names"], a["se_plugin"], np.atleast_1d(b["se_plugin"]), strict=True):
            close(case, f"se {name}", x, y)
        for name, x, y in zip(a["coef_names"], a["se_adjusted"], np.atleast_1d(b["se_adjusted"]), strict=True):
            close(case, f"KR se {name}", x, y)
        theirs_tests = {t["term"]: t for t in b["tests"]}
        for t in a["tests"]:
            r = theirs_tests[t["term"]]
            comparison.check(case, f"{t['term']} num df", int(t["num_df"]), int(r["num_df"]))
            close(case, f"{t['term']} F", t["F"], r["F"])
            close(case, f"{t['term']} den df", t["den_df"], r["den_df"])
            close(case, f"{t['term']} F scale", t["scale"], r["scale"])
            comparison.check(case, f"{t['term']} p", float(t["p"]), float(r["p"]),
                             tol=KR_RTOL * max(float(r["p"]), 1e-6))


def check_dsd(ours: list[dict], comparison: Comparison) -> None:
    """No R reference is installed, so check the published defining properties."""
    for entry in ours:
        k = entry["n_factors"]
        comparison.check(entry["case"], "runs == 2m+1", entry["n_runs"], 2 * entry["conference_order"] + 1)
        comparison.check(entry["case"], "three levels", entry["levels"], [-1.0, 0.0, 1.0])
        comparison.check(
            entry["case"],
            "main effects clear of 2FIs",
            entry["max_main_by_2fi_inner_product"],
            0.0,
            tol=1e-9,
        )
        comparison.check(entry["case"], "conference order >= factors", entry["conference_order"] >= k, True)


def main() -> int:
    tool_path, r_path = RESULTS_DIR / "tool.json", RESULTS_DIR / "r.json"
    if not tool_path.exists():
        print("missing validation/results/tool.json — run: python3 -m validation.run_tool", file=sys.stderr)
        return 2
    ours = json.loads(tool_path.read_text(encoding="utf-8"))

    print("=" * 72)
    print("doe-advisor cross-validation")
    print("=" * 72)

    overall: list[Comparison] = []

    dsd = Comparison()
    check_dsd(ours["dsd"], dsd)
    dsd.report("Definitive screening designs (published properties)")
    overall.append(dsd)

    if not r_path.exists():
        print("\nNo R results found — run: Rscript validation/validate_r.R")
        print("Skipping the R comparison.")
    else:
        theirs = json.loads(r_path.read_text(encoding="utf-8"))
        print(f"\nReference: R {theirs.get('r_version')}, FrF2 {theirs['packages']['FrF2']}, "
              f"rsm {theirs['packages']['rsm']}")

        for title, key, fn in (
            ("Fractional factorials vs FrF2", "fractional", compare_fractional),
            ("Box-Behnken vs rsm::bbd", "box_behnken", compare_box_behnken),
            ("Central composite vs rsm::ccd", "central_composite", compare_central_composite),
            ("Power per term vs R (model.matrix + pt)", "power", compare_power),
            ("Blocked factorials vs FrF2(blocks=)", "blocking", compare_blocking),
        ):
            comparison = Comparison()
            fn(ours.get(key, []), theirs.get(key, []), comparison)
            comparison.report(title)
            overall.append(comparison)

    kr_path = RESULTS_DIR / "r_kr.json"
    if not kr_path.exists():
        print("\nNo R Kenward-Roger results found — run: Rscript validation/validate_kr.R")
    else:
        theirs_kr = json.loads(kr_path.read_text(encoding="utf-8"))
        print(f"\nReference: R {theirs_kr.get('r_version')}, lme4 {theirs_kr['packages']['lme4']}, "
              f"pbkrtest {theirs_kr['packages']['pbkrtest']}")
        comparison = Comparison()
        compare_kenward_roger(ours.get("kenward_roger", []), theirs_kr["kenward_roger"], comparison)
        comparison.report("Mixed model + Kenward-Roger vs lme4/pbkrtest")
        overall.append(comparison)

    failures = sum(len(c.failures) for c in overall)
    checks = sum(len(c.rows) for c in overall)
    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED: {failures} of {checks} checks disagree.")
        return 1
    print(f"PASSED: all {checks} checks agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
