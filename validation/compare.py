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

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOL = 1e-9


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
        ):
            comparison = Comparison()
            fn(ours.get(key, []), theirs.get(key, []), comparison)
            comparison.report(title)
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
