"""Render a statistical report to PDF, with an optional data-driven appendix.

The Markdown -> PDF rendering itself lives in :mod:`pdstat.report`, shared with
the DoE advisor so both halves of the package produce identically styled
documents. This module adds what is specific to a statistical report: building
the figures/tables appendix from the dataset and gluing it onto the Markdown
before rendering.

    python3 -m stat_board.report <input.md> [output.pdf]

If the output path is omitted it is the input path with a .pdf suffix.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pdstat.report import markdown_to_pdf

# Re-exported so existing callers (stat_board.transcript, the desktop GUI) can
# keep importing it from here.
__all__ = ["markdown_to_pdf", "convert_file", "main"]



def convert_file(in_path: str | Path, out_path: str | Path | None = None, *,
                 data_path: str | None = None, group_col: str | None = None,
                 value_col: str | None = None, alpha: float = 0.05,
                 formula: str | None = None, factors: list[str] | None = None,
                 typ: int = 2) -> Path:
    """Render a Markdown report to PDF. If ``data_path`` is given, a deterministic
    appendix (figures + detailed statistics tables) computed from that dataset is
    generated and appended before rendering. Pass ``formula`` for a multi-factor/
    DoE analysis (regression/two-way-ANOVA/ANCOVA all reduce to one fitted patsy
    formula) instead of ``group_col``/``value_col`` — the two paths are mutually
    exclusive; ``formula`` takes precedence when both are given."""
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_suffix(".pdf")
    out_path = Path(out_path)
    md_text = in_path.read_text()
    image_root: Path | None = None

    if formula and data_path:
        from . import appendix  # local import: keeps matplotlib off the fast path
        built = appendix.build_multifactor(
            data_path, formula, factors=factors, typ=typ, alpha=alpha,
            assets_dir=out_path.parent / f"{out_path.stem}_assets")
        if built is not None:
            appendix_md, image_root = built
            md_text = md_text.rstrip() + "\n" + appendix_md
    elif data_path:
        from . import appendix  # local import: keeps matplotlib off the fast path
        built = appendix.build(data_path, group_col=group_col, value_col=value_col,
                               alpha=alpha, assets_dir=out_path.parent / f"{out_path.stem}_assets")
        if built is not None:
            appendix_md, image_root = built
            md_text = md_text.rstrip() + "\n" + appendix_md

    title = next((ln[2:].strip() for ln in md_text.splitlines() if ln.startswith("# ")), None)
    return markdown_to_pdf(md_text, out_path, title=title, image_root=image_root)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m stat_board.report",
        description="Render a Markdown report to PDF (optionally with a data-driven "
                    "appendix of figures and detailed statistics tables).")
    parser.add_argument("input", help="Markdown report file.")
    parser.add_argument("output", nargs="?", help="Output PDF (default: input with .pdf).")
    parser.add_argument("--data", help="Dataset to build the figures/tables appendix from.")
    parser.add_argument("--group-col", help="Long-format: group-label column.")
    parser.add_argument("--value-col", help="Long-format: value column.")
    parser.add_argument("--formula", help="Multi-factor/DoE analysis: the fitted patsy formula "
                                          "(regression/two-way-ANOVA/ANCOVA all reduce to one). "
                                          "Requires --data. Takes precedence over --group-col/"
                                          "--value-col.")
    parser.add_argument("--factor", action="append", dest="factors",
                       help="Multi-factor/DoE: a factor column (repeat for each). Enables the "
                            "design-coverage/curvature/optimum-ranking appendix sections.")
    parser.add_argument("--type", type=int, choices=[1, 2, 3], default=2, dest="typ",
                       help="--formula: ANOVA SS type (default 2).")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.formula and not args.data:
        parser.error("--formula requires --data")

    out = convert_file(args.input, args.output, data_path=args.data,
                       group_col=args.group_col, value_col=args.value_col, alpha=args.alpha,
                       formula=args.formula, factors=args.factors, typ=args.typ)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
