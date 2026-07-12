"""Render a Markdown report to PDF — pure Python, no LaTeX or headless browser.

Pipeline: Markdown -> HTML (python-markdown, with tables) -> PDF (PyMuPDF's
Story layout engine). Used by the /stat-board skill so the final report always
ships as a PDF.

    python3 -m stat_board.report <input.md> [output.pdf]

If the output path is omitted it is the input path with a .pdf suffix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown as _md
import pymupdf

# A print stylesheet Story understands (it supports a practical subset of CSS).
_CSS = """
* { font-family: sans-serif; }
body { font-size: 10pt; line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 6pt 0; color: #ffffff;
     background-color: #0f5e6e; padding: 10pt 12pt; }
h2 { font-size: 13pt; margin: 14pt 0 4pt 0; color: #0b3d47;
     background-color: #cbeef0; padding: 5pt 10pt; }
h3 { font-size: 11pt; margin: 10pt 0 3pt 0; color: #0f5e6e; }
p { margin: 4pt 0; }
em { color: #555; }
h1 em, h1 code { color: #ffc55c; font-style: normal; background: none; }
code { font-family: monospace; background: #f2f2f2; font-size: 9pt; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0; font-size: 9pt; }
th { background: #f0f0f0; text-align: left; padding: 4pt 6pt;
     border: 1px solid #cccccc; }
td { padding: 4pt 6pt; border: 1px solid #dddddd; vertical-align: top; }
li { margin: 2pt 0; }
"""


# Background-fill colours used by the stylesheet (H1 teal, H2 light-teal, th/code
# grays). PyMuPDF's Story re-draws these fills onto continuation pages where the
# element does not actually belong — heading bands bleed into the top margin, and
# table-header fills bleed mid-page behind the figures. Both are removed below.
_BG_FILLS = [(0.06, 0.37, 0.43), (0.80, 0.93, 0.94), (0.94, 0.94, 0.94), (0.95, 0.95, 0.95)]


def _is_bg_fill(f) -> bool:
    return f is not None and any(
        all(abs(a - b) < 0.04 for a, b in zip(f, c)) for c in _BG_FILLS)


def _strip_top_bleed(doc) -> None:
    """Remove Story's phantom background bands from continuation pages, leaving
    text and images intact. A fill is a phantom if it either intrudes into the top
    margin (a bled heading band over running text) or is a stylesheet background
    colour with NO text on it (a bled table header — a legitimate band always has
    its heading/header text). Best-effort; never fails the render."""
    try:
        frame_top = 54.0  # matches the 0.75in top margin used when placing the story
        for pno in range(1, doc.page_count):
            pg = doc[pno]
            strays = []
            for dr in pg.get_drawings():
                f = dr.get("fill")
                if f is None:
                    continue
                r = dr["rect"]
                if r.width < 8 or r.height < 3:
                    continue
                top_bleed = r.y0 < frame_top - 3 and r.width > 120
                phantom = (_is_bg_fill(f) and r.height < 40
                           and not pg.get_textbox(r).strip())
                if top_bleed or phantom:
                    strays.append(r)
            if not strays:
                continue
            for r in strays:
                pg.add_redact_annot(r)
            pg.apply_redactions(text=pymupdf.PDF_REDACT_TEXT_NONE,
                                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
    except Exception:
        pass


def _shrink(path: Path) -> None:
    """Strip the Story top-margin bleed, subset embedded fonts, and recompress.
    PyMuPDF's Story embeds full fonts, which dominate the file size (a Unicode
    fallback face alone can be >1 MB); subsetting to the glyphs actually used
    typically shrinks the PDF ~10x. Never fails the render over optimization."""
    try:
        doc = pymupdf.open(path)
        _strip_top_bleed(doc)
        try:
            doc.subset_fonts(verbose=False)
        except Exception:
            pass  # older PyMuPDF without subset_fonts — deflate below still helps
        tmp = path.with_suffix(".slim.pdf")
        doc.save(tmp, garbage=4, deflate=True, clean=True)
        doc.close()
        tmp.replace(path)
    except Exception:
        pass


def markdown_to_pdf(md_text: str, out_path: str | Path, *, title: str | None = None,
                    image_root: str | Path | None = None) -> Path:
    """Convert Markdown text to a PDF file. Returns the output path. If
    ``image_root`` is given, ``<img src="name.png">`` tags resolve against it."""
    out_path = Path(out_path)
    body = _md.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
    )
    head = f"<title>{title}</title>" if title else ""
    html = f"<html><head>{head}<style>{_CSS}</style></head><body>{body}</body></html>"

    archive = pymupdf.Archive(str(image_root)) if image_root else None
    story = pymupdf.Story(html=html, archive=archive)
    writer = pymupdf.DocumentWriter(str(out_path))
    mediabox = pymupdf.paper_rect("letter")
    frame = mediabox + (54, 54, -54, -54)  # 0.75in margins

    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(frame)
        story.draw(device)
        writer.end_page()
    writer.close()
    _shrink(out_path)
    return out_path


def convert_file(in_path: str | Path, out_path: str | Path | None = None, *,
                 data_path: str | None = None, group_col: str | None = None,
                 value_col: str | None = None, alpha: float = 0.05) -> Path:
    """Render a Markdown report to PDF. If ``data_path`` is given, a deterministic
    appendix (figures + detailed statistics tables) computed from that dataset is
    generated and appended before rendering."""
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_suffix(".pdf")
    out_path = Path(out_path)
    md_text = in_path.read_text()
    image_root: Path | None = None

    if data_path:
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
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    out = convert_file(args.input, args.output, data_path=args.data,
                       group_col=args.group_col, value_col=args.value_col, alpha=args.alpha)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
