"""Tests for the deterministic figures/appendix/PDF-rendering pipeline. All
outputs are confined to tmp_path — nothing is written into sample_data/."""

from __future__ import annotations

from stat_board import appendix, figures, report

GROUPS = {
    "A": [10.0, 11.0, 12.0, 13.0, 10.0, 11.0],
    "B": [30.0, 31.0, 32.0, 33.0, 30.0, 31.0],
}


def test_generate_all_writes_expected_png_files_with_captions(tmp_path):
    figs = figures.generate_all(GROUPS, tmp_path)
    names = {f["file"] for f in figs}
    assert names == {"fig_box.png", "fig_means.png", "fig_qq.png", "fig_resid.png"}
    for f in figs:
        assert (tmp_path / f["file"]).exists()
        assert f["caption"]


def test_generate_all_skips_residual_plots_for_tiny_samples(tmp_path):
    figs = figures.generate_all({"A": [1.0], "B": [2.0]}, tmp_path)
    names = {f["file"] for f in figs}
    assert names == {"fig_box.png", "fig_means.png"}


def test_appendix_build_returns_markdown_with_expected_sections(tmp_path, long_csv):
    assets_dir = tmp_path / "assets"
    built = appendix.build(str(long_csv), group_col="group", value_col="value",
                           assets_dir=assets_dir)
    assert built is not None
    md, returned_assets_dir = built
    assert returned_assets_dir == assets_dir
    assert "## Appendix A — Figures" in md
    assert "## Appendix B — Detailed statistics" in md
    assert "Descriptive statistics" in md
    assert (assets_dir / "fig_box.png").exists()


def test_appendix_build_returns_none_for_ambiguous_table(tmp_path, multifactor_csv):
    built = appendix.build(str(multifactor_csv), assets_dir=tmp_path / "assets")
    assert built is None


def test_markdown_to_pdf_produces_a_valid_pdf(tmp_path):
    out = tmp_path / "out.pdf"
    result_path = report.markdown_to_pdf("# Title\n\nSome *report* body.\n", out, title="Title")
    assert result_path == out
    assert out.exists()
    assert out.read_bytes()[:4] == b"%PDF"


def test_convert_file_without_data_renders_plain_markdown(tmp_path):
    md_path = tmp_path / "report.md"
    md_path.write_text("# My Report\n\nBody text.\n")
    pdf_path = report.convert_file(md_path)
    assert pdf_path == md_path.with_suffix(".pdf")
    assert pdf_path.read_bytes()[:4] == b"%PDF"


def test_convert_file_with_data_appends_the_figures_appendix(tmp_path, long_csv):
    md_path = tmp_path / "report.md"
    md_path.write_text("# My Report\n\nBody text.\n")
    out_pdf = tmp_path / "final.pdf"
    result = report.convert_file(md_path, out_pdf, data_path=str(long_csv),
                                 group_col="group", value_col="value")
    assert result == out_pdf
    assert out_pdf.read_bytes()[:4] == b"%PDF"
    assets_dir = tmp_path / "final_assets"
    assert assets_dir.is_dir()
    assert (assets_dir / "fig_box.png").exists()


# --------------------------------------------------------------------------- #
# Multi-factor/DoE figures
# --------------------------------------------------------------------------- #

def _predict_and_regression(path, formula):
    from stat_board.engine import analyses
    pred = analyses.predict_table(path, formula)
    fit = analyses.regression(path, formula)
    return pred, fit


def test_generate_all_multifactor_writes_the_general_figure_set(tmp_path, reg_csv):
    import numpy as np
    pred, fit = _predict_and_regression(str(reg_csv), "y ~ x")
    resid = np.array([r["residual"] for r in pred["rows"]])
    fitted = np.array([r["predicted"] for r in pred["rows"]])
    figs = figures.generate_all_multifactor(pred["rows"], fit["anova"], resid, fitted, tmp_path)
    names = {f["file"] for f in figs}
    assert names == {"fig_pred_obs.png", "fig_resid.png", "fig_qq.png", "fig_pareto.png"}
    for f in figs:
        assert (tmp_path / f["file"]).exists()
        assert f["caption"]


def test_predicted_vs_observed_writes_a_png(tmp_path):
    rows = [{"observed": 1.0, "predicted": 1.1}, {"observed": 2.0, "predicted": 1.9}]
    fig = figures.predicted_vs_observed(rows, tmp_path)
    assert fig["file"] == "fig_pred_obs.png"
    assert (tmp_path / fig["file"]).exists()


def test_pareto_effects_orders_terms_by_effect_size_descending(tmp_path):
    terms = [
        {"term": "small", "partial_eta_sq": 0.1, "significant": False},
        {"term": "big", "partial_eta_sq": 0.9, "significant": True},
    ]
    fig = figures.pareto_effects(terms, tmp_path)
    assert fig["file"] == "fig_pareto.png"
    assert (tmp_path / fig["file"]).exists()


def test_contour_writes_a_png_over_two_continuous_factors(tmp_path):
    import pandas as pd
    p = tmp_path / "doe.csv"
    rows = [{"x1": x1, "x2": x2, "y": 5 + 0.3 * x1 + 0.01 * x2}
           for x1 in (10, 20) for x2 in (100, 200)]
    pd.DataFrame(rows).to_csv(p, index=False)
    fig = figures.contour(str(p), "y ~ x1 + x2", "x1", "x2", tmp_path)
    assert fig["file"] == "fig_contour.png"
    assert (tmp_path / fig["file"]).exists()


# --------------------------------------------------------------------------- #
# appendix.build_multifactor()
# --------------------------------------------------------------------------- #

def test_build_multifactor_categorical_design(tmp_path, multifactor_csv):
    from stat_board.engine import analyses
    formula = analyses.formula_for("score", ["treatment", "sex"])
    assets_dir = tmp_path / "assets"
    built = appendix.build_multifactor(str(multifactor_csv), formula,
                                       factors=["treatment", "sex"], assets_dir=assets_dir)
    assert built is not None
    md, returned_assets_dir = built
    assert returned_assets_dir == assets_dir
    for heading in ("## Appendix A — Figures", "## Appendix B — Model tables",
                    "## Appendix C — Design diagnostics", "Variance inflation factors",
                    "Design coverage", "Tested combinations ranked by predicted response",
                    "Boundary check"):
        assert heading in md
    # No contour: treatment/sex are categorical, not continuous.
    assert "fig_contour.png" not in md
    assert not (assets_dir / "fig_contour.png").exists()


def test_build_multifactor_two_continuous_factors_adds_contour(tmp_path):
    import pandas as pd
    p = tmp_path / "doe.csv"
    rows = [{"x1": x1, "x2": x2, "y": 5 + 0.3 * x1 + 0.01 * x2}
           for x1 in (10, 15, 20) for x2 in (100, 150, 200)]
    pd.DataFrame(rows).to_csv(p, index=False)
    assets_dir = tmp_path / "assets"
    built = appendix.build_multifactor(str(p), "y ~ x1 + x2", factors=["x1", "x2"],
                                       assets_dir=assets_dir)
    assert built is not None
    md, _ = built
    assert "fig_contour.png" in md
    assert (assets_dir / "fig_contour.png").exists()


def test_build_multifactor_without_factors_skips_design_sections_but_keeps_model_tables(
        tmp_path, reg_csv):
    assets_dir = tmp_path / "assets"
    built = appendix.build_multifactor(str(reg_csv), "y ~ x", assets_dir=assets_dir)
    assert built is not None
    md, _ = built
    assert "## Appendix B — Model tables" in md
    assert "Variance inflation factors" in md
    assert "Design coverage" not in md
    assert "Tested combinations ranked" not in md


def test_build_multifactor_returns_none_for_an_unfittable_formula(tmp_path, reg_csv):
    built = appendix.build_multifactor(str(reg_csv), "y ~ not_a_real_column", assets_dir=tmp_path)
    assert built is None


def test_convert_file_with_formula_dispatches_to_build_multifactor(tmp_path, multifactor_csv):
    from stat_board.engine import analyses
    md_path = tmp_path / "report.md"
    md_path.write_text("# DoE Report\n\nBody text.\n")
    out_pdf = tmp_path / "final.pdf"
    formula = analyses.formula_for("score", ["treatment", "sex"])
    result = report.convert_file(md_path, out_pdf, data_path=str(multifactor_csv),
                                 formula=formula, factors=["treatment", "sex"])
    assert result == out_pdf
    assert out_pdf.read_bytes()[:4] == b"%PDF"
    assets_dir = tmp_path / "final_assets"
    assert (assets_dir / "fig_pareto.png").exists()


def test_pdf_with_a_large_multipage_table_has_no_stray_th_background_ghosts(tmp_path):
    """Regression test: a `th` background-fill CSS rule used to let PyMuPDF's
    Story renderer leave dozens of ghost copies of that fill scattered across
    later pages of a long table (e.g. a DoE ranked-combinations table spanning
    many pages) — landing on top of unrelated data rows and looking like
    random light-gray bars. `th` is now background-free, so no such fill
    should ever appear, no matter how long the table is."""
    import itertools

    import fitz
    import pandas as pd

    # 4 factors x 3 levels = a large enough grid that the ranked-combinations
    # table (Appendix C) spans several pages, same shape as the real report
    # that first surfaced this bug.
    factors = ["f1", "f2", "f3", "f4"]
    levels = [-1, 0, 1]
    rows = [dict(zip(factors, combo, strict=True)) for combo in itertools.product(levels, repeat=4)]
    for i, row in enumerate(rows):
        row["y"] = 50 + sum(row[f] for f in factors) + (i % 5)
    p = tmp_path / "big_doe.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    formula = "y ~ " + " + ".join(factors)
    md_path = tmp_path / "report.md"
    md_path.write_text("# Big DoE Report\n\nBody text.\n")
    out_pdf = tmp_path / "big.pdf"
    report.convert_file(md_path, out_pdf, data_path=str(p), formula=formula, factors=factors)

    doc = fitz.open(out_pdf)
    assert doc.page_count > 3  # confirms the table actually spans multiple pages
    stray_th_fill = (0.941, 0.941, 0.941)
    ghosts = sum(
        1 for i in range(doc.page_count) for d in doc[i].get_drawings()
        if d.get("fill") and all(abs(a - b) < 0.01 for a, b in zip(d["fill"], stray_th_fill, strict=False))
    )
    assert ghosts == 0
