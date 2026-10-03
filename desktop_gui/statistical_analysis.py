"""Desktop GUI for quick group comparisons (ANOVA, mean comparison, TOST, and
more) with automatic test recommendation.

All statistics come from the shared ``stat_board`` engine — the same core the
statistical board's agents use — so the GUI and the board can never disagree on
a number. The engine is located automatically by walking up from this file to
the repo root; set STAT_BOARD_HOME to override.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for flat imports below
from _bootstrap import ensure_stat_board_importable  # noqa: E402

_ENGINE_OK = ensure_stat_board_importable()

if _ENGINE_OK:
    import pandas as pd
    from PIL import Image, ImageTk

    from stat_board import figures, report
    from stat_board.engine import analyses
    from stat_board.engine.data import DataError, load_groups

    import recommend

    TEST_SPECS: dict[str, tuple[str, object]] = {
        "Student's t-test": ("two", lambda g, a: analyses.ttest(g, equal_var=True, alpha=a)),
        "Welch's t-test": ("two", lambda g, a: analyses.ttest(g, equal_var=False, alpha=a)),
        "Mann-Whitney U": ("two", lambda g, a: analyses.mann_whitney(g, alpha=a)),
        "Bayesian t-test": ("two", lambda g, a: analyses.bayes_ttest(g, alpha=a)),
        "Correlation": ("two", lambda g, a: analyses.correlation(g, alpha=a)),
        "TOST (equivalence)": ("tost", None),
        "One-way ANOVA": ("all", lambda g, a: analyses.anova(g, alpha=a)),
        "Welch ANOVA": ("all", lambda g, a: analyses.welch_anova(g, alpha=a)),
        "Kruskal-Wallis": ("all", lambda g, a: analyses.kruskal(g, alpha=a)),
        "Tukey HSD": ("all", lambda g, a: analyses.tukey_posthoc(g, alpha=a)),
    }
else:
    TEST_SPECS = {}

_TEST_DISPLAY_NAMES = {
    "ttest": "t-test",
    "mann_whitney": "Mann-Whitney U test",
    "anova": "one-way ANOVA",
    "welch_anova": "Welch's ANOVA",
    "kruskal": "Kruskal-Wallis test",
}

_NUM_SPLIT = re.compile(r"[,\s]+")


def _parse_numbers(text: str) -> tuple[list[float], int]:
    """Split free-form pasted text into floats, silently skipping and counting
    anything that isn't a number rather than blocking on the first bad token."""
    tokens = [t for t in _NUM_SPLIT.split(text.strip()) if t]
    values: list[float] = []
    skipped = 0
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            skipped += 1
    return values, skipped


def _compose_markdown(analysis: dict) -> str:
    test_name = _TEST_DISPLAY_NAMES.get(analysis["chosen_test"], analysis["chosen_test"])
    lines = [
        "# Statistical Analysis Report",
        "",
        f"**Recommended test:** {test_name} (alpha={analysis['alpha']:g})",
        "",
        "## Summary",
        "",
        recommend.plain_summary(analysis),
        "",
        "## Why this test",
        "",
    ]
    lines += [f"- {r}" for r in analysis["rationale"]]
    if analysis["caveats"]:
        lines += ["", "## Caveats", ""]
        lines += [f"- {c}" for c in analysis["caveats"]]
    lines += [
        "",
        "*The appendix below (figures and detailed tables) is generated directly "
        "from the data as supporting descriptive detail — it is not itself the "
        "recommendation above.*",
    ]
    return "\n".join(lines)


class StatisticalAnalysisApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Statistical Analysis Tool")
        self.root.geometry("1050x720")
        self.root.minsize(820, 600)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.datasets: dict[str, list[float]] = {}
        self._editing_name: str | None = None
        self.last_analysis: dict | None = None
        self._preview_photo = None
        self._preview_dir = Path(tempfile.mkdtemp(prefix="stat_gui_preview_"))

        self.alpha_var = tk.StringVar(value="0.05")
        self.entry_feedback_var = tk.StringVar(value="")
        self.recommend_hint_var = tk.StringVar(value="")
        self.ds1_var = tk.StringVar()
        self.ds2_var = tk.StringVar()
        self.advanced_test_var = tk.StringVar(
            value=next(iter(TEST_SPECS), "")
        )

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.data_tab = ttk.Frame(self.notebook, padding=10)
        self.results_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.data_tab, text="1. Data")
        self.notebook.add(self.results_tab, text="2. Results & Report")

        self._build_data_tab()
        self._build_results_tab()

        self._refresh_dataset_list()
        self._refresh_preview()
        self._update_recommend_button_state()

        if not _ENGINE_OK:
            messagebox.showwarning(
                "Engine not found",
                "Could not import the stat_board engine. Run this from inside the "
                "statistical-board repo, or set STAT_BOARD_HOME to the repo path.",
            )

    # ---- layout ------------------------------------------------------------

    def _build_data_tab(self) -> None:
        tab = self.data_tab
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Significance level (alpha):").pack(side="left")
        ttk.Spinbox(
            top, from_=0.001, to=0.5, increment=0.01, width=6, textvariable=self.alpha_var
        ).pack(side="left", padx=(6, 0))

        self.hint_label = ttk.Label(
            tab,
            text="Paste numbers below to add your first dataset, or use Import CSV/JSON.",
            foreground="#555555",
        )
        self.hint_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        entry_frame = ttk.LabelFrame(tab, text="Add / edit dataset", padding=8)
        entry_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 6))
        entry_frame.columnconfigure(0, weight=1)
        entry_frame.rowconfigure(2, weight=1)

        name_row = ttk.Frame(entry_frame)
        name_row.grid(row=0, column=0, sticky="ew")
        name_row.columnconfigure(1, weight=1)
        ttk.Label(name_row, text="Name:").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(name_row)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        ttk.Label(entry_frame, text="Values (comma, space, or newline separated):").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.paste_text = tk.Text(entry_frame, height=8, wrap="word")
        self.paste_text.grid(row=2, column=0, sticky="nsew", pady=(2, 4))

        self.entry_feedback_label = ttk.Label(
            entry_frame, textvariable=self.entry_feedback_var,
            foreground="#0a6e31", wraplength=280, justify="left",
        )
        self.entry_feedback_label.grid(row=3, column=0, sticky="w")

        btn_row = ttk.Frame(entry_frame)
        btn_row.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.add_update_btn = ttk.Button(
            btn_row, text="Add Dataset", command=self._add_or_update_dataset
        )
        self.add_update_btn.pack(side="left")
        ttk.Button(btn_row, text="Clear Fields", command=self._clear_entry_fields).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(btn_row, text="Import CSV/JSON...", command=self._import_file).pack(
            side="left", padx=(12, 0)
        )

        right_frame = ttk.Frame(tab)
        right_frame.grid(row=2, column=1, sticky="nsew", padx=(6, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        list_frame = ttk.LabelFrame(right_frame, text="Datasets (double-click to edit)", padding=8)
        list_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            list_frame, columns=("name", "n", "mean"), show="headings", height=6
        )
        for col, label, width, anchor in (
            ("name", "Dataset", 140, "w"), ("n", "n", 50, "center"), ("mean", "Mean", 90, "center"),
        ):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, anchor=anchor)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", self._on_dataset_double_click)
        tree_btns = ttk.Frame(list_frame)
        tree_btns.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(tree_btns, text="Remove Selected", command=self._remove_selected).pack(side="left")
        ttk.Button(tree_btns, text="Clear All", command=self._clear_all).pack(side="left", padx=(6, 0))

        preview_frame = ttk.LabelFrame(right_frame, text="Preview", padding=8)
        preview_frame.grid(row=1, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(
            preview_frame, text="Add at least one dataset to see a preview.", anchor="center"
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(tab)
        bottom.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.recommend_btn = ttk.Button(
            bottom, text="Recommend & Run Best Test →",
            command=self._recommend_and_run, state="disabled",
        )
        self.recommend_btn.pack(side="left")
        ttk.Label(bottom, textvariable=self.recommend_hint_var, foreground="#8a4b00").pack(
            side="left", padx=(10, 0)
        )

    def _build_results_tab(self) -> None:
        tab = self.results_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        summary_frame = ttk.LabelFrame(tab, text="Summary", padding=8)
        summary_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        summary_frame.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(summary_frame, height=6, wrap="word", state="disabled")
        self.summary_text.grid(row=0, column=0, sticky="ew")

        why_frame = ttk.LabelFrame(tab, text="Why this test", padding=8)
        why_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        why_frame.columnconfigure(0, weight=1)
        self.why_text = tk.Text(why_frame, height=6, wrap="word", state="disabled")
        self.why_text.grid(row=0, column=0, sticky="ew")

        adv_frame = ttk.LabelFrame(tab, text="Advanced: run a specific test", padding=8)
        adv_frame.grid(row=2, column=0, sticky="nsew")
        adv_frame.columnconfigure(0, weight=1)
        adv_frame.rowconfigure(4, weight=1)

        combo_row = ttk.Frame(adv_frame)
        combo_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(combo_row, text="Test:").pack(side="left")
        test_combo = ttk.Combobox(
            combo_row, textvariable=self.advanced_test_var,
            values=list(TEST_SPECS.keys()), state="readonly", width=22,
        )
        test_combo.pack(side="left", padx=(6, 0))
        test_combo.bind("<<ComboboxSelected>>", self._on_advanced_test_change)

        self.pair_frame = ttk.Frame(adv_frame)
        self.pair_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(self.pair_frame, text="Dataset 1:").pack(side="left")
        self.ds1_combo = ttk.Combobox(self.pair_frame, textvariable=self.ds1_var, state="readonly", width=16)
        self.ds1_combo.pack(side="left", padx=(4, 12))
        ttk.Label(self.pair_frame, text="Dataset 2:").pack(side="left")
        self.ds2_combo = ttk.Combobox(self.pair_frame, textvariable=self.ds2_var, state="readonly", width=16)
        self.ds2_combo.pack(side="left", padx=(4, 0))

        self.correlation_hint = ttk.Label(
            adv_frame,
            text="Correlation treats the two datasets as paired (x, y) — same length, matched order.",
            foreground="#8a4b00",
        )
        self.correlation_hint.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.correlation_hint.grid_remove()

        self.tost_frame = ttk.Frame(adv_frame)
        self.tost_frame.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(self.tost_frame, text="Margin % of dataset 1 mean:").pack(side="left")
        self.tost_margin_entry = ttk.Entry(self.tost_frame, width=6)
        self.tost_margin_entry.insert(0, "20")
        self.tost_margin_entry.pack(side="left", padx=(4, 12))
        ttk.Label(self.tost_frame, text="or explicit bounds — low:").pack(side="left")
        self.tost_low_entry = ttk.Entry(self.tost_frame, width=8)
        self.tost_low_entry.pack(side="left", padx=(4, 8))
        ttk.Label(self.tost_frame, text="high:").pack(side="left")
        self.tost_high_entry = ttk.Entry(self.tost_frame, width=8)
        self.tost_high_entry.pack(side="left", padx=(4, 0))
        self.tost_frame.grid_remove()

        ttk.Button(adv_frame, text="Run", command=self._run_advanced_test).grid(
            row=1, column=1, rowspan=3, sticky="n", padx=(12, 0)
        )

        self.advanced_results = scrolledtext.ScrolledText(
            adv_frame, height=8, wrap="word", state="disabled"
        )
        self.advanced_results.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

        ttk.Button(tab, text="Export PDF Report...", command=self._export_pdf).grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )

        self._on_advanced_test_change()

    # ---- data entry ----------------------------------------------------------

    def _alpha(self) -> float:
        try:
            return float(self.alpha_var.get())
        except (tk.TclError, ValueError):
            return 0.05

    def _clear_entry_fields(self) -> None:
        self.name_entry.delete(0, tk.END)
        self.paste_text.delete("1.0", tk.END)
        self._editing_name = None
        self.add_update_btn.configure(text="Add Dataset")
        self.entry_feedback_var.set("")

    def _add_or_update_dataset(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Missing name", "Enter a name for this dataset.")
            return
        values, skipped = _parse_numbers(self.paste_text.get("1.0", tk.END))
        if not values:
            messagebox.showerror("No values", "Paste at least one number.")
            return

        if self._editing_name is None and name in self.datasets:
            messagebox.showerror(
                "Name already used",
                f"A dataset named '{name}' already exists. Pick a different name, "
                "or double-click it in the list to edit it.",
            )
            return
        if self._editing_name is not None and self._editing_name != name:
            del self.datasets[self._editing_name]

        self.datasets[name] = values
        feedback = f"Saved '{name}' with {len(values)} value(s)."
        if skipped:
            feedback += f" ({skipped} entry/entries ignored — not numbers.)"
        self._clear_entry_fields()
        self.entry_feedback_var.set(feedback)
        self._refresh_dataset_list()
        self._refresh_preview()
        self._update_recommend_button_state()

    def _on_dataset_double_click(self, event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        name = sel[0]
        values = self.datasets[name]
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, name)
        self.paste_text.delete("1.0", tk.END)
        self.paste_text.insert("1.0", ", ".join(str(v) for v in values))
        self._editing_name = name
        self.add_update_btn.configure(text="Update Selected")
        self.entry_feedback_var.set(f"Editing '{name}' — change values and click Update Selected.")

    def _remove_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected", "Select a dataset in the list first.")
            return
        for name in sel:
            self.datasets.pop(name, None)
        if self._editing_name in sel:
            self._clear_entry_fields()
        self._refresh_dataset_list()
        self._refresh_preview()
        self._update_recommend_button_state()

    def _clear_all(self) -> None:
        if not self.datasets:
            return
        if not messagebox.askyesno("Clear all", "Remove every entered/imported dataset?"):
            return
        self.datasets.clear()
        self.last_analysis = None
        self._clear_entry_fields()
        self._refresh_dataset_list()
        self._refresh_preview()
        self._update_recommend_button_state()

    def _import_file(self) -> None:
        if not _ENGINE_OK:
            return
        path = filedialog.askopenfilename(
            title="Import dataset(s)",
            filetypes=[("CSV or JSON", "*.csv *.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            groups = load_groups(path)
        except DataError as e:
            if "ambiguous multi-column table" in str(e):
                groups = self._prompt_columns_and_load(path)
                if groups is None:
                    return
            else:
                messagebox.showerror("Import failed", str(e))
                return
        except Exception as e:
            messagebox.showerror("Import failed", str(e))
            return

        added = 0
        for name, values in groups.items():
            final_name = name
            suffix = 2
            while final_name in self.datasets:
                final_name = f"{name} ({suffix})"
                suffix += 1
            self.datasets[final_name] = values
            added += 1

        self._refresh_dataset_list()
        self._refresh_preview()
        self._update_recommend_button_state()
        messagebox.showinfo("Import complete", f"Imported {added} dataset(s) from {Path(path).name}.")

    def _prompt_columns_and_load(self, path: str) -> dict | None:
        try:
            preview_cols = list(pd.read_csv(path, nrows=5).columns)
        except Exception as e:
            messagebox.showerror("Import failed", f"Could not read columns: {e}")
            return None

        dialog = tk.Toplevel(self.root)
        dialog.title("Choose columns")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(
            dialog,
            text="This file has multiple columns — pick which one names the\n"
                 "group and which one holds the numeric value.",
            justify="left",
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 6), sticky="w")
        ttk.Label(dialog, text="Group column:").grid(row=1, column=0, sticky="e", padx=10)
        group_var = tk.StringVar(value=preview_cols[0])
        ttk.Combobox(dialog, textvariable=group_var, values=preview_cols, state="readonly").grid(
            row=1, column=1, padx=10, pady=4
        )
        ttk.Label(dialog, text="Value column:").grid(row=2, column=0, sticky="e", padx=10)
        value_var = tk.StringVar(value=preview_cols[-1])
        ttk.Combobox(dialog, textvariable=value_var, values=preview_cols, state="readonly").grid(
            row=2, column=1, padx=10, pady=4
        )

        choice: dict = {}

        def on_ok():
            choice["group_col"] = group_var.get()
            choice["value_col"] = value_var.get()
            dialog.destroy()

        btns = ttk.Frame(dialog)
        btns.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

        dialog.wait_window()

        if not choice:
            return None
        try:
            return load_groups(path, group_col=choice["group_col"], value_col=choice["value_col"])
        except DataError as e:
            messagebox.showerror("Import failed", str(e))
            return None

    # ---- shared refresh helpers ---------------------------------------------

    def _refresh_dataset_list(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for name, values in self.datasets.items():
            n = len(values)
            mean = sum(values) / n if n else float("nan")
            self.tree.insert("", tk.END, iid=name, values=(name, n, f"{mean:.4g}"))
        if self.datasets:
            self.hint_label.grid_remove()
        else:
            self.hint_label.grid()
        self._refresh_dataset_combos()

    def _refresh_dataset_combos(self) -> None:
        names = list(self.datasets.keys())
        self.ds1_combo.configure(values=names)
        self.ds2_combo.configure(values=names)
        if self.ds1_var.get() not in names:
            self.ds1_var.set(names[0] if names else "")
        if self.ds2_var.get() not in names:
            self.ds2_var.set(names[1] if len(names) > 1 else (names[0] if names else ""))

    def _refresh_preview(self) -> None:
        if not _ENGINE_OK or not self.datasets:
            self.preview_label.configure(image="", text="Add at least one dataset to see a preview.")
            self._preview_photo = None
            return
        try:
            info = figures.boxplot(dict(self.datasets), self._preview_dir)
            img = Image.open(self._preview_dir / info["file"])
            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self._preview_photo, text="")
        except Exception as e:
            self.preview_label.configure(image="", text=f"Preview unavailable: {e}")
            self._preview_photo = None

    def _update_recommend_button_state(self) -> None:
        eligible = (
            _ENGINE_OK and len(self.datasets) >= 2 and all(len(v) >= 2 for v in self.datasets.values())
        )
        self.recommend_btn.configure(state="normal" if eligible else "disabled")
        if eligible or not self.datasets:
            self.recommend_hint_var.set("")
        elif len(self.datasets) < 2:
            self.recommend_hint_var.set("Add at least 2 datasets to get a recommendation.")
        else:
            small = [n for n, v in self.datasets.items() if len(v) < 2]
            self.recommend_hint_var.set("Need at least 2 values in: " + ", ".join(small))

    # ---- recommend & run -----------------------------------------------------

    def _recommend_and_run(self) -> None:
        try:
            result = recommend.analyze(dict(self.datasets), alpha=self._alpha())
        except Exception as e:
            messagebox.showerror("Analysis failed", str(e))
            return
        self.last_analysis = result
        self._render_results(result)
        self.notebook.select(self.results_tab)

    def _render_results(self, analysis: dict) -> None:
        summary = recommend.plain_summary(analysis)
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", summary)
        self.summary_text.configure(state="disabled")

        self.why_text.configure(state="normal")
        self.why_text.delete("1.0", tk.END)
        for line in analysis["rationale"]:
            self.why_text.insert(tk.END, f"• {line}\n")
        for line in analysis["caveats"]:
            self.why_text.insert(tk.END, f"⚠ {line}\n")
        self.why_text.configure(state="disabled")

    # ---- advanced (manual) tests ----------------------------------------------

    def _on_advanced_test_change(self, event=None) -> None:
        name = self.advanced_test_var.get()
        kind = TEST_SPECS.get(name, ("two", None))[0]
        (self.pair_frame.grid_remove if kind == "all" else self.pair_frame.grid)()
        (self.tost_frame.grid if kind == "tost" else self.tost_frame.grid_remove)()
        (self.correlation_hint.grid if name == "Correlation" else self.correlation_hint.grid_remove)()

    def _run_advanced_test(self) -> None:
        name = self.advanced_test_var.get()
        if name not in TEST_SPECS:
            return
        kind, fn = TEST_SPECS[name]
        alpha = self._alpha()

        try:
            if kind == "all":
                if len(self.datasets) < 2:
                    messagebox.showerror("Not enough data", "Need at least 2 datasets.")
                    return
                result = fn(dict(self.datasets), alpha)
            else:
                d1, d2 = self.ds1_var.get(), self.ds2_var.get()
                if not d1 or not d2 or d1 == d2 or d1 not in self.datasets or d2 not in self.datasets:
                    messagebox.showerror("Pick two datasets", "Choose two different datasets to compare.")
                    return
                pair = {d1: self.datasets[d1], d2: self.datasets[d2]}
                if kind == "tost":
                    low_s = self.tost_low_entry.get().strip()
                    high_s = self.tost_high_entry.get().strip()
                    if low_s and high_s:
                        result = analyses.tost(pair, low=float(low_s), high=float(high_s), alpha=alpha)
                    else:
                        margin = float(self.tost_margin_entry.get().strip() or 20)
                        result = analyses.tost(pair, margin_pct=margin, alpha=alpha)
                else:
                    result = fn(pair, alpha)
        except Exception as e:
            messagebox.showerror("Test failed", str(e))
            return

        self._append_advanced_result(name, result)

    def _append_advanced_result(self, name: str, result: dict) -> None:
        self.advanced_results.configure(state="normal")
        self.advanced_results.insert(tk.END, f"— {name} —\n")
        if "conclusion" in result:
            self.advanced_results.insert(tk.END, result["conclusion"] + "\n")
        if name == "Tukey HSD":
            for p in result["pairs"]:
                verdict = "differ" if p["reject_null"] else "no significant difference"
                self.advanced_results.insert(
                    tk.END,
                    f"  {p['group1']} vs {p['group2']}: mean diff={p['mean_diff']:.4g}, "
                    f"p_adj={p['p_adj']:.4g} ({verdict})\n",
                )
        self.advanced_results.insert(tk.END, "\n")
        self.advanced_results.configure(state="disabled")
        self.advanced_results.see(tk.END)

    # ---- PDF export ------------------------------------------------------------

    def _export_pdf(self) -> None:
        if not _ENGINE_OK:
            return
        if self.last_analysis is None:
            messagebox.showinfo(
                "Run an analysis first",
                "Click 'Recommend & Run Best Test' before exporting a report.",
            )
            return
        chosen = filedialog.asksaveasfilename(
            title="Save PDF report", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")]
        )
        if not chosen:
            return
        out_path = Path(chosen)

        try:
            with tempfile.TemporaryDirectory() as tmp_name:
                tmp = Path(tmp_name)
                wide = pd.DataFrame({k: pd.Series(v) for k, v in self.datasets.items()})
                long_df = wide.melt(var_name="group", value_name="value").dropna(subset=["value"])
                csv_path = tmp / "data.csv"
                long_df.to_csv(csv_path, index=False)

                md_path = tmp / "report.md"
                md_path.write_text(_compose_markdown(self.last_analysis))

                report.convert_file(
                    md_path, out_path,
                    data_path=csv_path, group_col="group", value_col="value",
                    alpha=self.last_analysis["alpha"],
                )
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            return

        messagebox.showinfo(
            "Report saved",
            f"Saved {out_path.name} (with a '{out_path.stem}_assets' folder of "
            f"figures next to it) to:\n{out_path.parent}",
        )

    # ---- lifecycle ---------------------------------------------------------

    def _on_close(self) -> None:
        shutil.rmtree(self._preview_dir, ignore_errors=True)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StatisticalAnalysisApp(root)
    root.mainloop()
