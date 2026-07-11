"""Desktop GUI for quick group comparisons (ANOVA, mean comparison, TOST).

All statistics come from the shared ``stat_board`` engine — the same core the
statistical board's agents use — so the GUI and the board can never disagree on
a number. The engine is located automatically by walking up from this file to
the repo root; set STAT_BOARD_HOME to override.
"""

import os
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np


def _load_engine():
    try:
        from stat_board.engine import analyses
        return analyses
    except ModuleNotFoundError:
        pass
    candidates = []
    if os.environ.get("STAT_BOARD_HOME"):
        candidates.append(Path(os.environ["STAT_BOARD_HOME"]))
    candidates += list(Path(__file__).resolve().parents)      # repo root (bundled)
    for cand in candidates:
        if cand and (Path(cand) / "stat_board" / "engine").is_dir():
            sys.path.insert(0, str(cand))
            try:
                from stat_board.engine import analyses
                return analyses
            except ModuleNotFoundError:
                continue
    return None


engine = _load_engine()


class StatisticalAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Statistical Analysis Tool")
        self.root.geometry("800x600")

        self.datasets = []
        self.current_data = []

        self.create_widgets()

        if engine is None:
            messagebox.showwarning(
                "Engine not found",
                "Could not import the stat_board engine. Run this from inside the "
                "statistical-board repo, or set STAT_BOARD_HOME to the repo path.",
            )

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        data_frame = ttk.LabelFrame(main_frame, text="Data Entry", padding="5")
        data_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.data_entry = ttk.Entry(data_frame)
        self.data_entry.grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(data_frame, text="Add Value", command=self.add_value).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="New Dataset", command=self.new_dataset).grid(row=0, column=2, padx=5)
        ttk.Button(data_frame, text="Clear All", command=self.clear_all).grid(row=0, column=3, padx=5)

        display_frame = ttk.LabelFrame(main_frame, text="Current Dataset", padding="5")
        display_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.data_display = scrolledtext.ScrolledText(display_frame, height=5)
        self.data_display.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        analysis_frame = ttk.LabelFrame(main_frame, text="Statistical Analysis", padding="5")
        analysis_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(analysis_frame, text="Run ANOVA", command=self.run_anova).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(analysis_frame, text="Compare Means", command=self.compare_means).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(analysis_frame, text="Run TOST (α=0.05)", command=lambda: self.run_tost(0.05)).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(analysis_frame, text="Run TOST (α=0.1)", command=lambda: self.run_tost(0.1)).grid(row=0, column=3, padx=5, pady=5)

        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.results_display = scrolledtext.ScrolledText(results_frame, height=10)
        self.results_display.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

    def add_value(self):
        try:
            value = float(self.data_entry.get())
            self.current_data.append(value)
            self.update_display()
            self.data_entry.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")

    def new_dataset(self):
        if self.current_data:
            self.datasets.append(self.current_data)
            self.current_data = []
            self.update_display()
            messagebox.showinfo("Success", f"Dataset {len(self.datasets)} created")

    def clear_all(self):
        self.datasets = []
        self.current_data = []
        self.update_display()
        self.results_display.delete(1.0, tk.END)

    def update_display(self):
        self.data_display.delete(1.0, tk.END)
        if self.current_data:
            self.data_display.insert(tk.END, "Current dataset: " +
                                   ", ".join(map(str, self.current_data)) + "\n")
        for i, dataset in enumerate(self.datasets):
            self.data_display.insert(tk.END, f"Dataset {i+1}: " +
                                   ", ".join(map(str, dataset)) + "\n")

    # ---- Analysis: numbers come from the shared stat_board engine ------------

    def _named_groups(self):
        return {f"Dataset {i+1}": ds for i, ds in enumerate(self.datasets)}

    def _require(self, minimum, label):
        if engine is None:
            messagebox.showerror("Engine not found",
                                 "The stat_board engine is not available. Set "
                                 "STAT_BOARD_HOME and relaunch.")
            return False
        if len(self.datasets) < minimum:
            messagebox.showerror("Error", f"Need at least {minimum} datasets for {label}")
            return False
        return True

    def run_anova(self):
        if not self._require(2, "ANOVA"):
            return
        r = engine.anova(self._named_groups())
        result = "One-way ANOVA (stat_board engine):\n"
        result += f"F-statistic: {r['F']:.4f}\np-value: {r['p']:.4f}\n"
        result += f"eta^2 (effect size): {r['eta_squared']:.4f} ({r['effect_magnitude']})\n"
        result += r["conclusion"] + "\n"
        self.results_display.insert(tk.END, result + "\n\n")

    def compare_means(self):
        if not self._require(2, "mean comparison"):
            return
        result = "Mean Comparison — Student's t-test (stat_board engine):\n"
        for i, dataset1 in enumerate(self.datasets):
            for j, dataset2 in enumerate(self.datasets[i+1:], i+1):
                pair = {f"Dataset {i+1}": dataset1, f"Dataset {j+1}": dataset2}
                r = engine.ttest(pair, equal_var=True)
                result += f"Dataset {i+1} vs Dataset {j+1}:\n"
                result += f"t-statistic: {r['t']:.4f}\np-value: {r['p']:.4f}\n"
                result += f"Cohen's d: {r['cohens_d']:.4f} ({r['effect_magnitude']})\n"
                result += ("Significant difference found\n" if r["significant"]
                           else "No significant difference found\n")
                result += "\n"
        self.results_display.insert(tk.END, result)

    def run_tost(self, alpha):
        if not self._require(2, "TOST"):
            return
        mean1 = np.mean(self.datasets[0])
        bound = 0.2 * abs(mean1)
        result = f"TOST equivalence (α={alpha}, ±20% of Dataset 1 mean, stat_board engine):\n"
        for i, dataset1 in enumerate(self.datasets):
            for j, dataset2 in enumerate(self.datasets[i+1:], i+1):
                pair = {f"Dataset {i+1}": dataset1, f"Dataset {j+1}": dataset2}
                r = engine.tost(pair, low=-bound, high=bound, alpha=alpha)
                result += f"Dataset {i+1} vs Dataset {j+1}:\n"
                result += f"Lower p-value: {r['p_lower']:.4f}\nUpper p-value: {r['p_upper']:.4f}\n"
                result += ("Datasets are equivalent\n" if r["equivalent"]
                           else "Datasets are not equivalent\n")
                result += "\n"
        self.results_display.insert(tk.END, result)


if __name__ == "__main__":
    root = tk.Tk()
    app = StatisticalAnalysisApp(root)
    root.mainloop()
