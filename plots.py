"""
plots.py - figures for the report and slides.

Colours (fixed per policy, never reshuffled): shared = blue, independent = orange,
attention = aqua, greedy reference = grey. Lines show the mean over training seeds,
faint dots show each seed. Hollow diamonds = policy trained directly on that team size.

Every function takes the same optional arguments so one set of data can be drawn at
different sizes:
    figsize  figure size in inches
    fs       font/line scale factor (1.0 = default; use ~1.2 for a paper, ~1.5 for slides)
    short    use short panel titles (put the long definitions in the caption instead)
    top      fraction of the figure height left for the legend row(s)
    ncol     legend columns
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLORS = {"shared": "#2a78d6", "independent": "#eb6834", "attention": "#1baf7a", "attention_knn": "#6250d6",
          "greedy": "#8a8a85"}
LABELS = {"shared": "Shared policy", "independent": "Independent policies",
          "attention": "Attention policy (sees all agents)", "attention_knn": "Attention policy (2 nearest only)",
          "greedy": "Greedy (no avoidance)"}
MARKERS = {"attention_knn": "^"}      # the ablation is drawn with triangles so it stays visible on top of "attention"
INK, GRID = "#0b0b0b", "#e3e2de"

METRICS = [
    ("min_separation", "Minimum separation\n(higher = safer)"),
    ("violation_rate", "Violation rate\n(fraction of steps with a collision, lower = safer)"),
    ("success_rate", "Success rate\n(agents on their goal at the end)"),
]
METRICS_SHORT = [
    ("min_separation", "Minimum separation ↑"),
    ("violation_rate", "Violation rate ↓"),
    ("success_rate", "Success rate ↑"),
]


def _rc(fs):
    return plt.rc_context({"font.size": 9 * fs, "axes.titlesize": 10 * fs, "axes.labelsize": 9 * fs,
                           "xtick.labelsize": 9 * fs, "ytick.labelsize": 9 * fs,
                           "legend.fontsize": 10 * fs})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _xticks(ax, sizes, short=False):
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n}\n(trained)" if i == 0 else f"{n}" for i, n in enumerate(sizes)])
    ax.set_xlabel("agents at test time" if short else "number of agents at test time", color=INK)


def _line(ax, results, name, sizes, metric, color, fs, linestyle="-", marker="o", seed_dots=True):
    ys = [np.mean(results[name][n][metric]) for n in sizes]
    ax.plot(sizes, ys, color=color, linewidth=2 * fs, linestyle=linestyle, marker=marker, markersize=7 * fs)
    if seed_dots:
        for n in sizes:
            v = results[name][n][metric]
            ax.scatter([n] * len(v), v, s=10 * fs * fs, color=color, alpha=0.45, zorder=3)


def plot_transfer(results, sizes, path, figsize=(13.5, 4.4), fs=1.0, short=False, top=0.92, ncol=None, dpi=160):
    kinds = [k for k in ("shared", "independent", "attention", "attention_knn", "greedy") if k in results]
    metrics = METRICS_SHORT if short else METRICS
    with _rc(fs):
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        has_ref = False
        for ax, (metric, title) in zip(axes, metrics):
            _style(ax)
            for kind in kinds:
                _line(ax, results, kind, sizes, metric, COLORS[kind], fs, marker=MARKERS.get(kind, "o"))
                for n, row in results.get(kind + "@ref", {}).items():      # trained directly on n agents
                    has_ref = True
                    ax.scatter([n], [np.mean(row[metric])], s=80 * fs * fs, marker="D", facecolors="none",
                               edgecolors=COLORS[kind], linewidths=2 * fs, zorder=4)
            _xticks(ax, sizes, short)
            ax.set_title(title, color=INK)
        handles = [Line2D([0], [0], color=COLORS[k], linewidth=2 * fs, marker=MARKERS.get(k, "o"),
                          label=LABELS[k]) for k in kinds]
        if has_ref:
            handles.append(Line2D([0], [0], color=INK, linestyle="", marker="D", markerfacecolor="none",
                                  markeredgewidth=2 * fs, label="Trained directly on that size"))
        fig.legend(handles=handles, frameon=False, ncol=ncol or len(handles), loc="upper center")
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.savefig(path, dpi=dpi)
        plt.close(fig)


def plot_training(run_dir, seeds, path, figsize=(10, 3.8), fs=1.0, top=1.0, dpi=160):
    with _rc(fs):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, (key, title) in zip(axes, [("mean_return", "Training return"),
                                           ("violation_rate", "Training collision rate")]):
            _style(ax)
            for kind in ("shared", "independent", "attention"):
                curves = []
                for s in seeds:
                    p = os.path.join(run_dir, f"{kind}_seed{s}_log.json")
                    if os.path.exists(p):
                        curves.append([row[key] for row in json.load(open(p))])
                if not curves:
                    continue
                c = np.array(curves)
                k = 10                                           # smooth with a moving average
                smooth = np.array([np.convolve(r, np.ones(k) / k, mode="valid") for r in c])
                x = np.arange(smooth.shape[1]) + k
                ax.plot(x, smooth.mean(0), color=COLORS[kind], linewidth=2 * fs, label=LABELS[kind])
                ax.fill_between(x, smooth.min(0), smooth.max(0), color=COLORS[kind], alpha=0.15)
            ax.set_xlabel("training iteration", color=INK)
            ax.set_title(title, color=INK)
        axes[0].legend(frameon=False)
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
