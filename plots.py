"""
plots.py - figures for the report and slides.

Colours (fixed per policy, never reshuffled): shared = blue, independent = orange,
attention = aqua, greedy reference = grey. Lines show the mean over training seeds,
faint dots show each seed. Hollow diamonds = policy trained directly on that team size.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLORS = {"shared": "#2a78d6", "independent": "#eb6834", "attention": "#1baf7a", "greedy": "#8a8a85"}
LABELS = {"shared": "Shared policy", "independent": "Independent policies",
          "attention": "Attention policy", "greedy": "Greedy (no avoidance)"}
INK, GRID = "#0b0b0b", "#e3e2de"

METRICS = [
    ("min_separation", "Minimum separation\n(higher = safer)"),
    ("violation_rate", "Violation rate\n(fraction of steps with a collision, lower = safer)"),
    ("success_rate", "Success rate\n(agents on their goal at the end)"),
]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _xticks(ax, sizes):
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n}\n(trained)" if i == 0 else f"{n}" for i, n in enumerate(sizes)])
    ax.set_xlabel("number of agents at test time", color=INK, fontsize=9)


def _line(ax, results, name, sizes, metric, color, linestyle="-", marker="o", seed_dots=True):
    ys = [np.mean(results[name][n][metric]) for n in sizes]
    ax.plot(sizes, ys, color=color, linewidth=2, linestyle=linestyle, marker=marker, markersize=7)
    if seed_dots:
        for n in sizes:
            v = results[name][n][metric]
            ax.scatter([n] * len(v), v, s=10, color=color, alpha=0.45, zorder=3)


def plot_transfer(results, sizes, path):
    kinds = [k for k in ("shared", "independent", "attention", "greedy") if k in results]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.5, 4.4))
    has_ref = False
    for ax, (metric, title) in zip(axes, METRICS):
        _style(ax)
        for kind in kinds:
            _line(ax, results, kind, sizes, metric, COLORS[kind])
            for n, row in results.get(kind + "@ref", {}).items():      # trained directly on n agents
                has_ref = True
                ax.scatter([n], [np.mean(row[metric])], s=80, marker="D", facecolors="none",
                           edgecolors=COLORS[kind], linewidths=2, zorder=4)
        _xticks(ax, sizes)
        ax.set_title(title, fontsize=10, color=INK)
    handles = [Line2D([0], [0], color=COLORS[k], linewidth=2, marker="o", label=LABELS[k]) for k in kinds]
    if has_ref:
        handles.append(Line2D([0], [0], color=INK, linestyle="", marker="D", markerfacecolor="none",
                              markeredgewidth=2, label="Trained directly on that size"))
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=len(handles), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_shield(results, sizes, path):
    kinds = [k for k in ("shared", "independent", "attention") if k + "+shield" in results]
    if not kinds:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (metric, title) in zip(axes, [("violation_rate", "Violation rate (lower = safer)"),
                                           ("success_rate", "Success rate (cost of safety)")]):
        _style(ax)
        for kind in kinds:
            _line(ax, results, kind, sizes, metric, COLORS[kind], "-", "o", seed_dots=False)
            _line(ax, results, kind + "+shield", sizes, metric, COLORS[kind], "--", "s", seed_dots=False)
        _xticks(ax, sizes)
        ax.set_title(title, fontsize=10, color=INK)
    handles = [Line2D([0], [0], color=COLORS[k], linewidth=2, label=LABELS[k]) for k in kinds]
    handles += [Line2D([0], [0], color=INK, linewidth=2, linestyle="-", label="without shield"),
                Line2D([0], [0], color=INK, linewidth=2, linestyle="--", label="with shield")]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=len(handles), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training(run_dir, seeds, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
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
            ax.plot(x, smooth.mean(0), color=COLORS[kind], linewidth=2, label=LABELS[kind])
            ax.fill_between(x, smooth.min(0), smooth.max(0), color=COLORS[kind], alpha=0.15)
        ax.set_xlabel("training iteration", color=INK)
        ax.set_title(title, fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
