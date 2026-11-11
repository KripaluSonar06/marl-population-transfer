"""
make_figures.py - redraw the figures of the report (paper size) and of the slides (bigger text)
from runs/results.json, so every figure can be regenerated exactly.

Usage:
    python make_figures.py --run-dir runs --out figures
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plots import (plot_transfer, plot_training, COLORS, LABELS, MARKERS, _style)


def load(path):
    #results.json stores team sizes as strings; plots.py wants integers."""
    raw = json.load(open(path))
    return {name: {int(n): m for n, m in by_n.items()} for name, by_n in raw.items()}


def plot_transfer_slide(results, sizes, path, figsize=(6.8, 3.0)):
    #Two panels for the slides: episodes with a collision (%) and goal success (%), large text, no seed dots."""
    kinds = [k for k in ("shared", "independent", "attention", "attention_knn", "greedy") if k in results]
    panels = [("episodes_with_collision", "Episodes with a collision (%)  \u2193"), ("success_rate", "Goal success (%)  \u2191")]
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 9.5}):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, (metric, title) in zip(axes, panels):
            _style(ax)
            for kind in kinds:
                ys = [100 * np.mean(results[kind][n][metric]) for n in sizes]
                ax.plot(sizes, ys, color=COLORS[kind], linewidth=2.4, marker=MARKERS.get(kind, "o"), markersize=6)
            ax.set_xticks(sizes)
            ax.set_xticklabels([f"{n}\n(trained)" if i == 0 else f"{n}" for i, n in enumerate(sizes)])
            ax.set_xlabel("agents at test time")
            ax.set_title(title)
            if metric == "episodes_with_collision":
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 25, 50, 75, 100])
        handles = [Line2D([0], [0], color=COLORS[k], linewidth=2.4, marker=MARKERS.get(k, "o"), label=LABELS[k]) for k in kinds]
        fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center")
        fig.tight_layout(rect=(0, 0, 1, 0.80))
        fig.savefig(path, dpi=200)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs")
    p.add_argument("--out", default="figures")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6])
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    results = load(os.path.join(a.run_dir, "results.json"))
    # report versions
    plot_transfer(results, a.sizes, f"{a.out}/transfer_metrics.png", figsize=(13.5, 5.0), fs=1.2, top=0.84, ncol=3)
    plot_training(a.run_dir, a.seeds, f"{a.out}/training_curves.png")
    # slide versions: drawn at the size they are shown, so the text stays readable
    plot_transfer_slide(results, a.sizes, f"{a.out}/transfer_slide.png")
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
