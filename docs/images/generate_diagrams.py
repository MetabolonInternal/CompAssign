"""
Generate documentation diagrams for README.

Outputs:
  - docs/images/system_overview.png
  - docs/images/al_loop.png

Usage:
  poetry run python docs/images/generate_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import Rectangle


IMG_DIR = Path(__file__).resolve().parent


def _add_labeled_box(ax, x: float, y: float, w: float, h: float, label: str,
                     edge: str = "#222", face: str = "#F5F7FA", fs: int = 10) -> dict:
    rect = Rectangle((x, y), w, h, linewidth=1.2, edgecolor=edge, facecolor=face)
    ax.add_patch(rect)
    cx, cy = x + w / 2, y + h / 2
    txt = ax.text(cx, cy, label, ha="center", va="center", fontsize=fs)
    txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
    return {"left": x, "right": x + w, "top": y + h, "bottom": y, "cx": cx, "cy": cy}


def generate_system_overview(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    W, H = 0.16, 0.36
    Y = 0.32
    xs = [0.04, 0.24, 0.44, 0.64, 0.84]
    labels = [
        "Peak list\n+ library",
        "Hierarchical RT\n(run covariates)",
        "Candidate\nwindows",
        "Peak assignment\n(calibrated probs)",
        "Human review",
    ]

    boxes = []
    for x, lab in zip(xs, labels):
        boxes.append(_add_labeled_box(ax, x, Y, W, H, lab))

    # Straight arrows left->right (edge to edge)
    for i in range(len(xs) - 1):
        x0 = boxes[i]["right"]
        x1 = boxes[i + 1]["left"]
        y = boxes[i]["cy"]
        ax.annotate(
            "",
            xy=(x1 - 0.01, y),
            xytext=(x0 + 0.01, y),
            arrowprops=dict(arrowstyle="->", color="#222", lw=1.4),
        )

    # Active learning elbow: from Human review top center to RT top center
    hr = boxes[-1]
    rt = boxes[1]
    color = "#1a73e8"
    pad_y = 0.12

    p0 = (hr["cx"], hr["top"])  # start at top center
    p1 = (hr["cx"], hr["top"] + pad_y)
    p2 = (rt["cx"], hr["top"] + pad_y)
    p3 = (rt["cx"], rt["top"] + 0.002)

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=1.8)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1.8)
    ax.annotate("", xy=p3, xytext=p2, arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

    mx = (p1[0] + p2[0]) / 2
    my = p1[1] + 0.02
    ax.text(
        mx,
        my,
        "Active learning",
        color=color,
        ha="center",
        va="bottom",
        fontsize=10,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def generate_al_panel(path: Path) -> None:
    # Compact figure to avoid excessive whitespace
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer frame to enclose title, main row, and optional box
    frame_x, frame_y, frame_w, frame_h = 0.02, 0.04, 0.96, 0.92
    frame = Rectangle((frame_x, frame_y), frame_w, frame_h,
                      linewidth=1.6, edgecolor="#1a73e8", facecolor="#EFF5FF")
    ax.add_patch(frame)
    # Title inside the frame near the bottom to avoid overlap
    ax.text(frame_x + frame_w / 2, frame_y + 0.02, "Active learning",
            ha="center", va="bottom", fontsize=12, color="#1a73e8",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Inner boxes (main path), no outer frame
    inner_y = 0.60
    inner_h = 0.28
    margin = 0.03
    base_x = 0.05
    total_w = 0.90
    labels = [
        "Model",
        "Request K labels\nfrom reviewer",
        "Update priors\n+ retrain",
        "Recompute\npredictions",
    ]
    inner_w = (total_w - margin * 5) / 4.0
    xs = [base_x + margin + i * (inner_w + margin) for i in range(4)]

    for x, lab in zip(xs, labels):
        r = Rectangle((x, inner_y), inner_w, inner_h, linewidth=1.0, edgecolor="#222", facecolor="white")
        ax.add_patch(r)
        ax.text(x + inner_w / 2, inner_y + inner_h / 2, lab, ha="center", va="center", fontsize=10)

    # Arrows between main boxes
    for i in range(3):
        x0 = xs[i] + inner_w
        x1 = xs[i + 1]
        y = inner_y + inner_h / 2
        ax.annotate(
            "",
            xy=(x1 - 0.005, y),
            xytext=(x0 + 0.005, y),
            arrowprops=dict(arrowstyle="->", color="#222", lw=1.2),
        )

    # Optional posterior update box (no connectors), centered below main row
    opt_w, opt_h = inner_w, 0.20
    opt_x = xs[2]
    opt_y = inner_y - (opt_h + 0.06)
    opt = Rectangle((opt_x, opt_y), opt_w, opt_h, linewidth=1.0, edgecolor="#777", facecolor="white")
    ax.add_patch(opt)
    ax.text(
        opt_x + opt_w / 2,
        opt_y + opt_h / 2,
        "Posterior update\n(optional)",
        ha="center",
        va="center",
        fontsize=10,
        color="#555",
    )

    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    generate_system_overview(IMG_DIR / "system_overview.png")
    generate_al_panel(IMG_DIR / "al_loop.png")


if __name__ == "__main__":
    main()
