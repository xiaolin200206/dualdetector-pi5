#!/usr/bin/env python3
"""
make_fig_schematic.py - Figure 1: the test bench and the timing quantities used in the paper.

  python -I analysis/make_fig_schematic.py --out results

Panel b uses the clean latencies of the main campaign (phased policy, Table 4) and an
offset of 0.42 s at P = 1.5 s, the smallest tested offset at which contact stopped.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_results as M  # noqa: E402

L_LEAF, L_PEST, P, O = 0.388, 0.154, 1.5, 0.42


def box(ax, x, y, w, h, text, fc="#F4F4F4", ec="#555555", fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, color="#444444"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=color, shrinkA=0, shrinkB=0))


def panel_a(ax):
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("a  Test bench (detectors run in ONNX Runtime on the CPU)", loc="left")
    box(ax, 0.00, 0.60, 0.19, 0.26, "Camera\n640 × 480\nMJPG\n(or replay)")
    box(ax, 0.235, 0.60, 0.17, 0.26, "Capture\nthread\n(newest\nframe)")
    box(ax, 0.45, 0.60, 0.17, 0.26, "Scheduler\n(release\nrule)", fc="#FFF3E0")
    box(ax, 0.68, 0.75, 0.32, 0.17, "Leaf detector\nYOLO11s", fc="#E3F2FD")
    box(ax, 0.68, 0.54, 0.32, 0.17, "Pest detector\nYOLO11n or YOLO11s", fc="#E3F2FD")
    arrow(ax, 0.19, 0.73, 0.235, 0.73)
    arrow(ax, 0.405, 0.73, 0.45, 0.73)
    arrow(ax, 0.62, 0.77, 0.68, 0.83)
    arrow(ax, 0.62, 0.69, 0.68, 0.625)
    box(ax, 0.00, 0.06, 0.40, 0.36,
        "Raspberry Pi 5 (shared)\n4 × Cortex-A76, 2.4 GHz\n2 MB shared L3 cache\nLPDDR4X memory\nActive Cooler",
        fc="#FAFAFA")
    box(ax, 0.45, 0.06, 0.55, 0.36,
        "Logs\nper inference: release, start,\nend, frame used\n0.5 s: temperature, clock,\nthrottle flags, fan\n"
        "2.0 s (+ 0–0.5 s after the main\ncampaign): PMIC rail power", fc="#FAFAFA")
    arrow(ax, 0.84, 0.54, 0.84, 0.42)


def dim(ax, x0, x1, y, label, color="black", above=True):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=color, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y + (0.06 if above else -0.07), label, ha="center",
            va="bottom" if above else "top", fontsize=7, color=color)


def panel_b(ax):
    ax.set_title("b  Release geometry", loc="left")
    y_leaf, y_pest, h = 1.0, 0.5, 0.28
    ax.barh(y_leaf, L_LEAF, left=0, height=h, color=M.C_CL)
    ax.barh(y_leaf, L_LEAF, left=P, height=h, color=M.C_CL)
    ax.barh(y_pest, L_PEST, left=O, height=h, color=M.C_OR)
    for t in (0, P):
        ax.axvline(t, color="#999999", lw=0.6, ls=":")
    ax.set_yticks([y_leaf, y_pest])
    ax.set_yticklabels(["leaf", "pest"])
    ax.set_xlim(-0.05, P + 0.5)
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlabel("Time from a leaf release (s)")
    ax.grid(False)
    dim(ax, 0, L_LEAF, 1.2, r"$L_{leaf}$")
    dim(ax, 0, O, 0.22, r"$o$", above=False)
    dim(ax, O, O + L_PEST, 0.22, r"$L_{pest}$", above=False)
    dim(ax, 0, P, -0.42, r"$P$", above=False)
    t_on = 1.0
    ax.plot([t_on, t_on], [0.25, 1.35], color=M.C_OV, lw=0.9)
    ax.text(t_on - 0.03, 1.42, "image onset", color=M.C_OV, fontsize=6.3, ha="right", va="bottom")
    dim(ax, t_on, P, 1.2, r"$w$", color=M.C_OV)
    ax.text(0.98, 0.98, "no overlap if\n" r"$o \geq L_{leaf}$ and $o + L_{pest} \leq P$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fd = os.path.join(a.out, "figures")
    os.makedirs(fd, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
    panel_a(axs[0])
    panel_b(axs[1])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fd, "fig01_schematic." + ext))
    plt.close(fig)


if __name__ == "__main__":
    main()
