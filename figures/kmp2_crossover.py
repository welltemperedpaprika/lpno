#!/usr/bin/env python3
"""KPNO-MP2 and canonical k-point MP2 crossover benchmarks.

Timings use 64 threads. Si, diamond, BN, and LiH use pob-TZVP; NH3/DZ uses
cc-pVDZ.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "axes.titlesize": "medium",
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset": "cm",
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.right": True,
    }
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "kmp2_crossover_data.csv"

STYLE = {
    "Si (3D)": ("C2", "o", r"Si (3D)"),
    "Diamond (3D)": ("C7", "v", r"diamond (3D)"),
    "BN (2D)": ("C4", "^", r"BN (2D)"),
    "NH3/DZ (3D)": ("C1", "P", r"NH$_3$/DZ (3D)"),
    "LiH (3D)": ("C0", "s", r"LiH (3D)"),
}


def main() -> None:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, ax = plt.subplots(figsize=(5.05, 3.35))
    for system, (color, marker, label) in STYLE.items():
        system_rows = sorted(
            (row for row in rows if row["system"] == system),
            key=lambda row: int(row["nk"]),
        )
        if len(system_rows) < 2:
            raise ValueError(f"{system}: need at least two current-driver points")
        xs = [int(row["nk"]) for row in system_rows]
        ys = [float(row["ratio"]) for row in system_rows]
        ax.loglog(
            xs,
            ys,
            ls="-",
            lw=1.15,
            marker=marker,
            ms=5.3,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=3,
            label=label,
        )

    ax.axhline(1.0, color="0.25", lw=0.85, ls="--", zorder=1)
    ax.text(105, 1.25, "canonical faster", fontsize=6.5, color="0.4")
    ax.text(105, 0.68, "KPNO-MP2 faster", fontsize=6.5, color="0.4")
    ax.set_xlabel(r"$N_k$")
    ax.set_ylabel(r"$t_\mathrm{KPNO}\,/\,t_\mathrm{KMP2}$")
    ax.set_xticks([8, 27, 64, 81, 343])
    ax.set_xticklabels(["8", "27", "64", "81", "343"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(7, 420)
    ax.set_ylim(0.08, 40)
    ax.legend(
        fontsize=6.2,
        loc="lower left",
        framealpha=0.9,
        ncol=2,
        handletextpad=0.4,
        labelspacing=0.3,
        columnspacing=0.8,
    )

    fig.savefig(HERE / "kmp2_crossover.png", dpi=220)
    fig.savefig(HERE / "kmp2_crossover.pdf")
    print("Wrote kmp2_crossover.png and kmp2_crossover.pdf")


if __name__ == "__main__":
    main()
