#!/usr/bin/env python3
"""Per-stage wall time vs orbitals per cell at a fixed 2x2x2 k-mesh.

Each system ran cc-pVDZ -> TZ -> QZ sequentially on one node, so the
section ratios carry no cross-node scatter (benzene and oxalic-alpha
are DZ/TZ 2-point series). T_PNO = 1e-8. Data: orbital_scaling_data.csv.
Panels: 'DF ERIs' (3-center DF transform), 'OSV build' (the basis-cubic
step), 'PNO build', 'Residual solve' (amplitude iterations).
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

SYSTEMS = [
    ('NH3', 'C1', 'P', r'NH$_3$'),
    ('CO2', 'C3', 'X', r'CO$_2$'),
    ('urea', 'C0', 's', 'urea'),
    ('oxalic-beta', 'C4', 'D', r'oxalic-$\beta$'),
    ('oxalic-alpha', 'C2', '^', r'oxalic-$\alpha$'),
    ('benzene', 'C5', 'o', 'benzene'),
]
SECTIONS = [
    ('wall_df', 'DF ERIs'),
    ('wall_osv', 'OSV build'),
    ('wall_pairdom', 'PNO build'),
    ('wall_resid', 'Residual solve'),
]


def main():
    data = {}
    with open(HERE / 'orbital_scaling_data.csv') as f:
        for row in csv.DictReader(f):
            data.setdefault(row['system'], []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
    for ax, (col, title) in zip(axes.flat, SECTIONS):
        slopes = []
        for system, color, marker, label in SYSTEMS:
            recs = data.get(system, [])
            x = np.array([float(r['n']) for r in recs])
            y = np.array([float(r[col]) for r in recs])
            slope = np.polyfit(np.log(x), np.log(y), 1)[0]
            slopes.append(slope)
            ax.loglog(x, y, marker=marker, ms=4, color=color, lw=1.0,
                      markerfacecolor='none', markeredgewidth=0.8,
                      label=label)
        ax.set_title(title, fontsize=9)
        ax.text(0.97, 0.05, rf'$n^{{{min(slopes):.1f}}}$--$n^{{{max(slopes):.1f}}}$',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    xg = np.array([300.0, 700.0])
    axes.flat[1].loglog(xg, 6e0 * (xg / xg[0]) ** 3, ls=':', color='gray', lw=0.9)
    axes.flat[1].annotate(r'$n^3$', (xg[-1], 6e0 * (xg[-1] / xg[0]) ** 3),
                          textcoords='offset points', xytext=(3, -3),
                          fontsize=6.5, color='gray')
    axes.flat[3].loglog(xg, 6e1 * (xg / xg[0]) ** 1, ls=':', color='gray', lw=0.9)
    axes.flat[3].annotate(r'$n^1$', (xg[-1], 6e1 * (xg[-1] / xg[0]) ** 1),
                          textcoords='offset points', xytext=(3, -3),
                          fontsize=6.5, color='gray')
    for ax in axes.flat:
        ax.set_xticks([100, 200, 400, 800])
        ax.set_xticklabels(['100', '200', '400', '800'])
        ax.set_xticks([], minor=True)
        ax.set_xlim(95, 1200)
    for ax in axes[1]:
        ax.set_xlabel(r'orbitals per cell $n$')
    for ax in axes[:, 0]:
        ax.set_ylabel('wall time (s)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, fontsize=6.5, loc='upper center',
               bbox_to_anchor=(0.5, 1.0), frameon=False,
               handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.2, h_pad=1.0)
    fig.savefig(HERE / 'orbital_scaling.png', dpi=220)
    print('Wrote orbital_scaling.png')


if __name__ == '__main__':
    main()
