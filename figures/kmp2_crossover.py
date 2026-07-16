#!/usr/bin/env python3
"""KPNO-MP2 vs canonical DF-KMP2: wall-time ratio vs N_k.

Both methods timed in the same job on the same node at T_PNO = 1e-8;
KPNO time is the full pipeline (transform + domains + residual solve).
Data: kmp2_crossover_data.csv.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

STYLE = {
    'Si (3D)': ('C2', 'o', r'Si (3D)'),
    'C (3D)': ('C7', 'v', r'C (3D)'),
    'BN (2D)': ('C4', '^', r'BN (2D)'),
    'SiC (2D)': ('C8', '<', r'SiC (2D)'),
    'C2H2 (1D)': ('C1', 's', r'C$_2$H$_2$ (1D)'),
    'C2HF (1D)': ('C6', '>', r'C$_2$HF (1D)'),
    'CO2/DZ (3D)': ('C3', 'X', r'CO$_2$/DZ (3D)'),
    'NH3/DZ (3D)': ('C5', 'P', r'NH$_3$/DZ (3D)'),
}


def main():
    series = {}
    with open(HERE / 'kmp2_crossover_data.csv') as f:
        for row in csv.DictReader(f):
            series.setdefault(row['system'], []).append(
                (int(row['nk']), float(row['ratio'])))

    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for system, (color, marker, label) in STYLE.items():
        pts = sorted(series.get(system, []))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, ls='-', lw=0.9, marker=marker, ms=4.5,
                  color=color, zorder=3, label=label)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.text(55, 1.15, 'canonical faster', fontsize=6, color='gray')
    ax.text(55, 0.68, 'KPNO-MP2 faster', fontsize=6, color='gray')
    ax.set_xlabel(r'$N_k$')
    ax.set_ylabel(r'$t_\mathrm{KPNO}\,/\,t_\mathrm{KMP2}$')
    ax.set_xticks([3, 9, 27, 81])
    ax.set_xticklabels(['3', '9', '27', '81'])
    ax.set_xticks([], minor=True)
    ax.set_ylim(0.015, 25)
    ax.legend(fontsize=5.8, loc='lower left', framealpha=0.9, ncol=2,
              handletextpad=0.4, labelspacing=0.3, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(HERE / 'kmp2_crossover.png', dpi=220)
    print('Wrote kmp2_crossover.png')


if __name__ == '__main__':
    main()
