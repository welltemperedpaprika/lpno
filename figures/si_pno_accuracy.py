#!/usr/bin/env python3
"""Silicon PNO convergence: KPNO-MP2 correlation-energy error vs canonical KMP2.

Absolute error in the KPNO-MP2 correlation energy of silicon (pob-TZVP)
relative to canonical periodic MP2, in kcal/mol per unit cell, at three
k-meshes and four PNO truncation thresholds (log-log). The dashed lines mark
99.90 / 99.94 / 99.99% of the canonical correlation energy, referenced to the
4x4x4 KMP2 value. Reference data accompanies the manuscript.

Usage:  python si_pno_accuracy.py   ->   si_pno_accuracy.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HA2KCAL = 627.5094740631

# Si pob-TZVP: (k-mesh, canonical KMP2 E_corr, [KPNO E_corr at T_PNO = 1e-6,1e-7,1e-8,1e-9])
SI = [
    (r'$2\times2\times2$', -0.141262809044582,
     [-0.140021227661803, -0.140730898555522, -0.141073859010153, -0.141212926185060]),
    (r'$3\times3\times3$', -0.160697991576758,
     [-0.159113323535782, -0.159853063167544, -0.160233461270831, -0.160525985989561]),
    (r'$4\times4\times4$', -0.171051771949921,
     [-0.168914168499940, -0.169691588036068, -0.170263371832774, -0.170756999490976]),
]
T_PNO = np.array([1e-6, 1e-7, 1e-8, 1e-9])
kmp2_ref_kcal = abs(SI[-1][1]) * HA2KCAL

fig, ax = plt.subplots(figsize=(4.2, 2.8))
for (label, kmp2, kpno_vals), color, marker in zip(SI, ('C0', 'C1', 'C3'), ('o', 's', '^')):
    err = np.abs(np.array(kpno_vals) - kmp2) * HA2KCAL
    ax.loglog(T_PNO, err, '-' + marker, color=color, ms=5, label=label)

for frac, lab in [(0.10e-2, r'$99.90\%$'), (0.06e-2, r'$99.94\%$'), (0.01e-2, r'$99.99\%$')]:
    y = frac * kmp2_ref_kcal
    ax.axhline(y, color='gray', linestyle='--', linewidth=0.8, alpha=0.55)
    ax.text(1.25e-9, y * 1.08, lab, fontsize=7, color='gray', ha='left', va='bottom')

ax.set_xlabel(r'$T_\mathrm{cut}^\mathrm{PNO}$')
ax.set_ylabel(r'$|\Delta E_\mathrm{corr}|$ (kcal/mol per cell)')
ax.invert_xaxis()
ax.set_ylim(8e-3, 2.0)
ax.legend(title=r'Si $k$-mesh', fontsize=8, title_fontsize=8, framealpha=0.95, loc='upper right')

fig.tight_layout()
out = Path(__file__).resolve().parent / 'si_pno_accuracy.png'
fig.savefig(out, dpi=200)
print(f'Wrote {out}')
