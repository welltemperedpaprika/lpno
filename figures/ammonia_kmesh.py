#!/usr/bin/env python3
"""Solid ammonia: correlation cohesive energy vs k-mesh.

Correlation contribution to the cohesive energy of solid ammonia against the
inverse k-mesh density 1/N_k, for cc-pVDZ, cc-pVTZ, and a two-point (DZ/TZ)
CBS extrapolation, with KPNO-MP2 (solid, filled markers) and canonical KMP2
(dashed, open markers) overlaid. The marker at 1/N_k = 0 is a linear-in-1/N_k
extrapolation to the thermodynamic limit. Reference data accompanies the
manuscript (see ammonia_kmesh_data.csv).

Usage:  python ammonia_kmesh.py   ->   ammonia_kmesh.png
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HA2KJMOL = 2625.5
Z = 4  # NH3 molecules per unit cell
DATA = Path(__file__).resolve().parent / 'ammonia_kmesh_data.csv'

crystal_kpno, crystal_kmp2, mol_corr = {}, {}, {}
with DATA.open(newline='') as fh:
    for row in csv.DictReader(fh):
        if row['kind'] == 'crystal':
            key = (row['basis'], int(row['nk']))
            crystal_kpno[key] = float(row['kpno_t8'])
            crystal_kmp2[key] = float(row['kmp2'])
        elif row['kind'] == 'mol':
            mol_corr[row['basis']] = float(row['mol_corr'])


def cbs_hk(X, e_outer, e_inner):
    """Two-point 1/X^3 correlation-energy extrapolation (outer = larger X)."""
    return (X**3 * e_outer - (X - 1)**3 * e_inner) / (X**3 - (X - 1)**3)


mol_cbs = cbs_hk(3, mol_corr['TZ'], mol_corr['DZ'])


def cbs_crystal(table, nk):
    if ('DZ', nk) in table and ('TZ', nk) in table:
        return cbs_hk(3, table[('TZ', nk)], table[('DZ', nk)])
    return None


def coh_corr(e_cryst, e_mol):
    return (e_mol - e_cryst / Z) * HA2KJMOL


def build_curves(table):
    out = {}
    out['DZ'] = ([8, 27, 64], [coh_corr(table[('DZ', k)], mol_corr['DZ']) for k in (8, 27, 64)])
    out['TZ'] = ([8, 27], [coh_corr(table[('TZ', k)], mol_corr['TZ']) for k in (8, 27)])
    cbs_vals = [(k, cbs_crystal(table, k)) for k in (8, 27)]
    out['CBS'] = ([k for k, v in cbs_vals if v is not None],
                  [coh_corr(v, mol_cbs) for _, v in cbs_vals if v is not None])
    return out


def linfit_intercept(xs, ys):
    a = np.vstack([np.asarray(xs, float), np.ones(len(xs))]).T
    slope, intercept = np.linalg.lstsq(a, np.asarray(ys, float), rcond=None)[0]
    return slope, intercept


kpno = build_curves(crystal_kpno)
kmp2 = build_curves(crystal_kmp2)

fig, ax = plt.subplots(figsize=(4.2, 2.9))
for basis, (color, marker) in {'DZ': ('C0', 'o'), 'TZ': ('C1', '^'), 'CBS': ('C2', 's')}.items():
    nk_k, e_k = kpno[basis]
    nk_m, e_m = kmp2[basis]
    inv_k = [1.0 / n for n in nk_k]
    inv_m = [1.0 / n for n in nk_m]
    _, b_k = linfit_intercept(inv_k, e_k)
    _, b_m = linfit_intercept(inv_m, e_m)
    name = 'CBS' if basis == 'CBS' else f'cc-pV{basis}'
    ax.plot(inv_k + [0.0], e_k + [b_k], '-' + marker, color=color, ms=5, label=name)
    ax.plot(inv_m + [0.0], e_m + [b_m], '--' + marker, color=color, ms=5,
            markerfacecolor='white', markeredgewidth=1.1)

ax.axvline(0, color='k', linestyle='--', lw=0.6, alpha=0.55)
ax.set_xticks([1 / 64, 1 / 27, 1 / 8, 0])
ax.set_xticklabels([r'$4^{-3}$', r'$3^{-3}$', r'$2^{-3}$', r'$\infty^{-3}$'])
ax.set_xlim(-0.012, 1 / 7.0)
ax.set_xlabel(r'$N_k^{-1}$')
ax.set_ylabel(r'$E_\mathrm{coh}^{(2)}$ (kJ/mol)')
ax.set_title('solid: KPNO-MP2 (filled) vs KMP2 (open)', fontsize=8)
ax.legend(fontsize=8, framealpha=0.95, loc='lower left')

fig.tight_layout()
out = Path(__file__).resolve().parent / 'ammonia_kmesh.png'
fig.savefig(out, dpi=200)
print(f'Wrote {out}')
