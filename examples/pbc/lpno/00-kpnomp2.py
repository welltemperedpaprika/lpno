#!/usr/bin/env python
"""Periodic PNO-MP2 calculation with threshold configuration."""
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.lpno import KPNOMP2, make_lo_kpipek

# 1. Mean-field calculation with time-reversal symmetry (TRS)
cell = gto.Cell()
cell.atom = 'Li 0 0 0; H 2.042 2.042 2.042'
cell.a = '''0     2.042 2.042
     2.042 0     2.042
     2.042 2.042 0'''
cell.basis = 'pobtzvp'
cell.verbose = 4
cell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh, time_reversal_symmetry=True)
kmf = scf.KRHF(cell, kpts).density_fit(auxbasis='weigend').run()

# 2. Localized occupied orbitals via periodic Pipek-Mezey (kpipek)
lo_coeff, kfrozen = make_lo_kpipek(cell, kmf, kpts, kmesh)

# 3. KPNOMP2 with customizable thresholds
mp = KPNOMP2(kmf, lo_coeff, frozen=kfrozen, kmesh=kmesh)
mp.thresh_pno = 1e-7        # PNO occupation truncation cutoff
mp.thresh_osv = 1e-5        # OSV truncation cutoff
mp.thresh_weakpair = 1e-4   # Weak-pair cutoff (semicanonical MP2 treatment)
mp.thresh_distpair = 1e-6   # Distant-pair cutoff (Ewald dipole estimate)
mp.kernel()

print(f"KPNOMP2 E_corr = {mp.e_corr:.8f} Ha")
