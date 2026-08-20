#!/usr/bin/env python
"""Periodic OSV-MP2 with dipole prescreening modes."""
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.lpno import kdOSVMP2, make_lo_kpipek

# 1. Mean-field calculation
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

# 2. Localized occupied orbitals
lo_coeff, kfrozen = make_lo_kpipek(cell, kmf, kpts, kmesh)

# 3. kdOSVMP2 with dipole screening configuration
mp = kdOSVMP2(kmf, lo_coeff, frozen=kfrozen, kmesh=kmesh)
mp.thresh_osv = 1e-5        # OSV truncation cutoff
mp.dipole_mode = 'ewald'    # 'ewald' (periodic minimum image) or 'bare'
mp.thresh_distpair = 1e-6   # Prescreening cutoff for distant pairs
mp.thresh_foo = 1e-5        # Inter-cell Fock coupling screening
mp.kernel()

print(f"kdOSVMP2 E_corr = {mp.e_corr:.8f} Ha")
