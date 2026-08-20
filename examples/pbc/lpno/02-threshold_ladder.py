#!/usr/bin/env python
"""Multi-threshold PNO scan sharing a single set of transformed integrals."""
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.lpno import KPNOMP2, make_lo_kpipek

# 1. Mean field & localization
cell = gto.Cell()
cell.atom = 'Li 0 0 0; H 2.042 2.042 2.042'
cell.a = '''0     2.042 2.042
     2.042 0     2.042
     2.042 2.042 0'''
cell.basis = 'pobtzvp'
cell.verbose = 3
cell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh, time_reversal_symmetry=True)
kmf = scf.KRHF(cell, kpts).density_fit(auxbasis='weigend').run()
lo_coeff, kfrozen = make_lo_kpipek(cell, kmf, kpts, kmesh)

# 2. Transform integrals once and evaluate multiple PNO thresholds
eris = None
owner = None
thresholds = [1e-5, 1e-6, 1e-7, 1e-8]
for tp in thresholds:
    mp = KPNOMP2(kmf, lo_coeff, frozen=kfrozen, kmesh=kmesh)
    mp.thresh_pno = tp
    if eris is None:
        owner = mp
        eris = mp.ao2mo()
    mp.kernel(eris=eris)
    print(f"thresh_pno = {tp:1.0e}  ->  E_corr = {mp.e_corr:.8f} Ha")

owner._clear_eris(eris)
