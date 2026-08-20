#!/usr/bin/env python
"""Compare KPNOMP2 against canonical KMP2 reference and toggle memory modes."""
import numpy as np
from pyscf.pbc import gto, scf, mp as pbcmp
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
nkpts = kpts.nkpts
frozen_per_kpt = 1
kmf = scf.KRHF(cell, kpts).density_fit(auxbasis='weigend').run()
lo_coeff, kfrozen = make_lo_kpipek(cell, kmf, kpts, kmesh, frozen_per_kpt=frozen_per_kpt)

# 2. Canonical KMP2 reference
kmp = pbcmp.KMP2(kmf, frozen=frozen_per_kpt).run()

# 3. KPNOMP2 with in-core / out-of-core memory modes
for mode in ['incore', 'outcore']:
    mp = KPNOMP2(kmf, lo_coeff, frozen=kfrozen, kmesh=kmesh)
    mp.thresh_pno = 1e-7
    mp.pbc_ao2mo_mode = mode   # 'incore' (faster) or 'outcore' (low memory footprint)
    mp.kernel()
    err_pct = abs(mp.e_corr - kmp.e_corr) / abs(kmp.e_corr) * 100
    print(f"KPNOMP2 ({mode:7s}): E_corr = {mp.e_corr:.8f} Ha (error: {err_pct:.3f}%)")

print(f"Canonical KMP2:        E_corr = {kmp.e_corr:.8f} Ha")
