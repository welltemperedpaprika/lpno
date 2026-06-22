#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Hsuan Liang <yhljason@berkeley.edu>
#         Gengzhi Yang
#         Hong-Zhou Ye
#         Timothy C. Berkelbach
#

"""KPNOMP2 on a small periodic cell with stock Pipek-Mezey localization.

This example runs with a released PySCF and does not require pyscf.pbc.lo.
For production-quality periodic localization, see 01-kdosvmp2_kpipek.py,
which uses periodic Pipek-Mezey (pyscf.pbc.lo.kpipek.KPM).

NOTE on localization: plain supercell PipekMezey on a highly symmetric
cell (e.g. single-atom He with [1,1,2] k-mesh) can produce delocalized
symmetric/antisymmetric combinations that violate the translation-symmetry
assumption of the KPNOMP2 driver.  We therefore build translation-symmetric
Wannier functions directly from the Bloch occupied orbitals via lattice
Fourier transform, which is guaranteed to satisfy translation symmetry and
yields a correct MP2 energy in the full-domain limit.
"""
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lpno import KPNOMP2

def make_translation_symmetric_lo(cell, kmf, kpts, kmesh):
    """Build translation-symmetric supercell occupied LOs.

    The Wannier functions are the lattice Fourier transform of the Bloch
    occupied orbitals::

        W_{R}(S mu, n) = (1/sqrt(Nk)) sum_k exp(-i k.T_R) phase[S,k] C_k(mu,n)

    They are translation-symmetric by construction.  Returns a real
    ``(nao_supercell, nocc_supercell)`` coefficient matrix.
    """
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nocc_cell = cell.nelectron // 2
    cocc_k = np.asarray([kmf.mo_coeff[k][:, kmf.mo_occ[k] > 0]
                         for k in range(nkpts)])
    _, phase = k2gamma.get_phase(cell, kpts, kmesh)
    ncell = phase.shape[0]
    csc = np.einsum('Sk,kmn->Smkn', phase, cocc_k).reshape(
        ncell * nao, nkpts * nocc_cell)
    tvec = k2gamma.translation_vectors_for_kmesh(cell, kmesh)
    u = np.zeros((nkpts * nocc_cell, ncell * nocc_cell), dtype=complex)
    for ik, k in enumerate(kpts):
        for ir, t in enumerate(tvec):
            ph = np.exp(-1j * np.dot(k, t)) / np.sqrt(nkpts)
            for n in range(nocc_cell):
                u[ik * nocc_cell + n, ir * nocc_cell + n] = ph
    w = csc @ u
    assert abs(w.imag).max() < 1e-8
    return np.asarray(w.real, order='C')


# ── Cell ──────────────────────────────────────────────────────────────────────
cell = gto.Cell()
cell.atom = 'He 0 0 0'
cell.a = np.eye(3) * 3.0
cell.basis = 'gth-dzvp'   # gth-szv has 0 virtual orbitals for He; dzvp gives 4
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build()

# ── Mean-field ────────────────────────────────────────────────────────────────
kmesh = [1, 1, 2]
kpts = cell.make_kpts(kmesh)
kmf = scf.KRHF(cell, kpts).density_fit().run()

# ── Localization ──────────────────────────────────────────────────────────────
# Build translation-symmetric Wannier LOs from the Bloch occupied orbitals.
# These satisfy the translation-symmetry assumption of KPNOMP2 exactly.
# (Production runs use periodic Pipek-Mezey; see 01-kdosvmp2_kpipek.py.)
lo_coeff = make_translation_symmetric_lo(cell, kmf, kpts, kmesh)

# ── KPNOMP2 ───────────────────────────────────────────────────────────────────
mp = KPNOMP2(kmf, lo_coeff, kmesh=kmesh)
mp.thresh_osv = 1e-7
mp.thresh_pno = 1e-7
mp.kernel()
print('KPNOMP2 E_corr =', mp.e_corr)
