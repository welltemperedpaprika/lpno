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

"""KPNOMP2 / kdOSVMP2 reproduce canonical KMP2 in the full-domain limit.

The periodic local drivers assume the supercell occupied LOs are
translation-symmetric, i.e. the LOs in cell ``R`` are exact lattice
translates of the reference-cell LOs.  A plain supercell ``PipekMezey``
run does *not* guarantee this (for highly symmetric cells it returns
delocalized symmetric/antisymmetric combinations), so we build genuine
translation-symmetric Wannier LOs from the k-point occupied orbitals.
In the full-domain limit any such occupied rotation reproduces canonical
MP2 exactly.
"""
import unittest
import numpy as np
from pyscf.pbc import gto, scf, mp as pbcmp
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lpno import KPNOMP2, kdOSVMP2


def make_translation_symmetric_lo(cell, kmf, kpts, kmesh):
    """Build translation-symmetric, site-localized supercell occupied LOs.

    The supercell Wannier functions are the lattice Fourier transform of the
    Bloch occupied orbitals,
        W_R(S mu, n) = (1/sqrt(Nk)) sum_k e^{-i k.T_R} phase[S,k] C_k(mu, n),
    which are translation-symmetric by construction.  Returns a real
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


class KMP2Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = 'He 0 0 0'
        cell.a = np.eye(3) * 3.0
        cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.verbose = 0
        cell.build()
        cls.kmesh = [1, 1, 2]
        cls.kpts = cell.make_kpts(cls.kmesh)
        cls.kmf = scf.KRHF(cell, cls.kpts).density_fit().run()
        cls.e_kmp2 = pbcmp.KMP2(cls.kmf).run().e_corr
        # translation-symmetric supercell localized occupied orbitals
        cls.lo_coeff = make_translation_symmetric_lo(
            cell, cls.kmf, cls.kpts, cls.kmesh)

    def test_kpno_full_domain(self):
        m = KPNOMP2(self.kmf, self.lo_coeff, kmesh=self.kmesh)
        m.thresh_osv = 1e-9
        m.thresh_pno = 1e-9
        m.thresh_distpair = 0  # no pair distance/dipole-screened to Dist bucket
        m.kernel()
        # assert the Dist-pair energy bucket is zero
        self.assertAlmostEqual(
            np.sum(np.abs(m._e_pair.get('Dist pair', np.zeros(1)))), 0.0,
            delta=1e-12, msg='Dist-pair bucket is non-zero despite thresh_distpair=0')
        self.assertAlmostEqual(m.e_corr, self.e_kmp2, delta=1e-6)

    def test_kdosv_full_domain(self):
        m = kdOSVMP2(self.kmf, self.lo_coeff, kmesh=self.kmesh)
        m.thresh_osv = 1e-9
        m.thresh_distpair = 0  # no pair distance/dipole-screened to Dist bucket
        m.kernel()
        # assert the Dist-pair energy bucket is zero
        dist_e_pair = m._e_pair.get('Dist pair', None)
        if dist_e_pair is not None:
            self.assertAlmostEqual(
                np.sum(np.abs(dist_e_pair)), 0.0,
                delta=1e-12, msg='Dist-pair bucket is non-zero despite thresh_distpair=0')
        self.assertAlmostEqual(m.e_corr, self.e_kmp2, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
