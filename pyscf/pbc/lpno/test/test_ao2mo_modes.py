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

"""incore and outcore pbc_ao2mo_mode paths give identical e_corr."""
import unittest
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lpno import kdOSVMP2


def make_translation_symmetric_lo(cell, kmf, kpts, kmesh):
    """Translation-symmetric, site-localized supercell occupied LOs.

    Lattice Fourier transform of the Bloch occupied orbitals; see
    ``test_kmp2_regression`` for the rationale.
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


def _run(kmf, lo_coeff, kmesh, mode):
    m = kdOSVMP2(kmf, lo_coeff, kmesh=kmesh)
    m.pbc_ao2mo_mode = mode
    m.thresh_osv = 1e-9
    m.kernel()
    return m.e_corr


class AO2MOModes(unittest.TestCase):
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
        cls.lo_coeff = make_translation_symmetric_lo(
            cell, cls.kmf, cls.kpts, cls.kmesh)
        cls.e_incore = _run(cls.kmf, cls.lo_coeff, cls.kmesh, 'incore')

    def test_incore_vs_outcore(self):
        e_outcore = _run(self.kmf, self.lo_coeff, self.kmesh, 'outcore')
        self.assertAlmostEqual(e_outcore, self.e_incore, delta=1e-8)


if __name__ == '__main__':
    unittest.main()
