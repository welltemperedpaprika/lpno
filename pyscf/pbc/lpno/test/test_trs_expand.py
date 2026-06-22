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

"""IBZ -> full BZ time-reversal-symmetry expansion adapter (``_expand_trs_mf``).

``_expand_trs_mf`` rebuilds a full-BZ KSCF mock from a TRS-reduced k-point MF.
It relies on ``pyscf.pbc.lo.base.remove_trs_mo``; when that module is not
present (older PySCF builds) the test is skipped.
"""
import unittest
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.lib.kpts import KPoints

from pyscf.pbc.lpno.kpno import _expand_trs_mf


def _has_remove_trs_mo():
    try:
        from pyscf.pbc.lo.base import remove_trs_mo  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_remove_trs_mo(),
                     'pyscf.pbc.lo.base.remove_trs_mo unavailable')
class TRSExpand(unittest.TestCase):
    def test_expand_trs_mf_normalizes_full_bz_kpoints_object(self):
        # Use H2 with a [1,1,4] k-mesh: TRS reduces the 4-point BZ to a
        # 3-point IBZ (k=(0,0,k_z) pairs with k=(0,0,-k_z), but Gamma and
        # the zone-boundary point are their own partners).  This gives
        # len(kpts)==3 < kpts.nkpts==4, so _expand_trs_mf genuinely exercises
        # the expansion path instead of hitting the early-return.
        cell = gto.Cell()
        cell.atom = "H 0 0 0; H 0 0 1.4"
        cell.a = np.diag([6.0, 6.0, 2.8])
        cell.basis = "sto-3g"
        cell.verbose = 0
        cell.build()
        nao = cell.nao_nr()
        nkpts_ibz = 3  # IBZ k-points for [1,1,4] on this cell
        nkpts_bz = 4   # full BZ k-points

        class FakeDF:
            auxbasis = "weigend"

        class FakeKMF:
            pass

        kmf = FakeKMF()
        kmf.cell = cell
        kmf.kpts = cell.make_kpts([1, 1, 4], time_reversal_symmetry=True)
        kmf.with_df = FakeDF()
        kmf.mo_coeff = [np.eye(nao) for _ in range(nkpts_ibz)]
        kmf.mo_energy = [np.arange(nao, dtype=float) for _ in range(nkpts_ibz)]
        kmf.mo_occ = [np.ones(nao) for _ in range(nkpts_ibz)]
        kmf.e_tot = -1.0
        kmf.converged = True

        expanded, nkpts = _expand_trs_mf(kmf)

        self.assertEqual(nkpts, nkpts_bz)
        self.assertNotIsInstance(expanded.kpts, KPoints)
        self.assertEqual(np.asarray(expanded.kpts).shape, (nkpts_bz, 3))
        self.assertIs(expanded.with_df, kmf.with_df)
        # extra assertions: the rebuilt mock preserves the SCF result
        self.assertEqual(expanded.e_tot, kmf.e_tot)
        self.assertTrue(expanded.converged)


if __name__ == '__main__':
    unittest.main()
