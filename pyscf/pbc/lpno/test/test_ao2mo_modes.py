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
# Author: Yu Hsuan Liang
#         Gengzhi Yang
#         Hong-Zhou Ye
#         Timothy C. Berkelbach
#

"""incore and outcore pbc_ao2mo_mode paths give identical e_corr."""
import unittest
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.lpno import kdOSVMP2
from pyscf.pbc.lpno.kpts2supcell import make_translation_symmetric_lo


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
