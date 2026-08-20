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

"""KPNOMP2 / kdOSVMP2 reproduce canonical KMP2 in the full-domain limit."""
import unittest
import numpy as np
from pyscf.pbc import gto, scf, mp as pbcmp
from pyscf.pbc.lpno import KPNOMP2, kdOSVMP2
from pyscf.pbc.lpno.kpts2supcell import make_translation_symmetric_lo


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
