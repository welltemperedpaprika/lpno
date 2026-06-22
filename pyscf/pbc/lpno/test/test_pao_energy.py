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

"""PAO full-domain energy equivalence: pbc_osv_mode='pao' vs default.

In the full-domain / no-truncation limit (thresh_osv=1e-9, thresh_distpair=0,
pbc_pao_pair_screening=False, pbc_pao_domain_mode='full') the PAO-basis OSV
expansion spans the full virtual space and must reproduce the canonical-virtual
result exactly (up to floating-point rounding).  This test pins that property
using the same He/gth-dzvp [1,1,2] fixture and translation-symmetric Wannier
LOs as test_kmp2_regression.
"""
import unittest
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.lpno import kdOSVMP2

# Reuse the translation-symmetric LO builder from the regression module so we
# do not re-introduce stock PipekMezey (which violates translation symmetry).
from pyscf.pbc.lpno.test.test_kmp2_regression import make_translation_symmetric_lo


class PAOEnergyEquivalenceTest(unittest.TestCase):
    """kdOSVMP2 PAO full-domain reproduces default (refcell) full-domain."""

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

    def _make_full_domain_driver(self, pbc_osv_mode):
        """Return a kdOSVMP2 instance configured for the full-domain limit."""
        m = kdOSVMP2(self.kmf, self.lo_coeff, kmesh=self.kmesh)
        m.thresh_osv = 1e-9        # no OSV truncation
        m.thresh_distpair = 0      # no pair distance/dipole-screening to Dist
        m.pbc_osv_mode = pbc_osv_mode
        if pbc_osv_mode == 'pao':
            m.pbc_pao_domain_mode = 'full'
            m.pbc_pao_pair_screening = False
        return m

    def test_pao_full_domain_matches_default_full_domain(self):
        """pbc_osv_mode='pao' full-domain e_corr agrees with refcell default."""
        m_default = self._make_full_domain_driver('refcell')
        m_default.kernel()
        e_default = m_default.e_corr

        m_pao = self._make_full_domain_driver('pao')
        m_pao.kernel()
        e_pao = m_pao.e_corr

        delta = abs(e_pao - e_default)
        self.assertAlmostEqual(
            e_pao, e_default, delta=1e-7,
            msg=(
                f"PAO full-domain e_corr ({e_pao:.10f}) differs from "
                f"default refcell e_corr ({e_default:.10f}) by {delta:.3e} Ha "
                f"(tolerance 1e-7 Ha)"
            ),
        )


if __name__ == '__main__':
    unittest.main()
