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

"""K2SDF must use the explicitly supplied (non-cubic) kmesh."""
import unittest
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.lpno.kpts2supcell import K2SDF


class ExplicitKmesh(unittest.TestCase):
    def test_noncubic_mesh_used(self):
        cell = gto.Cell()
        cell.atom = 'H 0 0 0; H 0 0 1.0'
        cell.a = np.diag([6.0, 6.0, 3.0])
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.verbose = 0
        cell.build()
        kmesh = [1, 1, 4]
        kpts = cell.make_kpts(kmesh)
        mf = scf.KRHF(cell, kpts).density_fit().run()
        k = K2SDF(mf.with_df, kmesh=kmesh)
        self.assertEqual(list(np.asarray(k.kmesh).ravel()), kmesh)
        self.assertTrue(k.naux > 0)


if __name__ == '__main__':
    unittest.main()
