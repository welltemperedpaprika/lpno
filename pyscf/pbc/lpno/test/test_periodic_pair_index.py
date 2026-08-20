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

"""PeriodicPairIndex algebra and reference-cell pair-energy bookkeeping."""
import unittest
import numpy as np

from pyscf.pbc.lpno.kdosv import kdOSVMP2
from pyscf.pbc.lpno.kpno import KPNOMP2
from pyscf.pbc.lpno.kpair_domain import PeriodicPairIndex


def _cell_sub_from_kmesh(kmesh):
    kmesh = np.asarray(kmesh, dtype=int)
    coords = np.array(
        [(a, b, c)
         for a in range(kmesh[0])
         for b in range(kmesh[1])
         for c in range(kmesh[2])],
        dtype=int,
    )
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(coords)}
    cell_sub = np.empty((len(coords), len(coords)), dtype=int)
    for a, coord_a in enumerate(coords):
        for b, coord_b in enumerate(coords):
            coord = tuple((coord_a - coord_b) % kmesh)
            cell_sub[a, b] = coord_to_index[coord]
    return cell_sub


class _FakeKdosvPairDomain:
    def __init__(self, raw_ed_by_pair):
        self.K = {pair: np.array([[ed]], dtype=float) for pair, ed in raw_ed_by_pair.items()}
        self.J = {pair: np.array([[0.0]], dtype=float) for pair in raw_ed_by_pair}
        self._pairs = list(raw_ed_by_pair)

    def loop_pair(self):
        return iter(self._pairs)


class PeriodicPairIndexTest(unittest.TestCase):
    def test_kdosv_pair_energy_keeps_distinct_self_inverse_2d_cells(self):
        mp = kdOSVMP2.__new__(kdOSVMP2)
        mp.nlo_per_cell = 1
        mp.kpts = [None] * 4
        mp._cell_sub = _cell_sub_from_kmesh([2, 2, 1])
        mp._pair_index = PeriodicPairIndex(mp.nlo_per_cell, len(mp.kpts),
                                           mp._cell_sub)
        mp._e_pair = {}
        mp._e_pair_ss = {}
        mp._e_pair_os = {}

        pair_domain = _FakeKdosvPairDomain({
            (0, 0): 1.0,
            (0, 1): 10.0,  # cell vector (0, 1, 0), self-inverse in a 2x2x1 mesh
            (0, 2): 20.0,  # cell vector (1, 0, 0), self-inverse
            (0, 3): 30.0,  # cell vector (1, 1, 0), self-inverse
        })
        t2 = {pair: np.array([[1.0]]) for pair in pair_domain.loop_pair()}

        mp._energy_corr_pair(pair_domain, t2)

        self.assertEqual(mp._e_pair_os["Strong pair"][0, 1], 20.0)
        self.assertEqual(mp._e_pair_os["Strong pair"][0, 2], 40.0)
        self.assertEqual(mp._e_pair_os["Strong pair"][0, 3], 60.0)

    def test_kdosv_pair_energy_does_not_average_conjugate_cells(self):
        mp = kdOSVMP2.__new__(kdOSVMP2)
        mp.nlo_per_cell = 1
        mp.kpts = [None] * 3
        mp._cell_sub = _cell_sub_from_kmesh([3, 1, 1])
        mp._pair_index = PeriodicPairIndex(mp.nlo_per_cell, len(mp.kpts),
                                           mp._cell_sub)
        mp._e_pair = {}
        mp._e_pair_ss = {}
        mp._e_pair_os = {}

        pair_domain = _FakeKdosvPairDomain({
            (0, 1): 10.0,  # R
            (0, 2): 30.0,  # -R
        })
        t2 = {pair: np.array([[1.0]]) for pair in pair_domain.loop_pair()}

        mp._energy_corr_pair(pair_domain, t2)

        self.assertEqual(mp._e_pair_os["Strong pair"][0, 1], 20.0)
        self.assertEqual(mp._e_pair_os["Strong pair"][0, 2], 60.0)

    def test_kdosv_pair_energy_does_not_average_ordered_orientations(self):
        mp = kdOSVMP2.__new__(kdOSVMP2)
        mp.nlo_per_cell = 2
        mp.kpts = [None]
        mp._cell_sub = _cell_sub_from_kmesh([1, 1, 1])
        mp._pair_index = PeriodicPairIndex(mp.nlo_per_cell, 2, mp._cell_sub)
        mp._e_pair = {}
        mp._e_pair_ss = {}
        mp._e_pair_os = {}

        pair_domain = _FakeKdosvPairDomain({
            (0, 1): 10.0,
            (1, 0): 30.0,
        })
        t2 = {pair: np.array([[1.0]]) for pair in pair_domain.loop_pair()}

        mp._energy_corr_pair(pair_domain, t2)

        self.assertEqual(mp._e_pair_os["Strong pair"][0, 1], 20.0)
        self.assertEqual(mp._e_pair_os["Strong pair"][1, 0], 60.0)

    def test_periodic_pair_index_uses_cell_sub_for_inverse_cells(self):
        index = PeriodicPairIndex(nlo_per_cell=1, nlo=4,
                                  cell_sub=_cell_sub_from_kmesh([2, 2, 1]))
        self.assertEqual([index.inverse_cell(cell) for cell in range(4)],
                         [0, 1, 2, 3])

        index = PeriodicPairIndex(nlo_per_cell=1, nlo=3,
                                  cell_sub=_cell_sub_from_kmesh([3, 1, 1]))
        self.assertEqual([index.inverse_cell(cell) for cell in range(3)],
                         [0, 2, 1])

    def test_kpno_energy_sum_half_weights_periodic_offdiagonal_arrays_and_pairs(self):
        mp = KPNOMP2.__new__(KPNOMP2)
        mp.nlo_per_cell = 1
        mp.nlo = 3
        mp._cell_sub = _cell_sub_from_kmesh([3, 1, 1])
        mp._pair_index = PeriodicPairIndex(mp.nlo_per_cell, mp.nlo,
                                           mp._cell_sub)

        dist_ss = np.array([[10.0, 20.0, 30.0]])
        dist_os = np.array([[1.0, 2.0, 3.0]])
        strong_ss = {(0, 0): 5.0, (0, 2): 8.0}
        strong_os = {(0, 0): 0.5, (0, 2): 0.8}

        mp._e_pair_ss = {
            "Dist pair": dist_ss,
            "Strong pair": strong_ss,
        }
        mp._e_pair_os = {
            "Dist pair": dist_os,
            "Strong pair": strong_os,
        }

        total, ss, os = mp._sum_e_corr()

        self.assertEqual(ss, 10.0 + 0.5 * 20.0 + 0.5 * 30.0 + 5.0 + 0.5 * 8.0)
        self.assertEqual(os, 1.0 + 0.5 * 2.0 + 0.5 * 3.0 + 0.5 + 0.5 * 0.8)
        self.assertEqual(total, ss + os)


if __name__ == '__main__':
    unittest.main()
