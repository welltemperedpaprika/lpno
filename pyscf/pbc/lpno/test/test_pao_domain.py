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

"""PAO-basis kdOSV pair-domain wiring (unit tests with fake PAO objects).

These exercise the ``pbc_osv_mode='pao'`` code path of ``kdOSVMP2`` by
monkeypatching the module-level PAO builders that ``kdosv`` reaches through
``kpao_mp2`` (imported as ``kpao_mp2_mod``) and ``kpao`` (imported inside
``kpao_mp2`` as ``kpao_mod``).  The PAO machinery itself is replaced by light
fakes so the test isolates the driver's pair-mask / work-ledger bookkeeping.
"""
import sys
import unittest
import numpy as np

from pyscf.pbc.lpno import kdosv as kdosv_mod
from pyscf.pbc.lpno.kdosv import kdOSVMP2
from pyscf.pbc.lpno.kpao_pair_domain import PairDomain_kdOSV_PAOBasis

# kdosv -> kpao_mp2 (as kpao_mp2_mod) -> kpao (as kpao_mod)
kpao_mp2_mod = kdosv_mod.kpao_mp2_mod
kpao_mod = kpao_mp2_mod.kpao_mod
# pd_mod (kpair_domain) is shared between kdosv and kpao_mp2
pd_mod = kpao_mp2_mod.pd_mod


class _FakePao:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_orbital_pao2vir_coeff(self, idx):
        return np.ones((1, 1))

    def pair_domain_sizes(self, pair_mask=None):
        if pair_mask is None:
            return np.array([1, 2], dtype=np.int32)
        return np.ones(np.count_nonzero(pair_mask), dtype=np.int32)


class _FakePaoOSV:
    def __init__(self, eris, occ_energy, pao, osv_param, verbose=None):
        self.eris = eris
        self.occ_energy = occ_energy
        self.pao = pao
        self.osv_param = osv_param
        self.verbose = verbose
        self.nosv = [1, 1]
        self.nocc = 2
        self.u = [np.ones((1, 1)), np.ones((1, 1))]

    def get_ovlp(self, pair_mask=None, force_update=False):
        return "fake overlap"


class _FakeEris:
    max_memory = 1000
    nvir = 1
    naux = 1

    def get_projected_blk(self, occ_idx, proj):
        block = np.ones((proj.shape[1], 1)) * (occ_idx + 1)
        return block, np.zeros_like(block)


class PAODomainTest(unittest.TestCase):
    def test_kdosv_make_osv_pao_mode_uses_pao_basis_osv_builder(self):
        old_pao = kpao_mod.PeriodicPAO
        old_osv = kpao_mod.PeriodicPAOBasisOSV
        try:
            kpao_mod.PeriodicPAO = _FakePao
            kpao_mod.PeriodicPAOBasisOSV = _FakePaoOSV

            mp = kdOSVMP2.__new__(kdOSVMP2)
            mp.pbc_osv_mode = "pao"
            mp.pbc_pao_pair_screening = False
            mp.pbc_pao_domain_mode = "full"
            mp.pbc_pao_bp_thresh = 0.999
            mp.pbc_ao2mo_mode = "incore"
            mp.nlo_per_cell = 1
            mp.nlo = 2
            mp._cell_sub = np.array([[0, 1], [1, 0]])
            mp.cell = object()
            mp.lo_coeff = np.eye(3, 2)
            mp._s1e = np.eye(3)
            mp.verbose = 0
            mp.chkfile = None
            mp.thresh_osv = -1
            mp.split_mo_coeff = lambda: (None, None, np.eye(3, 1))
            mp.split_mo_energy = lambda: (None, None, np.array([1.0]))
            mp._foo = np.diag([-1.0, -1.0])

            eris = object()
            osv = mp.make_osv(eris)

            self.assertIsInstance(osv, _FakePaoOSV)
            self.assertIs(osv.pao, mp._pao)
            self.assertEqual(osv.osv_param["nlo_per_cell"], 1)
            self.assertEqual(osv.osv_param["nlo"], 2)
            self.assertEqual(mp._pao.kwargs["nlo_per_cell"], 1)
            self.assertEqual(mp._pao.kwargs["nlo"], 2)
            self.assertIsNone(mp._pao.kwargs["ref_atom_domains"])
        finally:
            kpao_mod.PeriodicPAO = old_pao
            kpao_mod.PeriodicPAOBasisOSV = old_osv

    def test_kdosv_make_pair_domain_pao_mode_uses_full_pao_basis_pair_domain(self):
        old_pao = kpao_mod.PeriodicPAO
        old_osv = kpao_mod.PeriodicPAOBasisOSV
        try:
            kpao_mod.PeriodicPAO = _FakePao
            kpao_mod.PeriodicPAOBasisOSV = _FakePaoOSV

            mp = kdOSVMP2.__new__(kdOSVMP2)
            mp.pbc_osv_mode = "pao"
            mp.pbc_pao_domain_mode = "full"
            mp.pbc_pao_bp_thresh = 0.999
            mp.pbc_ao2mo_mode = "incore"
            mp.nlo_per_cell = 1
            mp.nlo = 2
            mp._cell_sub = np.array([[0, 1], [1, 0]])
            mp.cell = object()
            mp.lo_coeff = np.eye(3, 2)
            mp._s1e = np.eye(3)
            mp.verbose = 0
            mp.stdout = sys.stdout
            mp.chkfile = None
            mp.thresh_osv = -1
            mp.thresh_distpair = -1
            mp.rmin_distpair = 20
            mp.screen_mode = "energy"
            mp.dipole_mode = "molecular"
            mp._pair_mask = None
            mp._e_pair = None
            mp._e_pair_ss = None
            mp._e_pair_os = None
            mp.split_mo_coeff = lambda: (None, None, np.eye(3, 1))
            mp.split_mo_energy = lambda: (None, None, np.array([1.0]))
            mp._foo = np.diag([-1.0, -1.0])

            pair_domain = mp.make_pair_domain(_FakeEris(), timer={})

            self.assertIsInstance(pair_domain, PairDomain_kdOSV_PAOBasis)
            np.testing.assert_array_equal(
                pair_domain.pair_mask, np.ones((1, 2), dtype=bool))
            np.testing.assert_array_equal(
                mp._pair_mask["Near pair"], np.ones((1, 2), dtype=bool))
            np.testing.assert_array_equal(
                mp._pair_mask["Residual pair"], np.ones((1, 2), dtype=bool))
            self.assertEqual(pair_domain.loop_pair(), [(0, 0), (0, 1)])
            np.testing.assert_array_equal(
                mp._pair_mask["Strong pair"], pair_domain.pair_mask)
            self.assertEqual(mp._pair_work_summary["near_pairs"], 2)
            self.assertEqual(mp._pair_work_summary["residual_pairs"], 2)
        finally:
            kpao_mod.PeriodicPAO = old_pao
            kpao_mod.PeriodicPAOBasisOSV = old_osv

    def test_kdosv_make_pair_domain_pao_pair_screening_updates_masks_and_work_ledger(self):
        old_pao = kpao_mod.PeriodicPAO
        old_osv = kpao_mod.PeriodicPAOBasisOSV
        old_dipole = pd_mod.pbc_pair_dipole_pao
        try:
            kpao_mod.PeriodicPAO = _FakePao
            kpao_mod.PeriodicPAOBasisOSV = _FakePaoOSV

            def fake_dipole(cell, lo_coeff, nlo_per_cell, moe_lo, pao, osv):
                self.assertIsNotNone(pao)
                return (
                    np.array([[0.0, 1.0e-8]]),
                    np.array([[0.0, 40.0]]),
                )

            pd_mod.pbc_pair_dipole_pao = fake_dipole

            mp = kdOSVMP2.__new__(kdOSVMP2)
            mp.pbc_osv_mode = "pao"
            mp.pbc_pao_pair_screening = True
            mp.pbc_pao_domain_mode = "full"
            mp.pbc_pao_bp_thresh = 0.999
            mp.pbc_ao2mo_mode = "incore"
            mp.nlo_per_cell = 1
            mp.nlo = 2
            mp._cell_sub = np.array([[0, 1], [1, 0]])
            mp.cell = object()
            mp.lo_coeff = np.eye(3, 2)
            mp._s1e = np.eye(3)
            mp.verbose = 0
            mp.stdout = sys.stdout
            mp.chkfile = None
            mp.thresh_osv = -1
            mp.thresh_distpair = 1.0e-6
            mp.rmin_distpair = 20
            mp.screen_mode = "energy"
            mp.dipole_mode = "molecular"
            mp._pair_mask = None
            mp._e_pair = None
            mp._e_pair_ss = None
            mp._e_pair_os = None
            mp.split_mo_coeff = lambda: (None, None, np.eye(3, 1))
            mp.split_mo_energy = lambda: (None, None, np.array([1.0]))
            mp._foo = np.diag([-1.0, -1.0])

            pair_domain = mp.make_pair_domain(_FakeEris(), timer={})

            expected_near = np.array([[True, False]])
            self.assertIsInstance(pair_domain, PairDomain_kdOSV_PAOBasis)
            np.testing.assert_array_equal(pair_domain.pair_mask, expected_near)
            np.testing.assert_array_equal(
                mp._pair_mask["Near pair"], expected_near)
            np.testing.assert_array_equal(
                mp._pair_mask["Residual pair"], expected_near)
            np.testing.assert_array_equal(
                mp._pair_mask["Dist pair"], ~expected_near)
            self.assertEqual(pair_domain.loop_pair(), [(0, 0)])

            summary = mp._pair_work_summary
            self.assertIs(summary["pao_pair_screening_enabled"], True)
            self.assertEqual(summary["near_pairs"], 1)
            self.assertEqual(summary["dist_pairs"], 1)
            self.assertEqual(summary["residual_pairs"], 1)
            self.assertEqual(summary["strong_pairs"], 1)
            self.assertEqual(summary["overlap_pair_blocks"], 2)
            self.assertEqual(summary["fock_pair_blocks"], 0)
        finally:
            kpao_mod.PeriodicPAO = old_pao
            kpao_mod.PeriodicPAOBasisOSV = old_osv
            pd_mod.pbc_pair_dipole_pao = old_dipole

    def test_kdosv_make_pao_bp_mode_passes_reference_atom_domains(self):
        mp = kdOSVMP2.__new__(kdOSVMP2)
        mp.cell = object()
        mp.lo_coeff = np.eye(4, 2)
        mp.nlo_per_cell = 1
        mp.nlo = 2
        mp._cell_sub = np.array([[0, 1], [1, 0]])
        mp._s1e = np.eye(4)
        mp.pbc_pao_domain_mode = "bp"
        mp.pbc_pao_bp_thresh = 0.8
        mp.split_mo_coeff = lambda: (None, None, np.eye(4, 2))
        mp.split_mo_energy = lambda: (None, None, np.array([1.0, 2.0]))

        old_pao = kpao_mod.PeriodicPAO
        old_domains = kpao_mod.reference_atom_domains
        try:
            kpao_mod.PeriodicPAO = _FakePao
            kpao_mod.reference_atom_domains = (
                lambda *args, **kwargs: [np.array([0, 1])])

            pao = mp.make_pao_full()

            self.assertIsInstance(pao, _FakePao)
            np.testing.assert_array_equal(
                pao.kwargs["ref_atom_domains"][0], [0, 1])
        finally:
            kpao_mod.PeriodicPAO = old_pao
            kpao_mod.reference_atom_domains = old_domains

    def test_kdosv_make_pao_principal_mode_passes_threshold(self):
        mp = kdOSVMP2.__new__(kdOSVMP2)
        mp.cell = object()
        mp.lo_coeff = np.eye(4, 2)
        mp.nlo_per_cell = 1
        mp.nlo = 2
        mp._cell_sub = np.array([[0, 1], [1, 0]])
        mp._s1e = np.eye(4)
        mp.pbc_pao_domain_mode = "principal"
        mp.pbc_pao_principal_thresh = 0.85
        mp.split_mo_coeff = lambda: (None, None, np.eye(4, 2))
        mp.split_mo_energy = lambda: (None, None, np.array([1.0, 2.0]))

        captured = {}
        old_pao = kpao_mod.PeriodicPAO
        old_domains = kpao_mod.reference_atom_domains
        try:
            kpao_mod.PeriodicPAO = _FakePao

            def fake_domains(*args, **kwargs):
                captured.update(kwargs)
                return [np.array([0, 1])]

            kpao_mod.reference_atom_domains = fake_domains

            pao = mp.make_pao_full()

            self.assertIsInstance(pao, _FakePao)
            self.assertEqual(captured["mode"], "principal")
            self.assertEqual(captured["principal_thr"], 0.85)
        finally:
            kpao_mod.PeriodicPAO = old_pao
            kpao_mod.reference_atom_domains = old_domains


if __name__ == '__main__':
    unittest.main()
