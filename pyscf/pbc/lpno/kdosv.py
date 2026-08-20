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

import numpy as np
from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.lpno import osv as osv_mod
from pyscf.pbc.lpno import kpair_domain as pd_mod
from pyscf.lpno.dosvmp2 import dOSVMP2
from pyscf.pbc.lpno.kpair_domain import (
    pbc_pair_dipole, pbc_pair_ewald_dipole, sum_periodic_pair_energy)
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.pbc.lpno.kpts2supcell import k2s_scf
from pyscf.lpno.tools import summarize_domain
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from pyscf.pbc.lpno.kpno import _expand_trs_mf
from pyscf.pbc.lpno import ao2mo as _pbc_ao2mo


class kdOSVMP2(dOSVMP2):
    '''Diagonal OSV-MP2 for periodic systems with k-point sampling.

    Args:
        kmf : KRHF object with density fitting (with_df attribute required)
        lo_coeff : ndarray of shape (nao_supercell, nlo)
            Localized occupied orbital coefficients in the supercell basis.
        thresh_osv : float
            OSV truncation threshold. Default 1e-5.
        frozen : int or list of int, optional
            Frozen core orbital indices.
        mf : SCF object, optional
            Supercell SCF object.
        kmesh : list of int [nx, ny, nz], optional
            k-point mesh dimensions.
    '''

    ao2mo = _pbc_ao2mo.get_eris
    OSV = osv_mod.OSV
    PairDomain = pd_mod.PairDomain_kdOSV

    def __init__(self, kmf, lo_coeff, thresh_osv=1e-5, frozen=None, mf=None, kmesh=None):
        if not hasattr(kmf, 'with_df'):
            raise RuntimeError("kdOSVMP2 requires a KSCF object with density fitting.")

        kmf, nkpts = _expand_trs_mf(kmf)

        self._kscf = kmf
        self.kpts = kmf.kpts
        if kmesh is None:
            kmesh = kpts_to_kmesh(self._kscf.cell, self.kpts)
        self.kmesh = kmesh
        self._kmesh = kmesh
        self.nlo = lo_coeff.shape[1]
        self.nlo_per_cell = self.nlo // nkpts
        if mf is None:
            mf = k2s_scf(kmf, kmesh=kmesh)
        super().__init__(mf, lo_coeff, thresh_osv=thresh_osv, frozen=frozen)
        self.with_df = kmf.with_df
        if self.with_df._cderi is None:
            raise NotImplementedError('DF Integrals not found. Rerun KSCF with DF.')

        self._scf = mf
        self.cell = mf.cell

        self.thresh_foo = 1e-5
        self.dipole_mode = 'ewald'
        self.thresh_distpair = 1e-6
        self.pbc_ao2mo_mode = 'incore'
        self.osv_re_pseudocano = False
        self._residual_plans = None
        self._cell_sub = get_kconserv(self._kscf.cell, self.kpts)[:, :, 0]
        self._pair_index = pd_mod.PeriodicPairIndex(
            self.nlo_per_cell, self.nlo, self._cell_sub)
        self._keys = self._keys.union(['kpts', '_kscf', 'nlo', 'nlo_per_cell',
                                       'pbc_ao2mo_mode', 'osv_re_pseudocano'])

    @property
    def e_tot(self):
        return self._kscf.e_tot + self.e_corr

    @property
    def emp2_scs(self):
        return self.e_corr_ss*1./3. + self.e_corr_os*1.2

    @property
    def e_tot_scs(self):
        return self._kscf.e_tot + self.emp2_scs

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('nlo_per_cell = %d', self.nlo_per_cell)
        log.info('pbc_ao2mo_mode = %s', self.pbc_ao2mo_mode)

    def _get_pair_index(self):
        pair_index = getattr(self, '_pair_index', None)
        if pair_index is None:
            pair_index = pd_mod.PeriodicPairIndex(
                self.nlo_per_cell, self.nlo, self._cell_sub)
            self._pair_index = pair_index
        return pair_index

    def make_osv(self, eris=None):
        log = logger.new_logger(self)
        if eris is None: eris = self.ao2mo()
        moe_lo = np.diag(self.foo)

        vir_coeff = self.split_mo_coeff()[2]
        vir_energy = self.split_mo_energy()[2]
        osv_param = self._get_osv_param()
        osv_param.update(self._get_refcell_osv_param(vir_coeff))
        osv = self.OSV(osv_param, eris, moe_lo,
                       vir_energy, verbose=self.verbose)

        summarize_domain(osv.nosv, log, 'OSV domain size')

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'osv/nosv', osv.nosv)

        return osv

    def get_osv_param(self):
        return {'thresh': self.thresh_osv}

    def _get_osv_param(self):
        return self.get_osv_param()

    def make_pair_domain(self, eris=None, timer=None):
        log = logger.new_logger(self)
        if timer is None: timer = {}

        cput0 = (logger.process_clock(), logger.perf_counter())
        if eris is None: eris = self.ao2mo()
        cput1 = log.timer('DF ERIs', *cput0)
        timer['DF ERIs'] = np.asarray(cput1) - np.asarray(cput0)

        osv = self.make_osv(eris)
        if getattr(eris, '_refcell_ovLR', None) is not None:
            eris._refcell_ovLR = None
            eris._refcell_ovLI = None
        cput2 = log.timer('OSV domain', *cput1)
        timer['OSV domain'] = np.asarray(cput2) - np.asarray(cput1)

        pair_mask, near_list, epair_dist = self.screen_pairs(osv)
        self._update_pair_mask('Dist pair', ~pair_mask)
        self._update_pair_energy('Dist pair', epair_dist, ~pair_mask)
        cput3 = log.timer('Pair prescreen', *cput2)
        timer['Pair prescreen'] = np.asarray(cput3) - np.asarray(cput2)

        pair_domain = pd_mod.PairDomain_kdOSV(eris, osv, pair_mask, self.nlo_per_cell, self.nlo, near_list,
                                                  cell_sub=self._cell_sub)
        self._update_pair_mask('Strong pair', pair_domain.pair_mask)

        self._clear_eris(eris)

        cput4 = log.timer('Pair domain (JK build)', *cput3)
        timer['Pair domain'] = np.asarray(cput4) - np.asarray(cput3)

        return pair_domain

    def _get_refcell_osv_param(self, vir_coeff):
        return {
            'nlo_per_cell': self.nlo_per_cell,
            'vir_coeff': vir_coeff,
            's_ao': self.s1e,
            'kmesh': self.kmesh,
            're_pseudocano': self.osv_re_pseudocano,
        }

    def _clear_eris(self, eris):
        if hasattr(eris, 'close'):
            eris.close()

    def _update_pair_mask(self, k, v):
        if self._pair_mask is None:
            self._pair_mask = {}
        self._pair_mask[k] = v

    def _update_pair_energy(self, k, v, mask=None):
        if self._e_pair is None:
            self._e_pair = {}
            self._e_pair_ss = {}
            self._e_pair_os = {}
        self._e_pair_ss[k] = getattr(v, 'e_corr_ss', 0)
        self._e_pair_os[k] = getattr(v, 'e_corr_os', 0)
        self._e_pair[k] = v

    def _get_dipole_energies(self, osv):
        lo_coeff = self.lo_coeff
        vir_coeff = self.split_mo_coeff()[2]
        moe_lo = np.diag(self.foo)
        if self.dipole_mode == 'ewald':
            return pbc_pair_ewald_dipole(self.cell, lo_coeff, self.nlo_per_cell,
                                         moe_lo, vir_coeff, osv)
        else:
            return pbc_pair_dipole(self.cell, lo_coeff, self.nlo_per_cell,
                                   moe_lo, vir_coeff, osv)

    def screen_pairs(self, osv):
        log = logger.new_logger(self)
        epair, Rpair = self._get_dipole_energies(osv)

        pair_mask = np.abs(epair) > self.thresh_distpair
        for i in range(self.nlo_per_cell):
            pair_mask[i, i] = True

        dist_pair_mask = ~pair_mask
        epair_dist = epair * dist_pair_mask
        epair_dist = lib.tag_array(epair_dist, e_corr_ss=epair_dist * 0.5,
                                   e_corr_os=epair_dist * 0.5)

        near_idx = np.where(pair_mask)
        near_list = list(zip(near_idx[0], near_idx[1]))

        nnear = len(near_list)
        ndist = np.count_nonzero(~pair_mask)
        log.info('Pair prescreen (%s, thresh=%.1e): %d Near | %d Dist | %d Total pairs',
                 self.dipole_mode, self.thresh_distpair, nnear, ndist, nnear + ndist)
        if ndist > 0:
            e_dist_total = np.sum(epair_dist)
            log.info('Dist-pair energy estimate (dipole approx): %.9f Ha', e_dist_total)

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'pair/mask_dist', pair_mask)
            lib.chkfile.dump(self.chkfile, 'pair/Edist', epair)
            lib.chkfile.dump(self.chkfile, 'pair/R', Rpair)

        return pair_mask, near_list, epair_dist

    screen_energy_pair = screen_pairs
    screen_dist_pair = screen_pairs

    def init_guess(self, pair_domain):
        t2 = pair_domain.new_array()
        moe_lo = np.diag(self.foo)
        osv = pair_domain.osv

        for i, j in pair_domain.loop_pair():
            e_osv_i = osv.e[i]
            e_osv_j = osv.e[j]
            denom = moe_lo[i] + moe_lo[j] - (e_osv_i[:, None] + e_osv_j)
            t2[i, j] = pair_domain.K[i, j] / denom

        return t2

    def update_amp(self, t2, pair_domain):
        t2new = t2.copy()
        foo_mask = abs(self.foo) > self.thresh_foo

        if self._residual_plans is None:
            self._residual_plans = self._build_residual_plans(
                t2, pair_domain, foo_mask)

        for i, j in pair_domain.loop_pair():
            plan = self._residual_plans[(i, j)]
            R = self._residual_prepacked(t2new, plan)
            t2new[i, j] = R / plan['denom']
        return t2new

    def _build_residual_plans(self, t2, pair_domain, foo_mask):
        S = pair_domain.S
        foo = self.foo
        moe_lo = np.diag(foo)
        osv = pair_domain.osv
        pair_index = self._get_pair_index()
        plans = {}

        for i, j in pair_domain.loop_pair():
            term1_keys = []
            term1_fock = []
            term1_S = []
            term2_keys = []
            term2_fock = []
            term2_S = []

            for k in pair_domain.loop_k(i, j):
                k_ref, R_idx_k = pair_index.ref_cell(k)
                j_shifted = pair_index.relative_lo(j, R_idx_k)

                if k != j and foo_mask[k, j] and (i, k) in t2:
                    S_kj = S[max(k, j), min(k, j)]
                    if k < j: S_kj = S_kj.T.conj()
                    term1_keys.append((i, k))
                    term1_fock.append(foo[k, j])
                    term1_S.append(np.ascontiguousarray(S_kj))

                if k != i and foo_mask[i, k]:
                    if (k_ref, j_shifted) in t2:
                        S_ik = S[max(i, k), min(i, k)]
                        if i < k: S_ik = S_ik.T.conj()
                        term2_keys.append((k_ref, j_shifted))
                        term2_fock.append(foo[i, k])
                        term2_S.append(np.ascontiguousarray(S_ik))

            e_osv_i = osv.e[i]
            e_osv_j = osv.e[j]
            denom = moe_lo[i] + moe_lo[j] - (e_osv_i[:, None] + e_osv_j)

            plan = {
                'K': pair_domain.K[i, j],
                'denom': denom,
                'term1_keys': term1_keys,
                'term1_fock': np.ascontiguousarray(term1_fock, dtype=np.float64),
                'term1_R': np.vstack(term1_S) if term1_S else None,
                'term2_keys': term2_keys,
                'term2_L': np.hstack([f * s for f, s in zip(term2_fock, term2_S)]) if term2_S else None,
            }
            plans[(i, j)] = plan
        return plans

    def _residual_prepacked(self, t2, plan):
        term1_keys = plan['term1_keys']
        term2_keys = plan['term2_keys']
        if not term1_keys and not term2_keys:
            return plan['K'].copy()
        R = plan['K'].copy()
        if term1_keys:
            L1 = np.hstack([f * t2[key]
                            for key, f in zip(term1_keys, plan['term1_fock'])])
            R -= lib.dot(L1, plan['term1_R'])
        if term2_keys:
            R2 = np.vstack([t2[key] for key in term2_keys])
            R -= lib.dot(plan['term2_L'], R2)
        return R

    def _energy_corr_pair(self, pair_domain, t2, with_ex=True):
        e_corr_os = {}
        e_corr_ss = {}
        for i, j in pair_domain.loop_pair():
            fac = 1 if i == j else 2
            ed = einsum('ab,ab->', t2[i, j], pair_domain.K[i, j]) * fac
            ex = (einsum('ab,ab->', t2[i, j], pair_domain.J[i, j]) * fac
                  if with_ex else 0)
            e_corr_ss[i, j] = ed - ex
            e_corr_os[i, j] = ed

        e_corr_total = {p: e_corr_ss[p] + e_corr_os[p] for p in e_corr_ss}
        e_corr = lib.tag_array(e_corr_total, e_corr_ss=e_corr_ss, e_corr_os=e_corr_os)
        self._update_pair_energy('Strong pair', e_corr)
        return e_corr

    def energy_corr(self, pair_domain, t2):
        self._energy_corr_pair(pair_domain, t2, with_ex=True)
        e_corr_sum, e_ss_sum, e_os_sum = self._sum_e_corr()
        return lib.tag_array(e_corr_sum,
                               e_corr_ss=e_ss_sum,
                               e_corr_os=e_os_sum)

    def _sum_e_corr(self):
        e_ss_sum = 0.0
        e_os_sum = 0.0
        for k in ['Strong pair', 'Dist pair']:
            if self._e_pair_ss and k in self._e_pair_ss:
                e_ss_sum += sum_periodic_pair_energy(self._e_pair_ss[k], self.nlo_per_cell)
                e_os_sum += sum_periodic_pair_energy(self._e_pair_os[k], self.nlo_per_cell)
        return float(e_ss_sum + e_os_sum), float(e_ss_sum), float(e_os_sum)

    def _finalize(self, timer):
        from pyscf.mp.mp2 import MP2
        log = logger.new_logger(self)
        MP2._finalize(self)
        self._timer_summary(timer)

        log.info('-' * 44)
        log.info('Correlation energy breakdown')
        log.info('-' * 44)

        e_total = self.e_corr
        for k in ['Strong pair', 'Dist pair']:
            if self._e_pair_ss and k in self._e_pair_ss:
                e_k = (sum_periodic_pair_energy(self._e_pair_ss[k], self.nlo_per_cell)
                       + sum_periodic_pair_energy(self._e_pair_os[k], self.nlo_per_cell))
                pct = e_k / e_total * 100 if e_total != 0 else 0
                log.info('%-15s  % 14.9f (%6.2f%%)', k, e_k, pct)

        log.info('-' * 44)
        log.info('%-15s  % 14.9f (100.00%%)', 'Total', e_total)
        log.info('-' * 44)
        log.info('')
