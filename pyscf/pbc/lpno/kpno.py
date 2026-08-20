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
from functools import reduce

from pyscf import lib
from pyscf.lib import logger

from pyscf.lpno import osv as osv_mod
from pyscf.pbc.lpno import kpair_domain as pd_mod
from pyscf.pbc.lpno.kpts2supcell import k2s_scf
from pyscf.pbc.lpno.kpair_domain import (
    pbc_pair_dipole, pbc_pair_ewald_dipole, sum_periodic_pair_energy)
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from pyscf.lpno.pnomp2 import PNOMP2
from pyscf.lpno.tools import summarize_domain
from pyscf.pbc.lpno import ao2mo as _pbc_ao2mo

def _expand_trs_mf(kmf):
    '''Expand IBZ -> full BZ if TRS KPoints detected. Returns (mf, nkpts).'''
    from pyscf.pbc.lib.kpts import KPoints
    from pyscf.pbc import scf
    kpts = kmf.kpts
    if not isinstance(kpts, KPoints):
        return kmf, len(np.asarray(kpts).reshape(-1, 3))
    if len(kpts) == kpts.nkpts:
        order = np.asarray(kpts.bz2ibz, dtype=int)
        mf_bz = scf.KRHF(kmf.cell, kpts=np.asarray(kpts.kpts)).density_fit(
            auxbasis=kmf.with_df.auxbasis)
        mf_bz.mo_coeff = [kmf.mo_coeff[x] for x in order]
        mf_bz.mo_energy = [kmf.mo_energy[x] for x in order]
        mf_bz.mo_occ = [kmf.mo_occ[x] for x in order]
        mf_bz.e_tot = kmf.e_tot
        mf_bz.converged = True
        mf_bz.with_df = kmf.with_df
        mf_bz.kpts = np.asarray(kpts.kpts)
        return mf_bz, kpts.nkpts

    nkpts = kpts.nkpts
    try:
        from pyscf.pbc.lo.base import remove_trs_mo
        mo_coeff_bz = list(remove_trs_mo(np.asarray(kmf.mo_coeff), kpts))
    except Exception:
        nkpts_ibz = kpts.nkpts_ibz
        mo_coeff_bz = [None] * nkpts
        for ki in range(nkpts_ibz):
            ids = np.where(kpts.bz2ibz == ki)[0]
            if ids.size == 1:
                mo_coeff_bz[ids[0]] = kmf.mo_coeff[ki]
            elif ids.size == 2:
                mo_coeff_bz[ids[0]] = np.asarray(kmf.mo_coeff[ki]).conj()
                mo_coeff_bz[ids[1]] = kmf.mo_coeff[ki]
            else:
                raise RuntimeError(f"Unexpected k-point count {ids.size} for IBZ {ki}")

    mf_bz = scf.KRHF(kmf.cell, kpts=np.asarray(kpts.kpts)).density_fit(
        auxbasis=kmf.with_df.auxbasis)
    mf_bz.mo_coeff = mo_coeff_bz
    mf_bz.mo_energy = list(kpts.transform_mo_energy(kmf.mo_energy))
    mf_bz.mo_occ = list(kpts.transform_mo_occ(kmf.mo_occ))
    mf_bz.e_tot = kmf.e_tot
    mf_bz.converged = True
    mf_bz.with_df = kmf.with_df
    mf_bz.kpts = np.asarray(kpts.kpts)
    return mf_bz, nkpts

einsum = lib.einsum


class KPNOMP2(PNOMP2):
    '''Pair natural orbital (PNO) MP2 for periodic systems with k-point sampling.

    Args:
        kmf : KRHF object with density fitting (with_df attribute required)
        lo_coeff : ndarray of shape (nao_supercell, nlo)
            Localized occupied orbital coefficients in the supercell basis.
        thresh_pno : float
            PNO truncation threshold based on occupation numbers. Default 3e-7.
        frozen : int or list of int, optional
            Frozen core orbital indices.
        mf : SCF object, optional
            Supercell SCF object.
        kmesh : list of int [nx, ny, nz], optional
            k-point mesh dimensions.
    '''
    ao2mo = _pbc_ao2mo.get_eris
    OSV = osv_mod.OSV
    PairDomain = pd_mod.PairDomain_kPNO

    def __init__(self, kmf, lo_coeff, thresh_pno=3e-7, frozen=None, mf=None, kmesh=None):
        if not hasattr(kmf, 'with_df'):
            raise RuntimeError("KPNOMP2 requires a KSCF object with density fitting.")

        # Expand IBZ → full BZ if TRS KPoints detected
        kmf, nkpts = _expand_trs_mf(kmf)

        self._kscf = kmf
        self.kpts = kmf.kpts
        self.nlo = lo_coeff.shape[1]
        self.nlo_per_cell = self.nlo // nkpts
        if kmesh is None:
            kmesh = kpts_to_kmesh(self._kscf.cell, self.kpts)
        self.kmesh = kmesh
        self._kmesh = kmesh

        if mf is None:
            mf = k2s_scf(kmf, kmesh=kmesh)

        super().__init__(mf, lo_coeff, thresh_pno=thresh_pno, frozen=frozen)

        self.with_df = kmf.with_df
        self._scf = mf
        self.cell = mf.cell

        self.pair_domain = None
        self.thresh_foo = 1e-5
        self.dipole_mode = 'ewald'
        self.thresh_distpair = 1e-6
        self._keys = self._keys.union(['kpts', '_kscf', 'nlo', 'nlo_per_cell'])

        self._k_lists_cache = None
        self._residual_plans = None
        self._t2_coupling = {}
        self.t2_clip_max = None
        self.pbc_ao2mo_mode = 'incore'
        self.promote_screened_pairs = True
        self._promote_screened = False
        self._keys = self._keys.union(['pbc_ao2mo_mode',
                                       'promote_screened_pairs'])

        self._cell_sub = get_kconserv(self._kscf.cell, self.kpts)[:, :, 0]
        self._pair_index = pd_mod.PeriodicPairIndex(
            self.nlo_per_cell, self.nlo, self._cell_sub)

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
        log.info('promote_screened_pairs = %s', self.promote_screened_pairs)

    def _get_pair_index(self):
        pair_index = getattr(self, '_pair_index', None)
        if pair_index is None:
            pair_index = pd_mod.PeriodicPairIndex(
                self.nlo_per_cell, self.nlo, self._cell_sub)
            self._pair_index = pair_index
        return pair_index

    def make_osv(self, eris, thresh_osv=None):
        log = logger.new_logger(self)
        if thresh_osv is None: thresh_osv = self.thresh_osv
        moe_lo = np.diag(self.foo)

        vir_energy = self.split_mo_energy()[2]
        osv_param = {'thresh': thresh_osv}
        osv_param.update(self._get_refcell_osv_param())
        osv = self.OSV(osv_param, eris, moe_lo, vir_energy)
        summarize_domain(osv.nosv, log, 'OSV domain size')
        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'osv/nosv', osv.nosv)
        return osv

    def make_pair_domain(self, eris=None, timer=None):
        log = logger.new_logger(self)
        if timer is None: timer = {}

        self._k_lists_cache = None

        cput0 = (logger.process_clock(), logger.perf_counter())
        owns_eris = eris is None
        if owns_eris: eris = self.ao2mo()
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

        promote_screened = self._should_promote_screened(pair_mask)
        self._promote_screened = promote_screened
        pair_mask_domain = (
            self._screened_promotion_mask(pair_mask)
            if promote_screened else pair_mask
        )
        if promote_screened:
            log.info('Screened-pair promotion: %d Energy-near | %d Promoted | %d Total pairs',
                     np.count_nonzero(pair_mask),
                     np.count_nonzero(pair_mask_domain & ~pair_mask),
                     np.count_nonzero(pair_mask_domain))

        pair_mask_full = self._expand_pair_mask(pair_mask_domain)
        osv.get_ovlp(pair_mask=pair_mask_full)
        osv.get_fock(pair_mask=pair_mask_full)

        moe_lo = np.diag(self.foo)
        vir_energy = self.split_mo_energy()[2]

        if getattr(eris, '_ovL_k_R', None) is not None and getattr(eris, '_U_k', None) is not None:
            eris._build_osv_cache([osv.u[j] for j in range(self.nlo)])

        pair_domain = self.PairDomain(eris, osv, self.pno_param, moe_lo, vir_energy,
                                      pair_mask=pair_mask_domain,
                                      nlo_per_cell=self.nlo_per_cell,
                                      nlo=self.nlo,
                                      thresh_psv_lindep=self.thresh_psv_lindep,
                                      thresh_weakpair=self.thresh_weakpair,
                                      with_ex_ene=self.with_ex_ene,
                                      compress_diagpair=self.compress_diagpair,
                                      cell_sub=self._cell_sub,
                                      keep_weak_domains=promote_screened)

        if owns_eris:
            self._clear_eris(eris)
        pair_domain.eris = None
        osv.F = None

        pair_mask_strong = pair_mask & pair_domain.pair_mask_strong
        pair_domain.pair_mask_residual = pair_mask_domain
        pair_domain.pair_mask_strong = pair_mask_strong

        self._update_pair_mask('Strong pair', pair_mask_strong)

        pair_mask_weak = pair_mask & (~pair_mask_strong)
        self._update_pair_mask('Weak pair', pair_mask_weak)

        if hasattr(pair_domain, 'epair_psv'):
            self._update_pair_energy('Weak pair',
                                     self._mask_pair_energy(pair_domain.epair_psv,
                                                            pair_mask_weak))

        if self.pno_param is not None and hasattr(pair_domain, 'epair_pno'):
            self._update_pair_energy('PNO truncation',
                                     self._mask_pair_energy(pair_domain.epair_pno,
                                                            pair_mask_strong))

        summarize_domain(pair_domain.npsv, log, 'PNO domain size')

        cput4 = log.timer('Pair domain (PNO build)', *cput3)
        timer['Pair domain'] = np.asarray(cput4) - np.asarray(cput3)

        return pair_domain

    def _should_promote_screened(self, pair_mask):
        if not bool(np.any(~pair_mask)):
            return False
        return bool(getattr(self, 'promote_screened_pairs', False))

    def _screened_promotion_mask(self, pair_mask):
        pair_mask_residual = pair_mask.copy()
        foo_mask = abs(self.foo) > self.thresh_foo
        near_i, near_j = np.where(pair_mask)
        pair_index = self._get_pair_index()

        for i, j in zip(near_i, near_j):
            for k in np.where(foo_mask[:, j])[0]:
                if k != j:
                    pair_mask_residual[i, k] = True

            for k in np.where(foo_mask[i, :])[0]:
                if k != i:
                    k_ref, k_cell = pair_index.ref_cell(k)
                    j_relative = pair_index.relative_lo(j, k_cell)
                    pair_mask_residual[k_ref, j_relative] = True

        return pair_mask_residual

    def _mask_pair_energy(self, e_pair, mask):
        e_pair_masked = np.asarray(e_pair) * mask
        e_pair_ss = getattr(e_pair, 'e_corr_ss', e_pair * 0.5) * mask
        e_pair_os = getattr(e_pair, 'e_corr_os', e_pair * 0.5) * mask
        return lib.tag_array(e_pair_masked, e_corr_ss=e_pair_ss, e_corr_os=e_pair_os)

    def _get_refcell_osv_param(self):
        osv_param = {
            'nlo_per_cell': self.nlo_per_cell,
            'vir_coeff': self.split_mo_coeff()[2],
            's_ao': self.s1e,
        }
        if self._kmesh is not None:
            osv_param['kmesh'] = self._kmesh
        elif getattr(self, '_kscf', None) is not None:
            from pyscf.pbc.tools import k2gamma
            osv_param['kmesh'] = k2gamma.kpts_to_kmesh(
                self._kscf.cell, self.kpts - self.kpts[0])
        return osv_param

    def _expand_pair_mask(self, pair_mask):
        return self._get_pair_index().expand_ref_pair_mask(pair_mask)

    def _clear_eris(self, eris):
        if hasattr(eris, 'close'):
            eris.close()
        if hasattr(eris, 'ovLR'):
            eris.ovLR = None
            eris.ovLI = None
        if getattr(eris, '_ovL_k_R', None) is not None:
            eris._ovL_k_R = None
            eris._ovL_k_I = None
            eris._U_k = None
            eris._phase_factor = None
            eris._kj_for_ki_qi = None
            eris._qi_ranges = None
        if getattr(eris, '_osv_cache_R', None) is not None:
            eris._osv_cache_R = None
            eris._osv_cache_I = None

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

    def _update_pair_energy(self, k, v, mask=None):
        if self._e_pair is None:
            self._e_pair = {}
            self._e_pair_ss = {}
            self._e_pair_os = {}

        v_ss = getattr(v, 'e_corr_ss', None)
        v_os = getattr(v, 'e_corr_os', None)

        if v_ss is None: v_ss = v * 0.5
        if v_os is None: v_os = v * 0.5

        self._e_pair_ss[k] = v_ss
        self._e_pair_os[k] = v_os
        self._e_pair[k] = v

    def _update_pair_mask(self, k, v):
        if self._pair_mask is None: self._pair_mask = {}
        self._pair_mask[k] = v


    def _energy_corr_pair(self, pair_domain, t2, with_ex=True):
        e_corr_os = {}
        e_corr_ss = {}
        for i, j in pair_domain.loop_strong_pairs():
            fac = 1 if i == j else 2
            Kij = pair_domain.K[i][j]
            t2ij = t2[i, j]
            ed = einsum('ab,ab->', t2ij, Kij) * fac
            if with_ex:
                ex = einsum('ab,ba->', t2ij, Kij) * fac
            else:
                ex = 0
            e_corr_ss[i,j] = ed - ex
            e_corr_os[i,j] = ed
        e_corr_total = {p: e_corr_ss[p] + e_corr_os[p] for p in e_corr_ss}
        e_corr = lib.tag_array(e_corr_total, e_corr_ss=e_corr_ss, e_corr_os=e_corr_os)

        self._update_pair_energy('Strong pair', e_corr)
        return e_corr

    def _sum_e_corr(self):
        e_ss_sum = 0.0
        e_os_sum = 0.0
        for k in ['Strong pair', 'Weak pair', 'PNO truncation', 'Dist pair']:
            if self._e_pair_ss and k in self._e_pair_ss:
                e_ss_sum += sum_periodic_pair_energy(self._e_pair_ss[k], self.nlo_per_cell)
                e_os_sum += sum_periodic_pair_energy(self._e_pair_os[k], self.nlo_per_cell)
        return float(e_ss_sum + e_os_sum), float(e_ss_sum), float(e_os_sum)

    def energy_corr(self, pair_domain, t2):
        self._energy_corr_pair(pair_domain, t2, with_ex=True)
        e_corr_sum, e_ss_sum, e_os_sum = self._sum_e_corr()
        return lib.tag_array(e_corr_sum,
                               e_corr_ss=e_ss_sum,
                               e_corr_os=e_os_sum)

    def init_guess(self, pair_domain):
        t2 = {}
        moe_lo = np.diag(self.foo)
        for i, j in pair_domain.loop_strong_pairs():
            vij = pair_domain.e[i][j]
            Kij = pair_domain.K[i][j]
            d = moe_lo[i] + moe_lo[j] - (vij[:, None] + vij)
            t2[i, j] = Kij / d
        self._t2_coupling = {}
        if getattr(self, '_promote_screened', False):
            pair_mask_residual = getattr(pair_domain, 'pair_mask_residual', None)
            if pair_mask_residual is not None:
                for i in range(self.nlo_per_cell):
                    for j in range(self.nlo):
                        if pair_domain.pair_mask_strong[i, j] or not pair_mask_residual[i, j]:
                            continue
                        if j not in pair_domain.e[i] or j not in pair_domain.K[i]:
                            continue
                        vij = pair_domain.e[i][j]
                        Kij = pair_domain.K[i][j]
                        d = moe_lo[i] + moe_lo[j] - (vij[:, None] + vij)
                        self._t2_coupling[i, j] = Kij / d
        return t2

    def _get_residual_t2(self, t2, i, j):
        if (i, j) in t2:
            return t2[i, j]
        return getattr(self, '_t2_coupling', {}).get((i, j))

    def _build_residual_plans(self, t2, pair_domain, foo_mask):
        '''Pre-resolve all residual overlap projections and workspace for strong pairs.'''
        foo = self.foo
        moe_lo = np.diag(foo)
        plans = {}
        pair_index = self._get_pair_index()
        osv = pair_domain.osv
        S_osv = osv.S
        w_all = pair_domain.w
        nosv = osv.nosv

        meta = {}
        for i, j in pair_domain.loop_strong_pairs():
            k_list = self._k_lists_cache.get((i, j))
            if k_list is None:
                k_list = pair_domain.loop_k(i, j, foo_mask)
            ent = []
            if i == j:
                for k in k_list:
                    if (k != i and foo_mask[k, i]
                            and self._get_residual_t2(t2, i, k) is not None):
                        ent.append(((i, k), (i, k), foo[k, i]))
            else:
                for k in k_list:
                    if (k != j and foo_mask[k, j]
                            and self._get_residual_t2(t2, i, k) is not None):
                        ent.append(((i, k), (i, k), foo[k, j]))
                    if k != i and foo_mask[i, k]:
                        k_ref, k_cell = pair_index.ref_cell(k)
                        j_rel = pair_index.relative_lo(j, k_cell)
                        if self._get_residual_t2(t2, k_ref, j_rel) is not None:
                            ent.append(((k_ref, j_rel), (k, j), foo[i, k]))
            meta[(i, j)] = ent

        def osv_S(x, y):
            blk = S_osv[max(x, y), min(x, y)]
            if x < y:
                blk = blk.T.conj()
            return blk

        H = {}

        def get_H(x, t2key, orb):
            key = (x,) + orb
            val = H.get(key)
            if val is None:
                p, q = t2key
                ko, lo = orb
                w = w_all[p][q]
                wt = w[:nosv[p]]
                wb = w[nosv[p]:]
                Sxk = osv_S(x, ko)
                val = lib.dot(Sxk, wt) if Sxk.size else None
                if wb.shape[0]:
                    Sxl = osv_S(x, lo)
                    if Sxl.size:
                        contrib = lib.dot(Sxl, wb)
                        val = contrib if val is None else val + contrib
                if val is None:
                    val = np.zeros((nosv[x], w.shape[1]))
                H[key] = val
            return val

        for (i, j), ent in meta.items():
            K = pair_domain.K[i][j]
            vij = pair_domain.e[i][j]
            denom = moe_lo[i] + moe_lo[j] - (vij[:, None] + vij)

            if not ent:
                plans[(i, j)] = {
                    'K': K,
                    'denom': denom,
                    'is_diag': i == j,
                    'entries': [],
                }
                continue

            HI = np.hstack([get_H(i, t2key, orb) for t2key, orb, _ in ent])
            w_tgt = w_all[i][j]
            wt = w_tgt[:nosv[i]]
            wb = w_tgt[nosv[i]:]
            S_concat = lib.dot(wt.T.conj(), HI)
            if wb.shape[0]:
                HJ = np.hstack([get_H(j, t2key, orb) for t2key, orb, _ in ent])
                S_concat += lib.dot(wb.T.conj(), HJ)

            offs = np.cumsum([0] + [w_all[p][q].shape[1] for (p, q), _, _ in ent])
            entries = [
                (S_concat[:, offs[n]:offs[n + 1]], t2key, f)
                for n, (t2key, _, f) in enumerate(ent)]
            fST = [np.ascontiguousarray(f * S.T) for S, _, f in entries]

            plans[(i, j)] = {
                'K': K,
                'denom': denom,
                'is_diag': i == j,
                'entries': entries,
                'S_concat_T': S_concat.T.conj(),
                't2_keys': [e[1] for e in entries],
                'fST': fST,
                'offs': offs,
                'BT': np.empty((offs[-1], K.shape[0])),
            }

        return plans

    def _residual_prepacked(self, t2, plan):
        '''Vectorized residual evaluation using precomputed plan.'''
        if not plan['entries']:
            return plan['K'].copy()

        BT = plan['BT']
        offs = plan['offs']
        fST = plan['fST']
        is_diag = plan['is_diag']
        get = self._get_residual_t2

        for idx, key in enumerate(plan['t2_keys']):
            t2_val = get(t2, *key)
            if is_diag:
                t2_val = t2_val + t2_val.T
            else:
                t2_val = t2_val.T
            np.matmul(t2_val, fST[idx], out=BT[offs[idx]:offs[idx + 1]])

        return plan['K'] - np.matmul(BT.T, plan['S_concat_T'])

    def update_amp(self, t2, pair_domain):
        t2new = t2.copy()
        foo_mask = abs(self.foo) > self.thresh_foo

        if self._residual_plans is None:
            if not hasattr(self, '_k_lists_cache') or self._k_lists_cache is None:
                self._k_lists_cache = pair_domain.precompute_k_lists(foo_mask)
            self._residual_plans = self._build_residual_plans(
                t2, pair_domain, foo_mask)

        for i, j in pair_domain.loop_strong_pairs():
            plan = self._residual_plans[(i, j)]
            R = self._residual_prepacked(t2new, plan)
            R /= plan['denom']

            if self.t2_clip_max is not None:
                t2_abs = np.abs(R)
                if np.max(t2_abs) > self.t2_clip_max:
                    R = R * np.minimum(1.0, self.t2_clip_max / np.maximum(t2_abs, 1e-10))

            t2new[i, j] = R

        return t2new
    

    def _finalize(self, timer):
        from pyscf.mp.mp2 import MP2
        log = logger.new_logger(self)
        MP2._finalize(self)
        self._timer_summary(timer)

        log.info('-' * 44)
        log.info('Correlation energy breakdown')
        log.info('-' * 44)

        e_total = self.e_corr
        for k in ['Strong pair', 'Weak pair', 'PNO truncation', 'Dist pair']:
            if self._e_pair_ss and k in self._e_pair_ss:
                e_k = (sum_periodic_pair_energy(self._e_pair_ss[k], self.nlo_per_cell)
                       + sum_periodic_pair_energy(self._e_pair_os[k], self.nlo_per_cell))
                pct = e_k / e_total * 100 if e_total != 0 else 0
                log.info('%-15s  % 14.9f (%6.2f%%)', k, e_k, pct)

        log.info('-' * 44)
        log.info('%-15s  % 14.9f (100.00%%)', 'Total', e_total)
        log.info('-' * 44)
        log.info('')
