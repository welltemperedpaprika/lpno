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

import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger

from pyscf.lpno import osv as osv_mod
from pyscf.pbc.lpno import kpair_domain as pd_mod
from pyscf.pbc.lpno.kpts2supcell import k2s_scf
from pyscf.pbc.lpno.kpair_domain import (pbc_pair_dipole,
                                          pbc_pair_ewald_dipole)
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from pyscf.lpno import pnomp2 as pno_mp2_slow
from pyscf.pbc.lpno import kpao_mp2 as kpao_mp2_mod
from pyscf.lpno.tools import summarize_domain
from pyscf.pbc.lpno import ao2mo as _pbc_ao2mo

def _expand_trs_mf(kmf):
    '''Expand IBZ → full BZ if TRS KPoints detected. Returns (mf, nkpts).'''
    from pyscf.pbc.lib.kpts import KPoints
    kpts = kmf.kpts
    if not isinstance(kpts, KPoints) or len(kpts) == kpts.nkpts:
        return kmf, len(np.asarray(kpts).reshape(-1, 3))

    from pyscf.pbc.lo.base import remove_trs_mo
    from pyscf.pbc import scf
    nkpts = kpts.nkpts

    mf_bz = scf.KRHF(kmf.cell, kpts=np.asarray(kpts.kpts)).density_fit(
        auxbasis=kmf.with_df.auxbasis)
    mf_bz.mo_coeff = list(remove_trs_mo(np.asarray(kmf.mo_coeff), kpts))
    mf_bz.mo_energy = list(kpts.transform_mo_energy(kmf.mo_energy))
    mf_bz.mo_occ = list(kpts.transform_mo_occ(kmf.mo_occ))
    mf_bz.e_tot = kmf.e_tot
    mf_bz.converged = True
    mf_bz.with_df = kmf.with_df
    return mf_bz, nkpts

einsum = lib.einsum
class KPNOMP2(pno_mp2_slow.PNOMP2):
    '''PNO-MP2 for periodic systems (k-point sampling).

    Performs periodic MP2 via an LO→supercell transformation followed by
    pair-natural-orbital (PNO) truncation and iterative amplitude equations.

    Args:
        kmf : KRHF object with density fitting (``with_df`` attribute required)
            A converged k-point restricted Hartree-Fock object.  If
            ``kmf.kpts`` is a symmetry-reduced ``KPoints`` object it is
            automatically expanded to the full Brillouin zone.
        lo_coeff : ndarray, shape (nao_supercell, nlo)
            Supercell occupied localized-orbital (LO) coefficient matrix.
            Must satisfy translation symmetry: ``lo_coeff[:, j + npc] ==
            T_1 lo_coeff[:, j]`` for all j, where ``npc`` is the number of
            LOs per cell.
        thresh_pno : float, optional
            Occupation-number threshold for PNO truncation.  Default 3e-7.
        frozen : int or list of int, optional
            Frozen core orbitals (see :class:`pyscf.mp.mp2.MP2`).
        mf : SCF object, optional
            Pre-built supercell SCF.  Constructed from ``kmf`` if not given.
        kmesh : list of int [nx, ny, nz], optional
            k-point mesh dimensions; inferred from ``kmf.kpts`` if not given.

    Attributes:
        thresh_osv : float
            OSV occupation-number threshold for the virtual-space pre-screen
            (inherited from :class:`~pyscf.lpno.pnomp2.PNOMP2`).
        thresh_pno : float
            PNO truncation threshold.
        pbc_ao2mo_mode : {'incore', 'outcore'}
            Strategy for building the periodic DF ERIs (default ``'incore'``).
        pbc_osv_mode : {'refcell', 'supercell', 'pao'}
            How the OSV virtual space is constructed (default ``'refcell'``).
        kmesh : list of int
            k-point mesh ``[nx, ny, nz]``.
        iterative_damping : float
            Fraction of new amplitudes mixed per iteration (default 1.0 = no
            damping).  Reduce below 1 to stabilise slow-converging cases.
            Only active when ``DIIS`` is not used.
        dipole_mode : {'ewald', 'direct'}
            Method for the long-range dipole pair-energy estimate used in pair
            pre-screening (default ``'ewald'``).

    Saved results:
        e_corr : float
            MP2 correlation energy per unit cell (Ha).
        e_tot : float
            Total energy = ``kmf.e_tot + e_corr``.
        t2 : dict
            Converged amplitude tensors keyed by pair index ``(i, j)``.
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
        self.rmin_distpair = 20
        self.thresh_foo = 1e-5
        self.dipole_mode = 'ewald'
        self.screen_mode = 'energy'
        self.thresh_distpair = 1e-6
        self.use_doi_screening = False
        self.thresh_doi = 1e-2
        self.doi_metric = 'ao'
        self._keys = self._keys.union(['kpts', '_kscf', 'nlo', 'nlo_per_cell',
                                       'use_doi_screening', 'thresh_doi',
                                       'doi_metric'])

        self._k_lists_cache = None
        self._residual_plans = None
        self._t2_coupling = {}
        self.t2_clip_max = None
        self.pbc_ao2mo_mode = 'incore'
        self.pbc_osv_mode = 'refcell'
        kpao_mp2_mod.set_default_options(self)
        self.use_stacked_residual = True
        self.use_scmp2_coupling = False
        self.enforce_coupling_closure = True
        self._apply_coupling_closure = False
        self._keys = self._keys.union(['pbc_ao2mo_mode', 'pbc_osv_mode',
                                       'enforce_coupling_closure',
                                       *kpao_mp2_mod.PAO_KEYS])

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
        log.info('pbc_osv_mode = %s', self.pbc_osv_mode)
        if self.pbc_osv_mode == 'pao':
            kpao_mp2_mod.dump_flags(self, log)
        log.info('use_doi_screening = %s', self.use_doi_screening)
        if self.use_doi_screening:
            log.info('doi_metric = %s', self.doi_metric)
            log.info('thresh_doi = %.1e', self.thresh_doi)
        log.info('use_scmp2_coupling = %s', self.use_scmp2_coupling)
        log.info('enforce_coupling_closure = %s', self.enforce_coupling_closure)

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
        if self.pbc_osv_mode == 'pao':
            return kpao_mp2_mod.make_osv(self, eris, moe_lo, thresh_osv)

        vir_energy = self.split_mo_energy()[2]
        osv_param = {'thresh': thresh_osv}
        if self.pbc_osv_mode == 'refcell':
            osv_param.update(self._get_refcell_osv_param())
        osv = self.OSV(osv_param, eris, moe_lo, vir_energy)
        summarize_domain(osv.nosv, log, 'OSV domain size')
        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'osv/nosv', osv.nosv)
        return osv

    def make_pao_full(self):
        return kpao_mp2_mod.make_pao_full(self)

    def make_pair_domain(self, eris=None, timer=None):
        log = logger.new_logger(self)
        if timer is None: timer = {}

        self._k_lists_cache = None

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

        if self.pbc_osv_mode == 'pao':
            return kpao_mp2_mod.make_kpno_pair_domain(
                self, eris, osv, timer, cput2, log)

        if self.screen_mode == 'energy':
            pair_mask, near_list, epair_dist = self.screen_energy_pair(osv)
        else:
            pair_mask, epair_dist = self.screen_dist_pair(osv)
        pair_mask, epair_dist, doi_promoted = self._apply_doi_pair_promotion(
            pair_mask, epair_dist)
        if getattr(self, 'use_doi_screening', False):
            log.info('DOI-near promotion (%s > %.1e): %d Promoted | %d Near | %d Dist | %d Total pairs',
                     self.doi_metric, self.thresh_doi,
                     np.count_nonzero(doi_promoted),
                     np.count_nonzero(pair_mask),
                     np.count_nonzero(~pair_mask),
                     pair_mask.size)
        self._update_pair_mask('Dist pair', ~pair_mask)
        self._update_pair_energy('Dist pair', epair_dist, ~pair_mask)

        cput3 = log.timer('Pair prescreen', *cput2)
        timer['Pair prescreen'] = np.asarray(cput3) - np.asarray(cput2)

        apply_coupling_closure = self._should_apply_coupling_closure(pair_mask)
        self._apply_coupling_closure = apply_coupling_closure
        pair_mask_domain = (
            self._residual_coupling_mask(pair_mask)
            if apply_coupling_closure else pair_mask
        )
        if apply_coupling_closure:
            log.info('Residual coupling-closure: %d Energy-near | %d Coupling-only | %d Total pairs',
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
                                      keep_weak_domains=apply_coupling_closure)

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

    def _should_apply_coupling_closure(self, pair_mask):
        if not bool(np.any(~pair_mask)):
            return False
        return bool(getattr(self, 'enforce_coupling_closure', False)
                    or self.use_scmp2_coupling)

    def _residual_coupling_mask(self, pair_mask):
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

    def _ao_density_doi_scores(self):
        coeff = np.asarray(self.lo_coeff)
        s_ao = np.asarray(self.s1e)
        prob = np.abs(coeff) ** 2
        metric = np.abs(s_ao) ** 2
        metric_prob = metric.dot(prob)
        raw_overlap = prob[:, :self.nlo_per_cell].T.dot(metric_prob)
        self_overlap = np.einsum('pi,pi->i', prob, metric_prob)

        norm = (
            np.sqrt(np.maximum(self_overlap[:self.nlo_per_cell, None], 0.0))
            * np.sqrt(np.maximum(self_overlap[None, :], 0.0))
        )
        rel_overlap = np.divide(
            raw_overlap, norm, out=np.zeros_like(raw_overlap, dtype=float),
            where=norm > 0)
        raw_score = np.sqrt(np.maximum(raw_overlap, 0.0))
        rel_score = np.sqrt(np.maximum(rel_overlap, 0.0))
        return raw_score, rel_score

    def _doi_pair_scores(self):
        metric = getattr(self, 'doi_metric', 'ao')
        raw_score, rel_score = self._ao_density_doi_scores()
        if metric in ('ao', 'ao_raw'):
            return raw_score
        if metric in ('ao_rel', 'rel'):
            return rel_score
        raise ValueError(f'Unknown DOI metric: {metric}')

    def _apply_doi_pair_promotion(self, pair_mask, epair_dist):
        final_mask = pair_mask.copy()
        promoted = np.zeros_like(pair_mask, dtype=bool)

        if getattr(self, 'use_doi_screening', False) and np.any(~pair_mask):
            scores = self._doi_pair_scores()
            if scores.shape != pair_mask.shape:
                raise ValueError(
                    f'DOI score shape {scores.shape} does not match pair mask '
                    f'shape {pair_mask.shape}')
            promoted = (~pair_mask) & (scores > self.thresh_doi)
            final_mask[promoted] = True
            if getattr(self, 'chkfile', None):
                lib.chkfile.dump(self.chkfile, 'pair/doi_score', scores)
                lib.chkfile.dump(self.chkfile, 'pair/mask_doi_promoted', promoted)

        final_epair_dist = self._mask_pair_energy(epair_dist, ~final_mask)
        return final_mask, final_epair_dist, promoted

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

    def screen_dist_pair(self, osv):
        log = logger.new_logger(self)
        epair, Rpair = self._get_dipole_energies(osv)

        pair_mask = Rpair <= self.rmin_distpair
        dist_pair_mask = ~pair_mask

        epair_dist = epair * dist_pair_mask
        epair_dist = lib.tag_array(epair_dist, e_corr_ss=epair_dist * 0.5, e_corr_os=epair_dist * 0.5)

        near_idx = np.where(pair_mask)
        near_list = list(zip(near_idx[0], near_idx[1]))

        nnear = len(near_list)
        ndist = np.count_nonzero(~pair_mask)
        log.info('Dist-pair prescreen: %d Near | %d Dist | %d Total pairs',
                 nnear, ndist, nnear + ndist)
        if ndist > 0:
            e_dist_total = np.sum(epair_dist)
            log.info('Dist-pair energy estimate (dipole approx): %.9f Ha', e_dist_total)

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'pair/mask_dist', pair_mask)
            lib.chkfile.dump(self.chkfile, 'pair/Edist', epair)
            lib.chkfile.dump(self.chkfile, 'pair/R', Rpair)

        return pair_mask, epair_dist

    def screen_energy_pair(self, osv):
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
        log.info('Energy-pair prescreen (thresh=%.1e): %d Near | %d Dist | %d Total pairs',
                 self.thresh_distpair, nnear, ndist, nnear + ndist)
        if ndist > 0:
            e_dist_total = np.sum(epair_dist)
            log.info('Dist-pair energy estimate (dipole approx): %.9f Ha', e_dist_total)

        return pair_mask, near_list, epair_dist

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

        for k in ['Dist pair', 'SC-MP2 pair', 'Weak pair', 'PNO truncation']:
            if k in self._e_pair_ss:
                arr_ss = self._e_pair_ss[k]
                arr_os = self._e_pair_os[k]
                for i in range(self.nlo_per_cell):
                    for j in range(self.nlo):
                        factor = 0.5 if i != j else 1.0
                        e_ss_sum += arr_ss[i, j].real * factor
                        e_os_sum += arr_os[i, j].real * factor

        if 'Strong pair' in self._e_pair_ss:
            strong_e_ss = self._e_pair_ss['Strong pair']
            strong_e_os = self._e_pair_os['Strong pair']
            for i, j in strong_e_ss.keys():
                factor = 0.5 if i != j else 1.0
                e_ss_sum += strong_e_ss.get((i, j), 0).real * factor
                e_os_sum += strong_e_os.get((i, j), 0).real * factor

        e_corr_sum = e_ss_sum + e_os_sum
        return e_corr_sum, e_ss_sum, e_os_sum

    def energy_corr(self, pair_domain, t2):
        self._energy_corr_pair(pair_domain, t2, with_ex=True)


        e_corr_sum, e_ss_sum, e_os_sum = self._sum_e_corr()

        e_corr = lib.tag_array(e_corr_sum,
                               e_corr_ss=e_ss_sum,
                               e_corr_os=e_os_sum)
        return e_corr

    def init_guess(self, pair_domain):
        t2 = {}
        moe_lo = np.diag(self.foo)
        for i, j in pair_domain.loop_strong_pairs():
            vij = pair_domain.e[i][j]
            Kij = pair_domain.K[i][j]
            d = moe_lo[i] + moe_lo[j] - (vij[:, None] + vij)
            t2[i, j] = Kij / d
        self._t2_coupling = {}
        if getattr(self, '_apply_coupling_closure', False) or self.use_scmp2_coupling:
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

    def residual(self, t2, pair_domain, i, j, foo_mask, k_list=None):
        K = pair_domain.K
        foo = self.foo
        R = K[i][j].copy()
        pair_index = self._get_pair_index()

        if k_list is None:
            k_list = pair_domain.loop_k(i, j, foo_mask)

        if i == j:
            for k in k_list:
                if k != i and foo_mask[k, i]:
                    t2_ik = self._get_residual_t2(t2, i, k)
                    if t2_ik is not None:
                        S = pair_domain.get_ovlp(i, i, i, k)
                        term = reduce(lib.dot, (S, t2_ik, S.T.conj()))
                        dR = foo[k, i] * term
                        dR += dR.T.conj()
                        R -= dR
        else:
            for k in k_list:
                if k != j and foo_mask[k, j]:
                    t2_ik = self._get_residual_t2(t2, i, k)
                    if t2_ik is not None:
                        S = pair_domain.get_ovlp(i, j, i, k)
                        term1 = reduce(lib.dot, (S, t2_ik, S.T.conj()))
                        R -= foo[k, j] * term1

                if k != i and foo_mask[i, k]:
                    k_ref, k_cell = pair_index.ref_cell(k)
                    j_relative = pair_index.relative_lo(j, k_cell)

                    t2_kj = self._get_residual_t2(t2, k_ref, j_relative)
                    if t2_kj is not None:
                        S = pair_domain.get_ovlp(i, j, k, j)
                        term2 = reduce(lib.dot, (S, t2_kj, S.T.conj()))
                        R -= foo[i, k] * term2

        return R

    def _get_residual_t2(self, t2, i, j):
        if (i, j) in t2:
            return t2[i, j]
        return getattr(self, '_t2_coupling', {}).get((i, j))

    def _build_residual_plans(self, t2, pair_domain, foo_mask):
        foo = self.foo
        plans = {}
        pair_index = self._get_pair_index()
        for i, j in pair_domain.loop_strong_pairs():
            k_list = self._k_lists_cache.get((i, j))
            if k_list is None:
                k_list = pair_domain.loop_k(i, j, foo_mask)

            entries = []
            S_list = []
            if i == j:
                for k in k_list:
                    t2_ik = self._get_residual_t2(t2, i, k)
                    if k != i and foo_mask[k, i] and t2_ik is not None:
                        S = pair_domain.get_ovlp(i, i, i, k)
                        entries.append((S, (i, k), foo[k, i]))
                        S_list.append(S)
            else:
                for k in k_list:
                    t2_ik = self._get_residual_t2(t2, i, k)
                    if k != j and foo_mask[k, j] and t2_ik is not None:
                        S = pair_domain.get_ovlp(i, j, i, k)
                        entries.append((S, (i, k), foo[k, j]))
                        S_list.append(S)

                    if k != i and foo_mask[i, k]:
                        k_ref, k_cell = pair_index.ref_cell(k)
                        j_relative = pair_index.relative_lo(j, k_cell)
                        t2_kj = self._get_residual_t2(t2, k_ref, j_relative)
                        if t2_kj is not None:
                            S = pair_domain.get_ovlp(i, j, k, j)
                            entries.append((S, (k_ref, j_relative), foo[i, k]))
                            S_list.append(S)

            S_concat_T = np.hstack(S_list).T.conj() if S_list else None
            t2_keys = [e[1] for e in entries]
            t2_is_fixed = [key not in t2 for key in t2_keys]
            fock_vals = np.ascontiguousarray(
                [e[2] for e in entries], dtype=np.float64)
            S_arrays_c = [np.ascontiguousarray(S) for S in S_list]
            plans[(i, j)] = {
                'K': pair_domain.K[i][j],
                'is_diag': i == j,
                'entries': entries,
                'S_concat_T': S_concat_T,
                'S_arrays': S_arrays_c,
                't2_keys': t2_keys,
                't2_is_fixed': t2_is_fixed,
                'fock_vals': fock_vals,
            }
        return plans

    def _residual_prepacked(self, t2, plan):
        entries = plan['entries']
        if not entries:
            return plan['K'].copy()
        R = plan['K'].copy()
        if plan['is_diag']:
            B_list = []
            for S, t2_key, f in entries:
                t2_val = self._get_residual_t2(t2, *t2_key)
                B_list.append(f * lib.dot(S, t2_val + t2_val.T.conj()))
        else:
            B_list = []
            for S, t2_key, f in entries:
                B_list.append(f * lib.dot(S, self._get_residual_t2(t2, *t2_key)))
        R -= lib.dot(np.hstack(B_list), plan['S_concat_T'])
        return R

    def update_amp(self, t2, pair_domain):
        t2new = t2.copy()
        moe_lo = np.diag(self.foo)
        foo_mask = abs(self.foo) > self.thresh_foo

        if not hasattr(self, '_k_lists_cache') or self._k_lists_cache is None:
            self._k_lists_cache = pair_domain.precompute_k_lists(foo_mask)

        use_prepacked = getattr(self, 'use_stacked_residual', False)
        if use_prepacked:
            if self._residual_plans is None:
                self._residual_plans = self._build_residual_plans(
                    t2, pair_domain, foo_mask)

        for i, j in pair_domain.loop_strong_pairs():
            vij = pair_domain.e[i][j]
            d = moe_lo[i] + moe_lo[j] - (vij[:, None] + vij)
            if use_prepacked:
                R = self._residual_prepacked(t2new,
                                             self._residual_plans[(i, j)])
            else:
                k_list = self._k_lists_cache.get((i, j))
                R = self.residual(t2new, pair_domain, i, j, foo_mask, k_list)
            t2new[i, j] = R / d

            if self.t2_clip_max is not None:
                t2_abs = np.abs(t2new[i, j])
                if np.max(t2_abs) > self.t2_clip_max:
                    scale = np.minimum(1.0, self.t2_clip_max / np.maximum(t2_abs, 1e-10))
                    t2new[i, j] = t2new[i, j] * scale

        return t2new


    def _finalize(self, timer):
        from pyscf.mp.mp2 import MP2
        log = logger.new_logger(self)
        MP2._finalize(self)
        self._timer_summary(timer)

        log.info('-' * 44)
        log.info('Correlation energy breakdown')
        log.info('-' * 44)

        e_strong = 0.0
        if 'Strong pair' in self._e_pair_ss:
            for (i, j) in self._e_pair_ss['Strong pair'].keys():
                e_ss = self._e_pair_ss['Strong pair'][(i, j)]
                e_os = self._e_pair_os['Strong pair'][(i, j)]
                e_pair = e_ss + e_os
                factor = 0.5 if i != j else 1.0
                e_strong += e_pair * factor

        e_total = self.e_corr

        log.info('%-15s  % 14.9f (%6.2f%%)', 'Strong pair', e_strong,
                 e_strong / e_total * 100 if e_total != 0 else 0)

        for k in ['SC-MP2 pair', 'Weak pair', 'PNO truncation', 'Dist pair']:
            e_k = 0.0
            if k in self._e_pair_ss:
                arr_ss = self._e_pair_ss[k]
                arr_os = self._e_pair_os[k]
                for i in range(self.nlo_per_cell):
                    for j in range(self.nlo):
                        factor = 0.5 if i != j else 1.0
                        e_k += (arr_ss[i, j] + arr_os[i, j]).real * factor
            if e_k != 0 or k in self._e_pair_ss:
                log.info('%-15s  % 14.9f (%6.2f%%)', k, e_k,
                         e_k / e_total * 100 if e_total != 0 else 0)

        log.info('-' * 44)
        log.info('%-15s  % 14.9f (100.00%%)', 'Total', e_total)
        log.info('-' * 44)
        log.info('')
