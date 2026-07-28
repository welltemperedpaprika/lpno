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
    if not isinstance(kpts, KPoints):
        return kmf, len(np.asarray(kpts).reshape(-1, 3))
    if len(kpts) == kpts.nkpts:
        # TRS KPoints whose IBZ already equals the full BZ (every point its
        # own time-reversal image, e.g. any 2x2x2 mesh): no conjugate
        # expansion needed, but downstream code requires a plain kpts array
        # -- normalize the wrapper, mapping mo data into BZ order.
        from pyscf.pbc import scf
        order = np.asarray(kpts.bz2ibz, dtype=int)
        mf_bz = scf.KRHF(kmf.cell, kpts=np.asarray(kpts.kpts)).density_fit(
            auxbasis=kmf.with_df.auxbasis)
        mf_bz.mo_coeff = [kmf.mo_coeff[x] for x in order]
        mf_bz.mo_energy = [kmf.mo_energy[x] for x in order]
        mf_bz.mo_occ = [kmf.mo_occ[x] for x in order]
        mf_bz.e_tot = kmf.e_tot
        mf_bz.converged = True
        mf_bz.with_df = kmf.with_df
        # KRHF.kpts proxies with_df.kpts; force the plain full-BZ array so
        # no KPoints object leaks through the reused DF object
        mf_bz.kpts = np.asarray(kpts.kpts)
        return mf_bz, kpts.nkpts

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
        '''Pre-resolve all residual contributions for each strong pair.

        Returns {(i,j): plan} where plan holds direct array references and
        precomputed constants so the per-iteration residual avoids conditionals,
        cache lookups, and index arithmetic.
        '''
        foo = self.foo
        moe_lo = np.diag(foo)
        plans = {}
        pair_index = self._get_pair_index()
        # The shared-half-product fast path reads the OSV overlaps and joint
        # transforms directly and is exact only for the plain periodic PNO
        # domain; subclasses (PAO basis) and test doubles keep the uniform
        # per-entry get_ovlp construction.
        fast = type(pair_domain) is pd_mod.PairDomain_kPNO
        if fast:
            osv = pair_domain.osv
            S_osv = osv.S
            w_all = pair_domain.w
            nosv = osv.nosv

        # ---- pass 1: entry metadata (source pair, Fock factor); no overlaps
        meta = {}
        for i, j in pair_domain.loop_strong_pairs():
            k_list = self._k_lists_cache.get((i, j))
            if k_list is None:
                k_list = pair_domain.loop_k(i, j, foo_mask)
            # each entry: (t2 key = translated source pair, original source
            # orbitals for the S-block lookups, Fock factor).  get_ovlp uses
            # the translated pair only for the w lookup; the OSV overlaps are
            # between the orbitals in their true cells.
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

        # ---- pass 2/3: shared half-products, then two fat GEMMs per target.
        # H[x, src=(p,q)] = S_xp w_top + S_xq w_bot depends only on the row
        # orbital and the SOURCE pair, so it is computed once and reused by
        # every target pair that couples to that source (the old per-entry
        # get_ovlp recomputed it per target: the 2026-07-27 Si A/B showed
        # this construction is ~90% of the residual stage).  The target's
        # transform is then applied to the hstacked halves in one GEMM per
        # row, producing S_concat directly.
        def osv_S(x, y):
            blk = S_osv[max(x, y), min(x, y)]
            if x < y:
                blk = blk.T.conj()
            return blk

        H = {}

        def get_H(x, t2key, orb):
            # fast path only (guarded by `fast` at the call sites)
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

        n_ent = 0
        for (i, j), ent in meta.items():
            entries = []
            S_concat = None
            if ent and fast:
                n_ent += len(ent)
                HI = np.hstack([get_H(i, t2key, orb)
                                for t2key, orb, _ in ent])
                w_tgt = w_all[i][j]
                wt = w_tgt[:nosv[i]]
                wb = w_tgt[nosv[i]:]
                S_concat = lib.dot(wt.T.conj(), HI)
                if wb.shape[0]:
                    HJ = np.hstack([get_H(j, t2key, orb)
                                    for t2key, orb, _ in ent])
                    S_concat += lib.dot(wb.T.conj(), HJ)
                offs_ = np.cumsum(
                    [0] + [w_all[p][q].shape[1] for (p, q), _, _ in ent])
                entries = [
                    (S_concat[:, offs_[n]:offs_[n + 1]], t2key, f)
                    for n, (t2key, _, f) in enumerate(ent)]
            elif ent:
                n_ent += len(ent)
                S_cols = [pair_domain.get_ovlp(i, j, *orb)
                          for _, orb, _ in ent]
                S_concat = np.hstack(S_cols)
                offs_ = np.cumsum([0] + [S.shape[1] for S in S_cols])
                entries = [
                    (S_concat[:, offs_[n]:offs_[n + 1]], t2key, f)
                    for n, (t2key, _, f) in enumerate(ent)]
            plans[(i, j)] = (entries, S_concat)

        if fast:
            logger.new_logger(self).info(
                'residual plans: %d targets, %d entries, %d shared '
                'half-products (sharing %.1fx)', len(meta), n_ent, len(H),
                2.0 * n_ent / max(1, len(H)))
        H = None

        # ---- pass 4: per-plan statics (unchanged layout)
        for (i, j), (entries, S_concat) in plans.items():
            S_list = [S for S, _, _ in entries]
            S_concat_T = S_concat.T.conj() if S_concat is not None else None
            t2_keys = [e[1] for e in entries]
            t2_is_fixed = [key not in t2 for key in t2_keys]
            K = pair_domain.K[i][j]
            try:
                vij = pair_domain.e[i][j]
                denom = moe_lo[i] + moe_lo[j] - (vij[:, None] + vij)
            except (AttributeError, KeyError, TypeError):
                denom = None
            plan = {
                'K': K,
                'denom': denom,
                'is_diag': i == j,
                'entries': entries,
                'S_concat_T': S_concat_T,
                't2_keys': t2_keys,
                't2_is_fixed': t2_is_fixed,
            }
            # Lean-path statics: fold f_k into S_k^T once and preallocate the
            # stacked B^T buffer, so the per-iteration inner loop is a single
            # allocation-free GEMM per coupling entry.  Small-domain residuals
            # are BLAS-dispatch bound, not flop bound; every Python call and
            # temporary removed here is a direct wall-time saving.  Dispatch
            # is size-hybrid: np.matmul for small entries (lib.dot carries a
            # ~100 us call floor on the rusty python/MKL stack), lib.ddot for
            # large entries and flop-heavy closing GEMMs (numpy's BLAS there
            # runs ~5x slower than MKL at n >~ 150; measured 2026-07-26,
            # bench job 6681399).
            if entries and not (np.iscomplexobj(K)
                                or any(np.iscomplexobj(S) for S in S_list)):
                fST = [np.ascontiguousarray(f * S.T)
                       for S, _, f in entries]
                offs = np.cumsum([0] + [S.shape[1] for S in S_list])
                plan['fST'] = fST
                plan['offs'] = offs
                plan['BT'] = np.empty((offs[-1], K.shape[0]))
                plan['entry_big'] = [S.shape[1] >= 128 for S in S_list]
                plan['close_libdot'] = (
                    K.shape[0] ** 2 * offs[-1] >= 2e6)
            plans[(i, j)] = plan
        return plans

    def _residual_prepacked(self, t2, plan):
        '''Residual using precomputed plan — no conditionals or cache lookups.'''
        entries = plan['entries']
        if not entries:
            return plan['K'].copy()
        BT = plan.get('BT')
        if BT is not None:
            # Lean path: BT[o_k:o_k+dk] = t2_k^T @ (f_k S_k)^T, written in
            # place; the flops concentrate in the single closing GEMM.
            # Per-entry GEMMs dispatch by size (see _build_residual_plans).
            offs = plan['offs']
            fST = plan['fST']
            big = plan['entry_big']
            is_diag = plan['is_diag']
            get = self._get_residual_t2
            for idx, key in enumerate(plan['t2_keys']):
                t2_val = get(t2, *key)
                if is_diag:
                    t2_val = t2_val + t2_val.T
                else:
                    t2_val = t2_val.T
                blk = BT[offs[idx]:offs[idx + 1]]
                if big[idx]:
                    lib.ddot(t2_val, fST[idx], 1, blk, 0)
                else:
                    np.matmul(t2_val, fST[idx], out=blk)
            if plan['close_libdot']:
                R = plan['K'] - lib.dot(BT.T, plan['S_concat_T'])
            else:
                R = plan['K'] - np.matmul(BT.T, plan['S_concat_T'])
            return R
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
                # The one-time plan construction (dominated by the PNO-PNO
                # get_ovlp triple products in the joint-OSV dimensions) is
                # logged separately: the 2026-07-26 Si 4^3 A/B showed it is
                # ~90% of the 'Residual eqn' stage, and unlike the iteration
                # it does not shrink with T_PNO.
                t0 = logger.perf_counter()
                self._residual_plans = self._build_residual_plans(
                    t2, pair_domain, foo_mask)
                logger.new_logger(self).info(
                    'residual plan build (PNO overlaps): %.2f s wall',
                    logger.perf_counter() - t0)

        for i, j in pair_domain.loop_strong_pairs():
            if use_prepacked:
                plan = self._residual_plans[(i, j)]
                d = plan['denom']
                if d is None:
                    vij = pair_domain.e[i][j]
                    d = moe_lo[i] + moe_lo[j] - (vij[:, None] + vij)
                R = self._residual_prepacked(t2new, plan)
                R /= d
                t2new[i, j] = R
            else:
                vij = pair_domain.e[i][j]
                d = moe_lo[i] + moe_lo[j] - (vij[:, None] + vij)
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
