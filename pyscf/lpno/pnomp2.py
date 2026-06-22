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

"""PNO-MP2 molecular driver."""

import json

import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger

einsum = lib.einsum

from pyscf.lpno.base import BASE
from pyscf.lpno import tools
from pyscf.lpno import osv as osv_mod
from pyscf.lpno import pair_domain as pd_mod
from pyscf.lpno.dosvmp2 import dOSVMP2


class PNOMP2(BASE):
    ''' Pair natural orbital (PNO) MP2

        Args:
            thresh_pno (alias: thresh): float
                PNO truncation threshold based on occupation number. This is the main
                control of PNOMP2. A rough relationship between thresh and accuracy is
                    -------------------------
                    thresh_pno      accuracy
                    -------------------------
                    3e-7            99.80 %
                    1e-7            99.85 %
                    3e-8            99.90 %
                    1e-8            99.94 %
                    3e-9            99.96 %
                    1e-9            99.97 %
                    -------------------------
                Default is 3e-7, which should give a quick and reasonable estimate for
                many problems. Seting it to `None` keeps all PNOs, in which case OSVMP2
                results are obtained.

            ** The following args have recommended defaults based on numerical tests **

            thresh_weakpair : float
                This threshold determines whether a pair is strong or weak based on their
                estimated pair energy at SC-MP2 level evaluated within pair OSVs. Strong
                pairs are used to solve the amplitude equation, while weak pairs contribute
                only by their SC-MP2 pair energy.
                Tests suggest that a value of 3e-6 introduces about 0.01% of error in Ecorr,
                which is good enough for thresh_pno >~ 1e-9. The following heuristics is used
                for automatic determination of this parameter:
                    thresh_weakpair = min(3e-6, thresh_pno**0.5/10)
            thresh_pno_ene : float
                Threshold for the percentage of PNO SC-MP2 energy. (Eqn 19, Werner et al.
                J. Chem. Theory Comput. 2015, 11, 484). The main purpose of this parameter
                is eliminating accidental null selection by `thresh_pno`.
                Default is 0.9, which is found good enough for avoiding null selection, but
                otherwise notably looser than the recommendation by Werner et al. (0.997).
            thresh_osv : float
                OSV truncation threshold. Since PNOs are obtained within pair OSVs, this
                parameter determines the highest possible accuracy PNOMP2 can achieve.
                Too loose a value leads to low accuracy, while too high increases cost.
                Tests suggest that 3e-5 is good for thresh_pno >~ 1e-8, while tighter
                thresh_pno requires 1e-5. The following heuristics makes the selection
                of this parameter smooth and automatic:
                    thresh_osv = numpy.clip(thresh_pno**0.5/3, 1e-5, 3e-5)
            thresh_psv_lindep : float
                Threshold for removing linear dependency in the pair OSV space.
                Default is 1e-6.
            compress_diagpair : bool
                Wether to compress diagonal pairs (i == j) using PNOs. Default it False.
    '''

    OSV = osv_mod.OSV
    PairDomain = pd_mod.PairDomain_PNO

    def __init__(self, mf, lo_coeff, thresh_pno=3e-7, frozen=None):

        BASE.__init__(self, mf, lo_coeff, frozen=frozen)

        self.thresh_pno = thresh_pno

        # Parameters with default
        self._thresh_osv = None
        self.thresh_pno_ene = 0.9
        self._thresh_weakpair = None
        self.thresh_psv_lindep = 1e-6
        self.compress_diagpair = False

        self.pno_feature_export = None
        self.pno_feature_molecule_id = None
        self.pno_feature_topk = 8
        self.pno_feature_export_pair_labels = False
        self._pno_feature_export_written = False

    @property
    def thresh_osv(self):
        if self._thresh_osv is None:
            return np.clip(self.thresh_pno**0.5 / 3, 1e-5, 3e-5)
        else:
            return self._thresh_osv

    @thresh_osv.setter
    def thresh_osv(self, x):
        self._thresh_osv = x

    @property
    def thresh_weakpair(self):
        if self._thresh_weakpair is None:
            return min(3e-6, self.thresh_pno**0.5 / 10)
        else:
            return self._thresh_weakpair

    @thresh_weakpair.setter
    def thresh_weakpair(self, x):
        self._thresh_weakpair = x

    @property
    def pno_param(self):
        if self.thresh_pno is None:
            return None
        else:
            return {'thresh': self.thresh_pno, 'thresh_ene': self.thresh_pno_ene}

    @property
    def with_ex_ene(self):
        return True

    def dump_flags(self, verbose=None):
        BASE.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('thresh_pno = %g', self.thresh_pno)
        log.info('thresh_pno_ene = %g', self.thresh_pno_ene)
        log.info('thresh_osv = %g', self.thresh_osv)
        log.info('thresh_weakpair = %g', self.thresh_weakpair)
        log.info('thresh_psv_lindep = %g', self.thresh_psv_lindep)
        log.info('compress_diagpair = %s', self.compress_diagpair)

    make_pair_domain = dOSVMP2.make_pair_domain
    make_osv = dOSVMP2.make_osv

    def _make_pair_domain(self, eris, osv, pair_mask, pno_param=None):
        log = logger.new_logger(self)

        osv.get_ovlp(pair_mask=pair_mask)
        osv.get_fock(pair_mask=pair_mask)

        nocc = self.nocc
        if pno_param is None:
            pno_param = self.pno_param
        moe_lo = np.diag(self.foo)
        vir_energy = self.split_mo_energy()[2]
        pair_domain = self.PairDomain(eris, osv, pno_param, moe_lo, vir_energy, pair_mask,
                                      thresh_psv_lindep=self.thresh_psv_lindep,
                                      thresh_weakpair=self.thresh_weakpair,
                                      with_ex_ene=self.with_ex_ene,
                                      compress_diagpair=self.compress_diagpair,
                                      pno_occ_topk=self.pno_feature_topk)

        pair_mask_dist = tools._to_tril(~pair_mask, nocc)
        pair_mask_strong = tools._to_tril(pair_domain.pair_mask, nocc)
        pair_mask_weak = ~pair_mask_strong & ~pair_mask_dist
        npairs = [np.count_nonzero(x) for x in [pair_mask_strong, pair_mask_weak, pair_mask_dist]]
        ntot = nocc * (nocc + 1) // 2
        assert(sum(npairs) == ntot)
        log.debug('Weak-pair prescreen: %d Strong | %d Weak | %d Dist | %d Total pairs',
                  *npairs, ntot)

        self._update_pair_mask('Weak pair', pair_mask_weak)
        self._update_pair_mask('Strong pair', pair_mask_strong)

        e_pair_weak = lib.tag_array(pair_domain.epair_psv,
                                    e_corr_ss=pair_domain.epair_psv_ss,
                                    e_corr_os=pair_domain.epair_psv_os)
        self._update_pair_energy('Weak pair', e_pair_weak, pair_mask_weak)
        if pno_param is not None:
            e_pair_pno = lib.tag_array(pair_domain.epair_pno,
                                       e_corr_ss=pair_domain.epair_pno_ss,
                                       e_corr_os=pair_domain.epair_pno_os)
            self._update_pair_energy('PNO truncation', e_pair_pno, pair_mask_strong)

        npsv_raw = [min(osv.nosv[i] + osv.nosv[j], self.nmo - self.nocc)
                    for i, j in pair_domain.loop_pair() if i != j] + \
                   [osv.nosv[i] for i in range(self.nocc)]
        tools.summarize_domain(npsv_raw, log, 'Joint OSV domain size')
        tools.summarize_domain(pair_domain.npsv_tril, log, 'PNO domain size')

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'psv/npsv', pair_domain.npsv_tril)
            lib.chkfile.dump(self.chkfile, 'pair/mask_strong', pair_mask_strong)
            lib.chkfile.dump(self.chkfile, 'pair/mask_weak', pair_mask_weak)
            lib.chkfile.dump(self.chkfile, 'pair/Eweak', pair_domain.epair_psv)
            if pno_param is not None:
                lib.chkfile.dump(self.chkfile, 'pair/dEpno', pair_domain.epair_pno)

        if self.pno_feature_export:
            self._write_pno_feature_export(
                pair_domain, pair_mask_strong, pair_mask_weak, pair_mask_dist)

        return pair_domain

    def _pno_feature_molecule_id(self):
        if self.pno_feature_molecule_id is not None:
            return str(self.pno_feature_molecule_id)
        mol = getattr(self, 'mol', None)
        if mol is not None and getattr(mol, 'name', None):
            return str(mol.name)
        return 'unknown_molecule'

    def _write_pno_feature_export(self, pair_domain, pair_mask_strong, pair_mask_weak, pair_mask_dist):
        import h5py

        nocc = self.nocc
        ntot = nocc * (nocc + 1) // 2
        pair_j, pair_i = np.tril_indices(nocc)
        # Export indices in the ML-side upper-triangular convention: i <= j.
        pair_class = np.empty(ntot, dtype='S16')
        pair_class[pair_mask_strong] = b'strong'
        pair_class[pair_mask_weak] = b'weak'
        pair_class[pair_mask_dist] = b'distant'

        if pair_domain.pno_occ_top is None:
            pno_occ_top = np.zeros((ntot, int(self.pno_feature_topk)), dtype=np.float64)
        else:
            pno_occ_top = np.asarray(pair_domain.pno_occ_top, dtype=np.float64)

        with h5py.File(self.pno_feature_export, 'w') as h5:
            h5.create_dataset('molecule_ids', data=np.asarray([self._pno_feature_molecule_id()], dtype='S'))
            h5.create_dataset('pair_molecule_index', data=np.zeros(ntot, dtype=np.int32))
            h5.create_dataset('pair_local_orbital_i', data=pair_i.astype(np.int32))
            h5.create_dataset('pair_local_orbital_j', data=pair_j.astype(np.int32))
            h5.create_dataset('pair_pno_occ_top', data=pno_occ_top)
            h5.create_dataset(
                'pair_pno_feature_names',
                data=np.asarray(['pno_occ_%02d' % i for i in range(pno_occ_top.shape[1])], dtype='S'))
            h5.create_dataset('pair_class', data=pair_class)
            h5.create_dataset('pair_mask_strong', data=pair_mask_strong.astype(np.bool_))
            h5.create_dataset('pair_mask_weak', data=pair_mask_weak.astype(np.bool_))
            h5.create_dataset('pair_mask_distant', data=pair_mask_dist.astype(np.bool_))
            h5.create_dataset('pair_energy_weak_mp2', data=np.asarray(pair_domain.epair_psv))
            if pno_occ_top.shape[1]:
                h5.create_dataset('pno_occ_topk', data=np.asarray([pno_occ_top.shape[1]], dtype=np.int32))
            if hasattr(pair_domain, 'epair_pno'):
                h5.create_dataset('pair_energy_pno_trunc_mp2', data=np.asarray(pair_domain.epair_pno))
            h5.create_dataset(
                'provenance_json',
                data=np.asarray(json.dumps({
                    'producer': 'pyscf.lpno.pnomp2.PNOMP2',
                    'molecule_id': self._pno_feature_molecule_id(),
                    'nocc': int(nocc),
                    'pair_order': 'packed lower-triangular rows; exported indices satisfy i <= j',
                    'pno_occ_source': 'eigenvalues of PNO pair-density matrix in get_pno1',
                    'thresh_pno': None if self.thresh_pno is None else float(self.thresh_pno),
                    'thresh_pno_ene': None if self.thresh_pno_ene is None else float(self.thresh_pno_ene),
                    'thresh_weakpair': float(self.thresh_weakpair),
                    'compress_diagpair': bool(self.compress_diagpair),
                    'pair_labels_exported': bool(self.pno_feature_export_pair_labels),
                }), dtype='S'))
        self._pno_feature_export_written = True

    def _update_pno_feature_export_pair_labels(self, e_corr):
        if (
            not self.pno_feature_export
            or not self._pno_feature_export_written
            or not self.pno_feature_export_pair_labels
        ):
            return
        import h5py

        with h5py.File(self.pno_feature_export, 'a') as h5:
            for name, data in [
                ('pair_energy_strong_mp2', np.asarray(e_corr)),
                ('pair_label_mp2', np.asarray(self.e_pair)),
            ]:
                if name in h5:
                    del h5[name]
                h5.create_dataset(name, data=data)

    def _energy_corr_pair(self, pair_domain, t2, with_ex=True):
        nocc = self.nocc
        e_corr_ss = np.zeros((nocc, nocc))
        e_corr_os = np.zeros((nocc, nocc))
        for ipair, (i, j) in enumerate(pair_domain.loop_pair()):
            fac = 1 if i == j else 2
            ed = einsum('ab,ab->', t2[i, j], pair_domain.K[i, j]) * fac
            if with_ex:
                ex = einsum('ab,ba->', t2[i, j], pair_domain.K[i, j]) * fac
            else:
                ex = 0
            e_corr_ss[i, j] = ed - ex
            e_corr_os[i, j] = ed
        e_corr_ss = lib.pack_tril(e_corr_ss)
        e_corr_os = lib.pack_tril(e_corr_os)

        e_corr = lib.tag_array(e_corr_ss + e_corr_os, e_corr_ss=e_corr_ss, e_corr_os=e_corr_os)

        self._update_pair_energy('Strong pair', e_corr, pair_domain.pair_mask)

        if 'PNO truncation' in self._e_pair:
            # PNO truncation energy in `self.e_pair` is overwritten by the update above
            # Here we add it back to `self.e_pair`
            self.e_pair += self._e_pair['PNO truncation']
            self.e_pair_ss += self._e_pair_ss['PNO truncation']
            self.e_pair_os += self._e_pair_os['PNO truncation']

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'pair/Estrong', e_corr)

        self._update_pno_feature_export_pair_labels(e_corr)
        return e_corr

    def init_guess(self, pair_domain):
        t2 = pair_domain.new_array()
        K = pair_domain.K
        moe_v = pair_domain.e
        moe_lo = np.diag(self.foo)
        for i, j in pair_domain.loop_pair():
            t2[i, j] = K[i, j] / (moe_lo[i] + moe_lo[j] - (moe_v[i, j][:, None] + moe_v[i, j]))
        return t2

    def residual(self, t2, pair_domain, i, j, foo_mask):
        K = pair_domain.K
        foo = self.foo
        R = K[i, j].copy()
        if i == j:
            for k in pair_domain.loop_k(i, i):
                if k != i and foo_mask[k, i]:
                    S = pair_domain.get_ovlp(i, i, i, k)
                    dR = foo[k, i] * reduce(lib.dot, (S, t2[i, k], S.T.conj()))
                    dR += dR.T.conj()
                    R -= dR
        else:
            for k in pair_domain.loop_k(i, j):
                if k != j and foo_mask[k, j]:
                    S = pair_domain.get_ovlp(i, j, i, k)
                    R -= foo[k, j] * reduce(lib.dot, (S, t2[i, k], S.T.conj()))
                if k != i and foo_mask[i, k]:
                    S = pair_domain.get_ovlp(i, j, k, j)
                    R -= foo[i, k] * reduce(lib.dot, (S, t2[k, j], S.T.conj()))
        return R

    def update_amp(self, t2, pair_domain):
        moe_lo = np.diag(self.foo)
        moe_v = pair_domain.e
        foo_mask = abs(self.foo) > self.thresh_foo
        t2new = t2.copy()
        for i, j in pair_domain.loop_pair():
            R = self.residual(t2new, pair_domain, i, j, foo_mask)
            t2new[i, j] = R / (moe_lo[i] + moe_lo[j] - (moe_v[i, j][:, None] + moe_v[i, j]))
        return t2new
