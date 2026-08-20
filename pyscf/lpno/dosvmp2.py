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

"""Direct OSV-MP2 molecular driver."""

import numpy as np

from pyscf import lib
from pyscf.lib import logger

from pyscf.lpno.base import BASE
from pyscf.lpno import tools
from pyscf.lpno import osv as osv_mod
from pyscf.lpno import pair_domain as pd_mod


class dOSVMP2(BASE):
    ''' Projection-based direct orbital-specific virtual (OSV) MP2

        Args:
            lo_coeff : np.ndarray
                AO coefficient matrix of (non-frozen) localized occupied orbitals.
            thresh_osv : float
                OSV truncation threshold based on occupation number. A rough relationship
                between thresh and accuracy is
                    -------------------------
                    thresh_osv      accuracy
                    -------------------------
                    3e-5            99.95 %
                    1e-5            99.80 %
                    3e-6            99.95 %
                    1e-6            99.99 %
                    -------------------------
                Default is 1e-5.
    '''

    OSV = osv_mod.OSV
    PairDomain = pd_mod.PairDomain_dOSV

    def __init__(self, mf, lo_coeff, thresh_osv=1e-5, frozen=None):
        BASE.__init__(self, mf, lo_coeff, frozen=frozen)

        self.thresh_osv = thresh_osv

    def dump_flags(self, verbose=None):
        BASE.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('thresh_osv = %g', self.thresh_osv)

    def make_pair_domain(self, eris=None, timer=None):
        '''
            1. OSV
            2. Pair prescreen
            3. Pair domain
        '''
        log = logger.new_logger(self)
        if timer is None:
            timer = {}

        cput0 = (logger.process_clock(), logger.perf_counter())
        if eris is None:
            eris = self.ao2mo()
        cput1 = log.timer('DF ERIs', *cput0)
        timer['DF ERIs'] = np.asarray(cput1) - np.asarray(cput0)

        # OSV
        osv = self.make_osv(eris)
        cput0 = log.timer('OSV domain', *cput1)
        timer['OSV domain'] = np.asarray(cput0) - np.asarray(cput1)

        # Pair prescreen
        pair_mask, epair_dist = self.screen_dist_pair(osv)
        self._update_pair_mask('Dist pair', ~pair_mask)
        self._update_pair_energy('Dist pair', epair_dist, ~pair_mask)
        cput1 = log.timer('Pair prescreen', *cput0)
        timer['Pair prescreen'] = np.asarray(cput1) - np.asarray(cput0)

        # Pair domain
        pair_domain = self._make_pair_domain(eris, osv, pair_mask)
        cput0 = log.timer('Pair domain', *cput1)
        timer['Pair domain'] = np.asarray(cput0) - np.asarray(cput1)

        return pair_domain

    def make_osv(self, eris, thresh_osv=None):
        log = logger.new_logger(self)
        if thresh_osv is None:
            thresh_osv = self.thresh_osv
        moe_lo = np.diag(self.foo)
        vir_energy = self.split_mo_energy()[2]
        osv = self.OSV({'thresh': thresh_osv}, eris, moe_lo, vir_energy)
        tools.summarize_domain(osv.nosv, log, 'OSV domain size')

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'osv/nosv', osv.nosv)

        return osv

    def _make_pair_domain(self, eris, osv, pair_mask):
        osv.get_ovlp(pair_mask=pair_mask)
        vir_energy = self.split_mo_energy()[2]
        pair_domain = self.PairDomain(eris, osv, vir_energy, pair_mask)
        self._update_pair_mask('Strong pair', pair_domain.pair_mask)

        return pair_domain

    def init_guess(self, pair_domain):
        t2 = pair_domain.new_array()
        K = pair_domain.K
        moe_osv = pair_domain.osv.e
        moe_lo = np.diag(self.foo)
        for i, j in pair_domain.loop_pair():
            t2[i, j] = K[i, j] / (moe_lo[i] + moe_lo[j] - (moe_osv[i][:, None] + moe_osv[j]))
        return t2

    def residual(self, t2, pair_domain, i, j, foo_mask):
        K, S = pair_domain.K, pair_domain.S
        foo = self.foo
        R = K[i, j].copy()
        for k in pair_domain.loop_k(i, j):
            if k != j and foo_mask[k, j]:
                R -= foo[k, j] * lib.dot(t2[i, k], S[k, j])
            if k != i and foo_mask[i, k]:
                R -= foo[i, k] * lib.dot(S[i, k], t2[k, j])
        return R

    def update_amp(self, t2, pair_domain):
        moe_lo = np.diag(self.foo)
        moe_osv = pair_domain.osv.e
        foo_mask = abs(self.foo) > self.thresh_foo
        t2new = t2.copy()
        for i, j in pair_domain.loop_pair():
            R = self.residual(t2new, pair_domain, i, j, foo_mask)
            t2new[i, j] = R / (moe_lo[i] + moe_lo[j] - (moe_osv[i][:, None] + moe_osv[j]))
        return t2new
